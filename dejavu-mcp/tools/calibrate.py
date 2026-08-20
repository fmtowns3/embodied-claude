"""この環境でのしきい値を、手元の素材から出す。

DINOv2 のコサイン類似度は別人ラインが高く、値の帯が素材とカメラに強く依存する。
他所で決めた定数（heishio 版の 0.35 / 0.45 / 0.80 など）を持ってきても機能しない。
移植のたびに、その場の素材で測り直す必要がある。そのための道具。

判定の基準は設計原則に合わせる:
    混在（別人を同じ群れに入れる）は絶対に避ける。過分割は許す。
したがって推奨値は「**混在ゼロを保てる最小のしきい値**」＝安全なまま群れが最も少なくなる点。

正解ラベルは vision.db の capture_path のファイル名接頭辞から取る
（`<名前>_<連番>.<拡張子>` という規約）。scene には場所の正解が無いので分布だけ出す。

使い方:
    .venv\\Scripts\\python.exe calibrate.py
"""

import random
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from dejavu_mcp.store import KINDS, VisionStore, decode

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 順序依存を潰すための試行回数。時刻順・ファイル名順に加えてこの回数だけ入れ替える
ORDER_TRIALS = 30


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# --scene-truth: ファイル名の接頭辞を「場所」の正解として scene も判定する
SCENE_TRUTH = "--scene-truth" in sys.argv


def load(store: VisionStore, kind: str):
    rows = store.conn.execute(
        """SELECT o.capture_path, o.timestamp, e.vector
           FROM observation_embeddings e JOIN observations o ON o.id = e.observation_id
           WHERE e.kind=? ORDER BY o.timestamp""", (kind,)).fetchall()
    out = []
    for r in rows:
        name = Path(r["capture_path"]).name
        out.append((name, name.rsplit("_", 1)[0],
                    datetime.fromisoformat(r["timestamp"]), decode(r["vector"])))
    return out


def greedy(items, threshold: float):
    centroids, counts, members = [], [], []
    for name, truth, _, vec in items:
        if centroids:
            sims = np.array([float(vec @ c) for c in centroids])
            b = int(sims.argmax())
            if sims[b] >= threshold:
                centroids[b] = (centroids[b] * counts[b] + vec)
                centroids[b] /= np.linalg.norm(centroids[b])
                counts[b] += 1
                members[b].append(truth)
                continue
        centroids.append(vec.copy())
        counts.append(1)
        members.append([truth])
    mixed = sum(1 for m in members if len(set(m)) > 1)
    return len(members), mixed


def main() -> int:
    from pathlib import Path as _P
    with VisionStore(_P(_opt('--db', 'vision.db'))) as store:
        for kind in KINDS:
            items = load(store, kind)
            if len(items) < 2:
                print(f"=== {kind} ===  データが {len(items)} 件しかありません\n")
                continue
            people = sorted({t for _, t, _, _ in items})
            print(f"=== {kind} ===  {len(items)} 件 / ラベル {len(people)} 種")

            same, diff = [], []
            same_gap, diff_gap = [], []
            for (n1, t1, d1, v1), (n2, t2, d2, v2) in combinations(items, 2):
                sim = float(v1 @ v2)
                gap = abs((d1 - d2).total_seconds())
                (same if t1 == t2 else diff).append(sim)
                (same_gap if t1 == t2 else diff_gap).append(gap)

            if kind == "scene" and not SCENE_TRUTH:
                # 場所の正解が無いので分布だけ。ラベルは人物であって場所ではない
                allsim = same + diff
                print(f"  類似度の分布のみ（場所の正解ラベルが無いため判定はしない）")
                print(f"    min {min(allsim):.4f} / 平均 {np.mean(allsim):.4f} / max {max(allsim):.4f}")
                print()
                continue

            L = "場所" if kind == "scene" else "人物"
            print(f"  同一{L}ペア {len(same):>3} 組: min {min(same):.4f} / 平均 {np.mean(same):.4f}")
            print(f"  別{L}ペア   {len(diff):>3} 組: 平均 {np.mean(diff):.4f} / max {max(diff):.4f}")
            headroom = min(same) - max(diff)
            if headroom > 0:
                print(f"  → 単純な閾値で完全分離できる（余裕 {headroom:+.4f}）")
            else:
                print(f"  → 完全分離する閾値は無い（重なり {headroom:+.4f}）。過分割で安全側に倒す")

            # ★貪欲な逐次クラスタリングは観測の到着順で結果が変わる。
            #   1つの順序だけで測ると、安全に見えて安全でない値が出る。
            #   実運用の順序（時刻順）に加えて、順序を無作為に入れ替えて検証する。
            orders = [items, sorted(items, key=lambda x: x[0])]
            for seed in range(ORDER_TRIALS):
                shuffled = items[:]
                random.Random(seed).shuffle(shuffled)
                orders.append(shuffled)

            best = None
            print(f"\n  {'しきい値':>8} {'群れ':>5} {'混在(時刻順)':>12} {'混在(全順序の最大)':>18}")
            for th in np.arange(0.60, 0.98, 0.01):
                n, mixed = greedy(items, float(th))
                worst = max(greedy(o, float(th))[1] for o in orders)
                if worst == 0 and best is None:
                    best = (float(th), n)
                if abs(th * 100 % 5) < 1e-6:
                    tail = "  ← どの順序でも混在ゼロの最小" if best and abs(best[0] - th) < 1e-9 else ""
                    print(f"  {th:>8.2f} {n:>5} {mixed:>12} {worst:>18}{tail}")
            if best:
                th, n = best
                print(f"\n  ★推奨しきい値 {th:.2f}（{n} 群れ・{len(orders)} 通りの順序すべてで混在ゼロ。"
                      f"ラベル {len(people)} 種に対し {n - len(people):+d} の過分割）")
            else:
                print("\n  ★どの順序でも混在ゼロにできるしきい値が範囲内に無い")

            # 時間の門が使えるか
            if same_gap and diff_gap:
                print(f"\n  時間の門: 同一人物の最小間隔 {min(same_gap):.0f}秒 / "
                      f"別人の最小間隔 {min(diff_gap):.0f}秒")
                if min(diff_gap) > min(same_gap):
                    print(f"    → {min(same_gap):.0f}〜{min(diff_gap):.0f}秒 の間に門を置ける"
                          f"（余裕 {min(diff_gap) - min(same_gap):.0f}秒）")
                else:
                    print("    → 別人のほうが近い時刻に現れており、時間だけでは切れない")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
