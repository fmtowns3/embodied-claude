"""既視感としての性能を測る。「混在ゼロ」ではなく「見た人を見たと言えた率」。

顔認証の指標（別人を混ぜない）を捨て、既視感の指標に入れ替える。
致命的な失敗は「別人を同一と判定」ではなく「**見た人を初めてと言う**」ほう。

2種類の抜き方で測る:
    leave-one-image-out  … その人の他の写真は DB にある
                           → 「見覚えがある」と言えるべき
    leave-one-person-out … その人の写真が DB に1枚も無い
                           → 「初めて」と言えるべき

出力は生の数字。段階の境目をここから決めるので、先に言葉にしない。

使い方:
    .venv\\Scripts\\python.exe evaluate.py [--threshold 0.80]
"""

import sys
from pathlib import Path

import numpy as np

from dejavu_mcp.vision import DINO_DIM, DINO_NAME, Vision, unit

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _opt(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def greedy(items, threshold):
    """items: [(person, vec)] → [(centroid, count, [person...])]"""
    out = []
    for person, vec in items:
        if out:
            sims = np.array([float(vec @ c) for c, _, _ in out])
            b = int(sims.argmax())
            if sims[b] >= threshold:
                c, n, ps = out[b]
                c = unit(c * n + vec)
                out[b] = (c, n + 1, ps + [person])
                continue
        out.append((vec.copy(), 1, [person]))
    return out


def probe(clusters, vec):
    """最近傍の類似度・その群れの枚数・その群れの多数派・2位との差・別人ライン。"""
    sims = sorted(((float(vec @ c), n, ps) for c, n, ps in clusters), reverse=True)
    top_sim, top_n, top_ps = sims[0]
    second = sims[1][0] if len(sims) > 1 else 0.0
    from collections import Counter
    majority = Counter(top_ps).most_common(1)[0][0]
    return top_sim, top_n, majority, top_sim - second


def floor_of(clusters):
    """別人ライン＝多数派ラベルが違う群れ同士の最大類似度。「別人でもこれくらいは似る」高さ。"""
    from collections import Counter
    best = 0.0
    for i, (ca, _, pa) in enumerate(clusters):
        ma = Counter(pa).most_common(1)[0][0]
        for cb, _, pb in clusters[i + 1:]:
            mb = Counter(pb).most_common(1)[0][0]
            if ma != mb:
                best = max(best, float(ca @ cb))
    return best


def main() -> int:
    th = _opt("--threshold", 0.80)
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "face"

    paths = sorted((p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS),
                   key=lambda p: p.stat().st_mtime)
    with Vision() as v:
        data = []
        for p in paths:
            obs = v.observe(p)
            vec = obs.vectors.get(kind)
            if vec is not None:
                data.append((p.name, p.stem.rsplit("_", 1)[0], vec))

    people = sorted({t for _, t, _ in data})
    print(f"kind={kind}  しきい値={th}  {len(data)} 枚 / {len(people)} 人\n")

    print("=== leave-one-image-out（その人の他の写真は DB にある → 見覚えがあるべき）===")
    print(f"{'file':<20} {'正解':<11} {'類似度':>8} {'別人ライン':>10} {'ラインとの差':>12} {'2位差':>7} {'枚数':>4} {'1位の多数派':<11} 一致")
    seen_rows = []
    for i, (name, truth, vec) in enumerate(data):
        rest = [(t, v) for j, (_, t, v) in enumerate(data) if j != i]
        cl = greedy(rest, th)
        fl = floor_of(cl)
        sim, n, maj, gap = probe(cl, vec)
        seen_rows.append((name, truth, sim, fl, sim - fl, gap, n, maj))
        print(f"{name:<20} {truth:<11} {sim:>7.4f} {fl:>7.4f} {sim - fl:>+8.4f} {gap:>+7.4f} {n:>4} {maj:<11} {'○' if maj == truth else '×'}")

    print("\n=== leave-one-person-out（その人は DB に1枚も無い → 初めてと言えるべき）===")
    print(f"{'file':<20} {'正解':<11} {'類似度':>8} {'別人ライン':>10} {'ラインとの差':>12} {'2位差':>7} {'1位の多数派':<11}")
    unseen_rows = []
    for person in people:
        rest = [(t, v) for _, t, v in data if t != person]
        cl = greedy(rest, th)
        fl = floor_of(cl)
        for name, truth, vec in data:
            if truth != person:
                continue
            sim, n, maj, gap = probe(cl, vec)
            unseen_rows.append((name, truth, sim, fl, sim - fl, gap, n, maj))
            print(f"{name:<20} {truth:<11} {sim:>7.4f} {fl:>7.4f} {sim - fl:>+8.4f} {gap:>+7.4f} {maj:<11}")

    print("\n=== 分布のまとめ（段階の境目を決める材料）===")
    for label, rows in (("見た人（ラインとの差）", seen_rows), ("知らない人（ラインとの差）", unseen_rows)):
        d = [r[4] for r in rows]
        g = [r[5] for r in rows]
        print(f"  {label:<24} min {min(d):+.4f} / 中央 {np.median(d):+.4f} / max {max(d):+.4f}"
              f"   2位差 中央 {np.median(g):+.4f} / max {max(g):+.4f}")
    hit = sum(1 for r in seen_rows if r[7] == r[1])
    print(f"\n  1位の多数派が正解と一致: {hit}/{len(seen_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
