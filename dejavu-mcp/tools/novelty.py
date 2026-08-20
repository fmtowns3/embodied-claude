"""scene を「どこか」ではなく「いつもと違うか」の検出器として測る。

場所当てとしての scene は失敗した。似た別の場所（同型の車内）が 0.95 で並び、
混在を避けようとすると全部バラバラになるまでしきい値を上げるしかなかった。

しかし実運用では、各拠点にカメラが1台ずつある。フレームがどのカメラから来たかは
最初から分かっているので、**場所を当てる必要がない**。scene に期待すべきは
「このカメラの、いつもの眺めと違うか」のほうだった。

この読み替えには大きな利点がある。**絶対しきい値が要らない。**
判定の基準を「そのカメラ自身の過去」から作れるので、環境ごとの較正が不要になる。

測り方（leave-one-out）:
    あるカメラの過去 = その場所の写真から1枚抜いた残り
    その1枚を「いつも通りの新着」として採点する
    他の場所の写真を「いつもと違う新着」として採点する
    採点は、過去のどれか1枚への最大類似度（重心ではなく max）。
    カメラが首を振れば構図は複数あるので、重心は意味を失う。

    さらに、過去の内部ばらつきで正規化する:
        z = (その点数 − 過去同士の点数の平均) / 標準偏差
    これで「そのカメラにとって、どれくらい外れているか」になる。

使い方:
    .venv\\Scripts\\python.exe novelty.py [--db scene.db]
"""

import sys
from pathlib import Path

import numpy as np

from dejavu_mcp.store import VisionStore, decode

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

MIN_HISTORY = 4  # 過去がこれ未満のカメラは扱わない


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def load_scene(db: Path):
    with VisionStore(db) as s:
        rows = s.conn.execute(
            """SELECT o.capture_path, e.vector FROM observation_embeddings e
               JOIN observations o ON o.id = e.observation_id
               WHERE e.kind='scene' ORDER BY o.timestamp""").fetchall()
    out = {}
    for r in rows:
        name = Path(r["capture_path"]).name
        out.setdefault(name.rsplit("_", 1)[0], []).append((name, decode(r["vector"])))
    return out


def main() -> int:
    groups = load_scene(Path(_opt("--db", "scene.db")))
    cameras = {k: v for k, v in groups.items() if len(v) >= MIN_HISTORY}
    others = {k: v for k, v in groups.items()}
    print(f"場所: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items())))
    print(f"カメラとして扱えるもの（過去 {MIN_HISTORY} 枚以上）: {sorted(cameras)}\n")

    for cam, members in sorted(cameras.items()):
        vecs = [v for _, v in members]
        names = [n for n, _ in members]

        # 過去同士の点数（各枚を他の過去と比べた最大類似度）
        internal = []
        for i in range(len(vecs)):
            rest = [vecs[j] for j in range(len(vecs)) if j != i]
            internal.append(max(float(vecs[i] @ r) for r in rest))
        mu, sd = float(np.mean(internal)), float(np.std(internal))

        print(f"=== カメラ {cam} （過去 {len(vecs)} 枚）===")
        print(f"  過去同士の最大類似度: 平均 {mu:.4f} / 標準偏差 {sd:.4f} / 最小 {min(internal):.4f}")

        # 「いつも通りの新着」= leave-one-out
        usual = []
        for i in range(len(vecs)):
            rest = [vecs[j] for j in range(len(vecs)) if j != i]
            s = max(float(vecs[i] @ r) for r in rest)
            usual.append((s, (s - mu) / sd if sd else 0.0, names[i]))

        # 「いつもと違う新着」= 他の場所すべて
        strange = []
        for other, om in others.items():
            if other == cam:
                continue
            for n, v in om:
                s = max(float(v @ r) for r in vecs)
                strange.append((s, (s - mu) / sd if sd else 0.0, n))

        u_z = [z for _, z, _ in usual]
        s_z = [z for _, z, _ in strange]
        print(f"  いつも通り  {len(usual):>2} 件: z 平均 {np.mean(u_z):+.2f} / 最小 {min(u_z):+.2f}")
        print(f"  いつもと違う {len(strange):>2} 件: z 平均 {np.mean(s_z):+.2f} / 最大 {max(s_z):+.2f}")
        margin = min(u_z) - max(s_z)
        if margin > 0:
            print(f"  ★分離できている（z のマージン {margin:+.2f}）")
            print(f"    しきい値は z = {max(s_z):.2f} 〜 {min(u_z):.2f} の間。"
                  f"このカメラ自身の過去から決まるので、外から与える必要がない")
        else:
            print(f"  ★重なっている（z のマージン {margin:+.2f}）")

        worst_u = min(usual)
        worst_s = max(strange)
        print(f"    いつも通りで最も外れた: {worst_u[2]} (sim {worst_u[0]:.4f}, z {worst_u[1]:+.2f})")
        print(f"    いつもと違うで最も紛れた: {worst_s[2]} (sim {worst_s[0]:.4f}, z {worst_s[1]:+.2f})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
