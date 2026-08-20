"""images/ を vision.db に投入し、名前を与えずに群れを作る。

やっていることは cluster.py と同じだが、結果をメモリ上ではなく DB に残す。
ラベル欄は空のまま。名前は後から `set_label` で与える。
「見覚え」が先にあって、名前が後から乗る、という順序を構造で表している。

使い方:
    .venv\\Scripts\\python.exe ingest.py            # 追記
    .venv\\Scripts\\python.exe ingest.py --reset    # vision.db を作り直してから
"""

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dejavu_mcp.store import CLUSTER_THRESHOLD, DEFAULT_DB, KINDS, VisionStore
from dejavu_mcp.vision import DINO_DIM, DINO_NAME, Vision

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


IMAGE_DIR = Path(_opt("--images", str(Path(__file__).resolve().parents[1] / "images")))
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    db = Path(_opt("--db", str(DEFAULT_DB)))
    if "--reset" in sys.argv and db.exists():
        db.unlink()
        print(f"{db.name} を作り直します")

    # ★実運用では観測は時刻順に届く。貪欲クラスタリングは到着順で結果が変わるので、
    #   投入もその順序に揃える（ファイル名順で回すと安全に見えて安全でない値が通る）。
    paths = sorted((p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS),
                   key=lambda p: p.stat().st_mtime)
    if not paths:
        print("images/ に画像がありません")
        return 1

    print(f"素材: {IMAGE_DIR}  /  DB: {db}")
    print(f"しきい値: {CLUSTER_THRESHOLD}")
    print(f"モデル: {DINO_NAME} ({DINO_DIM}d)\n")

    with Vision() as vision, VisionStore(db) as store:
        print(f"device: {vision.device}\n")
        for path in paths:
            obs = vision.observe(path)
            # ★撮影時刻。ここでは素材のファイル更新時刻で代用している。
            #   本来はカメラが撮った時刻。投入時刻を入れると時間的近接が測れない。
            captured = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            # --camera-from-prefix: ファイル名の接頭辞をカメラ名として記録する。
            #   新奇性の判定は「そのカメラの過去」との比較なので、どのカメラかが要る。
            camera = path.stem.rsplit('_', 1)[0] if '--camera-from-prefix' in sys.argv else None
            obs_id = store.add_observation(obs, DINO_NAME, DINO_DIM,
                                           camera_position=camera,
                                           timestamp=captured.isoformat(timespec='seconds'))
            marks = []
            for kind in KINDS:
                vec = obs.vectors.get(kind)
                if vec is None:
                    marks.append(f"{kind}=なし")
                    continue
                cid, is_new, sim = store.assign_cluster(obs_id, kind, vec, DINO_NAME, DINO_DIM)
                marks.append(f"{kind}={'新規' if is_new else f'既存({sim:.3f})'}")
            print(f"  {path.name:<20} 人物率{obs.person_ratio:>6.3f}  " + "  ".join(marks))

        print()
        for kind in KINDS:
            rows = store.clusters(kind)
            print(f"=== {kind} の群れ: {len(rows)} 個（しきい値 {CLUSTER_THRESHOLD[kind]}）===")
            for row in rows:
                members = store.members_of(row["id"])
                names = [Path(m["capture_path"]).name for m in members]
                # 検証用：ファイル名の接頭辞が実際の人物。
                # ただし scene は「同じ場所」の群れなので、人物が混ざるのは正常。
                # 混在を問題として見るのは person / face だけ。
                truths = Counter(n.rsplit("_", 1)[0] for n in names)
                mixed = "  ★混在" if (kind != "scene" and len(truths) > 1) else ""
                label = row["label"] or "（名前なし）"
                print(f"  {row['id']}  n={row['member_count']}  {label}{mixed}")
                print(f"      {', '.join(names)}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
