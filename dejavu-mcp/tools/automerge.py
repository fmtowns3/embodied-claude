"""時間の門をくぐった統合候補だけを、自動で畳む。

重心の距離だけで畳むと、正しい統合より上に誤った統合が来る（README 参照）。
そこへ「人は瞬間移動しない」という外からの信号を足す。連続して観測された
2つの群れなら、同じ人である公算が高い。

畳んだあとにもう一度候補を探す。統合で重心が動くと、新しく門をくぐる組が
出ることがあるため。変化がなくなるまで繰り返す。

使い方:
    .venv\\Scripts\\python.exe automerge.py            # 既定 60秒 / 類似度 0.70
    .venv\\Scripts\\python.exe automerge.py 30 0.75    # 秒数と類似度を指定
"""

import sys
from collections import Counter
from pathlib import Path

from dejavu_mcp.store import KINDS, VisionStore

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

MAX_GAP = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
SIM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70


def report(store: VisionStore, kind: str) -> None:
    rows = store.clusters(kind)
    mixed = 0
    print(f"  {kind}: {len(rows)} 群れ")
    for row in rows:
        names = [Path(m["capture_path"]).name for m in store.members_of(row["id"])]
        truths = Counter(n.rsplit("_", 1)[0] for n in names)
        bad = kind != "scene" and len(truths) > 1
        mixed += bad
        mark = "  ★混在" if bad else ""
        print(f"    n={row['member_count']}  {', '.join(names)}{mark}")
    if kind != "scene":
        print(f"    → 混在 {mixed} 件")


def main() -> int:
    print(f"時間の門: {MAX_GAP:.0f} 秒以内 / 類似度: {SIM} 以上\n")
    with VisionStore() as store:
        for kind in KINDS:
            print(f"=== {kind} ===")
            print("  【統合前】")
            report(store, kind)

            merged_total = 0
            for round_no in range(1, 11):
                pairs = store.suggest_merges(kind, SIM, max_gap_seconds=MAX_GAP)
                if not pairs:
                    break
                a, b, sim, gap = pairs[0]  # いちばん近い組から1つずつ畳む
                na = [Path(m["capture_path"]).name for m in store.members_of(a)]
                nb = [Path(m["capture_path"]).name for m in store.members_of(b)]
                print(f"  第{round_no}回: 類似度 {sim:.4f} / 間隔 {gap:.0f}秒")
                print(f"    {na} ← {nb}")
                store.merge_clusters(a, b)
                merged_total += 1

            print(f"  【統合後】{merged_total} 回畳んだ")
            report(store, kind)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
