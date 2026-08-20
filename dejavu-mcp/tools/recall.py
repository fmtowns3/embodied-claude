"""「この画像、見覚えある？」に答える。

**これは顔認証ではない。**「これは誰か」を断定する装置ではなく、
「見覚えがあるかどうか」の感覚を返す。名前は候補として添えるだけ。

なぜそうするか（2026-08-19 の盲検で分かったこと）:
    ある画像に対し、システムは間違った人を指しながら、
    「自信がない」ことは正しく伝えていた。
    名前だけを返す設計なら、私は誤った名前を復唱していた。
    **確信度は答えより先に届く。** だから強度が主で、候補は従。

段階は5つ。4つは強度の階段で、最後の1つは階段の外にある:

    強く見覚えがある      よく見かける人物・いつもの人物
    見覚えがある          たぶん一度は見たことがある
    見た気がする          見たことあるかも？
    初めて                何も引っかからない
    むしろ見た気がする     ★既視感。似ているものは無いのに、なぜか一つだけ浮いている

最後のものが「既視感」の本来の意味に近い——実際には初めてなのに見た気がする錯覚。
強度は別人ラインを下回っているのに鋭さだけが立っているとき、この状態になる。
実測では、DB に居ない人物に対してこれが2件出た。**どちらも指す先は誤り。**
錯覚であることまで再現している。

3つのチャンネルは別々のことを知っている:
    face   … 誰か。日をまたいでも生きるが、顔が撮れないと黙る
    person … 誰か。顔が見えなくても答えるが、着替えると別人になる
    scene  … どこか。人物とは別の問い
畳まずに3つとも返す。どのチャンネルがそう言ったかが分からない再認は使えない。

使い方:
    .venv/Scripts/python.exe recall.py                     # images/ 全部
    .venv/Scripts/python.exe recall.py path/to/photo.jpg  # 1枚
"""

import sys
from pathlib import Path

from dejavu_mcp.store import VisionStore
from dejavu_mcp.vision import Vision

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def describe(store, kind: str, vec, missing_reason: str | None = None) -> str:
    """1つのチャンネルの答えを1行にする。強度が主、候補は従。"""
    if vec is None:
        return f"{kind}: {missing_reason or 'ベクトルを作れなかった'}"
    f = store.familiarity(kind, vec)
    level = f["level"]
    if level == "初めて":
        return f"{kind}: 初めて"
    who = f.get("candidate")
    tail = f"候補は「{who}」" if who else f"名前は知らない（{f['cluster_id']}）"
    extra = ""
    if level == "強く見覚えがある":
        extra = f"・{f['seen_count']}回この群れで見ている"
    elif level == "むしろ見た気がする":
        extra = "・似ているものは無いのに、なぜか一つだけ浮いている"
    return (f"{kind}: {level}。{tail}{extra}"
            f"  〈別人ラインとの差 {f['margin']:+.3f} / 鋭さ {f['sharpness']:+.3f}〉")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = [Path(a) for a in args] if args else sorted(
        (p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS),
        key=lambda p: p.stat().st_mtime)

    with Vision() as vision, VisionStore() as store:
        if not store.clusters():
            print("vision.db に群れがありません。先に ingest.py を実行してください。")
            return 1
        for path in targets:
            if not path.exists():
                print(f"{path}: ファイルがありません")
                continue
            obs = vision.observe(path)
            print(f"--- {path.name} ---")
            print("  " + describe(store, "face", obs.vectors.get("face"),
                                  "顔が撮れていない（人物の向きか、誤検出として捨てた）"))
            print("  " + describe(store, "person", obs.vectors.get("person"),
                                  "人物が写っていない"))
            print("  " + describe(store, "scene", obs.vectors.get("scene")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
