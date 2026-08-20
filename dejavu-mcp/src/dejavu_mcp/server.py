"""dejavu-mcp — Claude Code に既視感を与える MCP サーバー。

stdio 型なので、ポートも常駐プロセスも持たない。HTTP デーモンは不要。
（「呼ばれていない間も見ている」受動チャンネルを足すときは常駐が要るが、いまは無い）

本家 memory-mcp の memory.db には一切書かない。参照が要るときだけ読み取り専用で
ATTACH する。vision.db は別ファイル。上流のスキーマは触らない。

環境変数:
    DEJAVU_DB         vision.db の場所（既定：dejavu-mcp/ 直下）
    DEJAVU_MODEL_DIR  .tflite モデルの場所（既定：dejavu-mcp/models/）
    DEJAVU_MEMORY_DB  本家 memory.db の場所（既定：~/.claude/memories/memory.db）

※ MCP SDK は 2.x の `MCPServer` を使っている。本家 memory-mcp は 1.x の
  `Server` + `@server.list_tools()` 系で書かれているので、取り込む場合は
  どちらに揃えるかを決める必要がある。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import MCPServer

from .store import KINDS, VisionStore
from .vision import DINO_DIM, DINO_NAME, Vision

# モデルのロードは重い（DINOv2 で約340MB）。最初に使うときまで遅らせる
_vision: Vision | None = None


def _v() -> Vision:
    global _vision
    if _vision is None:
        _vision = Vision()
    return _vision


def _resolve(image_path: str) -> Path:
    p = Path(image_path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if not p.exists():
        raise ValueError(f"ファイルがありません: {p}")
    return p


def _phrase(f: dict, kind: str) -> str:
    """1つのチャンネルの答えを1行にする。強度が主、候補は従。"""
    if f["level"] == "初めて":
        return f"{kind}: 初めて"
    who = f.get("candidate")
    tail = f"候補は「{who}」" if who else f"名前は知らない（{f['cluster_id']}）"
    extra = ""
    if f["level"] == "強く見覚えがある":
        extra = f"・{f['seen_count']}回この群れで見ている"
    elif f["level"] == "むしろ見た気がする":
        extra = "・似ているものは無いのに、なぜか一つだけ浮いている"
    return (f"{kind}: {f['level']}。{tail}{extra}"
            f"  〈別人ラインとの差 {f['margin']:+.3f} / 鋭さ {f['sharpness']:+.3f}〉")


_MISSING = {"face": "顔が撮れていない（向きが横か、誤検出として捨てた）",
            "person": "人物が写っていない",
            "scene": "背景が取れていない"}


def _render(store: VisionStore, obs, header: str) -> str:
    lines = [header]
    for kind in KINDS:
        vec = obs.vectors.get(kind)
        lines.append("  " + (f"{kind}: {_MISSING[kind]}" if vec is None
                             else _phrase(store.familiarity(kind, vec), kind)))
    return "\n".join(lines)


app = MCPServer(
    name="dejavu",
    version="0.1.0",
    instructions=(
        "見たものを、名前が付く前から覚えておく。\n"
        "**これは顔認証ではない。**返るのは判定ではなく感覚で、"
        "「強く見覚えがある／見覚えがある／見た気がする／初めて／"
        "むしろ見た気がする（既視感）」の5段階に根拠の数値が添う。\n"
        "名前は『候補』であって断定ではない。実測では、間違った人を指しながら"
        "『自信がない』ことは正しく伝えていた例がある。"
        "**確信度は答えより先に届く。**弱い信号は弱いまま受け取り、"
        "他の材料（いま居るはずの人・直前の発話・時刻）と重みをつけて判断すること。"),
)


@app.tool()
def observe(image_path: str, camera: str | None = None) -> str:
    """画像を見て記録し、見覚えの強さを返す。見たものが覚えられる。

    Args:
        image_path: 画像ファイルのパス
        camera: どのカメラで撮ったか。省略可。views で「いつもと違うか」を見るときに要る
    """
    path = _resolve(image_path)
    obs = _v().observe(path)
    with VisionStore() as store:
        out = _render(store, obs, f"{path.name}（記録した）")
        captured = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        obs_id = store.add_observation(obs, DINO_NAME, DINO_DIM, camera_position=camera,
                                       timestamp=captured.isoformat(timespec="seconds"))
        for kind, vec in obs.vectors.items():
            store.assign_cluster(obs_id, kind, vec, DINO_NAME, DINO_DIM)
    return out


@app.tool()
def recall(image_path: str) -> str:
    """画像の見覚えだけを返す。**記録はしない。**

    「これ見たことある？」と確かめたいだけで、覚えさせたくないときに使う。

    Args:
        image_path: 画像ファイルのパス
    """
    path = _resolve(image_path)
    obs = _v().observe(path)
    with VisionStore() as store:
        return _render(store, obs, f"{path.name}（記録していない）")


@app.tool()
def name(cluster_id: str, label: str) -> str:
    """群れに名前を与える。ここで初めて「見覚え」が「誰か」になる。

    **何に名前を付けたのかを必ず返す。**取りこぼしを避けるためにしきい値を下げて
    あるので、別人が同じ群れに入っていることがある。返ってきた中身を見て、
    混ざっていたら付け直すこと。

    Args:
        cluster_id: observe / recall が返した群れのID
        label: 与える名前
    """
    with VisionStore() as store:
        members = store.set_label(cluster_id, label)
        if not members:
            return f"{cluster_id} という群れは見つからなかった"
        lines = [f"「{label}」と名前を付けた。その群れの中身は:"]
        lines += [f"  {Path(m).name}" for m in members]
        lines.append("★別人が混ざっていないか確かめること。混ざっていたら付け直す。")
        return "\n".join(lines)


@app.tool()
def views(camera: str, image_path: str | None = None) -> str:
    """そのカメラの「いつもの眺め」を数え、画像を渡せば「いつもと違うか」を返す。

    場所を当てる道具ではない（よく似た別の場所は区別できない）。同じカメラの
    過去と比べて外れているかを見る道具。判定の基準はそのカメラ自身の過去から
    作るので、しきい値を外から与えない。

    Args:
        camera: カメラ名（observe に渡したもの）
        image_path: 省略すると眺めの数だけ返す
    """
    with VisionStore() as store:
        views_ = store.scene_views(camera)
        lines = [f"カメラ「{camera}」が覚えている眺め: {len(views_)} 通り"]
        if image_path:
            path = _resolve(image_path)
            vec = _v().observe(path).vectors.get("scene")
            if vec is None:
                lines.append(f"{path.name}: 背景が取れなかった")
            else:
                r = store.novelty(camera, vec)
                if r is None:
                    lines.append(f"{path.name}: 過去が乏しくて判断できない")
                else:
                    sim, z, unusual = r
                    lines.append(f"{path.name}: {'★いつもと違う' if unusual else 'いつも通り'}"
                                 f"（いつもの眺めとの近さ {sim:.3f} / 外れ具合 z={z:+.2f}）")
        return "\n".join(lines)


def main() -> None:
    if sys.platform == "win32":
        sys.stderr.reconfigure(encoding="utf-8")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
