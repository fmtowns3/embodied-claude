"""保存層：vision.db への読み書きと、逐次クラスタリング。

本家 memory-mcp の memory.db には一切触らない。参照が要るときは ATTACH する。
memory.db は journal_mode=wal なので、attach をまたぐ書き込みの原子性は
保証されない。★書くのは vision.db だけ、memory.db は読むだけ。

しきい値は用途で決まる。ここは既視感（déjà vu）の器なので、
「別人を混ぜない」ではなく「**見た人を初めてと言わない**」を優先する:
  face   0.80  … 取りこぼしゼロ優先。leave-one-image-out 16枚すべてが「初めて」を免れる
  person 0.75  … 同上
  scene  0.70  … 「そのカメラのいつもの眺め」のレパートリーを作るための値

★以前 face を 0.87 にしていたのは、指標が「混在ゼロ」だったから。それは顔認証の
要件であって既視感の要件ではない。0.87 では 6人に11群れまで割れ、何度も見た人を
「初めて見る」と答える装置になっていた。目的関数を戻して下げた。

なお貪欲な逐次クラスタリングは観測の到着順で結果が変わる。しきい値を検証する
ときは実運用の順序（時刻順）と無作為な順序の両方で回すこと（`calibrate.py`）。
DINOv2 のコサイン類似度は別人ラインが高いので、他所の絶対値は持ってきても機能しない。
だから段階の判定は生の類似度ではなく、別人ライン（別人でもこれくらいは似る、という
高さ）との差で見る。
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SCHEMA = ROOT / "schema.sql"
# MCP サーバーは任意の作業ディレクトリから起動されるので、DB も環境変数で決める。
# 既定はこのコンポーネント直下（dejavu-mcp/vision.db）。
DEFAULT_DB = Path(os.environ.get("DEJAVU_DB",
                                 Path(__file__).resolve().parents[2] / "vision.db"))
MEMORY_DB = Path(os.environ.get("DEJAVU_MEMORY_DB",
                                Path.home() / ".claude" / "memories" / "memory.db"))

# scene だけ用途が違う。face/person は「別人を混ぜない」ための値だが、
# scene は「そのカメラのいつもの眺めのレパートリー」を作るための値。
# 場所当てとしての scene は成立しなかった（似た別拠点が 0.95 で並ぶ）ので、
# 混在を恐れる必要がない。実測では 0.70 まで畳んでも新奇性の検出力は保たれ、
# 枚数の少ないカメラではむしろ改善した（冗長な眺めが内部分布を歪めなくなる）。
# ★2026-08-20 目的関数を「顔認証」から「既視感」に戻した。
# 以前の face 0.87 は「混在ゼロ」＝別人を混ぜないための値だったが、それは
# 顔認証の要件。既視感の致命的な失敗は「見た人を初めてと言う」ほうなので、
# 取りこぼしゼロを優先して下げた。0.80 で leave-one-image-out 16枚すべてが
# 「初めて」を免れる（誤検出は知らない人 16枚中 8 件。既視感としては許容）。
CLUSTER_THRESHOLD = {"face": 0.80, "person": 0.75, "scene": 0.70}

# 既視感の段階。境目は 16 枚から出した暫定値で、環境が変われば calibrate.py で出し直す。
#   強度＝別人ラインとの差。別人ラインは「別人でもこれくらいは似る」高さ＝ラベルの違う
#         群れ同士の最大類似度。生の絶対値ではなくこの差で見るので、多少は移植できる
#   鋭さ＝1位と2位の差。強度が低いのに鋭いときが「むしろ見た気がする（既視感）」
FAMILIAR_MARGIN = +0.05   # これ以上なら「見覚えがある」
OFTEN_SEEN_N = 3          # 群れの枚数がこれ以上なら「よく見かける人物」
MAYBE_MARGIN = -0.035     # これ以上なら「見た気がする」（見た人の最小値に合わせた）
DEJAVU_SHARPNESS = +0.07  # 強度が低いのにこれ以上鋭ければ「むしろ見た気がする」
KINDS = ("scene", "person", "face")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class VisionStore:
    def __init__(self, path: Path = DEFAULT_DB):
        self.path = path
        # uri=True にしないと ATTACH に file:...?mode=ro を渡せず、
        # 読み取り専用で繋ぐ手段が無くなる（先頭が file: でなければ通常のパス扱い）
        self.conn = sqlite3.connect(str(path), uri=True)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VisionStore":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.commit()
        self.conn.close()

    # --- 書き込み ---

    def add_observation(self, obs, model: str, dim: int,
                        camera_position: str | None = None,
                        timestamp: str | None = None) -> str:
        """timestamp は【撮影時刻】。省略すると現在時刻になるが、それは投入時刻であって
        撮影時刻ではない。時間的近接で群れを畳むには撮影時刻が要る。"""
        obs_id = uuid.uuid4().hex
        self.conn.execute(
            """INSERT INTO observations
               (id, capture_path, timestamp, camera_position, person_ratio, face_confidence)
               VALUES (?,?,?,?,?,?)""",
            (obs_id, obs.capture_path, timestamp or now(), camera_position,
             obs.person_ratio, obs.face_confidence))
        for kind, vec in obs.vectors.items():
            self.conn.execute(
                "INSERT INTO observation_embeddings VALUES (?,?,?,?,?)",
                (obs_id, kind, model, dim, encode(vec)))
        return obs_id

    def assign_cluster(self, obs_id: str, kind: str, vec: np.ndarray,
                       model: str, dim: int) -> tuple[str, bool, float]:
        """既存クラスタに入れるか、新しく作る。(cluster_id, 新規か, 類似度)"""
        rows = self.conn.execute(
            "SELECT id, centroid, member_count FROM clusters WHERE kind=? AND model=?",
            (kind, model)).fetchall()

        best_id, best_sim = None, -1.0
        for row in rows:
            sim = float(vec @ decode(row["centroid"]))
            if sim > best_sim:
                best_id, best_sim = row["id"], sim

        threshold = CLUSTER_THRESHOLD[kind]
        if best_id is not None and best_sim >= threshold:
            row = self.conn.execute(
                "SELECT centroid, member_count FROM clusters WHERE id=?", (best_id,)).fetchone()
            n = row["member_count"]
            merged = decode(row["centroid"]) * n + vec
            merged = merged / np.linalg.norm(merged)
            self.conn.execute(
                "UPDATE clusters SET centroid=?, member_count=?, updated_at=? WHERE id=?",
                (encode(merged), n + 1, now(), best_id))
            self._link(best_id, obs_id, best_sim)
            return best_id, False, best_sim

        cluster_id = f"{kind[:3]}-{uuid.uuid4().hex[:12]}"
        stamp = now()
        self.conn.execute(
            """INSERT INTO clusters
               (id, kind, model, dim, centroid, member_count, label, freshness, created_at, updated_at)
               VALUES (?,?,?,?,?,?,'',1.0,?,?)""",
            (cluster_id, kind, model, dim, encode(vec), 1, stamp, stamp))
        self._link(cluster_id, obs_id, 1.0)
        return cluster_id, True, best_sim

    def _link(self, cluster_id: str, obs_id: str, similarity: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cluster_members VALUES (?,?,?,?)",
            (cluster_id, obs_id, max(-1.0, min(1.0, similarity)), now()))

    def merge_clusters(self, keep_id: str, drop_id: str) -> None:
        """過分割を回収する。重心は件数で重み付けして混ぜ、メンバーを移す。

        混在（別人を同じ群れに入れる）は取り返しがつかないが、過分割は
        こうして後から畳める。しきい値を高めに置く根拠がこれ。
        """
        keep = self.conn.execute("SELECT * FROM clusters WHERE id=?", (keep_id,)).fetchone()
        drop = self.conn.execute("SELECT * FROM clusters WHERE id=?", (drop_id,)).fetchone()
        if keep is None or drop is None or keep["kind"] != drop["kind"]:
            raise ValueError("統合できない組み合わせ")

        merged = decode(keep["centroid"]) * keep["member_count"] \
            + decode(drop["centroid"]) * drop["member_count"]
        merged = merged / np.linalg.norm(merged)
        label = keep["label"] or drop["label"]
        self.conn.execute(
            "UPDATE clusters SET centroid=?, member_count=?, label=?, updated_at=? WHERE id=?",
            (encode(merged), keep["member_count"] + drop["member_count"], label, now(), keep_id))
        self.conn.execute(
            "UPDATE OR REPLACE cluster_members SET cluster_id=? WHERE cluster_id=?",
            (keep_id, drop_id))
        self.conn.execute("DELETE FROM clusters WHERE id=?", (drop_id,))

    def cluster_times(self, cluster_id: str) -> list[datetime]:
        rows = self.conn.execute(
            """SELECT o.timestamp FROM cluster_members m
               JOIN observations o ON o.id = m.observation_id
               WHERE m.cluster_id=?""", (cluster_id,)).fetchall()
        return [datetime.fromisoformat(r["timestamp"]) for r in rows]

    def min_gap_seconds(self, a_id: str, b_id: str) -> float:
        """2つの群れの、いちばん近いメンバー同士の時間差（秒）。"""
        ta, tb = self.cluster_times(a_id), self.cluster_times(b_id)
        if not ta or not tb:
            return float("inf")
        return min(abs((x - y).total_seconds()) for x in ta for y in tb)

    def suggest_merges(self, kind: str, threshold: float,
                       max_gap_seconds: float | None = None
                       ) -> list[tuple[str, str, float, float]]:
        """重心が近い群れの組を挙げる。決めるのは人。

        max_gap_seconds を渡すと、いちばん近いメンバー同士がその秒数以内に
        観測された組だけを残す。人は瞬間移動しないので、連続して撮れた像は
        同じ人である公算が高い——という外からの信号を、距離に足す。
        """
        rows = self.clusters(kind)
        pairs = []
        for i, a in enumerate(rows):
            va = decode(a["centroid"])
            for b in rows[i + 1:]:
                sim = float(va @ decode(b["centroid"]))
                if sim < threshold:
                    continue
                gap = self.min_gap_seconds(a["id"], b["id"])
                if max_gap_seconds is not None and gap > max_gap_seconds:
                    continue
                pairs.append((a["id"], b["id"], sim, gap))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def set_label(self, cluster_id: str, label: str) -> list[str]:
        """名前を後から与える。ここで初めて「見覚え」が「誰か」になる。

        ★何に名前を付けたのかを返す。取りこぼしゼロを優先してしきい値を下げた結果、
        別人が同じ群れに入ることがある。混在の自動検出は成立しなかった（枚数と
        区別できない）ので、**名前を付ける瞬間に人が中身を見る**のが唯一の防波堤。
        """
        self.conn.execute("UPDATE clusters SET label=?, updated_at=? WHERE id=?",
                          (label, now(), cluster_id))
        return [m["capture_path"] for m in self.members_of(cluster_id)]

    # --- 読み出し ---

    def clusters(self, kind: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM clusters"
        args: tuple = ()
        if kind:
            sql += " WHERE kind=?"
            args = (kind,)
        return self.conn.execute(sql + " ORDER BY kind, member_count DESC", args).fetchall()

    def members_of(self, cluster_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT o.capture_path, m.similarity
               FROM cluster_members m JOIN observations o ON o.id = m.observation_id
               WHERE m.cluster_id=? ORDER BY m.added_at""", (cluster_id,)).fetchall()

    # --- 既視感（見覚えの強さ）---

    def familiarity(self, kind: str, vec: np.ndarray) -> dict:
        """見覚えの強さを言葉で返す。名前は「候補」であって断定しない。

        ★2026-08-19 の盲検で分かったこと：システムは間違った人を指しながら、
        「自信がない」ことは正しく伝えていた。名前だけを返す設計なら誤答を
        復唱していた。**確信度は答えより先に届く。** だから強度が主で、候補は従。
        """
        clusters = self.clusters(kind)
        if not clusters:
            return {"level": "初めて", "reason": "群れがまだ無い"}

        scored = sorted(((float(vec @ decode(c["centroid"])), c) for c in clusters),
                        key=lambda x: x[0], reverse=True)
        top_sim, top = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        sharpness = top_sim - second

        # 別人ライン＝ラベルの違う群れ同士の最大類似度。「別人でもこれくらいは似る」
        floor = 0.0
        for i, (_, a) in enumerate(scored):
            for _, b in scored[i + 1:]:
                if a["label"] and b["label"] and a["label"] == b["label"]:
                    continue
                floor = max(floor, float(decode(a["centroid"]) @ decode(b["centroid"])))
        margin = top_sim - floor

        if margin >= FAMILIAR_MARGIN and top["member_count"] >= OFTEN_SEEN_N:
            level = "強く見覚えがある"
        elif margin >= FAMILIAR_MARGIN:
            level = "見覚えがある"
        elif margin >= MAYBE_MARGIN:
            level = "見た気がする"
        elif sharpness >= DEJAVU_SHARPNESS:
            level = "むしろ見た気がする"
        else:
            level = "初めて"

        return {"level": level, "candidate": top["label"] or None,
                "cluster_id": top["id"], "seen_count": top["member_count"],
                "similarity": round(top_sim, 4), "margin": round(margin, 4),
                "sharpness": round(sharpness, 4)}

    def cohesion(self, cluster_id: str) -> float | None:
        """群れのまとまり＝メンバーが重心にどれだけ近いかの最小値。低いほど幅が広い。"""
        row = self.conn.execute("SELECT centroid, kind FROM clusters WHERE id=?",
                                (cluster_id,)).fetchone()
        if row is None:
            return None
        c = decode(row["centroid"])
        sims = self.conn.execute(
            """SELECT e.vector FROM cluster_members m
               JOIN observation_embeddings e ON e.observation_id = m.observation_id
               WHERE m.cluster_id=? AND e.kind=?""", (cluster_id, row["kind"])).fetchall()
        if len(sims) < 2:
            return None
        return min(float(decode(r["vector"]) @ c) for r in sims)

    # ★2026-08-20 に「群れの幅」で混在を自動検出しようとして失敗した記録。
    #   cohesion（メンバーと重心の最小類似度）は枚数と強く相関するので、
    #   12枚の正常な群れが「幅が広い」と判定され、3枚の混在した群れは通ってしまった。
    #   そもそも混在の自動検出は「混在を避ける」のと同じ問題で、それが簡単なら
    #   クラスタリング自体がそうしている。**混在の判定には外からの信号が要る。**
    #   → 自動判定はやめ、`set_label` が中身を返して人が見る形にした。

    # --- 新奇性（いつもと違うか）---

    def scene_views(self, camera: str) -> list[np.ndarray]:
        """そのカメラの「いつもの眺め」のレパートリー（scene クラスタの重心）。"""
        rows = self.conn.execute(
            """SELECT DISTINCT c.id, c.centroid FROM clusters c
               JOIN cluster_members m ON m.cluster_id = c.id
               JOIN observations o ON o.id = m.observation_id
               WHERE c.kind='scene' AND o.camera_position=?""", (camera,)).fetchall()
        return [decode(r["centroid"]) for r in rows]

    def novelty(self, camera: str, vec: np.ndarray) -> tuple[float, float, bool] | None:
        """新着が「いつもと違う」か。(最大類似度, z, いつもと違うか) を返す。

        ★しきい値を外から与えない。判定の基準はそのカメラ自身の過去から作る。
        過去の各観測が「他の眺め」にどれだけ似ていたかの分布を取り、
        新着がその分布からどれだけ下に外れているかを z で測る。
        絶対値は環境をまたがないが、この形なら較正が要らない。

        過去が乏しい（眺めが2つ未満）ときは None。判断材料が無い。
        """
        views = self.scene_views(camera)
        if len(views) < 2:
            return None
        rows = self.conn.execute(
            """SELECT e.vector FROM observation_embeddings e
               JOIN observations o ON o.id = e.observation_id
               WHERE e.kind='scene' AND o.camera_position=?""", (camera,)).fetchall()
        past = [decode(r["vector"]) for r in rows]
        if len(past) < 3:
            return None

        internal = []
        for v in past:
            sims = [float(v @ c) for c in views]
            sims.sort(reverse=True)
            # 自分自身が入っている重心を避けきれないので、上位2つ目を使う
            internal.append(sims[1] if len(sims) > 1 else sims[0])
        mu, sd = float(np.mean(internal)), float(np.std(internal))

        sim = max(float(vec @ c) for c in views)
        z = (sim - mu) / sd if sd else 0.0
        # 過去のどれよりも外れていれば「いつもと違う」
        return sim, z, z < (min((x - mu) / sd for x in internal) if sd else 0.0)

    def attach_memory(self, memory_db: Path = MEMORY_DB) -> bool:
        """本家 memory.db を読み取り専用で繋ぐ。書き込みは絶対にしない。"""
        if not memory_db.exists():
            return False
        # URI にはスラッシュ区切りが要る。Windows のバックスラッシュのままだと開けない
        self.conn.execute("ATTACH DATABASE ? AS mem",
                          (f"file:{memory_db.as_posix()}?mode=ro",))
        return True
