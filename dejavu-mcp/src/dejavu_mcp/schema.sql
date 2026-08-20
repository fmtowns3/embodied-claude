-- vision.db  —  視覚観測の保存先
--
-- 本家 memory-mcp の memory.db とは【別ファイル】にする。
-- 参照が要るときは ATTACH して JOIN する:
--     ATTACH DATABASE '~/.claude/memories/memory.db' AS mem;
--     SELECT o.label, m.content FROM observations o
--       JOIN mem.memories m ON m.id = o.memory_id;
--
-- memory.db は journal_mode=wal なので、attach をまたぐ書き込みの原子性は
-- 保証されない。★こちらは vision.db にだけ書き、memory.db は読むだけにする。
-- 命名は memory-mcp/src/memory_mcp/store.py の流儀に合わせた
-- （複数形スネーク / vector BLOB / FK 明示 / NOT NULL DEFAULT 徹底 / CHECK / idx_<table>_<column>）。

-- 1枚の画像 = 1観測
CREATE TABLE IF NOT EXISTS observations (
    id              TEXT PRIMARY KEY,
    capture_path    TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    -- 本家 memories と同じ列名を使う（ATTACH して突き合わせるため）
    camera_position TEXT,
    pan_angle       REAL,
    tilt_angle      REAL,
    person_ratio    REAL NOT NULL DEFAULT 0.0 CHECK(person_ratio >= 0.0 AND person_ratio <= 1.0),
    face_confidence REAL NOT NULL DEFAULT 0.0 CHECK(face_confidence >= 0.0 AND face_confidence <= 1.0),
    freshness       REAL NOT NULL DEFAULT 1.0,
    -- mem.memories(id)。別 DB なので FK は張れない。張れないことをここに書き残す
    memory_id       TEXT,
    label           TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_memory_id ON observations(memory_id);
CREATE INDEX IF NOT EXISTS idx_observations_label     ON observations(label);

-- ベクトル。本家 embeddings に倣って列名は vector、種別は行で分ける。
--   scene  … 背景（どこにいたか）      heishio の flow_vector
--   person … 人物セグメント（服を含む） heishio の delta_vector
--   face   … 顔クロップ                heishio の face_vector
-- ★3本を分けて持つのは、「同じ人だ」がどのチャンネル由来かを言えるようにするため。
--   混ぜると、顔で当てたのか服の色で当てたのかが判別できなくなる。
-- ★model / dim を必ず持たせる。
--   モデルを替えると次元も意味も変わり、既存ベクトルは全部無効になる。
--   heishio 版には記録が無く、512d→768d の移行に専用スクリプトが要った。
--   model を主キーに含めるので、新旧を並べて置いてから古いほうを消せる（無停止移行）。
CREATE TABLE IF NOT EXISTS observation_embeddings (
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK(kind IN ('scene', 'person', 'face')),
    model          TEXT NOT NULL,
    dim            INTEGER NOT NULL CHECK(dim > 0),
    vector         BLOB NOT NULL,
    PRIMARY KEY (observation_id, kind, model)
);
CREATE INDEX IF NOT EXISTS idx_observation_embeddings_kind  ON observation_embeddings(kind);
CREATE INDEX IF NOT EXISTS idx_observation_embeddings_model ON observation_embeddings(model);

-- 重心クラスタ。「名前が付く前の見覚え」の器
CREATE TABLE IF NOT EXISTS clusters (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK(kind IN ('scene', 'person', 'face')),
    -- 重心も「どのモデル空間の重心か」を持たないと、混ざった瞬間に無意味になる
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL CHECK(dim > 0),
    centroid     BLOB NOT NULL,
    member_count INTEGER NOT NULL DEFAULT 0,
    -- 名前が付くのは後。既定は空文字（本家の NULL 回避に合わせる）
    label        TEXT NOT NULL DEFAULT '',
    freshness    REAL NOT NULL DEFAULT 1.0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clusters_kind  ON clusters(kind);
CREATE INDEX IF NOT EXISTS idx_clusters_label ON clusters(label);

-- ★テーブル名を composite_members にしない。
--   本家 heishio 版では memory 側の合成記憶と画像側で同名を共有し、
--   3列目が added_at / contribution_weight で食い違っていた。
--   別 DB なら衝突はしないが、ATTACH して並べたときに読み手が混乱する。
CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id     TEXT NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    similarity     REAL NOT NULL CHECK(similarity >= -1.0 AND similarity <= 1.0),
    added_at       TEXT NOT NULL,
    PRIMARY KEY (cluster_id, observation_id)
);
CREATE INDEX IF NOT EXISTS idx_cluster_members_observation ON cluster_members(observation_id);

-- 次元削減の基底（後で入れる場合に備えて）
CREATE TABLE IF NOT EXISTS pca_bases (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK(kind IN ('scene', 'person', 'face')),
    mean_vector BLOB NOT NULL,
    components  BLOB NOT NULL,
    eigenvalues BLOB NOT NULL,
    n_dims      INTEGER NOT NULL,
    n_samples   INTEGER NOT NULL,
    updated_at  TEXT NOT NULL
);
