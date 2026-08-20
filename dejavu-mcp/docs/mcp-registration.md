# dejavu-mcp の登録手順

Claude Code に MCP サーバーとして登録する手順。**この手順は既存の環境を壊さない範囲で
書いてあるが、`.mcp.json` / `~/.claude.json` を書き換えるので、実行前にバックアップを取る。**

## この MCP サーバーの性質

- **stdio 型。ポートを開かない。常駐プロセスも持たない。**
  Claude Code がセッションごとに子プロセスとして起動し、終われば落ちる。
  `memory-mcp` は HTTP recall のために 18900 番を常駐させるが、**こちらはポートを
  持たないので、常駐させるための仕掛けが要らない**。
- 初回起動時に DINOv2（約 340MB）を HuggingFace から取得する。**その間は応答が遅い。**
- CUDA は必須ではない。`torch.cuda.is_available()` が偽なら CPU で動く。

## 1. 前提を確認する

```powershell
python --version          # 3.12 以上
uv --version              # 無ければ https://docs.astral.sh/uv/ から
nvidia-smi                # 任意。無くても動く
```

ディスクは **4GB 程度**空けておく（torch + CUDA ランタイムで約 3GB、モデルで約 400MB）。

## 2. 取得して環境を作る

`dejavu-mcp` は embodied-claude の1コンポーネント。**このディレクトリの中に**
venv を作る（他のコンポーネントと同じく、依存はコンポーネント単位で閉じる）。

```powershell
git clone https://github.com/lifemate-ai/embodied-claude.git
cd embodied-claude\dejavu-mcp
uv venv --python 3.12

# torch は CUDA 版か CPU 版かを選ぶ
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU

uv pip install -e .
```

> `transformers` の `AutoImageProcessor` は `torchvision` を要求する。
> `torch` だけ入れると `ImportError` になる。

## 3. モデルを取得する

`.tflite` は同梱していない。どちらも Apache-2.0。

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L -o models/blaze_face_short_range.tflite `
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
curl.exe -L -o models/selfie_segmenter.tflite `
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
```

確認（先頭に `TFL3` が見えれば正しい tflite）:

```powershell
Get-Content models/blaze_face_short_range.tflite -Encoding Byte -TotalCount 8
```

DINOv2 は初回実行時に自動で取得される。

## 4. 起動を単体で確かめる（登録より先に）

**登録する前に、サーバーが単体で立ち上がるか確かめる。**ここで落ちるなら登録しても無駄。

```powershell
.venv\Scripts\python.exe -c "from dejavu_mcp.server import app; print('ok', app.name)"
```

道具が4つ見えるか:

```powershell
.venv\Scripts\python.exe -c @'
import asyncio
from dejavu_mcp.server import app
print([t.name for t in asyncio.run(app.list_tools())])
'@
```

`['observe', 'recall', 'name', 'views']` が出れば正常。

## 5. 登録する

**先にバックアップを取る。**

```powershell
Copy-Item $env:USERPROFILE\.claude.json "$env:USERPROFILE\.claude.json.bak-dejavu" -Force
```

`claude mcp add` を使うのが安全（JSON を手で壊さずに済む）。

```powershell
claude mcp add dejavu -- <embodied-claude>\dejavu-mcp\.venv\Scripts\python.exe -m dejavu_mcp.server
```

プロジェクト単位にしたい場合は `.mcp.json` に書く:

```json
{
  "mcpServers": {
    "dejavu": {
      "command": "D:\\path\\to\\embodied-claude\\dejavu-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "dejavu_mcp.server"],
      "env": {
        "DEJAVU_DB": "D:\\path\\to\\embodied-claude\\dejavu-mcp\\vision.db",
        "DEJAVU_MODEL_DIR": "D:\\path\\to\\embodied-claude\\dejavu-mcp\\models"
      }
    }
  }
}
```

### 環境変数

MCP サーバーは任意の作業ディレクトリから起動されるので、**相対パスは当てにならない**。
既定は dejavu-mcp/ 直下だが、明示しておくほうが安全。

| 変数 | 既定 | 意味 |
|---|---|---|
| `DEJAVU_DB` | `dejavu-mcp/vision.db` | 観測と群れの保存先 |
| `DEJAVU_MODEL_DIR` | `dejavu-mcp/models/` | `.tflite` の置き場 |
| `DEJAVU_MEMORY_DB` | `~/.claude/memories/memory.db` | 本家の記憶。**読み取り専用でしか開かない** |

## 6. 確認する

```powershell
claude mcp list
```

`dejavu` が `✔ Connected` になっていること。**初回は DINOv2 の取得で時間がかかる**ので、
すぐ繋がらなくても数分待つ。

Claude Code のセッションから、画像1枚で試す:

```
recall で this/is/a/photo.jpg を見て
```

`face: 初めて` のように返れば通っている（DB が空なら全部「初めて」で正しい）。

## 7. 失敗したときの切り分け

| 症状 | 見るところ |
|---|---|
| `✘ Failed to connect` | 手順4を単体で実行する。ここで落ちるならサーバー側の問題 |
| `ImportError: ... torchvision` | `uv pip install torchvision` を忘れている |
| `ファイルがありません` | `.tflite` を取得していない（手順3）／`DEJAVU_MODEL_DIR` が違う |
| 起動が異常に遅い | 初回の DINOv2 取得（約340MB）。2回目以降は速い |
| 応答が全部「初めて」 | `vision.db` が空。`observe` で記録してから `recall` する |

ログは Claude Code 側に出る。サーバーは stderr を UTF-8 にしてあるので日本語も読める。

## 8. 取り消す

```powershell
claude mcp remove dejavu
Copy-Item "$env:USERPROFILE\.claude.json.bak-dejavu" $env:USERPROFILE\.claude.json -Force
```

`vision.db` を消せば覚えたものは全部消える。**本家の `memory.db` には何も書いていない**ので、
そちらは触らなくてよい。

## 注意

- **顔写真は個人情報。**`images/` と `vision.db` は `.gitignore` 済みだが、ベクトルは
  元の顔を推定できる情報を含みうる。`vision.db` を共有しないこと。
- `name` で群れに名前を付けると、その名前は以後 `候補` として返る。取りこぼしを避ける
  ためにしきい値を下げてあるので、**別人が同じ群れに入っていることがある**。
  `name` は付けた群れの中身を返すので、混ざっていないか確かめること。
- MCP SDK は 2.x の `MCPServer` を使っている。本家 memory-mcp は 1.x の書き方なので、
  同じ環境に両方を入れる場合はバージョンの相性を確認すること。
