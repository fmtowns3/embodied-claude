# Infracubus の登録手順

Claude Code の hook として登録する手順。**`settings.json` を書き換えるので、
実行前にバックアップを取る。**

## この daemon の性質

- **常駐しない。**hook が呼んだときだけ走り、終われば消える。ポートも開かない
- **依存は標準ライブラリだけ。**`uv` も仮想環境も要らない
- **書き込むのは 2 か所だけ。**`<memory>/MEMORY.md` の自分の区画と `<memory>/nightmare-*.md`
- **落ちない。**`doctor.py` が無い・走らない・遅い、いずれも終了コード 0 で戻る。
  眠りに憑いている以上、ここで例外を投げても誰も気づけないため

## 1. 単体で走るか確かめる（登録より先に）

**登録する前に確かめる。**ここで動かないなら登録しても無駄。

```powershell
python infracubus\infracubus.py --repo . --dry-run
```

`doctor.py exit=... / 拾った件数 N` が出れば通っている。
`--dry-run` は走らせるだけで**何も書かない**。

書き込みまで含めて見るときは `--wake`。

```powershell
python infracubus\infracubus.py --repo . --wake
```

## 2. バックアップを取る

```powershell
Copy-Item $env:USERPROFILE\.claude\settings.json `
          "$env:USERPROFILE\.claude\settings.json.bak-infracubus" -Force
```

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.bak-infracubus
```

## 3. 登録する

`~/.claude/settings.json` の `hooks` に足す。**既に `hooks` がある場合は、
そのキーを消さずにイベントを追加する。**

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "timeout": 60,
            "command": "python \"D:/path/to/embodied-claude/infracubus/infracubus.py\" --repo \"D:/path/to/embodied-claude\""
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "timeout": 60,
            "command": "python \"D:/path/to/embodied-claude/infracubus/infracubus.py\" --repo \"D:/path/to/embodied-claude\""
          }
        ]
      }
    ]
  }
}
```

### どちらのイベントに憑けるか

| イベント | いつ来るか | 性格 |
|---|---|---|
| `PreCompact` | コンテキストが圧縮される直前 | **1 セッションで何度も来る。**長い作業の途中で落ちる眠り |
| `SessionEnd` | セッションが終わるとき | 1 セッションに 1 回。確実に来るとは限らない（強制終了など） |

**両方に憑けてよい。**同じ日に何度走っても `×n夜` は 1 しか増えない。

`matcher` で絞ることもできる（`PreCompact` は `manual` / `auto`、
`SessionEnd` は `clear` / `resume` / `logout` / `prompt_input_exit` / `other`）。
省略すると全部にマッチする。

### `--cwd` は要らない

hook は stdin に JSON を渡してくる。

```json
{ "session_id": "...", "cwd": "D:\\D\\Project\\CC", "hook_event_name": "PreCompact" }
```

ここから `cwd` を取るので、**どのプロジェクトの記憶置き場に書くかは自動で決まる。**
`hook_event_name` も拾うので、どの眠りで見た夢かも残る。

手で叩いたとき（stdin が無いとき）は `--cwd`、省略時はカレントディレクトリを使う。

### `--python` について

`doctor.py` は Python 3.13 を要求する。別の Python で走らせたいときは指定する。

```
--python "C:/path/to/python3.13.exe"
```

省略すると Infracubus 自身と同じ Python を使う。

## 4. 確認する

登録したら、**実際にコンパクションを起こすか、セッションを終える。**
その後で記憶置き場を見る。

```powershell
Get-Content "$env:USERPROFILE\.claude\projects\<平坦化したパス>\memory\MEMORY.md"
```

`<!-- infracubus:begin -->` の区画があれば通っている。
**`doctor.py` が緑なら何も書かれない**ので、その場合は手で `--wake` して確かめる。

hook そのものが呼ばれているかは、Claude Code の `/hooks` から確認できる。

## 5. 失敗したときの切り分け

| 症状 | 見るところ |
|---|---|
| 何も書かれない | `doctor.py` が緑（正常）。`--wake` で確かめる |
| `doctor.py が無い` | `--repo` が違う。`<repo>/scripts/doctor.py` が要る |
| `hook からの入力が JSON として読めない` | hook 側の問題。この場合カレントディレクトリに書く |
| `auto memory の置き場が無い` | `~/.claude/projects/` が無い。auto memory 未使用 |
| 出力が文字化けする | 子プロセスには `PYTHONIOENCODING=utf-8` を渡しているが、hook を呼ぶ側の locale も cp932 のことがある |
| 眠りが遅い | `doctor.py` の MCP live probe。`timeout` を下げる |

台帳を見れば、いま何を見ているか分かる。

```powershell
Get-Content "$env:USERPROFILE\.claude\infracubus\dreams.json"
```

## 6. 取り消す

```powershell
Copy-Item "$env:USERPROFILE\.claude\settings.json.bak-infracubus" `
          $env:USERPROFILE\.claude\settings.json -Force
```

書かれた夢を消すなら、記憶置き場の `nightmare-*.md` と `MEMORY.md` の
`<!-- infracubus:begin -->` 〜 `<!-- infracubus:end -->` を消す。
台帳は `~/.claude/infracubus/dreams.json`。

## 注意

- **`MEMORY.md` の自分の区画だけ**を書き換える。他の行には触らない。ただし
  **その区画の中に手で書いたものは、次に走ったとき消える**
- 記憶置き場の探索規則（`~/.claude/projects/<平坦化したパス>/`）は**公開仕様ではない**。
  既存ディレクトリを優先し、推測は最後の手段にしている。将来変わりうる
- 書かれるのは**夢であって事実ではない**。`doctor.py` の判定が正しいとは限らないし、
  `×5夜` は「5 日直していない」であって「5 日壊れている」とは限らない
