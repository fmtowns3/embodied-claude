#!/usr/bin/env python3
"""Infracubus — doctor.py を眠りに憑ける daemon。

succubus = sub(下) + cubare(横たわる)、incubus = in(上に) + cubare。
infra(さらに下・基盤) + cubare で「さらに下に横たわるもの」。性別を持たない夢魔。

問いは 2 つだけだった。

  1. `doctor.py` はいつ走るのか  → **走る時機を誰も決めていない**
  2. 出したエラーは誰が読むのか  → **誰も読まない**

前者に「眠りに落ちる瞬間」を、後者に「目覚めた自分」を与える。それだけをやる。

`doctor.py` 以外は診ない。lifemate-ai はマルチプラットフォームなので、
S.M.A.R.T. のような環境依存のものへ風呂敷を広げない。

  眠りに落ちる                            目覚める
  ┌─ PreCompact / SessionEnd ─┐        ┌─ SessionStart ─┐
  │  doctor.py を走らせる       │        │  MEMORY.md が   │
  │  [error] を夢にする         │──────▶│  勝手に読まれる  │
  │  （体験されない）            │        │ （出所を知らない）│
  └────────────────────────────┘        └─────────────────┘

依存は標準ライブラリだけ。眠りに憑くものが重い依存を持つと、眠りが遅くなる。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HOME = Path.home()
# 見た夢の台帳。INFRACUBUS_STATE で移せる（試すときと、家ごとに分けたいとき）
DREAMS_PATH = Path(os.environ.get("INFRACUBUS_STATE")
                   or HOME / ".claude" / "infracubus" / "dreams.json")

# doctor.py は `[status] name: detail` の形で吐く。name 自体がコロンを含む
# （`config:file` / `state:C:\Users\...`）ので、**コロン+スペースの最初の出現**で切る。
# name の中のコロンには空白が続かないため、これで正しく分かれる。
LINE_RE = re.compile(r"^\[(?P<level>error|warn)\]\s+(?P<rest>.*)$")


@dataclass
class Dream:
    """doctor.py が上げた 1 件。眠りの中では、これが 1 つの夢になる。"""

    level: str          # "error" | "warn"
    key: str            # doctor.py の検査名（config:file など）
    detail: str
    hint: str = ""      # doctor.py が続けて出す "  -> ..." の行


# ---------------------------------------------------------------- 走らせる

def run_doctor(repo: Path, python: str | None = None) -> tuple[int | None, list[Dream], str]:
    """doctor.py を走らせて [error]/[warn] を拾う。

    **doctor.py は stderr に何も出さない。**判定は exit code、中身は stdout の
    `[error]` / `[warn]` 行にある。stderr を見張る作りでは永久に何も起きない。

    戻り値は (exit code, 拾った夢, 走らせられなかった理由)。
    exit code が None なら doctor.py 自体が走らなかった。
    """
    script = repo / "scripts" / "doctor.py"
    if not script.exists():
        return None, [], f"doctor.py が無い: {script}"

    # Windows のタスク配下では locale が cp932 になり、doctor.py の出力が化ける。
    # 化けるだけならまだしも、読み取り側が例外で落ちると夢を見ないまま朝になる。
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            [python or sys.executable, str(script)],
            cwd=str(repo), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        return None, [], "doctor.py が 180 秒で終わらなかった"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, [], f"doctor.py を走らせられなかった: {exc}"

    return proc.returncode, parse_doctor(proc.stdout), ""


def parse_doctor(stdout: str) -> list[Dream]:
    dreams: list[Dream] = []
    for raw in stdout.splitlines():
        line = raw.rstrip()
        if line.startswith("  ->") and dreams:      # 直前の行への助言
            dreams[-1].hint = line[4:].strip()
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        rest = m.group("rest")
        key, _, detail = rest.partition(": ")
        dreams.append(Dream(m.group("level"), key.strip() or rest.strip(), detail.strip()))
    return dreams


# ---------------------------------------------------------------- 憶えておく

def read_hook_input() -> dict:
    """hook から呼ばれたとき、stdin に JSON が来る。

    `cwd` / `hook_event_name` / `session_id` などが入っている。おかげで
    登録側にパスを埋め込まなくて済む。**stdin が無い（手で叩いた）場合も落ちない。**
    """
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return {}
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        # 黙って「手動」に落ちると、**間違った置き場へ夢を書きうる。**
        # 落ちはしないが、気づけるように一言だけ残す。
        print(f"infracubus: hook からの入力が JSON として読めない: {exc}",
              file=sys.stderr)
        return {}
    return payload if isinstance(payload, dict) else {}


def load_dreams() -> dict:
    if not DREAMS_PATH.exists():
        return {}
    try:
        return json.loads(DREAMS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_dreams(seen: dict) -> None:
    DREAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DREAMS_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2),
                           encoding="utf-8")


# ---------------------------------------------------------------- 書き残す

def _slug(key: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    return (s[:60] or "unnamed")


def find_memory_dir(cwd: Path) -> Path | None:
    """cwd に対応する auto memory の置き場を探す。

    Claude Code はプロジェクトのパスを平坦化して `~/.claude/projects/<名>/`
    を作る。**推測で作らず既存を優先する。**規則は公開仕様ではないため。
    """
    root = HOME / ".claude" / "projects"
    if not root.exists():
        return None
    slug = re.sub(r"[:\\/]", "-", str(cwd))
    for cand in (root / slug, root / slug.replace(".", "-")):
        if cand.exists():
            return cand / "memory"
    for d in root.iterdir():          # 大文字小文字だけ違う既存を拾う
        if d.is_dir() and d.name.lower() == slug.lower():
            return d / "memory"
    return (root / slug) / "memory"


def _render(name: str, d: Dream, rec: dict, stamp: str) -> str:
    nights = rec["nights"]
    returns = int(rec.get("returns", 0))
    weight = ""
    if nights >= 3:
        weight = (f"\n**{nights} 夜つづけて同じ夢を見ている。**"
                  f"繰り返す夢ほど重い。起きているあいだに手を付けたほうがいい。\n")
    if returns:
        weight += (f"\n**一度は覚めた夢が {returns} 度戻ってきている。**"
                   f"直したはずのものが戻るなら、直し方のほうを疑う。\n")
    hint = f"\n`doctor.py` の助言: {d.hint}\n" if d.hint else ""
    return (
        "---\n"
        f"name: {name}\n"
        f"description: Infracubus が眠りの中で見た doctor.py の {d.level} — {d.key}\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        f"**[nightmare]** `doctor.py` が `{d.key}` で **{d.level}** を出している。\n\n"
        f"> {d.detail}\n"
        f"{hint}\n"
        "| | |\n|---|---|\n"
        f"| 初めて見た夜 | {rec['first'][:16]} |\n"
        f"| 直近 | {stamp[:16]} |\n"
        f"| 連続 | {nights} 夜 |\n"
        f"| 診た家 | `{rec.get('repo', '?')}` |\n"
        f"| 見た眠り | {rec.get('event', '手動')} |\n"
        f"{weight}\n"
        "> これは [[infracubus]] が眠りの中で書いたもの。書いた瞬間を誰も体験していない。\n"
        "> **起きている自分が確かめるまで、これは夢であって事実ではない。**\n"
        "> 直ったかどうかは `python infracubus/infracubus.py --repo <家> --wake` で確かめる。\n"
    )


def _rewrite_index(memory_dir: Path, seen: dict) -> None:
    """MEMORY.md の Infracubus 区画だけを書き換える。**他人の行には触らない。**"""
    index = memory_dir / "MEMORY.md"
    head, tail = "<!-- infracubus:begin -->", "<!-- infracubus:end -->"

    lines = [head, "", "### 眠りの中で見たもの（Infracubus）", ""]
    if seen:
        # 長く続いている夢を上に。重い夢ほど目に入る場所へ。
        for key, rec in sorted(seen.items(), key=lambda kv: -int(kv[1].get("nights", 0))):
            name = f"nightmare-{_slug(key)}"
            back = f"・{rec['returns']}度目の再来" if rec.get("returns") else ""
            if rec.get("awoken"):
                state = f"[awoken] 覚めた（{rec['awoken'][:10]}）{back}"
            else:
                state = f"×{rec.get('nights', 1)}夜{back}"
            lines.append(f"- [nightmare: {key}]({name}.md) — {rec.get('level', 'error')}・{state}")
    else:
        lines.append("- （まだ何も見ていない）")
    lines += ["", tail]
    block = "\n".join(lines)

    old = index.read_text(encoding="utf-8") if index.exists() else ""
    if head in old and tail in old:
        new = re.sub(re.escape(head) + r".*?" + re.escape(tail),
                     lambda _: block, old, flags=re.S)
    elif old.strip():
        new = old.rstrip() + "\n\n" + block + "\n"
    else:
        new = block + "\n"
    index.write_text(new, encoding="utf-8")


def write_dreams(memory_dir: Path, dreams: list[Dream], seen: dict,
                 stamp: str, repo: Path, event: str = "手動") -> list[str]:
    """夢の断片を置く。索引には一行、中身は別ファイルに。

    仕様（合意済み）:
      1. 健康な夜は何も書かない
      2. 同じ夢は行を増やさず ×n夜 を回す
      3. 覚めても消さない。[awoken] を付けて残す
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    # ひとつの記憶置き場に複数の家の夢が集まりうる。家の名前で名前空間を分ける。
    # 分けないと、別の家の同名エラー（`python` など）が同じ夢に潰れる。
    scope = f"{repo.name}:"
    alive = {scope + d.key for d in dreams}
    touched: list[str] = []

    # 覚めた夢に印を付ける（**消さない**）。
    # **今回診た家の夢だけを対象にする。**診ていない家の夢を覚ましてはいけない。
    for key, rec in seen.items():
        if key.startswith(scope) and key not in alive and not rec.get("awoken"):
            rec["awoken"] = stamp
            touched.append(f"[awoken] {key}")

    for d in dreams:
        key = scope + d.key
        rec = seen.setdefault(key, {"nights": 0, "first": stamp})
        if rec.get("awoken"):
            # 覚めた夢がまた来た。**連続は切れているので数え直す。**
            # ×n夜 は「いま何夜つづいているか」であって通算ではない。
            # ただし何度も戻ってくること自体が重い情報なので、再来だけは数える。
            rec["awoken"] = None
            rec["returns"] = int(rec.get("returns", 0)) + 1
            rec["nights"] = 0
            rec["first"] = stamp

        # **1 日に何度眠っても 1 夜。**PreCompact は 1 セッションで何度も来るので、
        # 回数をそのまま数えると「3 夜つづけて」の重みが嘘になる。
        if rec.get("last", "")[:10] != stamp[:10]:
            rec["nights"] = int(rec.get("nights", 0)) + 1
        rec["nights"] = max(1, int(rec.get("nights", 0)))
        rec["last"] = stamp
        rec["level"] = d.level
        rec["repo"] = str(repo)
        rec["event"] = event          # どの眠りで見たか（PreCompact / SessionEnd）

        name = f"nightmare-{_slug(key)}"
        (memory_dir / f"{name}.md").write_text(
            _render(name, d, rec, stamp), encoding="utf-8")
        touched.append(f"[nightmare] {key} ×{rec['nights']}夜")

    _rewrite_index(memory_dir, seen)
    return touched


# ---------------------------------------------------------------- 憑く

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Infracubus — doctor.py を眠りに憑ける daemon")
    ap.add_argument("--repo", type=Path, required=True,
                    help="doctor.py を持つリポジトリ（診る家）")
    ap.add_argument("--cwd", type=Path, default=Path.cwd(),
                    help="どのプロジェクトの眠りか（auto memory の置き場を決める）")
    ap.add_argument("--python", default=None,
                    help="doctor.py を走らせる python（既定: このスクリプトと同じ）")
    ap.add_argument("--dry-run", action="store_true", help="走らせるが何も書かない")
    ap.add_argument("--wake", action="store_true", help="起きたまま走らせて結果を見る")
    args = ap.parse_args()

    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except AttributeError:
                pass

    # hook から呼ばれたなら stdin に cwd と event が来ている。手で叩いたなら引数を使う。
    hook = read_hook_input()
    cwd = Path(hook.get("cwd") or args.cwd)
    event = str(hook.get("hook_event_name") or "手動")

    stamp = datetime.now().isoformat(timespec="seconds")
    repo = args.repo.expanduser().resolve()
    code, dreams, failure = run_doctor(repo, args.python)
    loud = args.wake or args.dry_run

    if loud:
        print(f"— {stamp}  {repo}  ({event}) —")
        if failure:
            print(f"  {failure}")
        else:
            print(f"  doctor.py exit={code} / 拾った件数 {len(dreams)}")
            for d in dreams:
                print(f"  [{d.level}] {d.key}: {d.detail[:70]}")
            if not dreams:
                print("  健康。今夜は夢を見ない。")

    if failure:
        # 夢魔が働けなかった夜。**黙って寝るほうが、嘘の安心より害が少ない。**
        # 眠りに憑いている以上、ここで落ちても誰も気づけないので握り潰す。
        print(f"infracubus: {failure}", file=sys.stderr)
        return 0
    if args.dry_run:
        return 0

    memory_dir = find_memory_dir(cwd)
    if memory_dir is None:
        print("infracubus: auto memory の置き場が無い。何も書かない。", file=sys.stderr)
        return 0

    seen = load_dreams()
    # 仕様1: 健康な夜は何も書かない。ただし**この家の**夢に覚めた印を付ける必要があれば書く
    scope = f"{repo.name}:"
    if dreams or any(k.startswith(scope) and not r.get("awoken")
                     for k, r in seen.items()):
        for line in write_dreams(memory_dir, dreams, seen, stamp, repo, event):
            if loud:
                print(f"  書いた: {line}")
        save_dreams(seen)
    elif loud:
        print("  何も書かなかった。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
