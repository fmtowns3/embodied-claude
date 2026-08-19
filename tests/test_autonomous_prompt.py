"""The autonomous prompt has to reach Claude as written.

PROMPT is assembled as a double-quoted shell string, so anything the prompt
says with backticks is command substitution rather than markup. The tool names
in the supplementary rules were being executed, failing, and substituted with
the empty string -- the rules arrived as arrows pointing at nothing.

Nothing surfaces that: `command not found` goes to stderr, which cron discards,
and the log records the prompt after the stripping.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]

# The script is mostly Japanese, so the encoding cannot be left to the locale.
SCRIPT = (ROOT / "autonomous-action.sample.sh").read_text(encoding="utf-8")

PROMPT_START = 'PROMPT="自律行動タイム'
PROMPT_END = '${INTEROCEPTION_SECTION}"'


def _prompt_block() -> str:
    start = SCRIPT.index(PROMPT_START)
    return SCRIPT[start : SCRIPT.index(PROMPT_END, start)]


def test_prompt_block_is_where_we_think_it_is() -> None:
    # If the prompt is ever rewritten as a quoted heredoc, the check below stops
    # being meaningful rather than starting to fail. Pin the assumption.
    assert SCRIPT.count(PROMPT_START) == 1
    assert SCRIPT.count(PROMPT_END) == 1
    assert 'PROMPT=$(cat <<' not in SCRIPT


def test_prompt_quotes_tool_names_without_running_them() -> None:
    unescaped = [match.start() for match in re.finditer(r"(?<!\\)`", _prompt_block())]
    assert unescaped == [], (
        "unescaped backtick in the prompt: the shell runs what it encloses and "
        "substitutes the empty string, so the tool name never reaches Claude"
    )


def test_the_tool_names_are_still_there() -> None:
    # Escaping is only half of it: deleting the names would satisfy the check
    # above and lose the rules the prompt is trying to state. Dropping the
    # backticks instead of escaping them is a fine way to fix this, so the
    # names are checked without them.
    prompt = _prompt_block()
    for tool in (
        "get_social_state",
        "evaluate_action",
        "review_social_post",
        "ingest_social_event",
        "append_daybook",
    ):
        assert tool in prompt
