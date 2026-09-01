"""The MCP surface: five tools and a way back.

Design notes that are easy to lose later:

**stdout belongs to the protocol.** The stdio transport says a server MUST NOT
write anything to stdout that is not an MCP message. One stray print corrupts
the stream. Logging goes to stderr, which the same spec explicitly allows.

**Tools are chosen by a model, not composed by a programmer.** One endpoint
carrying `down`, `up`, `click`, `double_click`, `wheel_up` and `wheel_down` as
optional parameters is a serviceable HTTP API and a poor tool: what happens
when several of them are set at once is knowable only by reading the handler,
and a caller who cannot read the handler has to guess. So each intention is its
own tool with its own arguments, and `drag` exists precisely so that a press
and its release cannot be issued separately.

**The hand is singular.** One cursor per desktop, but stdio servers are spawned
one per client session. The first process to start claims the hand; the others
still run, and say so when asked to move.

**A failure we saw coming is a `ToolError`.** The SDK sends a `ToolError`
message through to the model and logs it without a traceback; every other
exception reaches the model as the bare string "Error executing tool <name>",
with nothing of the original left. A hand that is busy, a button that does not
exist, input refused by the OS -- these are all things the caller can act on,
so they are raised as `ToolError` and keep their text. The layers below know
nothing about MCP: `hand.py` raises `ValueError` and `OSError`, and
`_anticipated` translates at the boundary.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .backend import UnsupportedPlatform, get_backend
from .hand import Hand, HandBusy

logging.basicConfig(
    level=os.environ.get("OEUFCOQUE_LOG", "INFO").upper(),
    stream=sys.stderr,  # never stdout: see the module docstring
    format="%(asctime)s oeufcoque %(levelname)s %(message)s",
)
log = logging.getLogger("oeufcoque")

mcp = MCPServer(
    name="oeufcoque",
    version="0.1.0",
    instructions=(
        "A hand that moves the mouse on this machine. It has no eyes: it takes "
        "screen coordinates and cannot tell you what is at them. To decide where "
        "to aim, capture the screen to an image and look at that yourself -- no "
        "separate vision service is involved. Capture again rather than trusting "
        "an earlier one, because the screen changes between looking and moving, "
        "and this hand will happily click whatever has taken that spot since. "
        "Coordinates are physical screen pixels on the virtual desktop, which may "
        "start at a negative origin on multi-monitor setups; call mouse_state to "
        "find out. Movement is deliberately unhurried -- a long move takes the "
        "better part of a second."
    ),
)

_backend = None
_hand: Hand | None = None
_have_hand = False


@contextmanager
def _anticipated():
    """Turn the failures we designed for into messages the model can read."""
    try:
        yield
    except (ValueError, HandBusy) as exc:
        raise ToolError(str(exc)) from exc
    except OSError as exc:
        raise ToolError(
            f"The OS refused the input: {exc.strerror or exc}"
        ) from exc


def _require_hand() -> Hand:
    if _hand is None:
        raise ToolError("Oeufcoque did not start correctly; see its stderr log.")
    if not _have_hand:
        raise ToolError(
            "Another Oeufcoque process is holding the cursor on this desktop. "
            "There is only one cursor, so only one hand may drive it. Close the "
            "other client session, or set OEUFCOQUE_ALLOW_MULTIPLE=1 if you "
            "genuinely want both."
        )
    return _hand


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    description=(
        "Report where the cursor is, which buttons are down, and how big the "
        "desktop is. Changes nothing. Use it to find the coordinate space "
        "before moving, or to check what the hand ended up holding."
    ),
)
def mouse_state() -> dict[str, Any]:
    if _hand is None:
        raise ToolError("Oeufcoque did not start correctly; see its stderr log.")
    with _anticipated():
        left, top, width, height = _backend.screen_bounds()
        return {
            **_hand.state(),
            "screen": {"left": left, "top": top, "width": width, "height": height},
            "double_click_threshold_s": round(_backend.double_click_time(), 3),
            "backend": _backend.name,
            "dpi_awareness": getattr(_backend, "dpi_awareness", "unknown"),
            "holds_the_hand": _have_hand,
        }


@mcp.tool(
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
    description=(
        "Move the cursor to (x, y) along a curved, slightly unsteady path, the "
        "way a hand would. Presses nothing. Returns the position it reached, "
        "read back from the OS."
    ),
)
async def mouse_move(x: int, y: int) -> dict[str, Any]:
    hand = _require_hand()
    with _anticipated():
        await hand.move_to(x, y)
        return hand.state()


@mcp.tool(
    annotations=ToolAnnotations(openWorldHint=False),
    description=(
        "Click a mouse button, optionally moving to (x, y) first. Set count=2 "
        "for a double click; the gap between clicks is kept inside the system's "
        "double-click threshold, so it registers as one. button is 'left', "
        "'right' or 'middle', where 'left' means the primary button."
    ),
)
async def mouse_click(
    button: str = "left",
    count: int = 1,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, Any]:
    hand = _require_hand()
    with _anticipated():
        if not 1 <= count <= 3:
            raise ValueError(f"count must be 1, 2 or 3, not {count}")
        return await hand.click(button, count, x, y)


@mcp.tool(
    annotations=ToolAnnotations(openWorldHint=False),
    description=(
        "Press a button, travel to (to_x, to_y), release. Optionally start by "
        "moving to (from_x, from_y). The release is guaranteed for every exit "
        "from this call, including cancellation, which is why there is no "
        "separate press tool."
    ),
)
async def mouse_drag(
    to_x: int,
    to_y: int,
    button: str = "left",
    from_x: int | None = None,
    from_y: int | None = None,
) -> dict[str, Any]:
    hand = _require_hand()
    with _anticipated():
        return await hand.drag(to_x, to_y, button, from_x, from_y)


@mcp.tool(
    annotations=ToolAnnotations(openWorldHint=False),
    description=(
        "Turn the wheel. direction is 'up' or 'down', amount is the number of "
        "notches. Optionally move to (x, y) first, since scrolling applies to "
        "whatever is under the cursor."
    ),
)
async def mouse_scroll(
    direction: str,
    amount: int = 3,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, Any]:
    hand = _require_hand()
    with _anticipated():
        if amount > 50:
            raise ValueError(f"amount must be 50 or less, not {amount}")
        return await hand.scroll(direction, amount, x, y)


@mcp.tool(
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
    description=(
        "Let go of any button this hand is holding. Normally unnecessary, since "
        "drag releases on its own. Use it if a previous session was killed "
        "mid-drag and a button appears stuck. It only releases buttons "
        "Oeufcoque pressed, never one a person is physically holding."
    ),
)
def mouse_release() -> dict[str, Any]:
    hand = _require_hand()
    with _anticipated():
        released = hand.release_all()
        return {**hand.state(), "released": released}


def main() -> None:
    global _backend, _hand, _have_hand

    try:
        _backend = get_backend()
    except UnsupportedPlatform as exc:
        log.error("%s", exc)
        raise SystemExit(1)

    _hand = Hand(backend=_backend)
    _hand.install_safety_net()

    if os.environ.get("OEUFCOQUE_ALLOW_MULTIPLE") == "1":
        _have_hand = True
        log.warning("OEUFCOQUE_ALLOW_MULTIPLE=1: not claiming the hand lock")
    else:
        _have_hand = _backend.acquire_hand_lock()
        if not _have_hand:
            log.warning(
                "another Oeufcoque process holds the cursor; "
                "starting read-only so mouse_state still answers"
            )

    log.info(
        "backend=%s dpi=%s hand=%s",
        _backend.name,
        getattr(_backend, "dpi_awareness", "unknown"),
        "held" if _have_hand else "busy",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
