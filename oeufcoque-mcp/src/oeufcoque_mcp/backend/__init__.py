"""Platform-dependent layer. This package is the only place that touches the OS.

Everything above it (`hand.py`, `server.py`) is platform-agnostic: Bezier paths,
easing, timing, held-button bookkeeping and the MCP tool surface all live there.

To port Oeufcoque to another platform, add one module here that satisfies the
`Backend` protocol below, and register it in `get_backend()`. Nothing else needs
to change.

  windows.py       user32.dll via ctypes           implemented
  darwin.py        Quartz Event Services (pyobjc)  not written yet
  linux_x11.py     XTEST via python-xlib           not written yet
  linux_wayland.py libei / RemoteDesktop portal    hard by design; see README
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

BUTTONS: tuple[str, ...] = ("left", "right", "middle")


class UnsupportedPlatform(RuntimeError):
    """Raised when no backend exists for the current platform."""


@runtime_checkable
class Backend(Protocol):
    """The smallest set of primitives a platform must provide.

    Deliberately dumb: no easing, no randomness, no state. A backend translates
    one call into one OS call and returns. All the behaviour that makes the hand
    look human is built on top of these, in platform-agnostic code.
    """

    name: str

    def get_position(self) -> tuple[int, int]:
        """Current cursor position in physical screen pixels."""

    def set_position(self, x: int, y: int) -> None:
        """Put the cursor at (x, y). No interpolation, no delay."""

    def get_buttons(self) -> dict[str, bool]:
        """Which buttons are currently down, as seen by the OS.

        This reflects physical and injected input alike, so it is the honest
        answer to "what is the hand holding right now?" even if something other
        than Oeufcoque pressed it.
        """

    def button_down(self, button: str) -> None: ...

    def button_up(self, button: str) -> None: ...

    def wheel(self, clicks: int) -> None:
        """Scroll by `clicks` notches. Positive is up/away from the user."""

    def double_click_time(self) -> float:
        """The system's double-click threshold, in seconds.

        Asked of the OS rather than assumed: a click pair spaced wider than this
        is not a double click, it is two clicks, and the caller deserves to know
        the real number.
        """

    def screen_bounds(self) -> tuple[int, int, int, int]:
        """Virtual desktop as (left, top, width, height). May start negative."""

    def acquire_hand_lock(self) -> bool:
        """Try to claim the single physical cursor for this process.

        There is one cursor per desktop, but a stdio MCP server is spawned once
        per client session, so two sessions can easily race for it. Returns True
        if this process now owns the hand, False if somebody else already does.
        The lock is released when the process dies, by the OS.
        """


def get_backend() -> Backend:
    """Pick the backend for the platform we are running on."""
    if sys.platform == "win32":
        from . import windows

        return windows.WindowsBackend()

    raise UnsupportedPlatform(
        f"No Oeufcoque backend for platform {sys.platform!r}. "
        f"Only Windows is implemented today. Adding one means writing a module "
        f"in oeufcoque_mcp/backend/ that satisfies the Backend protocol -- "
        f"roughly nine small functions. Patches welcome."
    )
