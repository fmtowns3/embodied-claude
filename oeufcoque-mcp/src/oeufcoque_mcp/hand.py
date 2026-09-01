"""The hand itself: paths, timing, and the promise to let go.

Everything here is platform-agnostic. It knows how to draw a human-looking
path between two points and how to keep track of what it is holding; it does
not know what an operating system is. That lives in `backend/`.

Two ideas run through this file.

**A press and its release belong to the same function.** There is no `press`
tool and no `release` tool, only `drag`, which does both inside one try/finally.
A button cannot be left down by a caller that forgot the second call, because
there is no second call to forget. This is a structural guarantee rather than a
promise in the documentation.

**The hand reports where it ended up.** Every operation returns the cursor
position and the button states afterwards, read back from the OS rather than
assumed from what we asked for. If something else moved the cursor, or if UIPI
swallowed the input, the return value says so.
"""

from __future__ import annotations

import asyncio
import atexit
import math
import random
import signal
from dataclasses import dataclass, field

from .backend import BUTTONS, Backend

# --- the shape of a human-looking move --------------------------------------
# These numbers were arrived at by looking at the result, which is the only way
# anybody arrives at numbers like these. Tune them by looking again, not by
# reasoning about them.
MIN_DISTANCE = 5.0  # below this, teleport; a curve would be noise
CONTROL_OFFSET = (0.10, 0.25)  # bulge, as a fraction of the distance
PIXELS_PER_STEP = 12
MIN_STEPS = 25
STEP_DELAY = (0.008, 0.012)  # seconds between waypoints
JITTER = 1.5  # peak hand-shake in pixels, decaying to zero at the target
JITTER_STOPS_BEFORE_END = 3  # last few steps are steady, like a real approach

# Between the two clicks of a double click, as a fraction of the system's
# double-click threshold. Staying under 1.0 is not decoration: a pair spaced
# wider than the threshold is not a double click, it is two clicks.
DOUBLE_CLICK_FRACTION = (0.25, 0.75)

# Between wheel notches. Reusing the double-click spacing here would put a
# fifth of a second to a full second between notches, which reads as hesitation
# rather than scrolling. A tenth of a second is closer to a real flick.
SCROLL_INTERVAL = (0.05, 0.12)


class HandBusy(RuntimeError):
    """Another process already owns the cursor."""


@dataclass
class Hand:
    backend: Backend
    rng: random.Random = field(default_factory=random.Random)
    _held: set[str] = field(default_factory=set, init=False)

    # --- proprioception -----------------------------------------------------

    def state(self) -> dict:
        """Where the hand is and what it is holding, read back from the OS."""
        x, y = self.backend.get_position()
        return {
            "x": x,
            "y": y,
            "buttons": self.backend.get_buttons(),
            "held_by_oeufcoque": sorted(self._held),
        }

    # --- movement -----------------------------------------------------------

    def plan_path(
        self, start: tuple[int, int], target: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Waypoints from start to target. Pure: nothing moves, nothing is read.

        A quadratic Bezier whose control point is thrown off the straight line
        at a random angle, walked with a smoothstep ease so the hand starts and
        arrives gently, plus a shake that fades out on approach.
        """
        sx, sy = start
        tx, ty = target
        distance = math.hypot(tx - sx, ty - sy)
        if distance < MIN_DISTANCE:
            return [(tx, ty)]

        offset = self.rng.uniform(*CONTROL_OFFSET) * distance
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        cx = (sx + tx) / 2.0 + math.cos(angle) * offset
        cy = (sy + ty) / 2.0 + math.sin(angle) * offset

        steps = max(MIN_STEPS, int(distance / PIXELS_PER_STEP))
        points: list[tuple[int, int]] = []
        for i in range(1, steps + 1):
            t = i / steps
            e = t * t * (3.0 - 2.0 * t)  # smoothstep
            u = 1.0 - e
            x = u * u * sx + 2.0 * u * e * cx + e * e * tx
            y = u * u * sy + 2.0 * u * e * cy + e * e * ty
            if i < steps - JITTER_STOPS_BEFORE_END:
                decay = 1.0 - t
                x += self.rng.uniform(-JITTER, JITTER) * decay
                y += self.rng.uniform(-JITTER, JITTER) * decay
            points.append((round(x), round(y)))

        points.append((tx, ty))  # land exactly, whatever the jitter did
        return points

    async def move_to(self, x: int, y: int) -> None:
        start = self.backend.get_position()
        for px, py in self.plan_path(start, (int(x), int(y))):
            self.backend.set_position(px, py)
            await asyncio.sleep(self.rng.uniform(*STEP_DELAY))

    # --- actions ------------------------------------------------------------

    async def click(
        self, button: str, count: int = 1, x: int | None = None, y: int | None = None
    ) -> dict:
        self._check_button(button)
        if x is not None and y is not None:
            await self.move_to(x, y)

        threshold = self.backend.double_click_time()
        intervals: list[float] = []
        for i in range(count):
            if i:
                gap = self.rng.uniform(*DOUBLE_CLICK_FRACTION) * threshold
                intervals.append(round(gap, 3))
                await asyncio.sleep(gap)
            self._press(button)
            self._release(button)

        return {
            **self.state(),
            "clicks": count,
            "intervals_s": intervals,
            "double_click_threshold_s": round(threshold, 3),
        }

    async def drag(
        self,
        to_x: int,
        to_y: int,
        button: str = "left",
        from_x: int | None = None,
        from_y: int | None = None,
    ) -> dict:
        """Press, travel, release. The release is in a finally block.

        If the move raises, or the client cancels the tool call, or the task is
        torn down, the button still comes up. The one case this cannot cover is
        the process being killed outright mid-drag -- no in-process mechanism
        can, so `release` exists as the way back from that.
        """
        self._check_button(button)
        if from_x is not None and from_y is not None:
            await self.move_to(from_x, from_y)

        origin = self.backend.get_position()
        self._press(button)
        try:
            await self.move_to(to_x, to_y)
        finally:
            self._release(button)

        return {**self.state(), "from": {"x": origin[0], "y": origin[1]}, "button": button}

    async def scroll(
        self,
        direction: str,
        amount: int = 3,
        x: int | None = None,
        y: int | None = None,
    ) -> dict:
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', not {direction!r}")
        if amount < 1:
            raise ValueError("amount must be at least 1")
        if x is not None and y is not None:
            await self.move_to(x, y)

        step = 1 if direction == "up" else -1
        for i in range(amount):
            if i:
                await asyncio.sleep(self.rng.uniform(*SCROLL_INTERVAL))
            self.backend.wheel(step)

        return {**self.state(), "direction": direction, "notches": amount}

    # --- letting go ---------------------------------------------------------

    def release_all(self) -> list[str]:
        """Let go of everything this hand believes it is holding.

        Called from the safety net on the way out, and exposed as a tool so a
        caller can recover from a previous process that died mid-drag. Only
        buttons Oeufcoque pressed are released: reaching for one a human is
        physically holding would break their drag, not repair ours.
        """
        released = []
        for button in sorted(self._held):
            try:
                self.backend.button_up(button)
                released.append(button)
            except OSError:
                pass  # dying anyway; a raised exception here helps nobody
        self._held.clear()
        return released

    def install_safety_net(self) -> None:
        """Release on the ordinary ways out of a process.

        atexit covers a normal return and an unhandled exception. SIGINT and
        SIGTERM cover a client that asks the process to stop. Nothing covers
        TerminateProcess or SIGKILL, and the README says so.
        """
        atexit.register(self.release_all)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)

                def handler(signum, frame, _previous=previous):
                    self.release_all()
                    if callable(_previous):
                        _previous(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
            except (ValueError, OSError, AttributeError):
                pass  # not the main thread, or the platform lacks the signal

    # --- internals ----------------------------------------------------------

    def _press(self, button: str) -> None:
        self.backend.button_down(button)
        self._held.add(button)

    def _release(self, button: str) -> None:
        try:
            self.backend.button_up(button)
        finally:
            self._held.discard(button)

    @staticmethod
    def _check_button(button: str) -> None:
        if button not in BUTTONS:
            raise ValueError(f"button must be one of {BUTTONS}, not {button!r}")
