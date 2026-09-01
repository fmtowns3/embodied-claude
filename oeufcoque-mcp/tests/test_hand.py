"""Tests for the parts that do not need a desktop.

A fake backend records what it was asked to do instead of doing it, so these
run anywhere -- CI, a headless box, a machine somebody is using -- and nothing
moves. The guarantee this file exists for is the one in `Hand.drag`: whatever
happens in the middle, the button comes back up.

    uv run python tests/test_hand.py
    uv run pytest tests/            # if you have pytest
"""

from __future__ import annotations

import asyncio
import math
import random

from oeufcoque_mcp.hand import Hand


class FakeBackend:
    """Records calls. Raises on the Nth move if you ask it to."""

    name = "fake"

    def __init__(self, explode_at: int | None = None):
        self.pos = (0, 0)
        self.held: set[str] = set()
        self.wheeled: list[int] = []
        self.moves = 0
        self.explode_at = explode_at

    def get_position(self): return self.pos
    def get_buttons(self): return {b: b in self.held for b in ("left", "right", "middle")}
    def double_click_time(self): return 0.5
    def screen_bounds(self): return (0, 0, 1920, 1080)
    def button_down(self, button): self.held.add(button)
    def button_up(self, button): self.held.discard(button)
    def wheel(self, clicks): self.wheeled.append(clicks)

    def set_position(self, x, y):
        self.moves += 1
        if self.explode_at is not None and self.moves >= self.explode_at:
            raise OSError(5, "simulated failure mid-move")
        self.pos = (int(x), int(y))


def _hand(seed=0, **kw):
    return Hand(backend=FakeBackend(**kw), rng=random.Random(seed))


# --- the path ---------------------------------------------------------------

def test_path_lands_exactly():
    h = _hand()
    for target in [(900, 600), (-400, 50), (1, 1), (1919, 1079)]:
        assert h.plan_path((100, 100), target)[-1] == target


def test_path_is_curved():
    """A straight line would mean the control point did nothing."""
    h = _hand(seed=7)
    start, target = (100, 100), (900, 600)
    points = h.plan_path(start, target)
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = math.hypot(dx, dy)
    worst = max(
        abs((x - start[0]) * dy - (y - start[1]) * dx) / length for x, y in points
    )
    assert worst > 10, f"path deviates only {worst:.1f}px; that is a straight line"


def test_tiny_move_does_not_draw_a_curve():
    h = _hand()
    assert h.plan_path((500, 500), (502, 501)) == [(502, 501)]


def test_step_count_scales_with_distance():
    h = _hand()
    assert len(h.plan_path((0, 0), (0, 30))) >= 25          # floor
    assert len(h.plan_path((0, 0), (1200, 0))) > 90         # 1200/12 + landing


# --- the promise to let go --------------------------------------------------

def test_drag_releases_when_the_move_raises():
    h = _hand(explode_at=10)
    try:
        asyncio.run(h.drag(900, 700, "left", from_x=0, from_y=0))
    except OSError:
        pass
    else:
        raise AssertionError("the fake backend was supposed to raise")
    assert h.backend.held == set(), "button left down after a failed drag"
    assert h._held == set()


def test_drag_releases_when_cancelled():
    async def run():
        h = _hand()
        task = asyncio.create_task(h.drag(1900, 1000, "left", from_x=0, from_y=0))
        await asyncio.sleep(0.15)
        mid = set(h.backend.held)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return mid, h.backend.held, h._held

    mid, after, tracked = asyncio.run(run())
    assert mid == {"left"}, "the drag was not holding the button mid-flight"
    assert after == set(), "button left down after cancellation"
    assert tracked == set()


def test_release_all_lets_go():
    h = _hand()
    h._press("left")
    h._press("right")
    assert h.release_all() == ["left", "right"]
    assert h.backend.held == set()


def test_release_all_ignores_buttons_it_did_not_press():
    """A button a person is physically holding is not ours to release."""
    h = _hand()
    h.backend.held.add("left")          # somebody else's press
    assert h.release_all() == []
    assert h.backend.held == {"left"}


# --- clicking ---------------------------------------------------------------

def test_double_click_stays_under_the_system_threshold():
    h = _hand()
    for seed in range(50):
        h.rng = random.Random(seed)
        result = asyncio.run(h.click("left", count=2))
        threshold = result["double_click_threshold_s"]
        for gap in result["intervals_s"]:
            assert gap < threshold, f"{gap}s gap exceeds the {threshold}s threshold"


def test_single_click_has_no_interval():
    assert asyncio.run(_hand().click("left", 1))["intervals_s"] == []


# --- arguments --------------------------------------------------------------

def test_unknown_button_is_rejected():
    for bad in ("pinky", "LEFT", ""):
        try:
            asyncio.run(_hand().click(bad))
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_scroll_direction_and_amount_are_checked():
    for kwargs in ({"direction": "sideways"}, {"direction": "up", "amount": 0}):
        try:
            asyncio.run(_hand().scroll(**kwargs))
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have been rejected")


def test_scroll_sign_follows_direction():
    h = _hand()
    asyncio.run(h.scroll("up", 2))
    asyncio.run(h.scroll("down", 3))
    assert h.backend.wheeled == [1, 1, -1, -1, -1]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [ok ] {t.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  [NG ] {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
