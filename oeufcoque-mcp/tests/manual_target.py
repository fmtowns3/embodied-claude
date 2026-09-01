"""End-to-end check against a target window this script puts up itself.

    uv run python tests/manual_target.py

This one really moves the cursor, so it is not part of the automatic suite. It
raises its own always-on-top window and clicks only inside that, which means no
window belonging to anybody else is touched, and the receiving end can confirm
that the input arrived at the coordinates it was aimed at.

The double-click count is judged by Tk's own <Double-Button-1> binding, which
fires on the system's threshold rather than on anything this project believes.
The last block repeats the trial with the gap picked from a plausible-looking
range instead of from the system threshold, so both numbers come from the same
referee and the only difference is whether the OS was consulted.

Measured on Windows 11 Pro 26200, Python 3.12.10, GetDoubleClickTime() = 500ms:

    gap taken from the system threshold   20/20
    gap taken from 0.2-1.0s                8/20   (12 of 20 gaps exceeded 500ms)
"""

from __future__ import annotations

import asyncio
import random
import sys
import threading
import time
import tkinter as tk

from oeufcoque_mcp.backend import get_backend
from oeufcoque_mcp.hand import Hand

WIN_W, WIN_H, WIN_X, WIN_Y = 700, 500, 300, 200
TRIALS = 20

events: dict = {"click": [], "double": 0, "motion": 0, "release": [], "wheel": []}
report: list[tuple[bool, str, str]] = []


def check(label: str, cond: bool, detail: str = "") -> bool:
    report.append((bool(cond), label, detail))
    return bool(cond)


def build_target() -> tk.Tk:
    root = tk.Tk()
    root.title("oeufcoque target")
    root.geometry(f"{WIN_W}x{WIN_H}+{WIN_X}+{WIN_Y}")
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, bg="#202020",
                       highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.create_text(WIN_W // 2, 30, fill="#888",
                       text="oeufcoque is aiming at this window; leave the mouse alone")
    canvas.bind("<Button-1>", lambda e: events["click"].append((e.x_root, e.y_root)))
    canvas.bind("<ButtonRelease-1>", lambda e: events["release"].append((e.x_root, e.y_root)))
    canvas.bind("<Double-Button-1>", lambda e: events.update(double=events["double"] + 1))
    canvas.bind("<B1-Motion>", lambda e: events.update(motion=events["motion"] + 1))
    canvas.bind("<MouseWheel>", lambda e: events["wheel"].append(e.delta))
    return root


async def drive(done: threading.Event) -> None:
    backend = get_backend()
    hand = Hand(backend=backend, rng=random.Random())
    home = backend.get_position()
    threshold = backend.double_click_time()
    cx, cy = WIN_X + WIN_W // 2, WIN_Y + WIN_H // 2

    try:
        await asyncio.sleep(0.6)  # let the target appear

        await hand.click("left", 1, cx, cy)
        await asyncio.sleep(0.3)
        got = events["click"][-1] if events["click"] else None
        check("a single click reaches the target", got is not None, f"received at {got}")
        check("at the coordinates it was aimed at", got == (cx, cy),
              f"sent ({cx},{cy}) received {got}")

        events["motion"] = 0
        before = len(events["release"])
        await hand.drag(cx + 200, cy + 120, "left", from_x=cx - 200, from_y=cy - 120)
        await asyncio.sleep(0.3)
        check("the button stays down while travelling", events["motion"] > 10,
              f"{events['motion']} B1-Motion events")
        check("and comes up at the end", len(events["release"]) > before,
              f"released at {events['release'][-1] if events['release'] else None}")

        events["wheel"].clear()
        await hand.scroll("down", 3, cx, cy)
        await asyncio.sleep(0.3)
        check("three wheel notches arrive", len(events["wheel"]) == 3,
              f"deltas {events['wheel']}")
        check("scrolling down is negative", all(d < 0 for d in events["wheel"]))

        events["double"] = 0
        for _ in range(TRIALS):
            await hand.click("left", 2, cx, cy)
            await asyncio.sleep(threshold * 1.6)
        ours = events["double"]
        check(f"oeufcoque double-click rate {ours}/{TRIALS}", ours == TRIALS,
              f"gaps held inside 25-75% of the {threshold}s threshold")

        events["double"] = 0
        rng = random.Random(20260901)
        gaps = []
        for _ in range(TRIALS):
            gap = rng.uniform(0.2, 1.0)  # a plausible range, chosen without asking
            gaps.append(gap)
            backend.button_down("left"); backend.button_up("left")
            await asyncio.sleep(gap)
            backend.button_down("left"); backend.button_up("left")
            await asyncio.sleep(threshold * 1.6)
        check(f"not asking the OS only manages {events['double']}/{TRIALS}",
              events["double"] < TRIALS,
              f"{sum(g > threshold for g in gaps)}/{TRIALS} gaps exceeded the threshold")

        backend.set_position(*home)
        check("the cursor is put back where it started", backend.get_position() == home)
        check("nothing is still being held",
              not any(backend.get_buttons().values())
              and hand.state()["held_by_oeufcoque"] == [])
    finally:
        done.set()


def main() -> int:
    root = build_target()
    done = threading.Event()
    threading.Thread(target=lambda: asyncio.run(drive(done)), daemon=True).start()

    started = time.time()
    while not done.is_set() and time.time() - started < 300:
        root.update()
        time.sleep(0.005)
    root.destroy()

    failed = 0
    for cond, label, detail in report:
        print(f"  [{'ok ' if cond else 'NG '}] {label}" + (f"  --  {detail}" if detail else ""))
        failed += not cond
    print(f"\n{len(report) - failed}/{len(report)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
