"""Windows backend: user32.dll through ctypes.

No compiled extension and no pip dependency: ctypes is in the standard library,
and user32.dll ships with the OS. Nothing here touches hardware -- these are all
requests to the operating system, which is why a scripting language is a
perfectly good place to make them from.

Two APIs are used for two different jobs:

  SetCursorPos  moves the cursor. The path-drawing code calls it many times in
                a row, so it wants the cheapest possible "put it exactly here".
  SendInput     presses buttons and turns the wheel. It is the current input
                synthesis API (mouse_event is documented as superseded), it is
                atomic per call, and it reports how many events were accepted,
                which gives us something to check instead of something to hope.

Known limits, stated rather than hidden:
  * UIPI. Input cannot be injected into a window running at a higher integrity
    level. Against an elevated application every call here reports success and
    nothing moves. See `send_was_accepted` on the error path.
  * A locked workstation or a session with no desktop has no cursor to move.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# --- virtual key codes ------------------------------------------------------
# VK_LBUTTON is the *primary* button, not the physically left one: it follows
# the "swap mouse buttons" setting. MOUSEEVENTF_LEFTDOWN follows it too, so
# "left" means the same thing whether we are reading or writing.
VK_LBUTTON, VK_RBUTTON, VK_MBUTTON = 0x01, 0x02, 0x04

# --- SendInput --------------------------------------------------------------
INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120

_DOWN_FLAG = {
    "left": MOUSEEVENTF_LEFTDOWN,
    "right": MOUSEEVENTF_RIGHTDOWN,
    "middle": MOUSEEVENTF_MIDDLEDOWN,
}
_UP_FLAG = {
    "left": MOUSEEVENTF_LEFTUP,
    "right": MOUSEEVENTF_RIGHTUP,
    "middle": MOUSEEVENTF_MIDDLEUP,
}

# --- GetSystemMetrics indices for the virtual desktop -----------------------
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79

ERROR_ALREADY_EXISTS = 183

# The cursor is a per-desktop resource, so the mutex is per-session on purpose:
# two desktops (RDP, fast user switching) have two cursors and may hold one each.
HAND_MUTEX_NAME = "Local\\oeufcoque-mcp-hand"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class INPUT(ctypes.Structure):
    class _Union(ctypes.Union):
        # Only the mouse member is declared. MOUSEINPUT is the largest of the
        # three INPUT variants, so the union -- and therefore INPUT itself --
        # still comes out the size the API expects. Asserted just below,
        # because SendInput answers a wrong cbSize by silently doing nothing.
        _fields_ = [("mi", MOUSEINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _Union)]


_EXPECTED_INPUT_SIZE = {4: 28, 8: 40}[ctypes.sizeof(ctypes.c_void_p)]
assert ctypes.sizeof(INPUT) == _EXPECTED_INPUT_SIZE, (
    "INPUT is %d bytes, expected %d"
    % (ctypes.sizeof(INPUT), _EXPECTED_INPUT_SIZE)
)

user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short  # SHORT, not int: sign matters
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetDoubleClickTime.argtypes = []
user32.GetDoubleClickTime.restype = wintypes.UINT
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int


def _declare_dpi_awareness() -> str:
    """Ask to be per-monitor DPI aware, so our pixels are the screen's pixels.

    Without this, a process on a scaled display is handed virtualised
    coordinates, and they will not line up with whatever component supplies the
    coordinates. Oeufcoque has no eyes of its own, so agreeing with the eyes is
    the entire job. Best effort: needs Windows 10 1703 or later, and it fails
    harmlessly when awareness has already been set for the process.
    """
    try:
        per_monitor_v2 = ctypes.c_void_p(-4)
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        if user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            return "per-monitor-v2"
        return "already-set"
    except (AttributeError, OSError):
        return "unavailable"


def _send(flags: int, mouse_data: int = 0) -> None:
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(
        dx=0,
        dy=0,
        mouseData=mouse_data & 0xFFFFFFFF,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(
            ctypes.get_last_error(),
            "SendInput accepted 0 of 1 events. Input is being blocked, most "
            "likely by UIPI (a window at a higher integrity level) or by "
            "another process holding a low-level mouse hook.",
        )


class WindowsBackend:
    name = "windows"

    def __init__(self) -> None:
        self.dpi_awareness = _declare_dpi_awareness()
        self._hand_mutex = None

    # --- reading ------------------------------------------------------------

    def get_position(self) -> tuple[int, int]:
        pt = POINT()
        if not user32.GetCursorPos(ctypes.byref(pt)):
            raise OSError(ctypes.get_last_error(), "GetCursorPos failed")
        return pt.x, pt.y

    def get_buttons(self) -> dict[str, bool]:
        return {
            "left": bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000),
            "right": bool(user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000),
            "middle": bool(user32.GetAsyncKeyState(VK_MBUTTON) & 0x8000),
        }

    def double_click_time(self) -> float:
        return user32.GetDoubleClickTime() / 1000.0

    def screen_bounds(self) -> tuple[int, int, int, int]:
        return (
            user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    # --- writing ------------------------------------------------------------

    def set_position(self, x: int, y: int) -> None:
        if not user32.SetCursorPos(int(x), int(y)):
            raise OSError(ctypes.get_last_error(), "SetCursorPos failed")

    def button_down(self, button: str) -> None:
        _send(_DOWN_FLAG[button])

    def button_up(self, button: str) -> None:
        _send(_UP_FLAG[button])

    def wheel(self, clicks: int) -> None:
        # mouseData is a DWORD that the receiving end reads back as signed, so a
        # negative notch count travels as two's complement. It looks wrong and
        # is correct; _send does the masking.
        _send(MOUSEEVENTF_WHEEL, clicks * WHEEL_DELTA)

    # --- the single-hand lock -----------------------------------------------

    def acquire_hand_lock(self) -> bool:
        handle = kernel32.CreateMutexW(None, True, HAND_MUTEX_NAME)
        if not handle:
            return False
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False
        self._hand_mutex = handle  # held until the process ends; the OS frees it
        return True
