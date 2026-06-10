import ctypes
import ctypes.wintypes as wt
import re
import time

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- Fallback-типы (Python 3.12: wintypes может быть неполным) ---
LRESULT = getattr(wt, "LRESULT", ctypes.c_ssize_t)
HICON = getattr(wt, "HICON", wt.HANDLE)
HCURSOR = getattr(wt, "HCURSOR", wt.HANDLE)
HBRUSH = getattr(wt, "HBRUSH", wt.HANDLE)

# --- Константы ---
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002

RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1

RIDEV_INPUTSINK = 0x00000100
RIDI_DEVICENAME = 0x20000007

RI_KEY_BREAK = 0x0001
RI_KEY_E0 = 0x0002
RI_KEY_E1 = 0x0004

# --- Подавление (глобальный hook) ---
SUPPRESS_OS = True
# F13..F24: 0x7C..0x87
SUPPRESS_VKEYS = set(range(0x7C, 0x88))

# Ваше устройство
FILTER_VIDPID = ("514C", "8850")

# --- ВАЖНО: прототипы WinAPI (иначе на x64 бывают OverflowError) ---
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]

user32.RegisterClassW.restype = wt.ATOM
user32.CreateWindowExW.restype = wt.HWND
user32.CreateWindowExW.argtypes = [
    wt.DWORD,
    wt.LPCWSTR,
    wt.LPCWSTR,
    wt.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wt.HWND,
    wt.HMENU,
    wt.HINSTANCE,
    wt.LPVOID,
]

user32.GetMessageW.restype = wt.BOOL
user32.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]

user32.TranslateMessage.restype = wt.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]

user32.DispatchMessageW.restype = LRESULT
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]

user32.PostQuitMessage.restype = None
user32.PostQuitMessage.argtypes = [ctypes.c_int]

user32.GetRawInputData.restype = wt.UINT
user32.GetRawInputData.argtypes = [
    wt.HANDLE,
    wt.UINT,
    wt.LPVOID,
    ctypes.POINTER(wt.UINT),
    wt.UINT,
]

user32.GetRawInputDeviceInfoW.restype = wt.UINT
user32.GetRawInputDeviceInfoW.argtypes = [
    wt.HANDLE,
    wt.UINT,
    wt.LPVOID,
    ctypes.POINTER(wt.UINT),
]

user32.RegisterRawInputDevices.restype = wt.BOOL

# --- Low-level keyboard hook прототипы ---
WH_KEYBOARD_LL = 13
HC_ACTION = 0
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

user32.SetWindowsHookExW.restype = wt.HHOOK
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, wt.HANDLE, wt.HINSTANCE, wt.DWORD]

user32.CallNextHookEx.restype = LRESULT
user32.CallNextHookEx.argtypes = [wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM]

user32.UnhookWindowsHookEx.restype = wt.BOOL
user32.UnhookWindowsHookEx.argtypes = [wt.HHOOK]

kernel32.GetModuleHandleW.restype = wt.HMODULE
kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]


# --- Raw Input структуры ---
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wt.USHORT),
        ("usUsage", wt.USHORT),
        ("dwFlags", wt.DWORD),
        ("hwndTarget", wt.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wt.DWORD),
        ("dwSize", wt.DWORD),
        ("hDevice", wt.HANDLE),
        ("wParam", wt.WPARAM),
    ]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wt.USHORT),
        ("Flags", wt.USHORT),
        ("Reserved", wt.USHORT),
        ("VKey", wt.USHORT),
        ("Message", wt.UINT),
        ("ExtraInformation", wt.ULONG),
    ]


class _RAWINPUTDATA(ctypes.Union):
    _fields_ = [("keyboard", RAWKEYBOARD)]


class RAWINPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("data", _RAWINPUTDATA),
    ]


user32.RegisterRawInputDevices.argtypes = [
    ctypes.POINTER(RAWINPUTDEVICE),
    wt.UINT,
    wt.UINT,
]


def vid_pid_from_path(path: str):
    m = re.search(r"VID_([0-9A-Fa-f]{4}).*PID_([0-9A-Fa-f]{4})", path)
    return (m.group(1).upper(), m.group(2).upper()) if m else None


def get_device_path(hDevice) -> str:
    pcbSize = wt.UINT(0)
    if (
        user32.GetRawInputDeviceInfoW(
            hDevice, RIDI_DEVICENAME, None, ctypes.byref(pcbSize)
        )
        == 0xFFFFFFFF
    ):
        return ""
    buf = ctypes.create_unicode_buffer(pcbSize.value)
    if (
        user32.GetRawInputDeviceInfoW(
            hDevice, RIDI_DEVICENAME, buf, ctypes.byref(pcbSize)
        )
        == 0xFFFFFFFF
    ):
        return ""
    return buf.value


def log(text: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {text}")


# --- Энкодеры (как было раньше; у вас они по-прежнему приходят как обычные клавиши) ---
# (MakeCode, E0) -> (knob, direction)
ENCODER_MAP = {
    (0x13, 0): ("LK", "CW"),  # R
    (0x19, 0): ("LK", "CCW"),  # P
    (0x1F, 0): ("RK", "CW"),  # S
    (0x16, 0): ("RK", "CCW"),  # U
}

device_cache = {}  # hDevice -> (path, vidpid)


def handle_raw_input(lParam):
    hRawInput = wt.HANDLE(lParam)

    dwSize = wt.UINT(0)
    if (
        user32.GetRawInputData(
            hRawInput,
            RID_INPUT,
            None,
            ctypes.byref(dwSize),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        == 0xFFFFFFFF
    ):
        return

    buf = ctypes.create_string_buffer(dwSize.value)
    if (
        user32.GetRawInputData(
            hRawInput,
            RID_INPUT,
            buf,
            ctypes.byref(dwSize),
            ctypes.sizeof(RAWINPUTHEADER),
        )
        == 0xFFFFFFFF
    ):
        return

    raw = RAWINPUT.from_buffer_copy(buf)
    if raw.header.dwType != RIM_TYPEKEYBOARD:
        return

    kb = raw.keyboard
    hdev = raw.header.hDevice

    if hdev not in device_cache:
        path = get_device_path(hdev)
        vp = vid_pid_from_path(path) if path else None
        device_cache[hdev] = (path, vp)

    _, vp = device_cache[hdev]
    if FILTER_VIDPID and vp != FILTER_VIDPID:
        return

    is_break = bool(kb.Flags & RI_KEY_BREAK)
    e0 = 1 if (kb.Flags & RI_KEY_E0) else 0

    make = int(kb.MakeCode)
    vkey = int(kb.VKey)

    # Энкодеры: считаем "тик" только на DOWN
    k = (make, e0)
    if k in ENCODER_MAP:
        if not is_break:
            knob, direction = ENCODER_MAP[k]
            log(f"ROTATE {knob} {direction}")
        return

    # Кнопки 1..12 теперь F13..F24 -> VKey 0x7C..0x87
    if 0x7C <= vkey <= 0x87:
        btn = vkey - 0x7B  # F13->1 ... F24->12
        log(f"KEY {btn:02d} {'UP' if is_break else 'DOWN'}")
        return

    log(
        f"UNKNOWN {'UP' if is_break else 'DOWN'} MakeCode=0x{make:02X} E0={e0} VKey=0x{vkey:02X}"
    )


# --- WndProc / message loop ---
WNDPROCTYPE = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


@WNDPROCTYPE
def WndProc(hwnd, msg, wParam, lParam):
    try:
        if msg == WM_INPUT:
            handle_raw_input(lParam)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wParam, lParam)
    except Exception as e:
        print("WndProc exception:", repr(e))
        return 0


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wt.UINT),
        ("lpfnWndProc", WNDPROCTYPE),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", HBRUSH),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
    ]


user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]


# --- Low-level hook структура ---
class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wt.DWORD),
        ("scanCode", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", wt.ULONG_PTR),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wt.WPARAM, wt.LPARAM)
_hook = None
_hook_proc = None


def install_suppress_hook():
    global _hook, _hook_proc

    @_hook_proc_type()
    def _proc(nCode, wParam, lParam):
        if nCode == HC_ACTION and wParam in (
            WM_KEYDOWN,
            WM_KEYUP,
            WM_SYSKEYDOWN,
            WM_SYSKEYUP,
        ):
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = int(kb.vkCode)
            if vk in SUPPRESS_VKEYS:
                return 1  # подавить
        return user32.CallNextHookEx(_hook, nCode, wParam, lParam)

    _hook_proc = _proc
    hInstance = kernel32.GetModuleHandleW(None)
    _hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, _hook_proc, hInstance, 0)
    if not _hook:
        raise ctypes.WinError()


def _hook_proc_type():
    return HOOKPROC


def uninstall_suppress_hook():
    global _hook
    if _hook:
        user32.UnhookWindowsHookEx(_hook)
        _hook = None


def main():
    hInstance = kernel32.GetModuleHandleW(None)

    wc = WNDCLASS()
    wc.lpfnWndProc = WndProc
    wc.hInstance = hInstance
    wc.lpszClassName = "CustomPadRawInputWindow"
    user32.RegisterClassW(ctypes.byref(wc))

    hwnd = user32.CreateWindowExW(
        0,
        wc.lpszClassName,
        "CustomPadRawInputWindow",
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        hInstance,
        None,
    )

    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01  # Generic Desktop Controls
    rid.usUsage = 0x06  # Keyboard
    rid.dwFlags = RIDEV_INPUTSINK
    rid.hwndTarget = hwnd

    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
        raise ctypes.WinError()

    if SUPPRESS_OS:
        install_suppress_hook()
        print("Suppression: ON (F13..F24 are swallowed globally).")
    else:
        print("Suppression: OFF.")

    print(
        "Logging custom keyboard (VID:PID=514C:8850). Buttons: 1..12 (F13..F24), knobs: LK/RK. Stop: Ctrl+C.\n"
    )

    msg = wt.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        uninstall_suppress_hook()


if __name__ == "__main__":
    main()
