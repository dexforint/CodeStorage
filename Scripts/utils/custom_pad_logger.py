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

# --- Фильтр устройства (ваша кастомная клавиатура) ---
FILTER_VIDPID = ("514C", "8850")


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


# --- Маппинг (по вашему логу, в порядке нажатий) ---
# Ключ: (MakeCode, E0)
# KEYMAP = {
#     (0x20, 0): 1,  # D
#     (0x23, 0): 2,  # H
#     (0x26, 0): 3,  # L
#     (0x2E, 0): 4,  # C
#     (0x22, 0): 5,  # G
#     (0x25, 0): 6,  # K
#     (0x30, 0): 7,  # B
#     (0x21, 0): 8,  # F
#     (0x24, 0): 9,  # J
#     (0x1E, 0): 10,  # A
#     (0x12, 0): 11,  # E
#     (0x17, 0): 12,  # I
# }
KEYMAP = {
    (0x64, 0): 1,  # D
    (0x65, 0): 2,  # H
    (0x66, 0): 3,  # L
    (0x67, 0): 4,  # C
    (0x68, 0): 5,  # G
    (0x69, 0): 6,  # K
    (0x6A, 0): 7,  # B
    (0x6B, 0): 8,  # F
    (0x6C, 0): 9,  # J
    (0x6D, 0): 10,  # A
    (0x6E, 0): 11,  # E
    (0x76, 0): 12,  # I
}

# Энкодеры (по вашему логу похоже на 2 пары кодов):
#  - левая крутилка: R/P  (0x13 / 0x19)
#  - правая крутилка: S/U (0x1F / 0x16)
# Направления CW/CCW при необходимости поменяйте местами.
ENCODER_MAP = {
    (0x13, 0): ("LK", "CW"),
    (0x19, 0): ("LK", "CCW"),
    (0x1F, 0): ("RK", "CW"),
    (0x16, 0): ("RK", "CCW"),
}

device_cache = {}  # hDevice -> (path, vidpid)


def log_event(text: str):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {text}")


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
    is_e0 = 1 if (kb.Flags & RI_KEY_E0) else 0

    make = int(kb.MakeCode)
    key = (make, is_e0)

    # Для энкодера считаем "тик" только по DOWN (иначе будет дублирование на UP)
    if key in ENCODER_MAP:
        if not is_break:
            knob, direction = ENCODER_MAP[key]
            log_event(f"ROTATE {knob} {direction}")
        return

    # Обычные кнопки: логируем DOWN/UP
    if key in KEYMAP:
        n = KEYMAP[key]
        action = "UP" if is_break else "DOWN"
        log_event(f"KEY {n:02d} {action}")
        return

    # На всякий случай: неизвестные события (если появятся новые)
    action = "UP" if is_break else "DOWN"
    log_event(
        f"UNKNOWN {action} MakeCode=0x{make:02X} E0={is_e0} VKey=0x{int(kb.VKey):02X}"
    )


# --- Window procedure / message loop ---
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
        # чтобы не было "Exception ignored on calling ctypes callback function"
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

    print("Logging custom keyboard (VID:PID=514C:8850).")
    print("Buttons: 1..12, knobs: LK/RK. Stop: Ctrl+C.\n")

    msg = wt.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
