import ctypes
import ctypes.wintypes as wt
import re
import time

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- Fallback-типы (на случай неполного wintypes в 3.12) ---
LRESULT = getattr(wt, "LRESULT", ctypes.c_ssize_t)
HICON = getattr(wt, "HICON", wt.HANDLE)
HCURSOR = getattr(wt, "HCURSOR", wt.HANDLE)
HBRUSH = getattr(wt, "HBRUSH", wt.HANDLE)

# --- WinAPI константы ---
WM_INPUT = 0x00FF
WM_DESTROY = 0x0002

RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1

RIDEV_INPUTSINK = 0x00000100

RIDI_DEVICENAME = 0x20000007

RI_KEY_BREAK = 0x0001
RI_KEY_E0 = 0x0002
RI_KEY_E1 = 0x0004

# --- Прототипы WinAPI (важно для x64, иначе OverflowError) ---
user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]

user32.RegisterClassW.restype = wt.ATOM
# argtypes зададим после объявления WNDCLASS

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

user32.RegisterRawInputDevices.restype = wt.BOOL
# argtypes зададим после объявления RAWINPUTDEVICE

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

user32.GetKeyNameTextW.restype = ctypes.c_int
user32.GetKeyNameTextW.argtypes = [wt.LPARAM, wt.LPWSTR, ctypes.c_int]


# --- Структуры Raw Input ---
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


# Теперь можно задать argtypes для функций, где используются эти структуры
user32.RegisterRawInputDevices.argtypes = [
    ctypes.POINTER(RAWINPUTDEVICE),
    wt.UINT,
    wt.UINT,
]


def get_device_path(hDevice) -> str:
    pcbSize = wt.UINT(0)
    res = user32.GetRawInputDeviceInfoW(
        hDevice, RIDI_DEVICENAME, None, ctypes.byref(pcbSize)
    )
    if res == 0xFFFFFFFF:
        return "<GetRawInputDeviceInfoW failed>"

    buf = ctypes.create_unicode_buffer(pcbSize.value)
    res = user32.GetRawInputDeviceInfoW(
        hDevice, RIDI_DEVICENAME, buf, ctypes.byref(pcbSize)
    )
    if res == 0xFFFFFFFF:
        return "<GetRawInputDeviceInfoW failed>"
    return buf.value


def vid_pid_from_path(path: str):
    m = re.search(r"VID_([0-9A-Fa-f]{4}).*PID_([0-9A-Fa-f]{4})", path)
    return (m.group(1).upper(), m.group(2).upper()) if m else None


def key_name_from_scancode(make_code: int, is_e0: bool) -> str:
    lparam = (make_code & 0xFF) << 16
    if is_e0:
        lparam |= 1 << 24
    buf = ctypes.create_unicode_buffer(64)
    n = user32.GetKeyNameTextW(lparam, buf, 64)
    return buf.value if n else ""


# Фильтр: только ваша кастомная клавиатура
FILTER_VIDPID = ("514C", "8850")

device_cache = {}  # hDevice -> (path, (vid,pid))


def handle_raw_input(lParam):
    dwSize = wt.UINT(0)
    hRawInput = wt.HANDLE(lParam)  # handle WM_INPUT

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
        vp = vid_pid_from_path(path)
        device_cache[hdev] = (path, vp)

    path, vp = device_cache[hdev]
    if FILTER_VIDPID and vp != FILTER_VIDPID:
        return

    is_break = bool(kb.Flags & RI_KEY_BREAK)
    is_e0 = bool(kb.Flags & RI_KEY_E0)
    is_e1 = bool(kb.Flags & RI_KEY_E1)

    make = int(kb.MakeCode)
    vkey = int(kb.VKey)
    name = key_name_from_scancode(make, is_e0)

    now = time.strftime("%H:%M:%S")
    action = "UP  " if is_break else "DOWN"
    print(
        f"[{now}] {action} "
        f"VKey=0x{vkey:02X} MakeCode=0x{make:02X} E0={int(is_e0)} E1={int(is_e1)} "
        f'Name="{name}"'
    )


# --- Окно + WndProc ---
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
    wc.lpszClassName = "RawInputHiddenWindow"

    user32.RegisterClassW(ctypes.byref(wc))  # если уже есть — вернёт 0, это не страшно

    hwnd = user32.CreateWindowExW(
        0,
        wc.lpszClassName,
        "RawInputHiddenWindow",
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
    rid.usUsagePage = 0x01  # Generic Desktop
    rid.usUsage = 0x06  # Keyboard
    rid.dwFlags = RIDEV_INPUTSINK
    rid.hwndTarget = hwnd

    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
        raise ctypes.WinError()

    print(
        "Слушаю кастомную клавиатуру VID:PID=514C:8850 (Raw Input). Нажимайте/крутите.\n"
    )

    msg = wt.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


if __name__ == "__main__":
    main()
