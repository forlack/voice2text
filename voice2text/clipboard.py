"""Clipboard support: system clipboard, tmux buffer, and OSC 52 fallback."""

import base64
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)


def _run(cmd: list[str], input_text: str) -> bool:
    """Run a command with text piped to stdin. Returns True on success."""
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        subprocess.run(
            cmd,
            input=input_text.encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            **kwargs,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _copy_system(text: str) -> bool:
    """Copy to system clipboard via wl-copy, xclip, pbcopy, kitten, or Win32 API."""
    # Windows
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes as w

            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002

            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32

            # Set proper argtypes/restype for 64-bit handle safety
            kernel32.GlobalAlloc.argtypes = [w.UINT, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = w.HGLOBAL
            kernel32.GlobalLock.argtypes = [w.HGLOBAL]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [w.HGLOBAL]
            user32.OpenClipboard.argtypes = [w.HWND]
            user32.SetClipboardData.argtypes = [w.UINT, w.HANDLE]

            encoded = text.encode("utf-16le") + b"\x00\x00"
            hmem = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
            if not hmem:
                return False
            ptr = kernel32.GlobalLock(hmem)
            if not ptr:
                return False
            ctypes.memmove(ptr, encoded, len(encoded))
            kernel32.GlobalUnlock(hmem)

            if not user32.OpenClipboard(None):
                return False
            user32.EmptyClipboard()
            user32.SetClipboardData(CF_UNICODETEXT, hmem)
            user32.CloseClipboard()
            return True
        except Exception:
            log.exception("Win32 clipboard failed")
            return False
    # Wayland
    if shutil.which("wl-copy"):
        return _run(["wl-copy"], text)
    # Kitty terminal
    if shutil.which("kitten"):
        return _run(["kitten", "clipboard"], text)
    # X11
    if shutil.which("xclip"):
        return _run(["xclip", "-selection", "clipboard"], text)
    if shutil.which("xsel"):
        return _run(["xsel", "--clipboard", "--input"], text)
    # macOS
    if shutil.which("pbcopy"):
        return _run(["pbcopy"], text)
    return False


def _copy_tmux(text: str) -> bool:
    """Copy to tmux paste buffer (+ system clipboard via OSC 52).

    Uses -w flag so tmux also sends OSC 52 to the outer terminal,
    which sets the system clipboard without needing wl-copy/xclip.
    Requires tmux set-clipboard on/external.
    """
    if not os.environ.get("TMUX"):
        return False
    if not shutil.which("tmux"):
        return False
    return _run(["tmux", "load-buffer", "-w", "-"], text)


def _copy_osc52(text: str) -> bool:
    """Copy via OSC 52 escape sequence written to /dev/tty.

    Last resort fallback — works in terminals that support OSC 52
    (kitty, foot, alacritty, wezterm, ghostty, etc.).
    """
    try:
        encoded = base64.b64encode(text.encode()).decode()
        osc = f"\033]52;c;{encoded}\a"
        tty_path = "CON" if sys.platform == "win32" else "/dev/tty"
        with open(tty_path, "wb") as tty:
            tty.write(osc.encode())
            tty.flush()
        return True
    except OSError:
        return False


def copy_to_clipboard(text: str) -> str:
    """Copy text to clipboard(s). Returns a status message."""
    sys_ok = _copy_system(text)
    tmux_ok = _copy_tmux(text)

    if sys_ok and tmux_ok:
        return "Copied to clipboard + tmux"
    if sys_ok:
        return "Copied to clipboard"
    if tmux_ok:
        return "Copied to clipboard"

    # Last resort: direct OSC 52
    if _copy_osc52(text):
        return "Copied to clipboard"

    return "Saved to file (clipboard unavailable)"
