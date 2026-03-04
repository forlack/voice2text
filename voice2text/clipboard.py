"""Clipboard support: system clipboard, tmux buffer, and OSC 52 fallback."""

import base64
import os
import shutil
import subprocess


def _run(cmd: list[str], input_text: str) -> bool:
    """Run a command with text piped to stdin. Returns True on success."""
    try:
        subprocess.run(
            cmd,
            input=input_text.encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _copy_system(text: str) -> bool:
    """Copy to system clipboard via wl-copy, xclip, pbcopy, or kitten."""
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
        with open("/dev/tty", "wb") as tty:
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
