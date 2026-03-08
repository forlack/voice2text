"""Post-process transcriptions via an external LLM CLI tool."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"

_DEFAULT_COMMAND = "claude"
_DEFAULT_PROMPT = (
    "Fix grammar, punctuation, and capitalization in this voice transcription. "
    "Return only the corrected text, nothing else."
)


def _load_config() -> dict:
    """Load post-processing config from config.toml if present."""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(_CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
    except Exception:
        return {}
    return config.get("post_processing", {})


def get_command() -> str:
    cfg = _load_config()
    return cfg.get("command", _DEFAULT_COMMAND)


def get_prompt() -> str:
    cfg = _load_config()
    return cfg.get("prompt", _DEFAULT_PROMPT)


def is_command_available() -> bool:
    """Check if the configured CLI tool is installed."""
    return shutil.which(get_command()) is not None


def correct(text: str) -> str:
    """Run the configured LLM CLI tool on text and return corrected output.

    Raises RuntimeError if the command fails or is not found.
    """
    command = get_command()
    prompt = get_prompt()

    exe = shutil.which(command)
    if exe is None:
        raise RuntimeError(
            f"'{command}' not found. Install it or configure "
            "[post_processing] command in config.toml"
        )

    full_prompt = f"{prompt}\n\n{text}"

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    result = subprocess.run(
        [exe, "-p", full_prompt],
        capture_output=True,
        text=True,
        timeout=30,
        **kwargs,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"{command} failed (exit {result.returncode}): {stderr}")

    corrected = result.stdout.strip()
    if not corrected:
        raise RuntimeError(f"{command} returned empty output")

    return corrected
