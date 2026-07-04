"""Shared config.toml loading/saving, preserving comments and formatting."""

from __future__ import annotations

from pathlib import Path

import tomlkit

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"


def load_config() -> tomlkit.TOMLDocument:
    """Load config.toml if present, else an empty document."""
    if not CONFIG_FILE.exists():
        return tomlkit.document()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return tomlkit.load(f)
    except Exception:
        return tomlkit.document()


def save_config_value(section: str, key: str, value: object) -> None:
    """Write a single config value to config.toml, preserving comments/formatting."""
    config = load_config()
    if section not in config:
        config[section] = tomlkit.table()
    config[section][key] = value
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        tomlkit.dump(config, f)
