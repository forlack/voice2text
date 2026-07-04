# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Regenerating `screenshot.png`

The README screenshot is a headless render of the real Textual TUI, not a mockup. To regenerate it without a mic or downloaded ASR model (e.g. in a sandbox):

1. Monkeypatch `Voice2TextApp._detect_and_load` to a no-op *before* instantiating the app — it's a `@work(thread=True)` method that probes for a mic/GPU and loads a model, none of which are available headlessly.
2. Textual dispatches `on_mount` (and other event handlers) to **every class in the MRO that defines it**, not just the most-derived override — so subclassing `Voice2TextApp` and overriding `on_mount` runs *both* the subclass's and the base class's `on_mount` (subclass first, then base). Don't rely on a subclass `on_mount` to set final display state; instead drive the app from outside via `App.run_test()`, `await pilot.pause()`, then set state directly (`app.history`, `app.model_manager._active`/`_backend`/`_asr_model`, widget `.update(...)` calls, `app._update_status(...)`) after the initial mount has settled.
3. Export with `app.export_screenshot(title=...)` (produces SVG), then rasterize with `rsvg-convert -w <width> screenshot.svg -o screenshot.png` and optionally shrink with `optipng -o4`.

## Test/dev dependencies and the download diagnostic script

- `pytest`/`pytest-asyncio` are declared under the `test` extra in `pyproject.toml` — install with `pip install -e .[test]` before running `pytest tests/test_app.py`. They are not part of the base `dependencies` list.
- `tests/debug_download.py` (not `test_*`, deliberately) is a manual diagnostic script that downloads real files from HuggingFace, including a ~640MB ONNX model. Run it directly with `python -m tests.debug_download`; it is intentionally excluded from pytest's default `test_*` discovery so plain `pytest` never triggers a network download.
- `tests/test_app.py::test_post_process_no_text` is flaky/pre-existing-broken: it relies on a fixed `asyncio.sleep(3)` timing window and fails independent of unrelated changes.

## History list width-based truncation

`HistoryItem` in `app.py` truncates its displayed preview to the item's own `self.size.width` (via `on_resize`, which Textual fires per-widget when its container resizes — not just once at mount), appending "…" when the full preview doesn't fit. `entry.preview`/`entry.full_text()` (`transcripts.py`) are never touched by this — the label's rendered text is display-only, and copy/save operations always read from `entry`. Tests can drive width changes with `pilot.resize_terminal(width, height)` inside `run_test`, and must read a `Label`'s current text via `label.content` — this Textual version (8.2.8) has no `label.renderable`.

## config.toml reading/writing

All reads and writes of `config.toml` go through `voice2text/config.py` (`load_config()` / `save_config_value()`), which uses `tomlkit` instead of `tomllib`/`tomli`. `tomlkit` round-trips comments and formatting, so a user who copies the heavily-commented `config.toml.example` and changes one setting via the in-app menu keeps their comments — a hand-rolled writer (previous implementation of `Voice2TextApp._save_config_value`) silently dropped them on every save. Don't reintroduce a local `tomllib`/`tomli` import in `app.py`/`models.py`/`postprocess.py`; route through `voice2text/config.py` instead.
