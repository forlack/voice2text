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

`HistoryItem` in `app.py` ellipsizes its preview via Textual CSS (`text-wrap: nowrap; text-overflow: ellipsis;` on the child `Label`, `width: 1fr` so it's actually given the panel's full width), **not** manual width tracking. Textual applies `text-overflow` at paint time against the widget's real box width on every frame, so it's correct from the very first rendered frame and self-corrects on any resize — there is no `self.size`/`on_resize`/`on_mount` state to go stale.

A prior implementation manually truncated text to `self.size.width` in `on_mount`/`on_resize`. It was broken in real usage: `self.size` reads as `(0, 0)` at the point `on_mount` runs (before the first layout pass), so the manual truncation saw a negative/zero width, skipped truncating, and left the *full* untruncated string as the Label's content — which the terminal then hard-clipped at paint time with no ellipsis and (for shorter entries added later at a since-corrected width) stale narrower truncation, leaving dead space. That version's own test passed anyway because it exercised a bespoke minimal `App`/`ListView`, not the real `Voice2TextApp` widget tree, and always drove a `pilot.pause()` + `resize_terminal()` cycle that happened to let the buggy `on_resize` self-correct in the test harness — a scenario a real fresh session (no user-initiated resize) never gets. Lesson: when testing widget sizing/layout bugs, exercise the real app's compose tree, not a minimal stand-in, and check the state that would be visible after normal startup, not after an artificial resize.

`entry.preview`/`entry.full_text()` (`transcripts.py`) are never touched by any of this — copy/save operations always read from `entry`; only the Label's rendered text is display-only. Rendered truncation is only observable via an actual paint (`widget.render_line(0)`), not via `label.content`/`label.render()`, which return the unclipped source renderable.

## config.toml reading/writing

All reads and writes of `config.toml` go through `voice2text/config.py` (`load_config()` / `save_config_value()`), which uses `tomlkit` instead of `tomllib`/`tomli`. `tomlkit` round-trips comments and formatting, so a user who copies the heavily-commented `config.toml.example` and changes one setting via the in-app menu keeps their comments — a hand-rolled writer (previous implementation of `Voice2TextApp._save_config_value`) silently dropped them on every save. Don't reintroduce a local `tomllib`/`tomli` import in `app.py`/`models.py`/`postprocess.py`; route through `voice2text/config.py` instead.
