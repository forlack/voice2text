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

`HistoryItem` in `app.py` ellipsizes its label to the panel width via Textual CSS (`DEFAULT_CSS`: `text-wrap: nowrap; text-overflow: ellipsis; width: 1fr` on the child `Label`) — **not** manual `self.size.width`/`on_resize` tracking. CSS `text-overflow` is applied at paint time against the widget's real box width, so it's correct on the very first frame and self-corrects on resize, with no size state to go stale.

**Critical gotcha that broke two prior fix attempts:** the label text must be the transcript's *full first line* (`HistoryItem._display_text` reads `entry.full_text()`, whitespace-collapsed, capped at `_MAX_DISPLAY_CHARS=512`), **not** `entry.preview`. `entry.preview` is hard-capped at 80 chars everywhere it's set (`transcripts.py`, and `app.py` grammar-correct/undo). On a **wide** terminal (History panel > 80 cols — the common real case) an 80-char string *fits* the panel, so CSS ellipsis never triggers: you get a mid-word 80-char slice, no "…", and dead space to the right. Both earlier attempts (manual width tracking, then CSS) only ever fed `preview`, and both "passed" their own headless verification because `App.run_test()`/`export_screenshot()` defaulted to an 80-col terminal (panel < 80 → ellipsis appears → bug hidden). They failed in the captain's real 100+-col `uv run voice2text` terminal. `entry.preview`/`entry.full_text()` themselves are never mutated by display truncation — copy/save always read from `entry`.

**How to verify History-panel rendering for real (not with `App.run_test()` alone):** run the actual app in a real pty and screen-scrape it. Launch a driver that monkeypatches `Voice2TextApp._detect_and_load` to a no-op and `voice2text.app.load_history`/`voice2text.transcripts.load_history` to return `TranscriptEntry`s backed by real temp files with long text, then `app.run()` inside `tmux new-session -d -s v2t -x <W> -y 30` and read it back with `tmux capture-pane -t v2t -p`. Test at **wide** widths (100, 140), not just 80 — a fresh session gets no user-driven resize, so the first settled frame at the real terminal width is what must be correct. In `pytest`, the equivalent guard is `test_history_item_ellipsizes_to_real_panel_width` (parametrized across 50/100/155): it drives the **real** `Voice2TextApp` tree with a real backing file and asserts on the **painted** strip (`widget.render_line(0)` joined) — `label.content`/`label.render()` return the *unclipped* source renderable, so CSS ellipsis is invisible to them. This Textual version is 8.2.8.

## Interactive (VAD) mode: waiting on in-flight segment workers

`_poll_vad` dispatches one `_transcribe_segment` `@work(thread=True)` worker per detected speech->silence transition, with no ordering/sync guarantee against when the user presses SPACE. `_stop_recording` must not call `_finalize_interactive` (which saves history + clipboard) while any of those workers are still running, or the saved/copied text is silently truncated by whatever text hadn't been appended to `_segment_texts` yet — this happened in production because `_transcribe_segment` typically takes 200-500ms, well within normal human reaction time after the last VAD-detected pause. `_stop_recording` and `action_toggle_record` are therefore `async def`; `_stop_recording` collects each dispatched worker in `self._segment_workers` and awaits `self.workers.wait_for_complete(self._segment_workers)` before finalizing. Don't revert these to sync or reintroduce a fire-and-forget call to `_transcribe_segment` without tracking its worker.

## config.toml reading/writing

All reads and writes of `config.toml` go through `voice2text/config.py` (`load_config()` / `save_config_value()`), which uses `tomlkit` instead of `tomllib`/`tomli`. `tomlkit` round-trips comments and formatting, so a user who copies the heavily-commented `config.toml.example` and changes one setting via the in-app menu keeps their comments — a hand-rolled writer (previous implementation of `Voice2TextApp._save_config_value`) silently dropped them on every save. Don't reintroduce a local `tomllib`/`tomli` import in `app.py`/`models.py`/`postprocess.py`; route through `voice2text/config.py` instead.
