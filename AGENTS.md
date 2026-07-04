# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Regenerating `screenshot.png`

The README screenshot is a headless render of the real Textual TUI, not a mockup. To regenerate it without a mic or downloaded ASR model (e.g. in a sandbox):

1. Monkeypatch `Voice2TextApp._detect_and_load` to a no-op *before* instantiating the app — it's a `@work(thread=True)` method that probes for a mic/GPU and loads a model, none of which are available headlessly.
2. Textual dispatches `on_mount` (and other event handlers) to **every class in the MRO that defines it**, not just the most-derived override — so subclassing `Voice2TextApp` and overriding `on_mount` runs *both* the subclass's and the base class's `on_mount` (subclass first, then base). Don't rely on a subclass `on_mount` to set final display state; instead drive the app from outside via `App.run_test()`, `await pilot.pause()`, then set state directly (`app.history`, `app.model_manager._active`/`_backend`/`_asr_model`, widget `.update(...)` calls, `app._update_status(...)`) after the initial mount has settled.
3. Export with `app.export_screenshot(title=...)` (produces SVG), then rasterize with `rsvg-convert -w <width> screenshot.svg -o screenshot.png` and optionally shrink with `optipng -o4`.
