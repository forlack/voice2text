"""Voice2Text TUI application."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Static,
)

from .clipboard import copy_to_clipboard
from .models import MODEL_REGISTRY, ModelInfo, ModelManager, delete_model_files, get_model_size_on_disk, is_model_downloaded, load_last_model, save_last_model
from .recorder import Recorder
from .transcripts import TranscriptEntry, load_history, save_transcript


# ── Custom Widgets ──────────────────────────────────────────────────────


class AudioLevelBar(Static):
    """Real-time audio level meter."""

    level: reactive[float] = reactive(0.0)
    recording: reactive[bool] = reactive(False)

    def render(self) -> str:
        width = max(self.size.width - 10, 10)
        filled = int(self.level * width)
        if self.recording:
            bar = "█" * filled + "░" * (width - filled)
            return f"  ▐{bar}▌ ● REC"
        bar = "█" * filled + "░" * (width - filled)
        return f"  ▐{bar}▌ Level"

    def watch_level(self, value: float) -> None:
        self.refresh()

    def watch_recording(self, value: bool) -> None:
        if value:
            self.add_class("recording")
        else:
            self.remove_class("recording")
        self.refresh()


class ModelPickerItem(ListItem):
    """A model entry in the picker list."""

    def __init__(self, info: ModelInfo, selected: bool = False) -> None:
        self.info = info
        downloaded = is_model_downloaded(info)
        icon = "●" if selected else "○"
        if downloaded:
            size = get_model_size_on_disk(info)
            status = f" ({size})"
        else:
            status = " ⬇ not downloaded"
        super().__init__(Label(f" {icon} {info.name}{status}"))


class HistoryItem(ListItem):
    """A transcript history entry."""

    def __init__(self, entry: TranscriptEntry) -> None:
        self.entry = entry
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        preview = entry.preview[:50] if entry.preview else "(empty)"
        super().__init__(Label(f" {ts}  {preview}"))


class DownloadProgress(Vertical):
    """Download progress display, hidden by default."""

    def compose(self) -> ComposeResult:
        yield Label("", id="download-label")
        yield ProgressBar(total=100, show_percentage=True, id="download-bar")

    def show_progress(self, fraction: float, text: str) -> None:
        self.query_one("#download-label", Label).update(text)
        self.query_one("#download-bar", ProgressBar).progress = fraction * 100
        self.display = True

    def hide_progress(self) -> None:
        self.display = False


# ── Download Confirmation Modal ─────────────────────────────────────────


class DownloadConfirmScreen(ModalScreen[bool]):
    """Modal dialog to confirm model download."""

    CSS = """
    DownloadConfirmScreen {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 2 4;
    }

    #confirm-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #confirm-body {
        width: 100%;
        content-align: center middle;
        margin-bottom: 2;
    }

    #confirm-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #confirm-buttons Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n", "confirm_no", "No", show=False),
        Binding("escape", "confirm_no", "Cancel", show=False),
    ]

    def __init__(self, info: ModelInfo) -> None:
        super().__init__()
        self.info = info

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static("Download Model?", id="confirm-title")
            yield Static(
                f"{self.info.name}\n{self.info.description}\nSize: {self.info.size_hint}",
                id="confirm-body",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes - Download (Y)", variant="success", id="btn-yes")
                yield Button("No - Cancel (N)", variant="error", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def action_confirm_yes(self) -> None:
        self.dismiss(True)

    def action_confirm_no(self) -> None:
        self.dismiss(False)


# ── Delete Confirmation Modal ──────────────────────────────────────────


class DeleteConfirmScreen(ModalScreen[bool]):
    """Modal dialog to confirm history deletion."""

    CSS = """
    DeleteConfirmScreen {
        align: center middle;
    }

    #delete-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 2 4;
    }

    #delete-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #delete-body {
        width: 100%;
        content-align: center middle;
        margin-bottom: 2;
    }

    #delete-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #delete-buttons Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n", "confirm_no", "No", show=False),
        Binding("escape", "confirm_no", "Cancel", show=False),
    ]

    def __init__(self, preview: str) -> None:
        super().__init__()
        self.preview = preview

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog"):
            yield Static("Delete Transcript?", id="delete-title")
            yield Static(self.preview, id="delete-body")
            with Horizontal(id="delete-buttons"):
                yield Button("Yes - Delete (Y)", variant="error", id="btn-yes")
                yield Button("No - Cancel (N)", variant="primary", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def action_confirm_yes(self) -> None:
        self.dismiss(True)

    def action_confirm_no(self) -> None:
        self.dismiss(False)


# ── Loading Modal ──────────────────────────────────────────────────────


class LoadingScreen(ModalScreen):
    """Modal overlay shown while a model is loading into memory."""

    CSS = """
    LoadingScreen {
        align: center middle;
    }

    #loading-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 2 4;
    }

    #loading-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #loading-body {
        width: 100%;
        content-align: center middle;
    }
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def compose(self) -> ComposeResult:
        with Vertical(id="loading-dialog"):
            yield Static("Loading Model...", id="loading-title")
            yield Static(
                f"{self.model_name}\nThis may take a few seconds.",
                id="loading-body",
            )


# ── Main App ────────────────────────────────────────────────────────────


class Voice2TextApp(App):
    """Voice-to-text TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        height: 1;
        dock: top;
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    #main-container {
        height: 1fr;
    }

    #left-panel {
        width: 2fr;
        height: 100%;
        border: solid $accent;
        padding: 1;
    }

    #right-panel {
        width: 1fr;
        height: 100%;
        border: solid $accent;
        padding: 1;
    }

    #mic-label {
        height: 1;
        color: $text-muted;
    }

    #level-bar {
        height: 1;
        margin-bottom: 1;
    }

    #level-bar.recording {
        color: #e06060;
    }

    #transcript-area {
        height: 1fr;
        border: round $primary;
        padding: 1;
        overflow-y: auto;
    }

    #model-picker-label {
        text-style: bold;
        margin-bottom: 1;
    }

    #model-list {
        height: auto;
        max-height: 8;
        border: round $secondary;
    }

    #download-progress {
        height: auto;
        margin-top: 1;
    }

    #download-progress ProgressBar {
        margin: 0 1;
    }

    #history-label {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }

    #history-list {
        height: 1fr;
        border: round $secondary;
    }
    """

    BINDINGS = [
        Binding("space", "toggle_record", "Record", show=True),
        Binding("i", "toggle_interactive", "Interactive Mode", show=True),
        Binding("x", "delete_selected", "Delete", show=True),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    TITLE = "Voice2Text"
    theme = "rose-pine"

    def __init__(self, force_cpu: bool = False) -> None:
        super().__init__()
        self.recorder = Recorder()
        self.model_manager = ModelManager(force_cpu=force_cpu)
        self.history: list[TranscriptEntry] = []
        self._record_start: float = 0.0
        self._level_task: asyncio.Task | None = None
        self._vad_task: asyncio.Task | None = None
        self._mic_name: str = ""
        self._interactive: bool = False
        self._vad = None  # VoiceActivityDetector instance during recording
        self._segment_texts: list[str] = []  # accumulated segment transcriptions
        self._segment_boundary: int = 0  # frame index of last segment end

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Starting...", id="status-bar")
        with Horizontal(id="main-container"):
            with Vertical(id="left-panel"):
                yield Static("  Mic: detecting...", id="mic-label")
                yield AudioLevelBar(id="level-bar")
                yield Static(
                    "Loading model, please wait...",
                    id="transcript-area",
                )
            with Vertical(id="right-panel"):
                yield Label("Model Picker", id="model-picker-label")
                yield ListView(
                    *self._build_model_items(),
                    id="model-list",
                )
                yield DownloadProgress(id="download-progress")
                yield Label("History", id="history-label")
                yield ListView(id="history-list")
        yield Footer()

    def _build_model_items(self) -> list[ModelPickerItem]:
        active = self.model_manager.active_model
        items = []
        for info in MODEL_REGISTRY:
            selected = active is not None and active.name == info.name
            items.append(ModelPickerItem(info, selected=selected))
        return items

    def on_mount(self) -> None:
        self.history = load_history()
        self._refresh_history()
        self.query_one("#download-progress", DownloadProgress).hide_progress()
        self._update_status("Detecting hardware...")
        # Kick off background loading — UI is already fully rendered above
        self._detect_and_load()

    @work(thread=True)
    def _detect_and_load(self) -> None:
        """Detect GPU, mic, and load first available model."""
        # Step 1: detect backend (fast)
        backend = self.model_manager.detect_backend()
        self.call_from_thread(
            self._update_status, f"Backend: {backend.upper()} | Detecting mic..."
        )

        # Step 2: detect mic (fast, but suppresses stderr)
        self._mic_name = Recorder.get_default_mic_name()
        self.call_from_thread(self._update_mic_label)
        self.call_from_thread(self._update_status, "Searching for models...")

        # Step 3: find and load a model (prefer last used)
        last = load_last_model()
        candidates = list(MODEL_REGISTRY)
        if last and is_model_downloaded(last):
            candidates = [last] + [m for m in candidates if m.name != last.name]
        for info in candidates:
            if is_model_downloaded(info):
                self.call_from_thread(
                    self._update_status, f"Loading {info.name}..."
                )
                self.call_from_thread(self.push_screen, LoadingScreen(info.name))
                try:
                    self.model_manager.load_model(info)
                    self.call_from_thread(self.pop_screen)
                    self.call_from_thread(self._on_model_loaded)
                    return
                except Exception as e:
                    self.call_from_thread(self.pop_screen)
                    self.call_from_thread(
                        self._update_status, f"Failed to load {info.name}: {e}"
                    )

        def _no_models():
            self._update_status(
                f"Backend: {backend.upper()} | No models downloaded. Select one from the picker."
            )
            self.query_one("#transcript-area", Static).update(
                "No models downloaded.\nSelect a model from the picker on the right to download it."
            )

        self.call_from_thread(_no_models)

    def _on_model_loaded(self) -> None:
        if self.model_manager.active_model:
            save_last_model(self.model_manager.active_model)
        self._refresh_model_list()
        self.query_one("#transcript-area", Static).update(
            "Press SPACE to start recording..."
        )
        self._update_status("Ready")

    def _update_status(self, text: str) -> None:
        """Update the top status bar with context info + message."""
        info = self.model_manager.active_model
        backend = self.model_manager.backend.upper()
        parts = []
        parts.append(backend)
        if self._interactive:
            parts.append("[bold blue]Interactive:[/bold blue] [bold]ON[/bold]")
        parts.append(text)
        self.query_one("#status-bar", Static).update(" | ".join(parts))

    def _update_mic_label(self) -> None:
        mic = self._mic_name or "detecting..."
        self.query_one("#mic-label", Static).update(f"  Mic: {mic}")

    def action_toggle_interactive(self) -> None:
        """Toggle interactive (chunked transcription) mode."""
        if self.recorder.is_recording:
            return  # don't toggle mid-recording
        self._interactive = not self._interactive
        self._update_status("Ready")

    def _refresh_model_list(self) -> None:
        lv = self.query_one("#model-list", ListView)
        lv.clear()
        for item in self._build_model_items():
            lv.append(item)

    def _refresh_history(self) -> None:
        lv = self.query_one("#history-list", ListView)
        lv.clear()
        for entry in self.history:
            lv.append(HistoryItem(entry))

    # ── History Delete ────────────────────────────────────────────────────

    def action_delete_selected(self) -> None:
        """Delete whatever is currently highlighted — model or history entry."""
        if self.recorder.is_recording:
            return

        # Check model list
        model_lv = self.query_one("#model-list", ListView)
        if model_lv.has_focus and model_lv.highlighted_child is not None:
            item = model_lv.highlighted_child
            if not isinstance(item, ModelPickerItem):
                return
            info = item.info
            if not is_model_downloaded(info):
                self._update_status(f"{info.name} is not downloaded")
                return

            size = get_model_size_on_disk(info)
            preview = f"{info.name} ({size})"

            def on_model_dismiss(result: bool) -> None:
                if not result:
                    return
                if (
                    self.model_manager.active_model
                    and self.model_manager.active_model.name == info.name
                ):
                    self.model_manager.unload()
                    self.query_one("#transcript-area", Static).update(
                        "No model loaded.\nSelect a model from the picker to download it."
                    )
                delete_model_files(info)
                self._refresh_model_list()
                self._update_status(f"Deleted {info.name}")

            self.push_screen(DeleteConfirmScreen(preview), callback=on_model_dismiss)
            return

        # Check history list
        history_lv = self.query_one("#history-list", ListView)
        if history_lv.has_focus and history_lv.highlighted_child is not None:
            item = history_lv.highlighted_child
            if not isinstance(item, HistoryItem):
                return
            entry = item.entry
            preview = entry.preview[:60] if entry.preview else "(empty)"

            def on_history_dismiss(result: bool) -> None:
                if not result:
                    return
                try:
                    entry.path.unlink()
                except FileNotFoundError:
                    pass
                self.history = [e for e in self.history if e.path != entry.path]
                self._refresh_history()
                self._update_status("Deleted transcript")

            self.push_screen(DeleteConfirmScreen(preview), callback=on_history_dismiss)

    # ── Recording ───────────────────────────────────────────────────────

    def action_toggle_record(self) -> None:
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if not self.model_manager.is_loaded:
            self._update_status("No model loaded! Select and download a model first.")
            return
        try:
            self.recorder.start()
        except Exception as e:
            self._update_status(f"Mic error: {e}")
            return

        self._record_start = time.monotonic()
        self._update_status("● Recording... | Press SPACE to stop")
        self.query_one("#transcript-area", Static).update("Recording...")
        bar = self.query_one("#level-bar", AudioLevelBar)
        bar.recording = True
        self._level_task = asyncio.ensure_future(self._poll_level())

        if self._interactive:
            from .vad import VoiceActivityDetector

            self._vad = VoiceActivityDetector()
            self._segment_texts = []
            self._segment_boundary = 0
            self._vad_task = asyncio.ensure_future(self._poll_vad())

    async def _poll_level(self) -> None:
        """Poll audio level while recording."""
        bar = self.query_one("#level-bar", AudioLevelBar)
        while self.recorder.is_recording:
            bar.level = self.recorder.level
            elapsed = time.monotonic() - self._record_start
            mins, secs = divmod(int(elapsed), 60)
            self._update_status(
                f"● Recording {mins:02d}:{secs:02d} | Press SPACE to stop"
            )
            await asyncio.sleep(0.05)
        bar.level = 0.0

    async def _poll_vad(self) -> None:
        """Poll audio chunks through VAD while recording in interactive mode."""
        from .vad import SILERO_CHUNK_SAMPLES

        silence_chunks = 0
        # ~1.5s of silence at 16kHz with 512-sample chunks = ~47 chunks
        silence_threshold = int(1.5 * 16000 / SILERO_CHUNK_SAMPLES)
        was_speech = False
        last_processed_frame = 0
        chunk_bytes = SILERO_CHUNK_SAMPLES * 2  # 16-bit = 2 bytes per sample

        while self.recorder.is_recording and self._vad is not None:
            current_frames = self.recorder.frame_count
            # Process any new frames
            while last_processed_frame < current_frames:
                with self.recorder._lock:
                    if last_processed_frame >= len(self.recorder._frames):
                        break
                    raw = self.recorder._frames[last_processed_frame]
                last_processed_frame += 1

                # Each frame is CHUNK=1024 samples = 2048 bytes
                # silero needs 512 samples = 1024 bytes, so split in two
                for offset in range(0, len(raw), chunk_bytes):
                    sub = raw[offset : offset + chunk_bytes]
                    if len(sub) < chunk_bytes:
                        break
                    is_speech = self._vad.is_speech(sub)
                    if is_speech:
                        was_speech = True
                        silence_chunks = 0
                    else:
                        silence_chunks += 1

                    if was_speech and silence_chunks >= silence_threshold:
                        # Speech→silence transition: extract and transcribe segment
                        seg_end = last_processed_frame
                        if seg_end > self._segment_boundary:
                            wav_seg = self.recorder.extract_segment(
                                self._segment_boundary, seg_end
                            )
                            self._segment_boundary = seg_end
                            self._transcribe_segment(wav_seg)
                        was_speech = False
                        silence_chunks = 0

            await asyncio.sleep(0.05)

    @work(thread=True)
    def _transcribe_segment(self, wav_data: bytes) -> None:
        """Transcribe a single segment and append to transcript area."""
        try:
            text = self.model_manager.transcribe(wav_data)
        except Exception:
            return
        if not text or not text.strip():
            return
        self._segment_texts.append(text.strip())

        def _update():
            combined = " ".join(self._segment_texts)
            self.query_one("#transcript-area", Static).update(combined)

        self.call_from_thread(_update)

    @work(thread=True)
    def _transcribe_final_segment(self, wav_data: bytes) -> None:
        """Transcribe the final remaining segment, then finalize."""
        try:
            text = self.model_manager.transcribe(wav_data)
            if text and text.strip():
                self._segment_texts.append(text.strip())
        except Exception:
            pass
        self.call_from_thread(self._finalize_interactive)

    def _finalize_interactive(self) -> None:
        """Combine all segment texts, save, and copy to clipboard."""
        full_text = " ".join(self._segment_texts)
        if not full_text.strip():
            self.query_one("#transcript-area", Static).update(
                "No speech detected. Press SPACE to record."
            )
            self._update_status("Ready")
            return

        entry = save_transcript(full_text)
        clip_msg = copy_to_clipboard(full_text)
        self.query_one("#transcript-area", Static).update(full_text)
        self.history.insert(0, entry)
        self._refresh_history()
        self._update_status(f"{clip_msg} | Ready")

    def _stop_recording(self) -> None:
        is_interactive = self._interactive and self._vad is not None

        # Cancel VAD polling before stopping recorder
        if self._vad_task:
            self._vad_task.cancel()
            self._vad_task = None

        if is_interactive:
            # Extract any remaining audio since last segment boundary
            final_boundary = self.recorder.frame_count
            remaining_wav = None
            if final_boundary > self._segment_boundary:
                remaining_wav = self.recorder.extract_segment(
                    self._segment_boundary, final_boundary
                )

        self._update_status("Transcribing...")
        wav_data = self.recorder.stop()
        if self._level_task:
            self._level_task.cancel()
            self._level_task = None
        bar = self.query_one("#level-bar", AudioLevelBar)
        bar.level = 0.0
        bar.recording = False
        self._vad = None

        if is_interactive:
            if remaining_wav:
                self._transcribe_final_segment(remaining_wav)
            else:
                self._finalize_interactive()
        else:
            self.query_one("#transcript-area", Static).update("Transcribing...")
            self._transcribe(wav_data)

    @work(thread=True)
    def _transcribe(self, wav_data: bytes) -> None:
        """Run transcription in background thread."""
        t0 = time.monotonic()
        try:
            text = self.model_manager.transcribe(wav_data)
        except Exception as e:
            self.call_from_thread(self._update_status, f"Transcription error: {e}")
            return
        elapsed = time.monotonic() - t0

        # Save and display
        entry = save_transcript(text)
        clip_msg = copy_to_clipboard(text)

        def _update():
            self.query_one("#transcript-area", Static).update(text)
            self.history.insert(0, entry)
            self._refresh_history()
            self._update_status(f"{clip_msg} | {elapsed:.1f}s | Ready")

        self.call_from_thread(_update)

    # ── Model Picker ────────────────────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item

        # Model picker selection
        if isinstance(item, ModelPickerItem):
            info = item.info
            if is_model_downloaded(info):
                if (
                    self.model_manager.active_model
                    and self.model_manager.active_model.name == info.name
                ):
                    return  # already active
                self._update_status(f"Switching to {info.name}...")
                self._load_model_async(info)
            else:
                self._show_download_confirm(info)
            return

        # History selection
        if isinstance(item, HistoryItem):
            entry = item.entry
            text = entry.full_text()
            self.query_one("#transcript-area", Static).update(text)
            clip_msg = copy_to_clipboard(text)
            self._update_status(f"{clip_msg} | Loaded from history")

    def _show_download_confirm(self, info: ModelInfo) -> None:
        """Show a modal dialog to confirm model download."""

        def on_dismiss(result: bool) -> None:
            if result:
                self._download_model(info)
            else:
                self._update_status("Download cancelled.")

        self.push_screen(DownloadConfirmScreen(info), callback=on_dismiss)

    @work(thread=True)
    def _download_model(self, info: ModelInfo) -> None:
        import traceback

        progress_widget = self.query_one("#download-progress", DownloadProgress)

        def on_progress(fraction: float, text: str) -> None:
            self.call_from_thread(progress_widget.show_progress, fraction, text)
            self.call_from_thread(self._update_status, text)

        log_path = Path(__file__).resolve().parent.parent / "error.log"

        try:
            self.model_manager.download_model(info, progress_cb=on_progress)
        except Exception as e:
            self.call_from_thread(progress_widget.hide_progress)
            tb = traceback.format_exc()
            log_path.write_text(f"DOWNLOAD FAILED: {info.name}\n\n{tb}")
            self.call_from_thread(self._update_status, f"Download failed: {e} (see error.log)")
            return

        try:
            self.call_from_thread(progress_widget.hide_progress)
            self.call_from_thread(self._update_status, f"Loading {info.name}...")
            self.call_from_thread(self.push_screen, LoadingScreen(info.name))
            self.model_manager.load_model(info)
            self.call_from_thread(self.pop_screen)
            self.call_from_thread(self._on_model_loaded)
        except Exception as e:
            try:
                self.call_from_thread(self.pop_screen)
            except Exception:
                pass
            tb = traceback.format_exc()
            log_path.write_text(f"LOAD FAILED: {info.name}\n\n{tb}")
            self.call_from_thread(self._update_status, f"Load failed: {e} (see error.log)")

    @work(thread=True)
    def _load_model_async(self, info: ModelInfo) -> None:
        self.call_from_thread(self.push_screen, LoadingScreen(info.name))
        try:
            self.model_manager.load_model(info)
            self.call_from_thread(self.pop_screen)
            self.call_from_thread(self._on_model_loaded)
        except Exception as e:
            self.call_from_thread(self.pop_screen)
            self.call_from_thread(
                self._update_status, f"Failed to load {info.name}: {e}"
            )

    def action_quit_app(self) -> None:
        if self.model_manager.active_model:
            save_last_model(self.model_manager.active_model)
        self.exit()


def main() -> None:
    import argparse

    # Pre-initialize tqdm's multiprocessing lock while fds are still valid.
    # tqdm creates a multiprocessing.RLock in __new__ which spawns the
    # resource tracker subprocess.  Inside Textual's worker threads the fd
    # table is redirected, so Python 3.14's fork_exec rejects the fds.
    # Doing it here (before app.run) ensures fds are in a good state.
    import tqdm.std
    tqdm.std.TqdmDefaultWriteLock()

    parser = argparse.ArgumentParser(description="Voice2Text TUI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (skip CUDA)")
    args = parser.parse_args()

    app = Voice2TextApp(force_cpu=args.cpu)
    app.run()


if __name__ == "__main__":
    main()
