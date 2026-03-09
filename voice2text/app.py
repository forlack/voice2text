"""Voice2Text TUI application."""

from __future__ import annotations

import asyncio
import logging
import shutil
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
        preview = entry.preview[:70] if entry.preview else "(empty)"
        super().__init__(Label(f" - {preview}"))


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


class MenuItem(ListItem):
    """A menu action entry."""

    def __init__(self, key: str, label: str, action: str) -> None:
        self.action_name = action
        super().__init__(Label(f" [{key}]  {label}"))


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


# ── Settings Modals ───────────────────────────────────────────────────


GRAMMAR_COMMANDS = ["claude", "gemini", "codex"]


class CommandPickerScreen(ModalScreen[str | None]):
    """Modal for selecting the grammar correction CLI tool."""

    CSS = """
    CommandPickerScreen {
        align: center middle;
    }

    #cmd-dialog {
        width: 45;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #cmd-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #cmd-list {
        height: auto;
        max-height: 10;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, current: str = "claude") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="cmd-dialog"):
            yield Static("Grammar Command", id="cmd-title")
            items = []
            for cmd in GRAMMAR_COMMANDS:
                selected = " *" if cmd == self._current else ""
                installed = shutil.which(cmd) is not None
                status = "" if installed else " [dim](not installed)[/dim]"
                items.append(ListItem(Label(f"  {cmd}{selected}{status}"), name=cmd))
            yield ListView(*items, id="cmd-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.name)

    def action_cancel(self) -> None:
        self.dismiss(None)


SILENCE_PRESETS = [0.3, 0.5, 0.7, 1.0, 1.5]


class SilencePickerScreen(ModalScreen[float | None]):
    """Modal for selecting silence threshold from presets."""

    CSS = """
    SilencePickerScreen {
        align: center middle;
    }

    #silence-dialog {
        width: 45;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #silence-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #silence-list {
        height: auto;
        max-height: 10;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, current: float = 0.5) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="silence-dialog"):
            yield Static("Silence Threshold (seconds)", id="silence-title")
            items = []
            for val in SILENCE_PRESETS:
                marker = " *" if abs(val - self._current) < 0.01 else ""
                items.append(ListItem(Label(f"  {val:.1f}s{marker}"), name=str(val)))
            yield ListView(*items, id="silence-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(float(event.item.name))

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Menu Modal ────────────────────────────────────────────────────────


class MenuScreen(ModalScreen[str | None]):
    """Main menu modal with actions."""

    CSS = """
    MenuScreen {
        align: center middle;
    }

    #menu-dialog {
        width: 55;
        height: auto;
        max-height: 22;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #menu-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #menu-list {
        height: auto;
        max-height: 18;
    }
    """

    BINDINGS = [
        Binding("escape", "close_menu", "Close", show=False),
        Binding("m", "close_menu", "Close", show=False),
        Binding("1", "pick_1", show=False),
        Binding("2", "pick_2", show=False),
        Binding("3", "pick_3", show=False),
        Binding("4", "pick_4", show=False),
        Binding("5", "pick_5", show=False),
        Binding("6", "pick_6", show=False),
        Binding("7", "pick_7", show=False),
        Binding("8", "pick_8", show=False),
        Binding("9", "pick_9", show=False),
        Binding("i", "pick_3", "Interactive", show=False),
        Binding("g", "pick_4", "Grammar", show=False),
        Binding("x", "pick_5", "Delete", show=False),
        Binding("q", "pick_9", "Quit", show=False),
    ]

    _ACTIONS = [
        "select_model",
        "delete_model",
        "toggle_interactive",
        "post_process",
        "delete_selected",
        "clear_history",
        "set_grammar_command",
        "set_silence_threshold",
        "quit_app",
    ]

    def __init__(
        self,
        interactive: bool = False,
        model_name: str = "",
        grammar_command: str = "",
        silence_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self._interactive = interactive
        self._model_name = model_name
        self._grammar_command = grammar_command
        self._silence_seconds = silence_seconds

    def compose(self) -> ComposeResult:
        interactive_state = "ON" if self._interactive else "OFF"
        model_display = self._model_name or "none"
        cmd_display = self._grammar_command or "claude"
        with Vertical(id="menu-dialog"):
            yield Static("Menu", id="menu-title")
            yield ListView(
                MenuItem("1", f"Select Model    \\[{model_display}]", "select_model"),
                MenuItem("2", "Delete Model", "delete_model"),
                MenuItem("3", f"Interactive     {interactive_state}", "toggle_interactive"),
                MenuItem("4", "Grammar Fix", "post_process"),
                MenuItem("5", "Delete History Entry", "delete_selected"),
                MenuItem("6", "Clear History", "clear_history"),
                MenuItem("7", f"Grammar Command \\[{cmd_display}]", "set_grammar_command"),
                MenuItem("8", f"Silence Delay   \\[{self._silence_seconds:.1f}s]", "set_silence_threshold"),
                MenuItem("9", "Quit", "quit_app"),
                id="menu-list",
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, MenuItem):
            self.dismiss(item.action_name)

    def _pick(self, idx: int) -> None:
        if idx < len(self._ACTIONS):
            self.dismiss(self._ACTIONS[idx])

    def action_pick_1(self) -> None:
        self._pick(0)

    def action_pick_2(self) -> None:
        self._pick(1)

    def action_pick_3(self) -> None:
        self._pick(2)

    def action_pick_4(self) -> None:
        self._pick(3)

    def action_pick_5(self) -> None:
        self._pick(4)

    def action_pick_6(self) -> None:
        self._pick(5)

    def action_pick_7(self) -> None:
        self._pick(6)

    def action_pick_8(self) -> None:
        self._pick(7)

    def action_pick_9(self) -> None:
        self._pick(8)

    def action_close_menu(self) -> None:
        self.dismiss(None)


# ── Model Picker Modal ───────────────────────────────────────────────


class ModelPickerScreen(ModalScreen[ModelInfo | None]):
    """Modal for selecting a model."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }

    #picker-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #picker-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #picker-list {
        height: auto;
        max-height: 14;
    }
    """

    BINDINGS = [
        Binding("escape", "close_picker", "Close", show=False),
    ]

    def __init__(self, active_model: ModelInfo | None = None) -> None:
        super().__init__()
        self._active_model = active_model

    def compose(self) -> ComposeResult:
        items = []
        for info in MODEL_REGISTRY:
            selected = (
                self._active_model is not None
                and self._active_model.name == info.name
            )
            items.append(ModelPickerItem(info, selected=selected))
        with Vertical(id="picker-dialog"):
            yield Static("Select Model", id="picker-title")
            yield ListView(*items, id="picker-list")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ModelPickerItem):
            self.dismiss(item.info)

    def action_close_picker(self) -> None:
        self.dismiss(None)


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

    #mic-label {
        height: 1;
        color: $text-muted;
        margin-top: 1;
    }

    #level-bar {
        height: 1;
    }

    #level-bar.recording {
        color: #e06060;
    }

    #transcript-area {
        height: 2fr;
        border: round $primary;
        padding: 1;
        overflow-y: auto;
    }

    #download-progress {
        height: auto;
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
        Binding("p", "toggle_pause", "Pause", show=True),
        Binding("i", "toggle_interactive", "Interactive", show=True),
        Binding("g", "post_process", "Grammar", show=True),
        Binding("x", "delete_selected", "Delete", show=True),
        Binding("m", "open_menu", "Menu", show=True),
        Binding("ctrl+z", "undo_correction", "Undo", show=False),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    TITLE = "Voice2Text"
    theme = "rose-pine"

    def __init__(self, force_cpu: bool = False) -> None:
        super().__init__()
        self._setup_file_logging()
        self.recorder = Recorder()
        self.model_manager = ModelManager(force_cpu=force_cpu)
        self.history: list[TranscriptEntry] = []
        self._record_start: float = 0.0
        self._level_task: asyncio.Task | None = None
        self._vad_task: asyncio.Task | None = None
        self._mic_name: str = ""
        self._interactive: bool = self._load_interactive_setting()
        self._vad = None  # VoiceActivityDetector instance during recording
        self._segment_texts: list[str] = []  # accumulated segment transcriptions
        self._segment_boundary: int = 0  # frame index of last segment end
        self._pre_correction_text: str | None = None  # for undo
        self._pre_correction_entry: TranscriptEntry | None = None  # for undo

    @staticmethod
    def _setup_file_logging() -> None:
        log_path = Path(__file__).resolve().parent.parent / "error.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger = logging.getLogger("voice2text")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Starting...", id="status-bar")
        with Vertical(id="main-container"):
            yield Static("  Mic: detecting...", id="mic-label")
            yield AudioLevelBar(id="level-bar")
            yield Static(
                "Loading model, please wait...",
                id="transcript-area",
            )
            yield DownloadProgress(id="download-progress")
            yield Label("History  \\[x] delete", id="history-label")
            yield ListView(id="history-list")
        yield Footer()

    def on_mount(self) -> None:
        self.history = load_history()
        self._refresh_history()
        self.query_one("#download-progress", DownloadProgress).hide_progress()
        self._update_status("Detecting hardware...")
        self._detect_and_load()

    @work(thread=True)
    def _detect_and_load(self) -> None:
        """Detect GPU, mic, and load first available model."""
        try:
            self._detect_and_load_inner()
        except Exception:
            pass  # App may have been shut down during background load

    def _detect_and_load_inner(self) -> None:
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
            self._update_status("No models downloaded")
            self.query_one("#transcript-area", Static).update(
                "No models downloaded.\nOpening model picker..."
            )
            self._open_model_picker()

        self.call_from_thread(_no_models)

    def _on_model_loaded(self) -> None:
        if self.model_manager.active_model:
            save_last_model(self.model_manager.active_model)

        self.query_one("#transcript-area", Static).update(
            "Press SPACE to start recording..."
        )
        self._update_status("Ready")

    def _update_status(self, text: str) -> None:
        """Update the top status bar with context info + message."""
        backend = self.model_manager.backend.upper()
        parts = [backend]
        if self.model_manager.active_model:
            parts.append(self.model_manager.active_model.name)
        if self._interactive:
            parts.append("[bold blue]Interactive[/bold blue]")
        parts.append(text)
        self.query_one("#status-bar", Static).update(" | ".join(parts))

    def _update_mic_label(self) -> None:
        mic = self._mic_name or "detecting..."
        self.query_one("#mic-label", Static).update(f"  Mic: {mic}")

    def action_open_menu(self) -> None:
        """Open the main menu modal."""
        if self.recorder.is_recording:
            return

        def on_menu_dismiss(action: str | None) -> None:
            if action is None:
                return
            if action == "select_model":
                self._open_model_picker()
            elif action == "delete_model":
                self._open_delete_model()
            elif action == "toggle_interactive":
                self.action_toggle_interactive()
            elif action == "post_process":
                self.action_post_process()
            elif action == "delete_selected":
                self.action_delete_selected()
            elif action == "clear_history":
                self._clear_history()
            elif action == "set_grammar_command":
                self._open_grammar_command_editor()
            elif action == "set_silence_threshold":
                self._open_silence_picker()
            elif action == "quit_app":
                self.action_quit_app()

        from .postprocess import get_command

        model_name = ""
        if self.model_manager.active_model:
            model_name = self.model_manager.active_model.name
        self.push_screen(
            MenuScreen(
                interactive=self._interactive,
                model_name=model_name,
                grammar_command=get_command(),
                silence_seconds=self._get_silence_seconds(),
            ),
            callback=on_menu_dismiss,
        )

    def _open_model_picker(self) -> None:
        """Open the model picker modal."""

        def on_picker_dismiss(info: ModelInfo | None) -> None:
            if info is None:
                return
            if is_model_downloaded(info):
                if (
                    self.model_manager.active_model
                    and self.model_manager.active_model.name == info.name
                ):
                    return
                self._update_status(f"Switching to {info.name}...")
                self._load_model_async(info)
            else:
                self._show_download_confirm(info)

        self.push_screen(
            ModelPickerScreen(active_model=self.model_manager.active_model),
            callback=on_picker_dismiss,
        )

    def _open_delete_model(self) -> None:
        """Open model picker to select a model to delete."""

        def on_picker_dismiss(info: ModelInfo | None) -> None:
            if info is None:
                return
            if not is_model_downloaded(info):
                self._update_status(f"{info.name} is not downloaded")
                return
            size = get_model_size_on_disk(info)
            preview = f"{info.name} ({size})"

            def on_confirm(result: bool) -> None:
                if not result:
                    return
                if (
                    self.model_manager.active_model
                    and self.model_manager.active_model.name == info.name
                ):
                    self.model_manager.unload()
            
                    self.query_one("#transcript-area", Static).update(
                        "No model loaded.\nPress M to open menu and select a model."
                    )
                delete_model_files(info)
                self._update_status(f"Deleted {info.name}")

            self.push_screen(DeleteConfirmScreen(preview), callback=on_confirm)

        self.push_screen(
            ModelPickerScreen(active_model=self.model_manager.active_model),
            callback=on_picker_dismiss,
        )

    def action_toggle_interactive(self) -> None:
        """Toggle interactive (chunked transcription) mode."""
        if self.recorder.is_recording:
            return  # don't toggle mid-recording
        self._interactive = not self._interactive
        self._save_config_value("interactive", "enabled", self._interactive)
        self._update_status("Ready")

    def action_toggle_pause(self) -> None:
        """Pause or resume recording."""
        if not self.recorder.is_recording:
            return
        if self.recorder.is_paused:
            self.recorder.resume()
            bar = self.query_one("#level-bar", AudioLevelBar)
            bar.recording = True
        else:
            self.recorder.pause()
            bar = self.query_one("#level-bar", AudioLevelBar)
            bar.recording = False

    @staticmethod
    def _load_config() -> dict:
        """Load config.toml if present."""
        config_file = Path(__file__).resolve().parent.parent / "config.toml"
        if not config_file.exists():
            return {}
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(config_file, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    @staticmethod
    def _get_silence_seconds() -> float:
        """Read silence_seconds from config.toml, default 0.5."""
        config = Voice2TextApp._load_config()
        try:
            return float(config.get("interactive", {}).get("silence_seconds", 0.5))
        except Exception:
            return 0.5

    @staticmethod
    def _load_interactive_setting() -> bool:
        """Read interactive enabled state from config.toml, default False."""
        config = Voice2TextApp._load_config()
        return bool(config.get("interactive", {}).get("enabled", False))

    # ── Post-Processing via LLM CLI ──────────────────────────────────

    def _get_transcript_text(self) -> str:
        """Get the current text from the transcript area."""
        widget = self.query_one("#transcript-area", Static)
        content = widget._Static__content  # noqa: WPS437
        return str(content) if content else ""

    def action_post_process(self) -> None:
        """Run grammar correction via external LLM CLI tool."""
        if self.recorder.is_recording:
            return
        text = self._get_transcript_text()
        if not text.strip():
            self._update_status("Nothing to correct")
            return
        placeholders = ("Press SPACE", "No model", "Loading", "Recording", "Transcribing", "No speech")
        if any(text.startswith(p) for p in placeholders):
            self._update_status("Nothing to correct")
            return

        from .postprocess import get_command

        self._update_status(f"Correcting with {get_command()}...")
        self._run_post_process(text)

    @work(thread=True)
    def _run_post_process(self, text: str) -> None:
        """Run post-processing in a background thread."""
        from .postprocess import correct

        try:
            corrected = correct(text)
        except Exception as e:
            self.call_from_thread(self._update_status, f"Post-process error: {e}")
            return

        if corrected.strip() == text.strip():
            self.call_from_thread(self._update_status, "No corrections needed | Ready")
            return

        def _apply():
            self._pre_correction_text = text
            if self.history:
                for entry in self.history:
                    if entry.full_text().strip() == text.strip():
                        self._pre_correction_entry = entry
                        entry.path.write_text(corrected, encoding="utf-8")
                        entry.preview = corrected[:80].replace("\n", " ").strip()
                        break
            self.query_one("#transcript-area", Static).update(corrected)
            self._refresh_history()

        self.call_from_thread(_apply)
        clip_msg = copy_to_clipboard(corrected)
        self.call_from_thread(
            self._update_status,
            f"{clip_msg} | Corrected | ctrl+z to undo",
        )

    def action_undo_correction(self) -> None:
        """Undo the last grammar correction."""
        if self._pre_correction_text is None:
            return
        original = self._pre_correction_text
        self._pre_correction_text = None

        # Revert history entry
        if self._pre_correction_entry is not None:
            entry = self._pre_correction_entry
            entry.path.write_text(original, encoding="utf-8")
            entry.preview = original[:80].replace("\n", " ").strip()
            self._pre_correction_entry = None
            self._refresh_history()

        self.query_one("#transcript-area", Static).update(original)
        self._copy_async(original, "Correction undone")

    def _refresh_history(self) -> None:
        lv = self.query_one("#history-list", ListView)
        lv.clear()
        for entry in self.history:
            lv.append(HistoryItem(entry))

    def _clear_history(self) -> None:
        """Delete all history entries after confirmation."""
        if not self.history:
            self._update_status("No history to clear")
            return

        def on_dismiss(result: bool) -> None:
            if not result:
                return
            for entry in self.history:
                try:
                    entry.path.unlink()
                except FileNotFoundError:
                    pass
            self.history.clear()
            self._refresh_history()
            self._update_status("History cleared")

        count = len(self.history)
        self.push_screen(
            DeleteConfirmScreen(f"Delete all {count} transcript(s)?"),
            callback=on_dismiss,
        )

    # ── Settings ──────────────────────────────────────────────────────────

    def _open_grammar_command_editor(self) -> None:
        """Open picker to change the grammar CLI command."""
        from .postprocess import get_command

        def on_dismiss(value: str | None) -> None:
            if value is None:
                return
            self._save_config_value("post_processing", "command", value)
            self._update_status(f"Grammar command set to '{value}'")

        self.push_screen(
            CommandPickerScreen(get_command()),
            callback=on_dismiss,
        )

    def _open_silence_picker(self) -> None:
        """Open picker to change silence threshold."""

        def on_dismiss(value: float | None) -> None:
            if value is None:
                return
            self._save_config_value("interactive", "silence_seconds", value)
            self._update_status(f"Silence delay set to {value:.1f}s")

        self.push_screen(
            SilencePickerScreen(self._get_silence_seconds()),
            callback=on_dismiss,
        )

    @staticmethod
    def _save_config_value(section: str, key: str, value: object) -> None:
        """Write a single config value to config.toml, preserving existing content."""
        config_file = Path(__file__).resolve().parent.parent / "config.toml"

        config: dict = {}
        if config_file.exists():
            try:
                try:
                    import tomllib
                except ModuleNotFoundError:
                    import tomli as tomllib  # type: ignore[no-redef]
                with open(config_file, "rb") as f:
                    config = tomllib.load(f)
            except Exception:
                pass

        if section not in config:
            config[section] = {}
        config[section][key] = value

        # Write back as TOML (simple serializer — no third-party writer needed)
        lines: list[str] = []
        # Write top-level [[models]] arrays first if present
        models = config.pop("models", None)
        for sect, values in config.items():
            if isinstance(values, dict):
                lines.append(f"[{sect}]")
                for k, v in values.items():
                    if isinstance(v, bool):
                        lines.append(f"{k} = {str(v).lower()}")
                    elif isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    elif isinstance(v, float):
                        lines.append(f"{k} = {v}")
                    elif isinstance(v, int):
                        lines.append(f"{k} = {v}")
                    else:
                        lines.append(f'{k} = "{v}"')
                lines.append("")
        if models:
            for model in models:
                lines.append("[[models]]")
                for k, v in model.items():
                    lines.append(f'{k} = "{v}"')
                lines.append("")

        config_file.write_text("\n".join(lines) + "\n")

    # ── History Delete ────────────────────────────────────────────────────

    def action_delete_selected(self) -> None:
        """Delete the highlighted history entry."""
        if self.recorder.is_recording:
            return

        history_lv = self.query_one("#history-list", ListView)
        if history_lv.highlighted_child is None:
            return
        item = history_lv.highlighted_child
        if not isinstance(item, HistoryItem):
            return
        entry = item.entry
        preview = entry.preview[:60] if entry.preview else "(empty)"

        def on_dismiss(result: bool) -> None:
            if not result:
                return
            try:
                entry.path.unlink()
            except FileNotFoundError:
                pass
            self.history = [e for e in self.history if e.path != entry.path]
            self._refresh_history()
            self._update_status("Deleted transcript")

        self.push_screen(DeleteConfirmScreen(preview), callback=on_dismiss)

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
            if self.recorder.is_paused:
                self._update_status(
                    f"⏸ Paused {mins:02d}:{secs:02d} | P to resume, SPACE to stop"
                )
            else:
                self._update_status(
                    f"● Recording {mins:02d}:{secs:02d} | P to pause, SPACE to stop"
                )
            await asyncio.sleep(0.05)
        bar.level = 0.0

    async def _poll_vad(self) -> None:
        """Poll audio chunks through VAD while recording in interactive mode."""
        from .vad import SILERO_CHUNK_SAMPLES

        silence_chunks = 0
        silence_seconds = self._get_silence_seconds()
        silence_threshold = int(silence_seconds * 16000 / SILERO_CHUNK_SAMPLES)
        was_speech = False
        last_processed_frame = 0
        chunk_bytes = SILERO_CHUNK_SAMPLES * 2  # 16-bit = 2 bytes per sample

        while self.recorder.is_recording and self._vad is not None:
            if self.recorder.is_paused:
                await asyncio.sleep(0.05)
                continue
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
        self.query_one("#transcript-area", Static).update(full_text)
        self.history.insert(0, entry)
        self._refresh_history()
        self._copy_async(full_text, "Ready")

    @work(thread=True)
    def _copy_async(self, text: str, suffix: str) -> None:
        """Copy text to clipboard in a background thread to avoid blocking the UI."""
        clip_msg = copy_to_clipboard(text)
        self.call_from_thread(self._update_status, f"{clip_msg} | {suffix}")

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

        # History selection
        if isinstance(item, HistoryItem):
            entry = item.entry
            text = entry.full_text()
            self.query_one("#transcript-area", Static).update(text)
            self._copy_async(text, "Loaded from history")

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
