"""End-to-end tests for the Voice2Text TUI application."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice2text.app import (
    AudioLevelBar,
    DownloadConfirmScreen,
    DownloadProgress,
    HistoryItem,
    ModelPickerItem,
    Voice2TextApp,
)
from voice2text.models import MODEL_REGISTRY, ModelInfo, ModelManager


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary models and transcripts directories."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    return models_dir, transcripts_dir


@pytest.fixture
def sample_transcripts(tmp_path):
    """Create sample transcript files for history tests."""
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    files = [
        ("2026-03-01_10-00-00.txt", "Hello world this is a test transcript"),
        ("2026-03-02_14-30-00.txt", "Second transcript with more content"),
        ("2026-03-03_09-15-00.txt", "Third and most recent transcript"),
    ]
    for name, content in files:
        (transcripts_dir / name).write_text(content)

    return transcripts_dir


# ── Test: App Startup & Widget Composition ──────────────────────────────


@pytest.mark.asyncio
async def test_app_composes_all_widgets():
    """App should mount all required widgets on startup."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

        # Core layout widgets (use `is not None` since empty widgets are falsy)
        assert app.query_one("#status-bar") is not None
        assert app.query_one("#transcript-area") is not None
        assert app.query_one("#level-bar") is not None
        assert app.query_one("#mic-label") is not None

        # Right panel widgets
        assert app.query_one("#model-picker-label") is not None
        assert app.query_one("#model-list") is not None
        assert app.query_one("#download-progress") is not None
        assert app.query_one("#history-label") is not None
        assert app.query_one("#history-list") is not None


@pytest.mark.asyncio
async def test_model_picker_shows_all_models():
    """Model picker should list all models from the registry."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

        items = list(app.query(ModelPickerItem))
        assert len(items) == len(MODEL_REGISTRY)
        model_names = {item.info.name for item in items}
        expected_names = {info.name for info in MODEL_REGISTRY}
        assert model_names == expected_names


@pytest.mark.asyncio
async def test_model_picker_items_persist_after_load():
    """Model picker should keep all items after background model load."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Wait for any background model loading to complete
        await asyncio.sleep(2)

        items = list(app.query(ModelPickerItem))
        assert len(items) == len(MODEL_REGISTRY), (
            f"Expected {len(MODEL_REGISTRY)} items, got {len(items)}"
        )


@pytest.mark.asyncio
async def test_download_progress_hidden_on_startup():
    """Download progress widget should be hidden initially."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        dp = app.query_one("#download-progress", DownloadProgress)
        assert dp.display is False


# ── Test: Download Confirmation Modal ───────────────────────────────────


@pytest.mark.asyncio
async def test_download_confirm_modal_yes():
    """Pressing Y on the download modal should return True."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("test")

        def on_mount(self):
            self.push_screen(
                DownloadConfirmScreen(MODEL_REGISTRY[0]),
                callback=self.exit,
            )

    app = TestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.press("y")
        await pilot.pause()

    assert app.return_value is True


@pytest.mark.asyncio
async def test_download_confirm_modal_no():
    """Pressing N on the download modal should return False."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("test")

        def on_mount(self):
            self.push_screen(
                DownloadConfirmScreen(MODEL_REGISTRY[0]),
                callback=self.exit,
            )

    app = TestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.press("n")
        await pilot.pause()

    assert app.return_value is False


@pytest.mark.asyncio
async def test_download_confirm_modal_escape():
    """Pressing Escape on the download modal should return False."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("test")

        def on_mount(self):
            self.push_screen(
                DownloadConfirmScreen(MODEL_REGISTRY[0]),
                callback=self.exit,
            )

    app = TestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.press("escape")
        await pilot.pause()

    assert app.return_value is False


@pytest.mark.asyncio
async def test_download_confirm_shows_model_info():
    """Modal should display the model name and size."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    info = MODEL_REGISTRY[0]

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("test")

        def on_mount(self):
            self.push_screen(DownloadConfirmScreen(info))

    app = TestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.2)

        # Query through the active screen (the modal)
        screen = app.screen
        body = screen.query_one("#confirm-body", Static)
        rendered = str(body.render())
        assert info.name in rendered
        assert info.size_hint in rendered

        await pilot.press("escape")
        await pilot.pause()


# ── Test: Audio Level Bar ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audio_level_bar_renders():
    """Audio level bar should render at different levels."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

        bar = app.query_one("#level-bar", AudioLevelBar)
        # Default level should be 0
        assert bar.level == 0.0

        # Set level and verify it updates
        bar.level = 0.5
        assert bar.level == 0.5

        bar.level = 1.0
        assert bar.level == 1.0


# ── Test: Quit Binding ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quit_binding():
    """Pressing Q should exit the app."""
    app = Voice2TextApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    # If we get here without timeout, quit worked


# ── Test: Clipboard Module ──────────────────────────────────────────────


def test_clipboard_copy_returns_status():
    """Clipboard copy should return a status message."""
    from voice2text.clipboard import copy_to_clipboard

    result = copy_to_clipboard("test text")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.startswith("Copied to ") or result == "Saved to file (clipboard unavailable)"


# ── Test: Transcript Storage ────────────────────────────────────────────


def test_save_and_load_transcript(tmp_path):
    """Save a transcript and verify it can be loaded back."""
    from voice2text.transcripts import TRANSCRIPTS_DIR, load_history, save_transcript

    with patch("voice2text.transcripts.TRANSCRIPTS_DIR", tmp_path):
        entry = save_transcript("Hello world test transcript")

        assert entry.path.exists()
        assert entry.path.suffix == ".txt"
        assert entry.full_text() == "Hello world test transcript"
        assert "Hello world" in entry.preview

        history = load_history()
        assert len(history) == 1
        assert history[0].full_text() == "Hello world test transcript"


def test_load_history_sorted_newest_first(tmp_path):
    """History should be sorted with newest transcripts first."""
    from voice2text.transcripts import load_history

    (tmp_path / "2026-03-01_10-00-00.txt").write_text("first")
    (tmp_path / "2026-03-03_10-00-00.txt").write_text("third")
    (tmp_path / "2026-03-02_10-00-00.txt").write_text("second")

    with patch("voice2text.transcripts.TRANSCRIPTS_DIR", tmp_path):
        history = load_history()

    assert len(history) == 3
    assert history[0].full_text() == "third"
    assert history[1].full_text() == "second"
    assert history[2].full_text() == "first"


def test_load_history_empty_dir(tmp_path):
    """Loading history from empty dir should return empty list."""
    from voice2text.transcripts import load_history

    with patch("voice2text.transcripts.TRANSCRIPTS_DIR", tmp_path):
        history = load_history()
    assert history == []


# ── Test: Model Registry ───────────────────────────────────────────────


def test_model_registry_has_entries():
    """Model registry should have at least the two parakeet models."""
    assert len(MODEL_REGISTRY) >= 2
    names = [m.name for m in MODEL_REGISTRY]
    assert "parakeet-tdt-0.6b-v2" in names
    assert "parakeet-tdt-0.6b-v3" in names


def test_is_model_downloaded_false_by_default(tmp_path):
    """Models should not be marked as downloaded by default."""
    from voice2text.models import is_model_downloaded

    with patch("voice2text.models.MODELS_DIR", tmp_path):
        for info in MODEL_REGISTRY:
            assert not is_model_downloaded(info)


def test_mark_downloaded(tmp_path):
    """Marking a model as downloaded should create the marker file."""
    from voice2text.models import _mark_downloaded, is_model_downloaded

    with patch("voice2text.models.MODELS_DIR", tmp_path):
        info = MODEL_REGISTRY[0]
        assert not is_model_downloaded(info)
        _mark_downloaded(info)
        assert is_model_downloaded(info)
        assert (tmp_path / info.name / ".downloaded").exists()


def test_detect_backend():
    """Backend detection should return 'CUDA' or 'CPU'."""
    mm = ModelManager()
    backend = mm.detect_backend()
    assert backend in ("CUDA", "CPU")


# ── Test: Recorder Module ──────────────────────────────────────────────


def test_recorder_initial_state():
    """Recorder should start in non-recording state."""
    from voice2text.recorder import Recorder

    rec = Recorder()
    assert not rec.is_recording
    assert rec.level == 0.0


# ── Test: HF_HUB_ENABLE_HF_TRANSFER ─────────────────────────────────


def test_hf_xet_disabled():
    """hf-xet should be disabled to prevent FDS_TO_KEEP errors."""
    # Import triggers the os.environ.setdefault
    import voice2text.models  # noqa: F401

    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"
