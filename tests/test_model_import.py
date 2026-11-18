"""Model import checks use metadata fixtures; never download model weights."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from voice2text import models
from voice2text.app import AddModelScreen, Voice2TextApp, ModelPickerScreen, DownloadConfirmScreen


@pytest.mark.parametrize("link", ["https://example.com/a/b", "https://huggingface.co/a/b/blob/main/model.onnx", "../bad", "a", "a/b?x=1"])
def test_reject_invalid_repository_links(link):
    with pytest.raises(ValueError):
        models.parse_huggingface_repo(link)


@pytest.fixture
def hub(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "nemo-conformer-tdt", "features_size": 128}))
    files = ["config.json", "vocab.txt", "int8/encoder-model.int8.onnx", "int8/decoder_joint-model.int8.onnx", "fp32/encoder-model.onnx", "scripts/test.py"]
    metadata = SimpleNamespace(siblings=[SimpleNamespace(rfilename=f, size=100) for f in files])
    def download(repo_id, filename):
        if filename == "config.json":
            return str(config)
        path = tmp_path / filename.replace("/", "_")
        path.write_text("fixture")
        return str(path)
    with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", side_effect=download) as fetch:
        api.return_value.model_info.return_value = metadata
        api.return_value.list_repo_files.return_value = files
        yield config, metadata, fetch


def test_inspect_and_download_ultra_layout(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    info = models.inspect_huggingface_model("https://huggingface.co/Olicorne/example/")
    assert info.onnx_asr_name == "nemo-conformer-tdt"
    assert [c.kwargs.get("filename", c.args[1] if len(c.args) > 1 else None) for c in hub[2].call_args_list] == ["config.json"]
    models.ModelManager().download_model(info)
    directory = models.MODELS_DIR / info.name
    assert (directory / "encoder-model.int8.onnx").is_file()
    assert (directory / "decoder_joint-model.int8.onnx").is_file()
    assert not (directory / "int8").exists()
    assert not (directory / "scripts").exists()
    assert models.is_model_downloaded(info)


def test_reject_raw_checkpoint_and_wrong_type(hub):
    with pytest.raises(ValueError, match="Expected"):
        models.inspect_huggingface_model("owner/model", "nemo-conformer-ctc")
    hub[0].write_text('{"model_type":"parakeet_tdt"}')
    with pytest.raises(ValueError, match="Unsupported model type"):
        models.inspect_huggingface_model("owner/model")


def test_incomplete_export_is_not_marked_downloaded(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.list_repo_files.return_value = ["config.json", "vocab.txt"]
        info = models._BUILTIN_MODELS[2]
        with pytest.raises(ValueError, match="missing"):
            models.ModelManager().download_model(info)
        assert not models.is_model_downloaded(info)


@pytest.mark.parametrize("model_type", models.IMPORT_MODEL_TYPES)
def test_import_layouts_include_tensor_sidecars(model_type):
    graphs = models.IMPORT_MODEL_TYPES[model_type][1]
    files = ["config.json", "vocab.txt", *["int8/" + f for f in graphs]]
    files += ["int8/" + graphs[0] + ".data.000"]
    plan = models._nemo_download_files(model_type, {"model_type": model_type}, files)
    assert plan[files[-1]] == graphs[0] + ".data.000"
    assert set(graphs).issubset(plan.values())


def test_custom_model_persists_comments_and_deduplicates(tmp_path, monkeypatch):
    from voice2text import config
    path = tmp_path / "config.toml"
    path.write_text("# keep this\n[interactive]\nenabled = true\n")
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    monkeypatch.setattr(models, "MODEL_REGISTRY", list(models._BUILTIN_MODELS))
    info = models.ModelInfo("custom", "nemo-conformer-tdt", "test", "1 MB", "owner/repo")
    assert models.register_custom_model(info) is info
    assert models.register_custom_model(info) is info
    assert models._load_custom_models() == [info]
    assert "# keep this" in path.read_text()
    assert len(config.load_config()["models"]) == 1


@pytest.mark.asyncio
async def test_add_model_validation_error_keeps_form_open():
    app = App()
    async with app.run_test(size=(100, 35)) as pilot:
        app.push_screen(AddModelScreen())
        await pilot.pause()
        app.screen.query_one("#model-link", Input).value = "bad link"
        await pilot.click("#check-model")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, AddModelScreen)
        assert not app.screen.query_one("#check-model", Button).disabled
        assert "link" in str(app.screen.query_one("#add-model-status", Static).render())
        await pilot.press("escape")
        assert not isinstance(app.screen, AddModelScreen)


@pytest.mark.asyncio
async def test_import_from_picker_reaches_download_confirmation(tmp_path, monkeypatch):
    from voice2text import config
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(models, "MODEL_REGISTRY", list(models._BUILTIN_MODELS))
    monkeypatch.setattr(Voice2TextApp, "_detect_and_load", lambda self: None)
    info = models.ModelInfo("custom", "nemo-conformer-tdt", "test", "1 MB", "owner/repo")
    with patch("voice2text.app.inspect_huggingface_model", return_value=info):
        app = Voice2TextApp()
        async with app.run_test(size=(100, 35)) as pilot:
            app._open_model_picker()
            await pilot.pause()
            assert isinstance(app.screen, ModelPickerScreen)
            await pilot.click("#add-model")
            await pilot.pause()
            await pilot.click("#check-model")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, DownloadConfirmScreen)
            assert app.screen.info == info
            await pilot.press("n")
            assert not models.is_model_downloaded(info)


@pytest.mark.asyncio
async def test_cancel_inflight_inspection_does_not_register_model():
    started, release = threading.Event(), threading.Event()
    def inspect(*args):
        started.set()
        release.wait(timeout=5)
        return models.ModelInfo("custom", "nemo-conformer-tdt", "test", "1 MB", "owner/repo")
    app = App()
    with patch("voice2text.app.inspect_huggingface_model", side_effect=inspect), patch("voice2text.app.register_custom_model") as register:
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(AddModelScreen())
            await pilot.pause()
            try:
                await pilot.click("#check-model")
                assert started.is_set()
                await pilot.press("escape")
            finally:
                release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            register.assert_not_called()
