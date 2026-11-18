"""Discovery and keyboard browsing tests with no Hub network traffic."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import App
from textual.widgets import Input, Static

from voice2text import model_search, models
from voice2text.app import DownloadConfirmScreen, Voice2TextApp
from voice2text.model_browser import ModelSearchScreen, SearchResultItem


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(model_search, "_cache", None)


def result(repo, downloads):
    return model_search.ModelSearchResult(
        models.ModelInfo(repo.replace("/", "--"), "nemo-conformer-tdt", "test", "668 MB", repo),
        downloads, ("en", "fr"),
    )


@pytest.mark.parametrize("query,text,expected", [
    ("prkt ult", "Olicorne/parakeet-ultra", True),
    ("ULTRA en", "parakeet-ultra en fr", True),
    ("", "parakeet", True),
    ("ultra de", "parakeet-ultra en fr", False),
    ("abc", "cba", False),
])
def test_fuzzy_filter(query, text, expected):
    assert model_search.fuzzy_matches(query, text) is expected


def hub_model(repo, downloads, files=None):
    files = files if files is not None else ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]
    return SimpleNamespace(id=repo, downloads=downloads, siblings=[SimpleNamespace(rfilename=f) for f in files], card_data={"language": ["en"]})


def test_discovery_filters_deduplicates_sorts_and_caches():
    candidates = [hub_model("a/low", 5), hub_model("a/high", 500), hub_model("a/raw", 1000, ["model.safetensors"]), hub_model("a/wrong", 100), hub_model("a/offline", 10)]
    def inspect(repo):
        if repo == "a/wrong":
            raise ValueError("wrong model type")
        if repo == "a/offline":
            raise RuntimeError("network unavailable")
        return result(repo, 0).info
    with patch("huggingface_hub.HfApi") as api, patch.object(model_search, "inspect_huggingface_model", side_effect=inspect) as inspect_mock:
        api.return_value.list_models.side_effect = [candidates, [candidates[1]]]
        catalog = model_search.discover_models()
        assert [entry.info.repo_id for entry in catalog.results] == ["a/high", "a/low"]
        assert catalog.checked == 5
        assert catalog.unavailable == 1
        assert inspect_mock.call_count == 4
        assert api.return_value.list_models.call_args.kwargs["sort"] == "downloads"
        assert model_search.discover_models() is catalog
        assert inspect_mock.call_count == 4
        api.return_value.list_models.side_effect = [[candidates[0]], []]
        assert len(model_search.discover_models(refresh=True).results) == 1


def test_cancelled_discovery_does_not_cache():
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.list_models.return_value = [hub_model("a/repo", 1)]
        catalog = model_search.discover_models(cancelled=lambda: True)
    assert not catalog.results
    assert model_search._cache is None


def test_curated_defaults_preserve_legacy_preferences(tmp_path, monkeypatch):
    from voice2text import config
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    legacy = replace(models._LEGACY_MODELS[0])
    monkeypatch.setattr(models, "MODEL_REGISTRY", [*[replace(info) for info in models._BUILTIN_MODELS], legacy])
    assert [info.name for info in models.picker_models()] == ["parakeet-ultra", "parakeet-tdt-0.6b-v2", "canary-180m-flash"]
    preference = tmp_path / ".last_model"
    preference.write_text(legacy.name)
    monkeypatch.setattr(models, "_LAST_MODEL_FILE", preference)
    assert models.load_last_model() is legacy
    models._mark_downloaded(legacy)
    assert legacy in models.picker_models()
    models.delete_model_files(legacy)
    assert legacy not in models.picker_models()
    models.register_custom_model(result(legacy.repo_id, 0).info)
    assert legacy in models.picker_models()
    assert models._load_custom_models()[0].featured


@pytest.mark.asyncio
async def test_keyboard_search_handoff_to_download(tmp_path, monkeypatch):
    from voice2text import config
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(models, "MODEL_REGISTRY", [])
    monkeypatch.setattr(Voice2TextApp, "_detect_and_load", lambda self: None)
    entries = [result("owner/parakeet-v3", 100), result("owner/parakeet-ultra", 10)]
    with patch("voice2text.model_browser.discover_models", return_value=model_search.ModelCatalog(entries, 5)):
        app = Voice2TextApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app._open_model_picker()
            await pilot.pause()
            await pilot.click("#search-models")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ModelSearchScreen)
            assert app.focused is app.screen.query_one(Input)
            assert len(app.screen.query(SearchResultItem)) == 2
            await pilot.press("down")
            assert app.focused is app.screen.query_one(Input)
            await pilot.press(*"prkt ult")
            await pilot.pause()
            assert [item.result.info.repo_id for item in app.screen.query(SearchResultItem)] == [entries[1].info.repo_id]
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, DownloadConfirmScreen)
            assert app.screen.info == entries[1].info
            assert models._load_custom_models() == [entries[1].info]
            assert not models.is_model_downloaded(entries[1].info)


@pytest.mark.asyncio
async def test_discovery_error_retries_and_empty_filter():
    app = App()
    with patch("voice2text.model_browser.discover_models", side_effect=[RuntimeError("offline"), model_search.ModelCatalog([result("a/parakeet", 1)], 1)]) as discover:
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(ModelSearchScreen())
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "offline" in str(app.screen.query_one("#model-search-status", Static).render())
            await pilot.click("#refresh-model-search")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert discover.call_args.kwargs["refresh"] is True
            app.screen.query_one(Input).value = "nonexistent"
            await pilot.pause()
            assert not app.screen.query(SearchResultItem)
            await pilot.press("enter")
            assert isinstance(app.screen, ModelSearchScreen)
            assert "No matches" in str(app.screen.query_one("#model-search-status", Static).render())
            await pilot.press("escape")
