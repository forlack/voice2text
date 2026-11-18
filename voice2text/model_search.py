"""Discover compatible Hub exports without downloading model weights."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from .models import IMPORT_MODEL_TYPES, ModelInfo, _nemo_download_files, inspect_huggingface_model


@dataclass
class ModelSearchResult:
    info: ModelInfo
    downloads: int
    languages: tuple[str, ...] = ()

    @property
    def search_text(self) -> str:
        family = IMPORT_MODEL_TYPES[self.info.onnx_asr_name][0]
        return f"{self.info.repo_id} {family} {' '.join(self.languages)}".casefold()


@dataclass
class ModelCatalog:
    results: list[ModelSearchResult]
    checked: int
    unavailable: int = 0


_cache: tuple[float, ModelCatalog] | None = None


def fuzzy_matches(query: str, text: str) -> bool:
    """Match each query word as a case-insensitive subsequence, preserving rank."""
    text = text.casefold()
    for word in query.casefold().split():
        chars = iter(text)
        if not all(any(char == candidate for candidate in chars) for char in word):
            return False
    return True


def discover_models(
    progress: Callable[[str], None] = lambda message: None,
    cancelled: Callable[[], bool] = lambda: False,
    *,
    refresh: bool = False,
) -> ModelCatalog:
    """Check up to 200 popular tagged exports; cache for five minutes.

    Only config/files compatible with our importer enter the catalog. This
    checks the export contract, not ONNX execution or transcription quality.
    """
    global _cache
    if not refresh and _cache is not None and time.monotonic() - _cache[0] < 300:
        return _cache[1]

    from huggingface_hub import HfApi

    progress("Finding popular ONNX speech models on Hugging Face...")
    candidates = {}
    api = HfApi()
    for filters in ("onnx-asr", ["automatic-speech-recognition", "onnx"]):
        for model in api.list_models(filter=filters, sort="downloads", limit=100, full=True, cardData=True):
            if cancelled():
                return ModelCatalog([], 0)
            candidates[model.id] = model

    # Cheap filename checks avoid fetching configs for Whisper, VAD, etc.
    promising = []
    for model in candidates.values():
        files = [file.rfilename for file in model.siblings or []]
        for model_type in IMPORT_MODEL_TYPES:
            try:
                _nemo_download_files(model_type, {"model_type": model_type}, files)
            except ValueError:
                continue
            promising.append(model)
            break

    results = []
    unavailable = 0
    pool = ThreadPoolExecutor(max_workers=4)
    pending = {pool.submit(inspect_huggingface_model, model.id): model for model in promising}
    try:
        for checked, future in enumerate(as_completed(pending), 1):
            if cancelled():
                return ModelCatalog([], 0)
            model = pending[future]
            try:
                info = future.result()
            except ValueError:
                pass  # Valid Hub repo, incompatible export.
            except Exception:
                unavailable += 1  # Unverified models stay hidden, including network failures.
            else:
                languages = (model.card_data or {}).get("language", []) or []
                if isinstance(languages, str):
                    languages = [languages]
                results.append(ModelSearchResult(info, model.downloads or 0, tuple(languages)))
            progress(f"Checking export layouts {checked}/{len(promising)} — {len(results)} compatible")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results.sort(key=lambda result: (-result.downloads, result.info.repo_id.casefold()))
    catalog = ModelCatalog(results, len(candidates), unavailable)
    if not cancelled():
        _cache = (time.monotonic(), catalog)
    return catalog
