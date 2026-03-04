"""Model registry, download, and transcription using onnx-asr."""

from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import onnxruntime as ort

# Suppress onnxruntime CUDA warnings (e.g. missing cuBLAS/cuDNN) from stderr
ort.set_default_logger_severity(3)  # 3 = ERROR only, suppresses WARNINGS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_LAST_MODEL_FILE = MODELS_DIR / ".last_model"
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class ModelInfo:
    name: str
    onnx_asr_name: str  # onnx-asr model identifier
    description: str
    size_hint: str  # human-readable download size
    repo_id: str = ""  # HuggingFace repo ID (auto-resolved if empty)
    language: str = ""  # language hint passed to recognize() (e.g. "en")


_BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="parakeet-tdt-0.6b-v2",
        onnx_asr_name="nemo-parakeet-tdt-0.6b-v2",
        description="English ASR – Parakeet TDT 0.6B v2",
        size_hint="~640 MB",
        repo_id="istupakov/parakeet-tdt-0.6b-v2-onnx",
    ),
    ModelInfo(
        name="parakeet-tdt-0.6b-v3",
        onnx_asr_name="nemo-parakeet-tdt-0.6b-v3",
        description="Multilingual ASR – Parakeet TDT 0.6B v3",
        size_hint="~640 MB",
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
    ),
    ModelInfo(
        name="canary-180m-flash",
        onnx_asr_name="istupakov/canary-180m-flash-onnx",
        description="EN/DE/FR/ES ASR + Translation – Canary 180M Flash",
        size_hint="~214 MB",
        repo_id="istupakov/canary-180m-flash-onnx",
        language="en",
    ),
]


def _load_custom_models() -> list[ModelInfo]:
    """Load custom models from config.toml if it exists."""
    if not _CONFIG_FILE.exists():
        return []
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with open(_CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
    except Exception:
        return []
    custom = []
    for m in config.get("models", []):
        try:
            custom.append(ModelInfo(
                name=m["name"],
                onnx_asr_name=m["onnx_asr_name"],
                description=m.get("description", ""),
                size_hint=m.get("size_hint", "unknown"),
                repo_id=m.get("repo_id", ""),
                language=m.get("language", ""),
            ))
        except KeyError:
            continue
    return custom


MODEL_REGISTRY: list[ModelInfo] = _BUILTIN_MODELS + _load_custom_models()


def save_last_model(info: ModelInfo) -> None:
    """Save the last-used model name for next startup."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _LAST_MODEL_FILE.write_text(info.name)


def load_last_model() -> ModelInfo | None:
    """Load the last-used model preference, or None."""
    try:
        name = _LAST_MODEL_FILE.read_text().strip()
    except FileNotFoundError:
        return None
    for info in MODEL_REGISTRY:
        if info.name == name:
            return info
    return None


def is_model_downloaded(info: ModelInfo) -> bool:
    """Check if a model has been downloaded."""
    marker = MODELS_DIR / info.name / ".downloaded"
    return marker.exists()


def get_model_size_on_disk(info: ModelInfo) -> str:
    """Return human-readable size of model on disk."""
    model_dir = MODELS_DIR / info.name
    if not model_dir.exists():
        return "0 MB"
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.1f} GB"
    return f"{total / 1_000_000:.0f} MB"


def delete_model_files(info: ModelInfo) -> None:
    """Delete a model's files from disk."""
    import shutil

    model_dir = MODELS_DIR / info.name
    if model_dir.exists():
        shutil.rmtree(model_dir)


def _mark_downloaded(info: ModelInfo) -> None:
    d = MODELS_DIR / info.name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".downloaded").touch()


def _detect_providers() -> tuple[list[str], str]:
    """Detect available ONNX Runtime providers. Returns (providers, label).

    Lists CUDA first if available, but onnxruntime will silently fall back
    to CPU if CUDA runtime libs (cuBLAS, cuDNN) aren't installed.
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA"
    return ["CPUExecutionProvider"], "CPU"


class ModelManager:
    """Handles model downloading, loading, and inference via onnx-asr."""

    def __init__(self, force_cpu: bool = False) -> None:
        self._asr_model = None
        self._active: ModelInfo | None = None
        self._backend: str = "cpu"
        self._force_cpu = force_cpu

    @property
    def active_model(self) -> ModelInfo | None:
        return self._active

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_loaded(self) -> bool:
        return self._asr_model is not None

    def detect_backend(self) -> str:
        """Detect CUDA vs CPU and return label."""
        if self._force_cpu:
            self._backend = "CPU"
            return self._backend
        _, self._backend = _detect_providers()
        return self._backend

    def download_model(
        self,
        info: ModelInfo,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> None:
        """Download model files via huggingface_hub with progress tracking.

        progress_cb receives (fraction 0-1, status_text).
        """
        import shutil

        from huggingface_hub import HfApi, hf_hub_download

        if progress_cb:
            progress_cb(0.0, f"Starting download of {info.name}...")

        model_dir = MODELS_DIR / info.name
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        repo_id = info.repo_id or info.onnx_asr_name
        if "/" not in repo_id:
            raise ValueError(f"No repo_id for model: {info.name}")

        # List files matching our patterns to download individually
        api = HfApi()
        all_files = api.list_repo_files(repo_id)
        import fnmatch

        patterns = ["*.int8.onnx", "config.json", "vocab.txt"]
        files = [
            f for f in all_files
            if any(fnmatch.fnmatch(f, p) for p in patterns)
        ]

        if not files:
            raise RuntimeError(f"No matching files found in {repo_id}")

        for i, filename in enumerate(files):
            fraction = i / len(files)
            short_name = filename.rsplit("/", 1)[-1]
            if progress_cb:
                progress_cb(fraction, f"Downloading {short_name} ({i+1}/{len(files)})...")

            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            dest = model_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, dest)

        _mark_downloaded(info)
        size = get_model_size_on_disk(info)

        if progress_cb:
            progress_cb(1.0, f"Download complete ({size} on disk)")

    def load_model(self, info: ModelInfo) -> None:
        """Load a model into memory for inference."""
        self.unload()
        self.detect_backend()

        import onnx_asr

        if self._force_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers, _ = _detect_providers()
        model_dir = MODELS_DIR / info.name

        # Suppress onnxruntime C++ stderr noise during model load
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            self._asr_model = onnx_asr.load_model(
                info.onnx_asr_name,
                path=str(model_dir),
                quantization="int8",
                providers=providers,
            )
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)
        self._active = info

    def unload(self) -> None:
        """Unload current model and free memory."""
        if self._asr_model is not None:
            del self._asr_model
            self._asr_model = None
            self._active = None
            gc.collect()

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes and return text."""
        if self._asr_model is None:
            raise RuntimeError("No model loaded")

        # Write WAV to temp file (onnx-asr can accept file paths)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            kwargs = {}
            if self._active and self._active.language:
                kwargs["language"] = self._active.language
            result = self._asr_model.recognize(tmp_path, **kwargs)
            return result
        finally:
            os.unlink(tmp_path)
