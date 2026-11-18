"""Model registry, download, and transcription using onnx-asr."""

from __future__ import annotations

import gc
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import sys

# Suppress onnxruntime CUDA warnings (e.g. missing cuBLAS/cuDNN) from stderr.
# ORT writes errors to C-level stderr (fd 2) during import and provider probes,
# bypassing Python logging entirely. We silence fd 2 during these calls.
_stderr_fd = sys.stderr.fileno()
_saved_fd = os.dup(_stderr_fd)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, _stderr_fd)
os.close(_devnull)
try:
    import onnxruntime as ort
finally:
    os.dup2(_saved_fd, _stderr_fd)
    os.close(_saved_fd)
del _stderr_fd, _saved_fd, _devnull

ort.set_default_logger_severity(3)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_LAST_MODEL_FILE = MODELS_DIR / ".last_model"


@dataclass
class ModelInfo:
    name: str
    onnx_asr_name: str  # onnx-asr model identifier
    description: str
    size_hint: str  # human-readable download size
    repo_id: str = ""  # HuggingFace repo ID (auto-resolved if empty)
    language: str = ""  # language hint passed to recognize() (e.g. "en")
    featured: bool = True  # keep old defaults available without promoting them


_BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="parakeet-ultra",
        onnx_asr_name="nemo-conformer-tdt",
        description="Recommended · 25-language ASR – Moondream Parakeet Ultra (Olicorne ONNX)",
        size_hint="~668 MB",
        repo_id="Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx",
    ),
    ModelInfo(
        name="parakeet-tdt-0.6b-v2",
        onnx_asr_name="nemo-parakeet-tdt-0.6b-v2",
        description="English baseline – Parakeet TDT 0.6B v2",
        size_hint="~640 MB",
        repo_id="istupakov/parakeet-tdt-0.6b-v2-onnx",
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

# Keep old names resolvable for installed models and .last_model preferences.
_LEGACY_MODELS = [ModelInfo(
    name="parakeet-tdt-0.6b-v3", onnx_asr_name="nemo-parakeet-tdt-0.6b-v3",
    description="Multilingual ASR – Parakeet TDT 0.6B v3", size_hint="~640 MB",
    repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
    featured=False,
)]


def _load_custom_models() -> list[ModelInfo]:
    """Load custom models from config.toml if it exists."""
    from .config import load_config

    config = load_config()
    custom = []
    for m in config.get("models", []):
        try:
            name = m["name"]
            if "/" in name or "\\" in name or ".." in name:
                continue  # reject path traversal in model names
            custom.append(ModelInfo(
                name=name,
                onnx_asr_name=m["onnx_asr_name"],
                description=m.get("description", ""),
                size_hint=m.get("size_hint", "unknown"),
                repo_id=m.get("repo_id", ""),
                language=m.get("language", ""),
                featured=m.get("featured", True),
            ))
        except KeyError:
            continue
    return custom


MODEL_REGISTRY: list[ModelInfo] = list({
    info.name: info for info in _BUILTIN_MODELS + _LEGACY_MODELS + _load_custom_models()
}.values())


def picker_models() -> list[ModelInfo]:
    """Show curated defaults, user-added models, and installed legacy models."""
    return [info for info in MODEL_REGISTRY if info.featured or is_model_downloaded(info)]

# Import only layouts understood by the existing NeMo ONNX engine.
IMPORT_MODEL_TYPES = {
    "nemo-conformer-tdt": ("Parakeet TDT", ("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx")),
    "nemo-conformer-rnnt": ("Parakeet RNNT", ("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx")),
    "nemo-conformer-ctc": ("Parakeet CTC", ("model.int8.onnx",)),
    "nemo-conformer-aed": ("Canary", ("encoder-model.int8.onnx", "decoder-model.int8.onnx")),
}


def parse_huggingface_repo(link: str) -> str:
    """Accept a model repository URL or owner/repo, never arbitrary download URLs."""
    value = link.strip()
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ValueError("Use an https://huggingface.co/owner/model repository link.")
        value = url.path.strip("/")
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(value)
    except ValueError as exc:
        raise ValueError("Paste a model repository link or owner/model (not a file link).") from exc
    if value.count("/") != 1:
        raise ValueError("Include both the Hugging Face owner and model name.")
    return value


def _nemo_download_files(model_type: str, config: dict, files: list[str]) -> dict[str, str]:
    """Map repo files to the flat layout required by onnx-asr's NeMo loader."""
    if config.get("model_type") != model_type:
        raise ValueError(f"Expected config.json model_type {model_type!r}.")
    required = ("config.json", "vocab.txt", *IMPORT_MODEL_TYPES[model_type][1])
    result = {}
    for dest in required:
        candidates = [dest] if dest in files else [f"int8/{dest}"]
        source = next((f for f in candidates if f in files), None)
        if source is None:
            raise ValueError(f"Not a compatible INT8 ONNX export: missing {dest} at root or in int8/.")
        result[source] = dest
        # Preserve optional ONNX external tensor data next to its graph.
        for filename in files:
            if filename.startswith(source + ".data"):
                suffix = filename[len(source):]
                if re.fullmatch(r"\.data(?:\.\d+)?", suffix):
                    result[filename] = dest + suffix
    return result


def inspect_huggingface_model(link: str, model_type: str = "auto", language: str = "") -> ModelInfo:
    """Validate small metadata before downloading weights. Runs in a worker."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = parse_huggingface_repo(link)
    metadata = HfApi().model_info(repo_id, files_metadata=True)
    files = [f.rfilename for f in metadata.siblings]
    if "config.json" not in files:
        raise ValueError("Missing config.json. Choose an onnx-asr compatible INT8 ONNX export.")
    with open(hf_hub_download(repo_id, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    detected = config.get("model_type")
    if model_type == "auto":
        model_type = detected
    if model_type not in IMPORT_MODEL_TYPES:
        raise ValueError("Unsupported model type. Use a NeMo Parakeet/Canary INT8 ONNX export; PyTorch, GGUF and CoreML are not supported.")
    plan = _nemo_download_files(model_type, config, files)
    sizes = {f.rfilename: f.size for f in metadata.siblings}
    total = sum(sizes[f] or 0 for f in plan)
    size = f"~{total / 1_000_000:.0f} MB" if all(sizes[f] is not None for f in plan) else "unknown"
    language = language.strip()
    if language and not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)?", language):
        raise ValueError("Use a language code such as en, fr or de.")
    if language and model_type != "nemo-conformer-aed":
        raise ValueError("Language selection is only supported for Canary; Parakeet detects language automatically.")
    return ModelInfo(
        name=repo_id.replace("/", "--"), onnx_asr_name=model_type,
        description=f"{IMPORT_MODEL_TYPES[model_type][0]} INT8 ONNX – {repo_id}",
        size_hint=size, repo_id=repo_id,
        language=(language or "en") if model_type == "nemo-conformer-aed" else "",
    )


def register_custom_model(info: ModelInfo) -> ModelInfo:
    """Persist an inspected model and make it available without restarting."""
    from dataclasses import asdict, replace
    from .config import save_custom_model

    for existing in MODEL_REGISTRY:
        if (existing.repo_id or existing.onnx_asr_name) == info.repo_id:
            if not existing.featured:
                promoted = replace(existing, featured=True)
                save_custom_model(asdict(promoted))
                existing.featured = True
            return existing
        if existing.name == info.name:
            raise ValueError(f"A model named {info.name} already exists.")
    save_custom_model(asdict(info))
    MODEL_REGISTRY.append(info)
    return info


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


def _suppress_stderr(func):
    """Call func with C-level stderr (fd 2) silenced.

    ORT writes errors directly to fd 2, bypassing Python's sys.stderr.
    Textual redirects sys.stderr so fileno() won't return 2 — we must
    hardcode fd 2 to actually suppress the native output.
    """
    try:
        saved_fd = os.dup(2)
    except OSError:
        return func()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return func()
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _detect_providers() -> tuple[list[str], str]:
    """Detect available ONNX Runtime providers. Returns (providers, label).

    Lists CUDA first if available, but onnxruntime will silently fall back
    to CPU if CUDA runtime libs (cuBLAS, cuDNN) aren't installed.
    """
    available = _suppress_stderr(ort.get_available_providers)
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

        destinations = {f: f for f in files}
        if "config.json" in all_files:
            with open(hf_hub_download(repo_id, "config.json"), encoding="utf-8") as f:
                config = json.load(f)
            model_type = config.get("model_type")
            if model_type in IMPORT_MODEL_TYPES:
                destinations = _nemo_download_files(model_type, config, all_files)
                files = list(destinations)

        if not files:
            raise RuntimeError(f"No matching files found in {repo_id}")

        for i, filename in enumerate(files):
            fraction = i / len(files)
            short_name = filename.rsplit("/", 1)[-1]
            if progress_cb:
                progress_cb(fraction, f"Downloading {short_name} ({i+1}/{len(files)})...")

            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            dest = (model_dir / destinations[filename]).resolve()
            if not str(dest).startswith(str(model_dir.resolve()) + os.sep):
                raise ValueError(f"Unsafe filename in repo: {filename}")
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

        self._asr_model = _suppress_stderr(
            lambda: onnx_asr.load_model(
                info.onnx_asr_name,
                path=str(model_dir),
                quantization="int8",
                providers=providers,
            )
        )
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
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows: file may still be held by ONNX Runtime
