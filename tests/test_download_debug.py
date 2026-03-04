"""Diagnostic script for model download issues.

Run directly:  python -m tests.test_download_debug
Or via pytest:  pytest tests/test_download_debug.py -s
"""

import os
import sys
import platform
import traceback


def _header():
    print("=" * 60)
    print("Download Debug Diagnostics")
    print("=" * 60)
    print(f"Python:    {sys.version}")
    print(f"Platform:  {platform.platform()}")
    print(f"Machine:   {platform.machine()}")
    print()


def _check_env_vars():
    print("--- Environment Variables ---")
    for var in [
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_XET_HIGH_PERFORMANCE",
        "HF_HUB_OFFLINE",
    ]:
        print(f"  {var}={os.environ.get(var, '<not set>')}")
    print()


def _check_packages():
    print("--- Package Versions ---")
    for pkg in ["huggingface_hub", "hf_xet", "hf_transfer", "onnx_asr", "onnxruntime"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed (no version)")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")
    print()


def _check_constants():
    print("--- huggingface_hub Constants ---")
    try:
        from huggingface_hub import constants
        print(f"  HF_HUB_DISABLE_XET: {constants.HF_HUB_DISABLE_XET}")
        print(f"  HF_XET_HIGH_PERFORMANCE: {constants.HF_XET_HIGH_PERFORMANCE}")
    except Exception as e:
        print(f"  ERROR reading constants: {e}")
    print()


def _test_list_repo_files():
    """Test basic HF API call (no download, just listing)."""
    print("--- Test: list_repo_files ---")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files("onnx-community/silero-vad")
        print(f"  OK: listed {len(files)} files")
    except Exception:
        print(f"  FAILED:")
        traceback.print_exc()
    print()


def _test_hf_hub_download_small():
    """Test downloading a tiny file via hf_hub_download."""
    print("--- Test: hf_hub_download (small file) ---")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="onnx-community/silero-vad",
            filename="onnx/model.onnx",
        )
        size = os.path.getsize(path)
        print(f"  OK: downloaded to {path} ({size} bytes)")
    except Exception:
        print(f"  FAILED:")
        traceback.print_exc()
    print()


def _test_model_download():
    """Test the actual voice2text model download flow."""
    print("--- Test: ModelManager.download_model ---")
    try:
        from voice2text.models import MODEL_REGISTRY, ModelManager
        mm = ModelManager()
        info = MODEL_REGISTRY[0]
        print(f"  Downloading: {info.name} ({info.repo_id})")
        mm.download_model(info, progress_cb=lambda f, t: print(f"  [{f:.0%}] {t}"))
        print(f"  OK: download complete")
    except Exception:
        print(f"  FAILED:")
        traceback.print_exc()
    print()


def run_all():
    _header()

    # Import voice2text first (sets env vars)
    print("--- Importing voice2text ---")
    try:
        import voice2text
        print(f"  OK")
    except Exception:
        print(f"  FAILED:")
        traceback.print_exc()
    print()

    _check_env_vars()
    _check_packages()
    _check_constants()
    _test_list_repo_files()
    _test_hf_hub_download_small()
    _test_model_download()

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    run_all()


# pytest entry point
def test_download_debug():
    """Run all download diagnostics (use pytest -s to see output)."""
    run_all()
