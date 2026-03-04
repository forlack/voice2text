"""Voice Activity Detection using Silero VAD ONNX model.

Uses onnxruntime (already installed) to run the ~2MB Silero VAD model directly,
avoiding the silero-vad-lite dependency which lacks Python 3.14 wheels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples (32ms) at 16kHz
SILERO_CHUNK_SAMPLES = 512
_CONTEXT_SIZE = 64  # context samples prepended to each chunk

_VAD_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "_silero_vad"
_VAD_MODEL_PATH = _VAD_MODEL_DIR / "silero_vad.onnx"
_VAD_REPO_ID = "onnx-community/silero-vad"
_VAD_FILENAME = "onnx/model.onnx"


def _ensure_vad_model() -> Path:
    """Download the Silero VAD ONNX model if not present (~2MB)."""
    if _VAD_MODEL_PATH.exists():
        return _VAD_MODEL_PATH
    from huggingface_hub import hf_hub_download

    _VAD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=_VAD_REPO_ID,
        filename=_VAD_FILENAME,
    )
    # Copy from HF cache to our local models dir
    import shutil
    shutil.copy2(downloaded, _VAD_MODEL_PATH)
    return _VAD_MODEL_PATH


class VoiceActivityDetector:
    """Wraps the Silero VAD ONNX model for speech/silence detection.

    Accepts 16-bit signed integer PCM audio (matching our recorder output)
    and converts to float32 internally for the model.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        model_path = _ensure_vad_model()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def process(self, audio_chunk: bytes) -> float:
        """Feed a 512-sample chunk of 16-bit PCM and return speech probability (0-1).

        The chunk must be exactly 512 samples = 1024 bytes of int16 PCM.
        """
        # Convert int16 PCM to float32 [-1, 1]
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        x = samples.reshape(1, -1)

        # Prepend context from previous chunk
        x_with_ctx = np.concatenate([self._context, x], axis=1)

        ort_inputs = {
            "input": x_with_ctx,
            "state": self._state,
            "sr": self._sr,
        }
        out, new_state = self._session.run(None, ort_inputs)

        # Update state and context
        self._state = new_state
        self._context = x_with_ctx[:, -_CONTEXT_SIZE:]

        return float(out.squeeze())

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Convenience: returns True if speech probability exceeds threshold."""
        return self.process(audio_chunk) > self.threshold

    def reset(self) -> None:
        """Reset internal state for a new recording session."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
