"""Grammar correction using T5 ONNX models.

Runs a T5 encoder-decoder model via raw ONNX Runtime + tokenizers library.
No torch/transformers dependency needed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

_GRAMMAR_DIR = Path(__file__).resolve().parent.parent / "models" / "_grammar"
_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.toml"

# Default model
_DEFAULT_REPO = "onnx-community/t5-base-grammar-correction-ONNX"
_DEFAULT_SIZE_HINT = "~570 MB"

# Files to download from the repo
_MODEL_FILES = [
    "onnx/encoder_model_int8.onnx",
    "onnx/decoder_model_int8.onnx",
    "onnx/decoder_with_past_model_int8.onnx",
    "tokenizer.json",
    "config.json",
]

# T5 token IDs (standard for t5-base)
_DECODER_START_ID = 0
_EOS_ID = 1
_PAD_ID = 0


def _load_grammar_config() -> dict:
    """Load grammar model config from config.toml if present."""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        with open(_CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
    except Exception:
        return {}
    return config.get("post_processing", {})


def _get_repo_id() -> str:
    cfg = _load_grammar_config()
    return cfg.get("model", _DEFAULT_REPO)


def is_grammar_downloaded() -> bool:
    return (_GRAMMAR_DIR / ".downloaded").exists()


def download_grammar_model(
    progress_cb: callable | None = None,
) -> None:
    """Download grammar model files from HuggingFace."""
    from huggingface_hub import hf_hub_download

    repo_id = _get_repo_id()
    _GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(_MODEL_FILES):
        fraction = i / len(_MODEL_FILES)
        short = filename.rsplit("/", 1)[-1]
        if progress_cb:
            progress_cb(fraction, f"Downloading {short} ({i+1}/{len(_MODEL_FILES)})...")

        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        dest = (_GRAMMAR_DIR / filename).resolve()
        if not str(dest).startswith(str(_GRAMMAR_DIR.resolve()) + os.sep):
            raise ValueError(f"Unsafe filename: {filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)

    (_GRAMMAR_DIR / ".downloaded").touch()

    if progress_cb:
        progress_cb(1.0, "Grammar model downloaded")


def get_grammar_size_hint() -> str:
    cfg = _load_grammar_config()
    return cfg.get("size_hint", _DEFAULT_SIZE_HINT)


class GrammarCorrector:
    """T5 grammar correction via raw ONNX Runtime."""

    def __init__(self) -> None:
        from tokenizers import Tokenizer

        model_dir = _GRAMMAR_DIR

        self._tokenizer = Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1

        # Suppress ORT CUDA stderr during session creation
        try:
            saved_fd = os.dup(2)
        except OSError:
            saved_fd = None

        if saved_fd is not None:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)

        try:
            self._encoder = ort.InferenceSession(
                str(model_dir / "onnx" / "encoder_model_int8.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            self._decoder = ort.InferenceSession(
                str(model_dir / "onnx" / "decoder_with_past_model_int8.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            self._decoder_init = ort.InferenceSession(
                str(model_dir / "onnx" / "decoder_model_int8.onnx"),
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
        finally:
            if saved_fd is not None:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)

    def correct(self, text: str, max_length: int = 512) -> str:
        """Run grammar correction on text. Returns corrected text.

        Processes sentence-by-sentence to prevent the model from
        dropping content when given long multi-sentence input.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= 1:
            return self._correct_single(text, max_length)
        corrected = []
        for sentence in sentences:
            if sentence.strip():
                corrected.append(self._correct_single(sentence, max_length))
        return " ".join(corrected)

    def _correct_single(self, text: str, max_length: int = 512) -> str:
        """Correct a single sentence/fragment."""
        # Tokenize
        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Encode
        encoder_out = self._encoder.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        encoder_hidden = encoder_out[0]

        # Build name mappings for KV cache: present.X.Y -> past_key_values.X.Y
        past_input_names = [
            inp.name for inp in self._decoder.get_inputs()
            if inp.name.startswith("past_key_values")
        ]
        init_output_names = [
            out.name for out in self._decoder_init.get_outputs()
            if out.name.startswith("present")
        ]
        # decoder_with_past outputs only decoder KV (no encoder KV)
        past_output_names = [
            out.name for out in self._decoder.get_outputs()
            if out.name.startswith("present")
        ]

        # First step: full decoder (produces encoder + decoder KV cache)
        decoder_input = np.array([[_DECODER_START_ID]], dtype=np.int64)
        results = self._decoder_init.run(None, {
            "input_ids": decoder_input,
            "encoder_attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden,
        })
        logits = results[0]
        # Map present.* outputs to past_key_values.* by position
        init_kv = {
            name.replace("present", "past_key_values"): results[i + 1]
            for i, name in enumerate(init_output_names)
        }

        output_ids = []
        next_id = int(np.argmax(logits[0, -1, :]))
        if next_id == _EOS_ID:
            return self._tokenizer.decode(output_ids, skip_special_tokens=True)
        output_ids.append(next_id)

        # Subsequent steps: decoder with past KV cache
        for _ in range(max_length - 1):
            decoder_input = np.array([[next_id]], dtype=np.int64)
            feeds = {
                "input_ids": decoder_input,
                "encoder_attention_mask": attention_mask,
            }
            # Add past KV: encoder keys stay from init, decoder keys update
            for name in past_input_names:
                feeds[name] = init_kv[name]

            results = self._decoder.run(None, feeds)
            logits = results[0]

            # Update decoder KV cache (encoder KV stays the same)
            for i, name in enumerate(past_output_names):
                kv_name = name.replace("present", "past_key_values")
                init_kv[kv_name] = results[i + 1]

            next_id = int(np.argmax(logits[0, -1, :]))
            if next_id == _EOS_ID:
                break
            output_ids.append(next_id)

        corrected = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return corrected
