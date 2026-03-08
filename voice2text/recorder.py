"""Audio recording module using PyAudio."""

import ctypes
import io
import os
import sys
import threading
import wave

import numpy as np
import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024


_alsa_error_handler = None  # prevent GC of ctypes callback


def _suppress_alsa_errors() -> None:
    """Suppress ALSA error/warning messages that PortAudio dumps to stderr."""
    global _alsa_error_handler
    if _alsa_error_handler is not None:
        return  # already installed
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        c_handler_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        )
        _alsa_error_handler = c_handler_type(lambda *args: None)
        asound.snd_lib_error_set_handler(_alsa_error_handler)
    except OSError:
        pass  # not on ALSA, nothing to suppress


class Recorder:
    """Records audio from the default microphone."""

    def __init__(self) -> None:
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._frames: list[bytes] = []
        self._recording = False
        self._paused = False
        self._lock = threading.Lock()
        self._level: float = 0.0  # 0.0-1.0 normalized RMS level

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def level(self) -> float:
        """Current audio level 0.0–1.0."""
        return self._level

    @property
    def frame_count(self) -> int:
        """Number of audio chunks recorded so far."""
        with self._lock:
            return len(self._frames)

    @staticmethod
    def get_default_mic_name() -> str:
        """Return the name of the default input device."""
        _suppress_alsa_errors()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            pa = pyaudio.PyAudio()
            try:
                info = pa.get_default_input_device_info()
                return info.get("name", "Unknown")
            except IOError:
                return "No microphone found"
            finally:
                pa.terminate()
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)

    def start(self) -> None:
        """Start recording from default mic."""
        if self._recording:
            return
        _suppress_alsa_errors()
        # Suppress JACK/ALSA stderr noise during PortAudio init
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            self._pa = pyaudio.PyAudio()
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)
        self._frames = []
        self._level = 0.0
        self._recording = True
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def pause(self) -> None:
        """Pause recording — stream stays open for level meter but frames are not saved."""
        self._paused = True

    def resume(self) -> None:
        """Resume recording after a pause."""
        self._paused = False

    def stop(self) -> bytes:
        """Stop recording and return WAV bytes."""
        self._recording = False
        self._paused = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        self._level = 0.0
        return self._build_wav()

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict,
        status_flags: int,
    ) -> tuple[None, int]:
        if in_data and self._recording:
            # Compute RMS level even while paused (visual feedback)
            samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples**2)) / 32768.0
            self._level = min(1.0, rms * 5.0)  # amplify for visual feedback
            # Only save frames when not paused
            if not self._paused:
                with self._lock:
                    self._frames.append(in_data)
        return (None, pyaudio.paContinue)

    def extract_segment(self, start_frame: int, end_frame: int) -> bytes:
        """Build a WAV from a slice of recorded frames without stopping.

        start_frame and end_frame are indices into the frames list (chunk indices).
        """
        buf = io.BytesIO()
        with self._lock:
            frames = list(self._frames[start_frame:end_frame])
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    def _build_wav(self) -> bytes:
        """Build a WAV file in memory from recorded frames."""
        buf = io.BytesIO()
        with self._lock:
            frames = list(self._frames)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()
