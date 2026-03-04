"""Transcript file storage and history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TRANSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "transcripts"


@dataclass
class TranscriptEntry:
    path: Path
    timestamp: datetime
    preview: str  # first line / truncated

    @property
    def filename(self) -> str:
        return self.path.name

    def full_text(self) -> str:
        return self.path.read_text(encoding="utf-8")


def save_transcript(text: str) -> TranscriptEntry:
    """Save transcript text to a timestamped .txt file."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    path = TRANSCRIPTS_DIR / filename
    path.write_text(text, encoding="utf-8")
    preview = text[:80].replace("\n", " ").strip()
    return TranscriptEntry(path=path, timestamp=now, preview=preview)


def load_history() -> list[TranscriptEntry]:
    """Load all transcripts from disk, sorted newest-first."""
    if not TRANSCRIPTS_DIR.exists():
        return []
    entries = []
    for p in sorted(TRANSCRIPTS_DIR.glob("*.txt"), reverse=True):
        try:
            # Parse timestamp from filename
            stem = p.stem  # e.g. "2026-03-03_14-30-00"
            ts = datetime.strptime(stem, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            ts = datetime.fromtimestamp(p.stat().st_mtime)
        text = p.read_text(encoding="utf-8")
        preview = text[:80].replace("\n", " ").strip()
        entries.append(TranscriptEntry(path=p, timestamp=ts, preview=preview))
    return entries
