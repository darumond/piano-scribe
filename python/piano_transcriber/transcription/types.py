"""Normalized, backend-independent transcription types."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class NoteEvent:
    onset_seconds: float
    pitch: int
    offset_seconds: float
    velocity: int = 80
    confidence: float = 1.0
    pedal: bool | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pitch, int)
            or isinstance(self.pitch, bool)
            or not 0 <= self.pitch <= 127
        ):
            raise ValueError("pitch must be a MIDI integer between 0 and 127")
        if not math.isfinite(self.onset_seconds) or self.onset_seconds < 0.0:
            raise ValueError("onset_seconds must be finite and non-negative")
        if not math.isfinite(self.offset_seconds) or self.offset_seconds <= self.onset_seconds:
            raise ValueError("offset_seconds must be finite and greater than onset_seconds")
        if (
            not isinstance(self.velocity, int)
            or isinstance(self.velocity, bool)
            or not 0 <= self.velocity <= 127
        ):
            raise ValueError("velocity must be an integer between 0 and 127")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    notes: tuple[NoteEvent, ...]
    model_name: str
    audio_duration_seconds: float

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not math.isfinite(self.audio_duration_seconds) or self.audio_duration_seconds < 0.0:
            raise ValueError("audio_duration_seconds must be finite and non-negative")
        if any(note.offset_seconds > self.audio_duration_seconds + 1e-6 for note in self.notes):
            raise ValueError("note events cannot extend beyond the audio duration")
