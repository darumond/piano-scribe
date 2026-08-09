"""Deterministic backend used by tests and pipeline demonstrations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from piano_transcriber.models.base import TranscriptionModel
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


class MockTranscriptionModel(TranscriptionModel):
    @property
    def name(self) -> str:
        return "mock"

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TranscriptionResult:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        duration = float(audio.size) / sample_rate
        if duration <= 0.0:
            return TranscriptionResult(notes=(), model_name=self.name, audio_duration_seconds=0.0)
        pitches = (60, 64, 67)
        note_length = min(0.4, duration / len(pitches))
        notes = tuple(
            NoteEvent(
                pitch=pitch,
                onset_seconds=index * note_length,
                offset_seconds=min(duration, (index + 1) * note_length),
                velocity=80 + index * 4,
                confidence=1.0,
            )
            for index, pitch in enumerate(pitches)
            if index * note_length < duration
        )
        return TranscriptionResult(
            notes=notes,
            model_name=self.name,
            audio_duration_seconds=duration,
        )
