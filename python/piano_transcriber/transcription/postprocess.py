"""Backend-independent note filtering and ordering."""

from __future__ import annotations

from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def postprocess_result(
    result: TranscriptionResult, *, minimum_confidence: float = 0.0
) -> TranscriptionResult:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")
    notes: tuple[NoteEvent, ...] = tuple(
        sorted(note for note in result.notes if note.confidence >= minimum_confidence)
    )
    return TranscriptionResult(
        notes,
        result.model_name,
        result.audio_duration_seconds,
        result.pedal_events,
    )
