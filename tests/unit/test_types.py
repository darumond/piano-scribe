from __future__ import annotations

import pytest
from piano_transcriber.transcription.types import NoteEvent, PedalEvent, TranscriptionResult


def test_valid_note_event() -> None:
    event = NoteEvent(pitch=60, onset_seconds=0.1, offset_seconds=0.5, velocity=90)
    assert event.pitch == 60
    assert event.confidence == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pitch": 128, "onset_seconds": 0.0, "offset_seconds": 1.0}, "pitch"),
        ({"pitch": 60.5, "onset_seconds": 0.0, "offset_seconds": 1.0}, "pitch"),
        ({"pitch": 60, "onset_seconds": -0.1, "offset_seconds": 1.0}, "onset"),
        ({"pitch": 60, "onset_seconds": 1.0, "offset_seconds": 1.0}, "offset"),
        (
            {"pitch": 60, "onset_seconds": 0.0, "offset_seconds": 1.0, "confidence": 1.1},
            "confidence",
        ),
    ],
)
def test_invalid_note_event(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NoteEvent(**kwargs)  # type: ignore[arg-type]


def test_result_rejects_note_beyond_audio() -> None:
    note = NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=2.0)
    with pytest.raises(ValueError, match="beyond"):
        TranscriptionResult((note,), "test", 1.0)


def test_transcription_result_preserves_raw_pedal_events() -> None:
    pedal = PedalEvent(0.1, 0.8)
    result = TranscriptionResult((), "test", 1.0, (pedal,))
    assert result.pedal_events == (pedal,)
