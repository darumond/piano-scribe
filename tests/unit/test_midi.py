from __future__ import annotations

from pathlib import Path

import mido
from piano_transcriber.midi.writer import write_midi
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def test_write_midi_contains_expected_notes(tmp_path: Path) -> None:
    result = TranscriptionResult(
        (NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=0.5, velocity=90),),
        "test",
        1.0,
    )
    output = write_midi(result, tmp_path / "nested" / "result.mid")
    assert output.is_file()
    messages = [message for track in mido.MidiFile(output).tracks for message in track]
    assert any(message.type == "note_on" and message.note == 60 for message in messages)
    assert any(message.type == "note_off" and message.note == 60 for message in messages)
