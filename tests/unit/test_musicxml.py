from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from piano_transcriber.notation.musicxml import write_musicxml
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def test_write_musicxml_contains_expected_pitch(tmp_path: Path) -> None:
    result = TranscriptionResult(
        (NoteEvent(pitch=61, onset_seconds=0.0, offset_seconds=0.5),), "test", 1.0
    )
    output = write_musicxml(result, tmp_path / "score.musicxml")
    root = ET.parse(output).getroot()
    assert root.tag == "score-partwise"
    assert root.findtext(".//pitch/step") == "C"
    assert root.findtext(".//pitch/alter") == "1"
    assert root.findtext(".//pitch/octave") == "4"
