from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mido
from piano_transcriber.config import ModelName, PipelineConfig
from piano_transcriber.transcription.pipeline import TranscriptionPipeline


def test_synthetic_audio_pipeline_creates_outputs(synthetic_wav: Path, tmp_path: Path) -> None:
    midi_path = tmp_path / "output.mid"
    musicxml_path = tmp_path / "output.musicxml"
    pipeline = TranscriptionPipeline(PipelineConfig(model=ModelName.MOCK))
    output = pipeline.run(synthetic_wav, midi_path=midi_path, musicxml_path=musicxml_path)

    assert midi_path.is_file() and midi_path.stat().st_size > 0
    assert musicxml_path.is_file() and musicxml_path.stat().st_size > 0
    assert [note.pitch for note in output.result.notes] == [60, 64, 67]

    midi_pitches = {
        message.note
        for track in mido.MidiFile(midi_path).tracks
        for message in track
        if message.type == "note_on"
    }
    xml_steps = [element.text for element in ET.parse(musicxml_path).findall(".//pitch/step")]
    assert midi_pitches == {60, 64, 67}
    assert xml_steps == ["C", "E", "G"]
