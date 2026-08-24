from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import mido
from piano_transcriber.midi.writer import write_score_midi
from piano_transcriber.notation.musicxml import write_score_musicxml
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.transcription.types import NoteEvent, PedalEvent, TranscriptionResult


def test_score_musicxml_uses_measures_chords_and_conventional_durations(tmp_path: Path) -> None:
    raw = TranscriptionResult(
        (
            NoteEvent(pitch=60, onset_seconds=0.01, offset_seconds=0.51),
            NoteEvent(pitch=64, onset_seconds=0.02, offset_seconds=0.52),
            NoteEvent(pitch=67, onset_seconds=3.75, offset_seconds=4.25),
        ),
        "test",
        5.0,
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    output = write_score_musicxml(score, tmp_path / "score.musicxml")
    root = ET.parse(output).getroot()

    assert root.findtext(".//time/beats") == "4"
    assert root.findtext(".//time/beat-type") == "4"
    assert root.find(".//sound").attrib["tempo"] == "60"  # type: ignore[union-attr]
    assert len(root.findall(".//measure")) == 2
    assert len(root.findall(".//chord")) == 1
    assert {element.text for element in root.findall(".//type")} <= {
        "whole",
        "half",
        "quarter",
        "eighth",
        "16th",
        "32nd",
    }
    assert all(int(element.text or "0") > 0 for element in root.findall(".//note/duration"))
    assert root.findall(".//tie[@type='start']")
    assert root.findall(".//tie[@type='stop']")


def test_triplet_musicxml_has_time_modification(tmp_path: Path) -> None:
    raw = TranscriptionResult(
        (NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=2 / 3),), "test", 1.0
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    root = ET.parse(write_score_musicxml(score, tmp_path / "triplet.musicxml")).getroot()
    assert root.findtext(".//time-modification/actual-notes") == "3"
    assert root.findtext(".//time-modification/normal-notes") == "2"


def test_score_midi_uses_quantized_beat_positions(tmp_path: Path) -> None:
    raw = TranscriptionResult(
        (NoteEvent(pitch=60, onset_seconds=0.49, offset_seconds=0.99),),
        "test",
        1.5,
        (PedalEvent(0.25, 1.25),),
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert score.notes[0].onset_beats == Fraction(1, 2)
    midi = mido.MidiFile(write_score_midi(score, tmp_path / "score.mid"))
    absolute_ticks = 0
    note_on_ticks = None
    for message in midi.tracks[0]:
        absolute_ticks += message.time
        if message.type == "note_on":
            note_on_ticks = absolute_ticks
            break
    assert note_on_ticks == 240
    pedal = [
        message
        for message in midi.tracks[0]
        if message.type == "control_change" and message.control == 64
    ]
    assert [message.value for message in pedal] == [127, 0]
