from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from piano_transcriber.notation.musicxml import write_score_musicxml
from piano_transcriber.score.chords import group_chords
from piano_transcriber.score.diagnostics import (
    piano_layout_data,
    write_staff_assignment_tsv,
    write_voice_assignment_tsv,
)
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.separation import (
    PianoLayoutMode,
    PianoSeparationConfig,
    separate_piano_score,
)
from piano_transcriber.score.types import PianoHand, ReconstructedScore
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def staff_score(events: list[tuple[int, float, float]]) -> ReconstructedScore:
    notes = tuple(
        NoteEvent(pitch=pitch, onset_seconds=onset, offset_seconds=onset + duration)
        for pitch, onset, duration in events
    )
    result = TranscriptionResult(
        notes,
        "synthetic",
        max(note.offset_seconds for note in notes) + 0.5,
    )
    return reconstruct_score(
        result,
        ReconstructionConfig(
            bpm=60.0,
            piano_separation=PianoSeparationConfig(mode=PianoLayoutMode.SEQUENCE),
        ),
    )


def test_simple_melody_and_bass_use_two_hands_and_staves() -> None:
    score = staff_score(
        [(48, beat, 0.75) for beat in range(4)] + [(72 + beat, beat, 0.75) for beat in range(4)]
    )
    bass = [note for note in score.notes if note.pitch == 48]
    melody = [note for note in score.notes if note.pitch >= 72]
    assert {(note.hand, note.staff) for note in bass} == {(PianoHand.LEFT, 2)}
    assert {(note.hand, note.staff) for note in melody} == {(PianoHand.RIGHT, 1)}
    assert {note.voice for note in bass + melody} == {1}


def test_right_hand_continuity_can_cross_below_middle_c() -> None:
    score = staff_score(
        [(43, beat, 0.75) for beat in range(4)]
        + [(67, 0, 0.75), (64, 1, 0.75), (59, 2, 0.75), (62, 3, 0.75)]
    )
    crossing_note = next(note for note in score.notes if note.pitch == 59)
    assert crossing_note.hand is PianoHand.RIGHT
    assert crossing_note.staff == 1


def test_left_hand_continuity_can_remain_above_middle_c() -> None:
    score = staff_score(
        [(52, 0, 0.75), (57, 1, 0.75), (61, 2, 0.75), (57, 3, 0.75)]
        + [(72, beat, 0.75) for beat in range(4)]
    )
    crossing_note = next(note for note in score.notes if note.pitch == 61)
    assert crossing_note.hand is PianoHand.LEFT
    assert crossing_note.staff == 2


def test_wide_chord_splits_jointly_between_hands() -> None:
    score = staff_score([(pitch, 0, 1.0) for pitch in (40, 48, 55, 64, 72, 79)])
    assert {note.pitch for note in score.notes if note.hand is PianoHand.LEFT} == {40, 48, 55}
    assert {note.pitch for note in score.notes if note.hand is PianoHand.RIGHT} == {64, 72, 79}
    assert len({note.onset_beats for note in score.notes}) == 1
    assert piano_layout_data(score)["chord_groups_split_between_hands"] == 1


def test_overlapping_right_hand_notes_use_two_voices_without_truncation() -> None:
    score = staff_score([(72, 0, 3.0), (64, 0, 0.75), (65, 1, 0.75), (67, 2, 0.75)])
    sustained = next(note for note in score.notes if note.pitch == 72)
    moving = [note for note in score.notes if note.pitch in {65, 67}]
    assert sustained.duration_beats == Fraction(3)
    assert sustained.voice == 1
    assert {note.voice for note in moving} == {2}
    assert all(note.onset_beats < sustained.offset_beats for note in moving)


def test_voice_aware_duration_refinement_can_restore_a_longer_note() -> None:
    raw = TranscriptionResult(
        (
            NoteEvent(pitch=72, onset_seconds=0.0, offset_seconds=3.0),
            NoteEvent(pitch=43, onset_seconds=0.0, offset_seconds=0.75),
            NoteEvent(pitch=65, onset_seconds=1.0, offset_seconds=1.75),
            NoteEvent(pitch=43, onset_seconds=1.0, offset_seconds=1.75),
        ),
        "synthetic",
        4.0,
    )
    base = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    shortened_notes = tuple(
        replace(note, duration_beats=Fraction(2)) if note.pitch == 72 else note
        for note in base.notes
    )
    shortened = replace(base, notes=shortened_notes, chords=group_chords(shortened_notes))
    score = separate_piano_score(
        shortened,
        PianoSeparationConfig(mode=PianoLayoutMode.SEQUENCE),
    )
    restored = next(note for note in score.notes if note.pitch == 72)
    moving = next(note for note in score.notes if note.pitch == 65)
    assert restored.voice != moving.voice
    assert restored.duration_beats == Fraction(3)
    assert restored.original_duration_beats == Fraction(2)
    assert restored.voice_duration_adjusted
    assert score.voice_duration_changes == 1


def test_rare_voice_crossing_is_allowed_without_allocating_extra_voices() -> None:
    score = staff_score([(67, 0, 2.0), (72, 0, 2.0), (74, 1, 0.75)])
    crossing = next(note for note in score.notes if note.pitch == 74)
    assert crossing.voice == 2
    assert max(note.voice for note in score.notes) == 2


def test_microtimed_chord_keeps_shared_written_onset_and_chord_identity() -> None:
    notes = (
        NoteEvent(pitch=48, onset_seconds=0.0, offset_seconds=0.75),
        NoteEvent(pitch=64, onset_seconds=0.008, offset_seconds=0.75),
        NoteEvent(pitch=67, onset_seconds=0.015, offset_seconds=0.75),
    )
    score = reconstruct_score(
        TranscriptionResult(notes, "synthetic", 1.5),
        ReconstructionConfig(
            bpm=60.0,
            piano_separation=PianoSeparationConfig(mode=PianoLayoutMode.SEQUENCE),
        ),
    )
    assert len({note.onset_beats for note in score.notes}) == 1
    assert len({note.chord_id for note in score.notes}) == 1


def test_grand_staff_musicxml_has_valid_voice_streams_and_boundary_ties(
    tmp_path: Path,
) -> None:
    score = staff_score([(48, 0, 1.0), (72, 3.5, 1.0), (67, 4.0, 0.5)])
    output = write_score_musicxml(score, tmp_path / "grand-staff.musicxml")
    root = ET.parse(output).getroot()
    assert root.findtext(".//staves") == "2"
    assert [(clef.attrib["number"], clef.findtext("sign")) for clef in root.findall(".//clef")] == [
        ("1", "G"),
        ("2", "F"),
    ]
    assert {element.text for element in root.findall(".//note/staff")} == {"1", "2"}
    assert root.findall(".//backup")
    assert root.findall(".//rest")
    assert root.findall(".//tie[@type='start']")
    assert root.findall(".//tie[@type='stop']")
    assert all(int(element.text or "0") > 0 for element in root.findall(".//note/duration"))
    divisions = int(root.findtext(".//divisions") or "1")
    assert all(
        Fraction(int(rest.findtext("duration") or "0"), divisions)
        >= score.minimum_explicit_rest_beats
        for rest in root.findall(".//note[rest]")
    )

    staff_tsv = write_staff_assignment_tsv(score, tmp_path / "staff-assignment.tsv")
    voice_tsv = write_voice_assignment_tsv(score, tmp_path / "voice-assignment.tsv")
    assert "assignment_confidence" in staff_tsv.read_text(encoding="utf-8").splitlines()[0]
    assert "duration_changed" in voice_tsv.read_text(encoding="utf-8").splitlines()[0]

    for measure in root.findall(".//measure"):
        expected = int(score.time_signature.measure_beats * divisions)
        streams: list[int] = [0]
        for element in measure:
            if element.tag == "backup":
                streams.append(0)
            elif (
                element.tag == "note" and element.find("chord") is None
            ) or element.tag == "forward":
                streams[-1] += int(element.findtext("duration") or "0")
        assert all(total == expected for total in streams)
