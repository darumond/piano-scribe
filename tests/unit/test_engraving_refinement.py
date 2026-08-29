from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from piano_transcriber.engraving.pipeline import EngravingConfig, EngravingMode, apply_engraving
from piano_transcriber.notation.musicxml import write_score_musicxml
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.tracking import fixed_beat_track
from piano_transcriber.score.types import PianoHand, ReconstructedScore, ScoreRest, TimeSignature
from piano_transcriber.transcription.types import NoteEvent, PedalEvent, TranscriptionResult


def _engraved(
    attacks: list[float],
    *,
    duration: float,
    meter: str = "4/4",
) -> ReconstructedScore:
    notes = tuple(
        NoteEvent(onset_seconds=onset, pitch=60 + index, offset_seconds=onset + duration)
        for index, onset in enumerate(attacks)
    )
    return reconstruct_score(
        TranscriptionResult(notes, "synthetic", max(note.offset_seconds for note in notes) + 0.1),
        ReconstructionConfig(
            bpm=60.0,
            time_signature=TimeSignature.parse(meter),
            engraving=EngravingConfig(mode=EngravingMode.REFINED),
        ),
    )


def test_eighth_notes_are_beamed_by_quarter_note_pulses() -> None:
    score = _engraved([index / 2 for index in range(8)], duration=0.5)
    assert len({item.group_id for item in score.beam_annotations}) == 4
    assert {item.value for item in score.beam_annotations} == {"begin", "end"}


def test_sixteenth_notes_receive_two_beam_levels() -> None:
    score = _engraved([index / 4 for index in range(4)], duration=0.25)
    assert {item.level for item in score.beam_annotations} == {1, 2}
    assert len({item.group_id for item in score.beam_annotations}) == 1


@pytest.mark.parametrize(("meter", "count"), [("6/8", 6), ("6/4", 12)])
def test_compound_meter_beams_follow_dotted_pulses(meter: str, count: int) -> None:
    score = _engraved([index / 2 for index in range(count)], duration=0.5, meter=meter)
    assert len({item.group_id for item in score.beam_annotations}) == 2


def test_beams_never_cross_measure_boundaries() -> None:
    score = _engraved([3.5, 4.0, 4.5], duration=0.5)
    assert {item.measure_index for item in score.beam_annotations} == {1}
    assert {item.source_index for item in score.beam_annotations} == {1, 2}


def test_triplet_group_has_start_stop_notations(tmp_path: Path) -> None:
    notes = tuple(
        NoteEvent(onset_seconds=index / 3, pitch=60 + index, offset_seconds=index / 3 + 0.3)
        for index in range(3)
    )
    raw = TranscriptionResult(notes, "synthetic", 1.2)
    score = reconstruct_score(
        raw,
        ReconstructionConfig(
            bpm=60.0,
            adaptive_quantization=True,
            engraving=EngravingConfig(mode=EngravingMode.REFINED),
        ),
        beat_track=fixed_beat_track(1.2, 60.0),
    )
    assert [item.value for item in score.tuplet_annotations] == ["start", "stop"]
    root = ET.parse(write_score_musicxml(score, tmp_path / "triplet.musicxml")).getroot()
    assert [item.attrib["type"] for item in root.findall(".//tuplet")] == ["start", "stop"]


def test_refined_rest_pass_omits_secondary_voice_boundary_padding() -> None:
    base = _engraved([1.0], duration=0.5)
    note = replace(base.notes[0], voice=2)
    padded = replace(
        base,
        notes=(note,),
        rests=(
            ScoreRest(Fraction(0), Fraction(1), 1, 2),
            ScoreRest(Fraction(3, 2), Fraction(1), 1, 2),
            ScoreRest(Fraction(0), Fraction(1, 2), 1, 1),
            ScoreRest(Fraction(1, 2), Fraction(1, 2), 1, 1),
        ),
        engraving_mode=EngravingMode.BASIC.value,
    )
    refined = apply_engraving(padded, EngravingConfig(mode=EngravingMode.REFINED))
    assert refined.rests == (ScoreRest(Fraction(0), Fraction(1), 1, 1),)
    assert sum(item.action == "omit" for item in refined.rest_decisions) == 2
    assert refined.merged_rest_count == 1


def test_musicxml_serializes_pedal_start_and_stop(tmp_path: Path) -> None:
    note = NoteEvent(onset_seconds=0.0, pitch=60, offset_seconds=1.0)
    raw = TranscriptionResult((note,), "synthetic", 2.0, (PedalEvent(0.25, 1.5),))
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    root = ET.parse(write_score_musicxml(score, tmp_path / "pedal.musicxml")).getroot()
    assert [item.attrib["type"] for item in root.findall(".//pedal")] == ["start", "stop"]


def test_span_diagnostics_distinguish_attack_and_sustained_overlap() -> None:
    score = _engraved([0.0], duration=1.0)
    low = replace(score.notes[0], pitch=60, hand=PianoHand.RIGHT)
    high = replace(score.notes[0], source_index=1, pitch=81, hand=PianoHand.RIGHT)
    diagnostic_score = apply_engraving(
        replace(score, notes=(low, high), engraving_mode=EngravingMode.BASIC.value),
        EngravingConfig(mode=EngravingMode.REFINED),
    )
    assert diagnostic_score.hand_span_diagnostics[0].attack_span_semitones == 21
    assert ">20" in diagnostic_score.hand_span_diagnostics[0].threshold_flags
