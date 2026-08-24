from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from piano_transcriber.score.diagnostics import (
    rhythm_optimization_data,
    score_diagnostics,
    write_rhythm_path_tsv,
)
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.rhythm import RhythmOptimizerMode
from piano_transcriber.score.tracking import fixed_beat_track
from piano_transcriber.score.types import EventDiagnostic, ReconstructedScore
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def optimized_score(
    onsets: list[float],
    *,
    offsets: list[float] | None = None,
    pitches: list[int] | None = None,
) -> ReconstructedScore:
    resolved_offsets = offsets or [onset + 0.32 for onset in onsets]
    resolved_pitches = pitches or [60 + index for index in range(len(onsets))]
    notes = tuple(
        NoteEvent(
            pitch=resolved_pitches[index],
            onset_seconds=onset,
            offset_seconds=resolved_offsets[index],
            velocity=80,
        )
        for index, onset in enumerate(onsets)
    )
    result = TranscriptionResult(notes, "synthetic", max(resolved_offsets) + 0.5)
    return reconstruct_score(
        result,
        ReconstructionConfig(
            bpm=60.0,
            adaptive_quantization=True,
            rhythm_optimizer=RhythmOptimizerMode.SEQUENCE,
        ),
        beat_track=fixed_beat_track(result.audio_duration_seconds, 60.0),
    )


def group_diagnostics(score: ReconstructedScore) -> list[EventDiagnostic]:
    seen: set[int] = set()
    groups = []
    for diagnostic in score.diagnostics:
        if diagnostic.action != "quantized" or diagnostic.rhythm_group_index in seen:
            continue
        seen.add(diagnostic.rhythm_group_index)
        groups.append(diagnostic)
    return groups


def test_human_timing_jitter_stays_in_straight_eighth_family() -> None:
    score = optimized_score([0.0, 0.53, 0.98, 1.52, 2.0, 2.47, 3.03, 3.49])
    groups = group_diagnostics(score)
    assert [item.quantized_onset_beats for item in groups] == [
        Fraction(index, 2) for index in range(8)
    ]
    assert {item.selected_rhythm_family for item in groups} == {"eighth-straight"}
    assert all("triplet" not in (item.selected_subdivision or "") for item in groups)


def test_clear_triplet_spacing_uses_supported_triplet_context() -> None:
    score = optimized_score([0.0, 1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3, 2.0])
    groups = group_diagnostics(score)
    assert [item.quantized_onset_beats for item in groups] == [
        Fraction(index, 3) for index in range(7)
    ]
    assert {item.selected_rhythm_family for item in groups} == {"eighth-triplet"}
    assert rhythm_optimization_data(score)["isolated_triplet_events"] == 0


def test_mixed_conventional_rhythm_uses_only_one_expected_family_change() -> None:
    score = optimized_score([0.0, 1.0, 1.5, 2.5, 3.5, 3.75, 4.0, 4.25])
    groups = group_diagnostics(score)
    assert [item.quantized_onset_beats for item in groups] == [
        Fraction(0),
        Fraction(1),
        Fraction(3, 2),
        Fraction(5, 2),
        Fraction(7, 2),
        Fraction(15, 4),
        Fraction(4),
        Fraction(17, 4),
    ]
    summary = rhythm_optimization_data(score)
    assert summary["rhythmic_family_switches"] == 1
    assert summary["straight_triplet_switches"] == 0


def test_one_noisy_onset_does_not_introduce_a_special_subdivision() -> None:
    score = optimized_score([0.0, 0.5, 1.0, 1.53, 2.0, 2.5, 3.0])
    groups = group_diagnostics(score)
    assert {item.selected_rhythm_family for item in groups} == {"eighth-straight"}
    assert groups[3].quantized_onset_beats == Fraction(3, 2)


def test_thirty_second_remains_available_when_timing_requires_it() -> None:
    score = optimized_score(
        [0.0, 0.125, 0.5, 1.0],
        offsets=[0.25, 0.25, 0.75, 1.25],
    )
    groups = group_diagnostics(score)
    assert groups[1].quantized_onset_beats == Fraction(1, 8)
    assert groups[1].selected_rhythm_family == "thirty-second"
    short_note = next(note for note in score.notes if note.onset_beats == Fraction(1, 8))
    assert short_note.duration_beats == Fraction(1, 8)


def test_measure_crossing_avoids_an_unnecessary_tie() -> None:
    score = optimized_score(
        [3.49, 4.0, 4.5],
        offsets=[3.98, 4.49, 4.99],
    )
    first = min(score.notes, key=lambda note: note.onset_beats)
    assert first.onset_beats == Fraction(7, 2)
    assert first.offset_beats == Fraction(4)
    assert rhythm_optimization_data(score)["tie_count"] == 0


def test_microtimed_chord_notes_share_one_selected_onset() -> None:
    score = optimized_score(
        [0.0, 0.008, 0.5, 0.512, 1.0, 1.006],
        pitches=[60, 64, 62, 65, 64, 67],
    )
    assert len(score.chords) == 3
    assert all(len(chord.notes) == 2 for chord in score.chords)
    assert [chord.onset_beats for chord in score.chords] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
    ]


def test_sustained_transition_from_straight_to_triplets_is_allowed() -> None:
    score = optimized_score([0.0, 0.5, 1.0, 1.5, 2.0, 7 / 3, 8 / 3, 3.0, 10 / 3, 11 / 3, 4.0])
    families = [item.selected_rhythm_family for item in group_diagnostics(score)]
    assert families[:5] == ["eighth-straight"] * 5
    assert families[5:] == ["eighth-triplet"] * 6
    assert rhythm_optimization_data(score)["straight_triplet_switches"] == 1


def test_sequence_diagnostics_expose_path_and_local_counterfactual(tmp_path: Path) -> None:
    score = optimized_score([0.0, 0.53, 0.98, 1.52, 2.0])
    output = write_rhythm_path_tsv(score, tmp_path / "rhythm-path.tsv")
    header = output.read_text(encoding="utf-8").splitlines()[0]
    assert "transition_cost" in header
    assert "local_best_subdivision" in header
    assert "changed_from_local" in header
    diagnostics = score_diagnostics(score)
    summary = diagnostics["rhythm_optimization"]
    assert summary["group_count"] == 5  # type: ignore[index]
    assert summary["events_changed_from_local"] > 0  # type: ignore[index]


def test_local_mode_remains_available_and_has_no_sequence_path(tmp_path: Path) -> None:
    result = TranscriptionResult(
        (
            NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=0.3),
            NoteEvent(pitch=62, onset_seconds=0.5, offset_seconds=0.8),
        ),
        "synthetic",
        1.0,
    )
    score = reconstruct_score(
        result,
        ReconstructionConfig(bpm=60.0, adaptive_quantization=True),
        beat_track=fixed_beat_track(1.0, 60.0),
    )
    assert score.rhythm_optimizer == "local"
    assert all(item.rhythm_group_index is None for item in score.diagnostics)
    with pytest.raises(ValueError, match="sequence optimization"):
        write_rhythm_path_tsv(score, tmp_path / "rhythm-path.tsv")
