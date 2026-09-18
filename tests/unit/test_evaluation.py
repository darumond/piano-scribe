from __future__ import annotations

from dataclasses import replace

import pytest
from piano_transcriber.evaluation.align import align_seconds
from piano_transcriber.evaluation.notes import match_notes
from piano_transcriber.evaluation.report import aggregate, compare
from piano_transcriber.evaluation.score import evaluate
from piano_transcriber.evaluation.types import (
    Alignment,
    EvaluationConfig,
    EvaluationNote,
    EvaluationScore,
)


def note(
    pitch: int = 60, onset: float = 0, duration: float = 0.5, **metadata: str
) -> EvaluationNote:
    return EvaluationNote(pitch, onset, duration, onset, duration, **metadata)


def score(*notes: EvaluationNote) -> EvaluationScore:
    return EvaluationScore(notes)


def metrics(prediction: EvaluationScore, reference: EvaluationScore) -> dict:
    return evaluate(prediction, reference, EvaluationConfig(alignment="none"))["metrics"]


def test_perfect_match() -> None:
    reference = score(note(), note(64, 1), note(67, 2))
    result = metrics(reference, reference)
    assert result["notes"]["true_positives"] == 3
    assert result["notes"]["f1"] == 1
    assert result["rhythm"]["rhythmic_value_accuracy"] == 1
    assert result["transcription"]["25ms"]["onset_error"]["mae"] == 0


def test_false_positive_and_negative() -> None:
    result = metrics(score(note(), note(67, 2)), score(note(), note(64, 1)))
    assert result["notes"]["true_positives"] == 1
    assert result["notes"]["false_positives"] == 1
    assert result["notes"]["false_negatives"] == 1
    assert result["notes"]["f1"] == 0.5
    assert result["pitch"]["false_positive_pitches"] == {"67": 1}
    assert result["pitch"]["false_negative_pitches"] == {"64": 1}


@pytest.mark.parametrize(("onset", "expected"), [(0.025, 1), (0.02501, 0), (0.05, 0)])
def test_onset_tolerance_boundary(onset: float, expected: int) -> None:
    assert (
        len(match_notes([note(onset=onset)], [note()], Alignment("seconds", "none"), 0.025))
        == expected
    )


def test_duplicate_notes_are_one_to_one() -> None:
    result = metrics(score(note(), note()), score(note()))
    assert result["notes"]["true_positives"] == 1
    assert result["notes"]["false_positives"] == 1


def test_maximum_cardinality_beats_greedy_nearest() -> None:
    pairs = match_notes(
        [note(onset=0.04), note(onset=0.09)],
        [note(onset=0), note(onset=0.05)],
        Alignment("seconds", "none"),
        0.05,
    )
    assert {(m.prediction, m.reference) for m in pairs} == {(0, 0), (1, 1)}


def test_assignment_minimizes_error_after_cardinality() -> None:
    pairs = match_notes(
        [note(onset=0.08), note(onset=0.02)],
        [note(onset=0), note(onset=0.1)],
        Alignment("seconds", "none"),
        0.1,
    )
    assert {(m.prediction, m.reference) for m in pairs} == {(0, 1), (1, 0)}


def test_octave_error_is_not_hidden_by_exact_pitch_matching() -> None:
    result = metrics(score(note(72)), score(note(60)))
    assert result["notes"]["f1"] == 0
    assert result["pitch"]["exact_pitch_accuracy"] == 0
    assert result["pitch"]["pitch_class_accuracy"] == 1
    assert result["pitch"]["octave_errors"] == 1
    assert result["pitch"]["semitone_errors"] == {"12": 1}


def test_duration_is_separate_from_onset_match() -> None:
    result = metrics(score(note(duration=2)), score(note(duration=0.5)))
    assert result["notes"]["f1"] == 1
    assert result["transcription"]["50ms"]["duration_error"]["mae"] == 1.5
    assert result["transcription"]["50ms"]["offset_error"]["mae"] == 1.5
    assert result["rhythm"]["rhythmic_value_accuracy"] == 0
    strict = evaluate(
        score(note(duration=2)),
        score(note()),
        EvaluationConfig(duration_tolerance_beats=0.1, alignment="none"),
    )
    assert strict["metrics"]["notes"]["true_positives"] == 0


def test_triplet_and_straight_rhythm_families_differ() -> None:
    result = metrics(score(note(duration=1 / 3)), score(note(duration=0.5)))
    assert result["rhythm"]["subdivision_family_accuracy"] == 0


def test_meter_pickup_and_grid_mismatch() -> None:
    prediction = replace(
        score(note()),
        time_signatures=((0, 4, 4),),
        pickup_beats=1,
        first_downbeat_beats=1,
        downbeats=(1, 5),
        measure_boundaries=(0, 1, 5),
    )
    reference = replace(
        score(note()),
        time_signatures=((0, 3, 4),),
        pickup_beats=0,
        first_downbeat_beats=0,
        downbeats=(0, 3, 6),
        measure_boundaries=(0, 3, 6),
    )
    result = metrics(prediction, reference)["meter"]
    assert result["correct_meter"] is False
    assert result["pickup_error_beats"] == 1
    assert result["first_full_downbeat_error_beats"] == 1
    assert result["downbeats"]["f1"] == 0
    assert result["measure_boundary_error_beats"]["mae"] > 0


def test_staff_does_not_imply_hand() -> None:
    result = metrics(score(note(staff="2")), score(note(staff="1")))
    assert result["staff"]["accuracy"] == 0
    assert result["staff"]["confusion_reference_rows_prediction_columns"] == {"1": {"2": 1}}
    assert result["hand"]["accuracy"] is None


def test_explicit_hand_labels() -> None:
    assert metrics(score(note(hand="left")), score(note(hand="left")))["hand"]["accuracy"] == 1


def test_voice_id_permutation() -> None:
    reference = score(
        note(60, 0, voice="1"),
        note(72, 0, voice="2"),
        note(61, 1, voice="1"),
        note(73, 1, voice="2"),
    )
    prediction = score(
        note(60, 0, voice="9"),
        note(72, 0, voice="4"),
        note(61, 1, voice="9"),
        note(73, 1, voice="4"),
    )
    voice = metrics(prediction, reference)["voice"]
    assert voice["assignment_accuracy"] == 1
    assert voice["track_purity"] == 1
    assert voice["voice_id_switches"] == 0
    assert voice["fragmentation"] == 0


def test_voice_fragmentation_is_distinct_from_purity() -> None:
    reference = score(note(60, 0, voice="1"), note(62, 1, voice="1"), note(64, 2, voice="1"))
    prediction = score(note(60, 0, voice="1"), note(62, 1, voice="2"), note(64, 2, voice="1"))
    result = metrics(prediction, reference)["voice"]
    assert result["fragmentation"] == 1
    assert result["voice_id_switches"] == 2
    assert result["assignment_accuracy"] == pytest.approx(2 / 3)
    assert result["track_purity"] == 1


def test_simultaneous_chord_split_does_not_create_order_dependent_switches() -> None:
    reference = score(
        note(60, 0, voice="1"),
        note(64, 0, voice="1"),
        note(62, 1, voice="1"),
        note(65, 1, voice="1"),
    )
    prediction = score(
        note(60, 0, voice="1"),
        note(64, 0, voice="2"),
        note(62, 1, voice="1"),
        note(65, 1, voice="2"),
    )
    result = metrics(prediction, reference)["voice"]
    assert result["voice_id_switches"] == 0
    assert result["per_reference_track"]["1:1"]["simultaneous_split_groups"] == 2


def test_chord_exact_and_partial_matches() -> None:
    reference = score(note(60), note(64), note(67), note(62, 1), note(65, 1))
    prediction = score(note(60), note(64), note(67), note(62, 1))
    result = metrics(prediction, reference)["chords"]
    assert result["reference_groups"] == 2
    assert result["exact_chord_set_matches"] == 1
    assert result["partial_chord_matches"] == 1
    assert result["f1"] == 0.5


def test_missing_metadata_is_unavailable() -> None:
    result = metrics(score(note()), score(note()))
    assert result["voice"]["assignment_accuracy"] is None
    assert result["staff"]["accuracy"] is None
    assert result["meter"]["correct_meter"] is None
    assert result["meter"]["pickup_error_beats"] is None
    assert result["structure"]["reference"]["rest_count"] is None


def test_global_offset_alignment() -> None:
    reference = EvaluationScore(
        tuple(EvaluationNote(p, i, 0.4) for i, p in enumerate((60, 64, 67, 72))),
        stage="transcription",
    )
    prediction = replace(
        reference,
        notes=tuple(
            replace(n, onset_seconds=float(n.onset_seconds or 0) + 2.345) for n in reference.notes
        ),
    )
    result = evaluate(prediction, reference)
    assert result["alignment"]["offset"] == pytest.approx(-2.345)
    assert result["metrics"]["transcription"]["25ms"]["f1"] == 1


def test_affine_alignment_small_tempo_mismatch() -> None:
    reference = EvaluationScore(
        tuple(EvaluationNote(48 + i % 30, i * 0.5, 0.25) for i in range(100)), stage="transcription"
    )
    prediction = replace(
        reference,
        notes=tuple(
            replace(
                n, onset_seconds=float(n.onset_seconds or 0) * 1.03 + 0.7, duration_seconds=0.2575
            )
            for n in reference.notes
        ),
    )
    alignment = align_seconds(prediction, reference, EvaluationConfig(alignment="affine"))
    assert alignment.scale == pytest.approx(1 / 1.03, abs=1e-4)
    assert alignment.offset == pytest.approx(-0.7 / 1.03, abs=1e-4)


def test_symbolic_alignment_ignores_tempo_mismatch_without_shifting_beats() -> None:
    reference = score(note(60, 0), note(64, 1), note(67, 2))
    prediction = replace(
        reference,
        notes=tuple(
            replace(n, onset_seconds=float(n.onset_seconds or 0) * 2) for n in reference.notes
        ),
    )
    result = evaluate(prediction, reference)
    assert result["alignment"]["domain"] == "beats"
    assert result["metrics"]["notes"]["f1"] == 1


def test_aggregation_macro_micro_and_missing_metadata() -> None:
    small = evaluate(score(note()), score(note()), EvaluationConfig(alignment="none"))
    large = evaluate(
        score(), score(*(note(60, i) for i in range(3))), EvaluationConfig(alignment="none")
    )
    result = aggregate([small, large])
    assert result["macro"]["notes.f1"]["mean"] == 0.5
    assert result["micro"]["notes"]["f1"] == 0.4
    assert "staff.accuracy" not in result["macro"]


def test_compare_rejects_different_reference_or_tolerance() -> None:
    original = evaluate(score(note()), score(note()), EvaluationConfig(alignment="none"))
    original["reference_sha256"] = "reference-one"
    with pytest.raises(ValueError, match="reference_sha256"):
        compare(original, {**original, "reference_sha256": "reference-two"})
    with pytest.raises(ValueError, match="config"):
        compare(original, {**original, "config": {}})


def test_compare_reports_dimensions_without_ranking_structure() -> None:
    original = evaluate(score(note(72)), score(note()), EvaluationConfig(alignment="none"))
    improved = evaluate(score(note()), score(note()), EvaluationConfig(alignment="none"))
    for item in (original, improved):
        item["reference_sha256"] = "same"
    result = compare(original, improved)
    assert any(x["metric"] == "notes.f1" for x in result["improved"])
    assert not any("structure" in x["metric"] for x in result["improved"])


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_tolerances_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        EvaluationConfig(onset_tolerances_ms=(value,))


def test_empty_scores_have_undefined_rates() -> None:
    result = metrics(score(), score())
    assert result["notes"]["true_positives"] == 0
    assert result["notes"]["f1"] is None


def test_missing_tempo_with_raw_prediction_returns_unavailable_not_false_errors() -> None:
    raw = EvaluationScore((EvaluationNote(60, 0, 1),), stage="transcription")
    reference = EvaluationScore((EvaluationNote(60, onset_beats=0, duration_beats=1),))
    report = evaluate(raw, reference)
    assert report["metrics"]["notes"]["f1"] is None
    assert report["metrics"]["notes"]["false_negatives"] is None
    assert report["metrics"]["structure"]["reference"]["note_count"] == 1
    assert aggregate([report])["micro"] == {}
