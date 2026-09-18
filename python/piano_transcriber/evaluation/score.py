"""Compose independent evaluation dimensions without a composite quality score."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

from piano_transcriber.evaluation.align import align_seconds, available, choose_domain
from piano_transcriber.evaluation.chords import chord_metrics
from piano_transcriber.evaluation.meter import meter_metrics
from piano_transcriber.evaluation.notes import detection, match_notes, matched_errors, pitch_metrics
from piano_transcriber.evaluation.rhythm import rhythm_metrics
from piano_transcriber.evaluation.types import Alignment, EvaluationConfig, EvaluationScore
from piano_transcriber.evaluation.voices import assignment_metrics, voice_metrics


def structure_metrics(score: EvaluationScore) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        name: None
        for name in (
            "measure_count",
            "staff_count",
            "voice_count",
            "tie_count",
            "rest_count",
            "beam_groups",
            "tuplet_groups",
        )
    }
    result.update(score.structure)
    result["note_count"] = len(score.notes)
    if score.notes and all(n.staff is not None for n in score.notes):
        result["staff_count"] = result["staff_count"] or len(
            {(n.part, n.staff) for n in score.notes}
        )
    if score.notes and all(n.voice is not None for n in score.notes):
        result["voice_count"] = len({(n.part, n.voice) for n in score.notes})
    return result


def evaluate(
    prediction: EvaluationScore, reference: EvaluationScore, config: EvaluationConfig | None = None
) -> dict[str, Any]:
    started = perf_counter()
    config = config or EvaluationConfig()
    domain = choose_domain(prediction, reference, config)
    if domain is None:
        reason = "No shared onset domain: supply explicit tempo or shared beat positions."
        unavailable: dict[str, Any] = {
            name: None
            for name in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
            )
        }
        unavailable.update(domain=None, reason=reason)
        return {
            "schema_version": 1,
            "kind": "piece",
            "config": asdict(config),
            "prediction_format": prediction.source_format,
            "reference_format": reference.source_format,
            "stage": prediction.stage if config.stage == "auto" else config.stage,
            "alignment": {"domain": None, "strategy": "unavailable", "reason": reason},
            "seconds_alignment": None,
            "warnings": [*prediction.warnings, *reference.warnings, reason],
            "metrics": {
                "notes": unavailable,
                "transcription": None,
                "pitch": None,
                "rhythm": None,
                "meter": None,
                "staff": None,
                "hand": None,
                "voice": None,
                "chords": None,
                "structure": {
                    "prediction": structure_metrics(prediction),
                    "reference": structure_metrics(reference),
                },
            },
            "matched_notes": [],
            "runtime_seconds": perf_counter() - started,
        }
    seconds_available = available(prediction, "seconds") and available(reference, "seconds")
    seconds_alignment = align_seconds(prediction, reference, config) if seconds_available else None
    alignment = (
        Alignment(
            "beats",
            "symbolic",
            warning="Beat origins are preserved; pickup and meter errors are not fitted away.",
        )
        if domain == "beats"
        else seconds_alignment
    )
    if alignment is None:
        raise ValueError("no usable alignment domain")
    tolerance = (
        config.onset_tolerance_beats
        if domain == "beats"
        else max(config.onset_tolerances_ms) / 1000
    )
    duration_tolerance = (
        config.duration_tolerance_beats if domain == "beats" else config.duration_tolerance_seconds
    )
    matches = match_notes(
        prediction.notes,
        reference.notes,
        alignment,
        tolerance,
        duration_tolerance=duration_tolerance,
    )
    seconds: dict[str, Any] | None = None
    if seconds_alignment is not None:
        seconds = {}
        for milliseconds in sorted(set(config.onset_tolerances_ms)):
            pairs = match_notes(
                prediction.notes,
                reference.notes,
                seconds_alignment,
                milliseconds / 1000,
                duration_tolerance=config.duration_tolerance_seconds,
            )
            seconds[f"{milliseconds:g}ms"] = {
                **detection(len(pairs), len(prediction.notes), len(reference.notes)),
                **matched_errors(prediction.notes, reference.notes, pairs, seconds_alignment),
            }
    symbolic = config.stage != "transcription" and (
        config.stage == "symbolic" or prediction.stage != "transcription"
    )
    return {
        "schema_version": 1,
        "kind": "piece",
        "config": asdict(config),
        "prediction_format": prediction.source_format,
        "reference_format": reference.source_format,
        "stage": "symbolic" if symbolic else "transcription",
        "alignment": asdict(alignment),
        "seconds_alignment": asdict(seconds_alignment) if seconds_alignment is not None else None,
        "warnings": [*prediction.warnings, *reference.warnings],
        "metrics": {
            "notes": {
                **detection(len(matches), len(prediction.notes), len(reference.notes)),
                "domain": domain,
                "onset_tolerance": tolerance,
                **matched_errors(prediction.notes, reference.notes, matches, alignment),
            },
            "transcription": seconds,
            "pitch": pitch_metrics(
                prediction.notes, reference.notes, matches, alignment, tolerance
            ),
            "rhythm": rhythm_metrics(prediction, reference, matches) if symbolic else None,
            "meter": meter_metrics(prediction, reference, config) if symbolic else None,
            "staff": assignment_metrics(prediction, reference, matches, "staff")
            if symbolic
            else None,
            "hand": assignment_metrics(prediction, reference, matches, "hand")
            if symbolic
            else None,
            "voice": voice_metrics(prediction, reference, matches) if symbolic else None,
            "chords": chord_metrics(
                prediction,
                reference,
                alignment,
                config.chord_group_beats if domain == "beats" else config.chord_group_seconds,
                tolerance,
            ),
            "structure": {
                "prediction": structure_metrics(prediction),
                "reference": structure_metrics(reference),
            },
        },
        "matched_notes": [asdict(m) for m in matches],
        "runtime_seconds": perf_counter() - started,
    }
