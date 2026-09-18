"""Meter and unshifted symbolic grid evaluation."""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

from piano_transcriber.evaluation.notes import detection, distribution, window_pairs
from piano_transcriber.evaluation.types import EvaluationConfig, EvaluationScore


def _grid(
    prediction: tuple[float, ...] | None, reference: tuple[float, ...] | None, tolerance: float
) -> dict[str, Any] | None:
    if prediction is None or reference is None:
        return None
    pairs = window_pairs(prediction, reference, tolerance)
    return {
        **detection(len(pairs), len(prediction), len(reference)),
        "matched_error_beats": distribution([prediction[p] - reference[r] for p, r in pairs]),
    }


def _boundary_error(
    prediction: tuple[float, ...] | None, reference: tuple[float, ...] | None
) -> dict[str, Any]:
    errors: list[float] = []
    if prediction and reference:
        ordered = sorted(prediction)
        for boundary in reference:
            index = bisect_left(ordered, boundary)
            candidates = ordered[max(0, index - 1) : index + 1]
            errors.append(min(candidates, key=lambda x: abs(x - boundary)) - boundary)
    return distribution(errors)


def meter_metrics(
    prediction: EvaluationScore, reference: EvaluationScore, config: EvaluationConfig
) -> dict[str, Any]:
    correct: bool | None = None
    if prediction.time_signatures and reference.time_signatures:
        correct = len(prediction.time_signatures) == len(reference.time_signatures) and all(
            abs(p[0] - r[0]) <= 1e-7 and p[1:] == r[1:]
            for p, r in zip(prediction.time_signatures, reference.time_signatures, strict=False)
        )
    pickup = (
        None
        if prediction.pickup_beats is None or reference.pickup_beats is None
        else prediction.pickup_beats - reference.pickup_beats
    )
    downbeat = (
        None
        if prediction.first_downbeat_beats is None or reference.first_downbeat_beats is None
        else prediction.first_downbeat_beats - reference.first_downbeat_beats
    )
    return {
        "correct_meter": correct,
        "predicted_time_signatures": prediction.time_signatures,
        "reference_time_signatures": reference.time_signatures,
        "pickup_error_beats": pickup,
        "first_full_downbeat_error_beats": downbeat,
        "downbeats": _grid(
            prediction.downbeats, reference.downbeats, config.boundary_tolerance_beats
        ),
        "measure_boundaries": _grid(
            prediction.measure_boundaries,
            reference.measure_boundaries,
            config.boundary_tolerance_beats,
        ),
        "measure_boundary_error_beats": _boundary_error(
            prediction.measure_boundaries, reference.measure_boundaries
        ),
        "beat_alignment": _grid(prediction.beats, reference.beats, config.boundary_tolerance_beats),
        "alternative_hypotheses": prediction.meter_hypotheses,
        "domain": "unshifted quarter-note grid; not acoustic downbeat timing",
    }
