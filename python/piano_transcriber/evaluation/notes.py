"""Window-indexed maximum-cardinality, minimum-cost one-to-one matching."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from piano_transcriber.evaluation.types import Alignment, EvaluationNote, Match


def assign_edges(
    prediction_count: int, reference_count: int, edges: list[tuple[int, int, float]]
) -> list[tuple[int, int]]:
    """Sparse assignment; private dummy columns make missed predictions legal.

    Real costs must be in [0, 1]. A dummy costs n+1, so cardinality takes priority
    over the total secondary cost. Positive weights avoid sparse zero-edge ambiguity.
    """
    if not prediction_count or not reference_count or not edges:
        return []
    rows = [p for p, _r, _cost in edges] + list(range(prediction_count))
    cols = [r for _p, r, _cost in edges] + [reference_count + p for p in range(prediction_count)]
    costs = [1.0 + cost for _p, _r, cost in edges] + [
        float(2 * prediction_count + 2)
    ] * prediction_count
    graph = csr_matrix(
        (costs, (rows, cols)), shape=(prediction_count, reference_count + prediction_count)
    )
    row, col = min_weight_full_bipartite_matching(graph)
    return [(int(p), int(r)) for p, r in zip(row, col, strict=True) if r < reference_count]


def match_notes(
    prediction: Sequence[EvaluationNote],
    reference: Sequence[EvaluationNote],
    alignment: Alignment,
    tolerance: float,
    *,
    pitch_equal: bool = True,
    duration_tolerance: float | None = None,
) -> tuple[Match, ...]:
    if tolerance < 0:
        raise ValueError("onset tolerance must be non-negative")
    groups: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for index, note in enumerate(reference):
        onset = note.onset(alignment.domain)
        if onset is not None:
            groups[note.pitch if pitch_equal else 0].append((onset, index))
    for group in groups.values():
        group.sort()
    edges: list[tuple[int, int, float]] = []
    errors: dict[tuple[int, int], float] = {}
    for p, note in enumerate(prediction):
        onset = note.onset(alignment.domain)
        if onset is None:
            continue
        onset = alignment.apply(onset)
        group = groups.get(note.pitch if pitch_equal else 0, [])
        first = bisect_left(group, (onset - tolerance - 1e-9, -1))
        last = bisect_right(group, (onset + tolerance + 1e-9, len(reference)))
        for ref_onset, r in group[first:last]:
            if duration_tolerance is not None:
                pd = note.duration(alignment.domain)
                rd = reference[r].duration(alignment.domain)
                if (
                    pd is None
                    or rd is None
                    or abs(pd * alignment.scale - rd) > duration_tolerance + 1e-9
                ):
                    continue
            error = onset - ref_onset
            onset_cost = min(1.0, abs(error) / max(tolerance, 1e-9))
            # Pitch is only a small deterministic tie-breaker in the separate timing pairing.
            cost = (
                onset_cost
                if pitch_equal
                else (0.99 * onset_cost + 0.01 * abs(note.pitch - reference[r].pitch) / 127)
            )
            edges.append((p, r, cost))
            errors[p, r] = error
    return tuple(
        Match(p, r, errors[p, r]) for p, r in assign_edges(len(prediction), len(reference), edges)
    )


def ratio(numerator: float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def detection(tp: int, predicted: int, reference: int) -> dict[str, Any]:
    return {
        "true_positives": tp,
        "false_positives": predicted - tp,
        "false_negatives": reference - tp,
        "precision": ratio(tp, predicted),
        "recall": ratio(tp, reference),
        "f1": ratio(2 * tp, predicted + reference),
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_signed": None,
            "mae": None,
            "median_absolute": None,
            "p95_absolute": None,
            "min_signed": None,
            "max_signed": None,
        }
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {
        "count": len(values),
        "mean_signed": float(np.mean(array)),
        "mae": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "p95_absolute": float(np.percentile(absolute, 95)),
        "min_signed": float(np.min(array)),
        "max_signed": float(np.max(array)),
    }


def matched_errors(
    prediction: Sequence[EvaluationNote],
    reference: Sequence[EvaluationNote],
    matches: Sequence[Match],
    alignment: Alignment,
) -> dict[str, Any]:
    durations: list[float] = []
    offsets: list[float] = []
    for match in matches:
        pd = prediction[match.prediction].duration(alignment.domain)
        rd = reference[match.reference].duration(alignment.domain)
        if pd is not None and rd is not None:
            error = pd * alignment.scale - rd
            durations.append(error)
            offsets.append(match.onset_error + error)
    return {
        "onset_error": distribution([m.onset_error for m in matches]),
        "offset_error": distribution(offsets),
        "duration_error": distribution(durations),
    }


def pitch_metrics(
    prediction: Sequence[EvaluationNote],
    reference: Sequence[EvaluationNote],
    matches: Sequence[Match],
    alignment: Alignment,
    tolerance: float,
) -> dict[str, Any]:
    timing_pairs = match_notes(prediction, reference, alignment, tolerance, pitch_equal=False)
    errors = [prediction[m.prediction].pitch - reference[m.reference].pitch for m in timing_pairs]
    matched_p = {m.prediction for m in matches}
    matched_r = {m.reference for m in matches}

    def counts(items: Sequence[EvaluationNote], selected: set[int]) -> dict[str, int]:
        return dict(
            sorted(Counter(str(n.pitch) for i, n in enumerate(items) if i not in selected).items())
        )

    histogram = Counter(errors)
    return {
        "timing_pair_count": len(errors),
        "pairing": "pitch-independent onset assignment",
        "exact_pitch_accuracy": ratio(sum(e == 0 for e in errors), len(errors)),
        "pitch_class_accuracy": ratio(sum(e % 12 == 0 for e in errors), len(errors)),
        "octave_errors": sum(e != 0 and e % 12 == 0 for e in errors),
        "semitone_errors": {str(k): v for k, v in sorted(histogram.items())},
        "recurring_semitone_errors": {
            str(k): v for k, v in sorted(histogram.items()) if k != 0 and v >= 2
        },
        "false_positive_pitches": counts(prediction, matched_p),
        "false_negative_pitches": counts(reference, matched_r),
    }


def window_pairs(
    prediction: Sequence[float],
    reference: Sequence[float],
    tolerance: float,
    eligible: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int]]:
    ordered = sorted((time, i) for i, time in enumerate(reference))
    edges: list[tuple[int, int, float]] = []
    for p, time in enumerate(prediction):
        start = bisect_left(ordered, (time - tolerance - 1e-9, -1))
        end = bisect_right(ordered, (time + tolerance + 1e-9, len(reference)))
        for other, r in ordered[start:end]:
            if eligible is None or eligible(p, r):
                edges.append((p, r, min(1.0, abs(time - other) / max(tolerance, 1e-9))))
    return assign_edges(len(prediction), len(reference), edges)
