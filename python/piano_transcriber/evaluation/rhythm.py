"""Written rhythm metrics in quarter-note units, independent of pitch spelling."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from piano_transcriber.evaluation.notes import distribution, ratio
from piano_transcriber.evaluation.types import EvaluationScore, Match


def rhythm_family(duration: float | None) -> str | None:
    if duration is None:
        return None
    value = Fraction(duration).limit_denominator(192)
    if abs(float(value) - duration) > 1e-7:
        return None
    for power in range(-6, 5):
        base = Fraction(2) ** power
        if value == base:
            return "straight"
        if value == base * Fraction(3, 2):
            return "dotted"
        if value == base * Fraction(2, 3):
            return "triplet"
    return None


def rhythm_metrics(
    prediction: EvaluationScore, reference: EvaluationScore, matches: tuple[Match, ...]
) -> dict[str, Any]:
    onsets: list[float] = []
    durations: list[float] = []
    families: list[bool] = []
    for match in matches:
        p, r = prediction.notes[match.prediction], reference.notes[match.reference]
        if p.onset_beats is not None and r.onset_beats is not None:
            onsets.append(p.onset_beats - r.onset_beats)
        if p.duration_beats is not None and r.duration_beats is not None:
            durations.append(p.duration_beats - r.duration_beats)
            pf, rf = rhythm_family(p.duration_beats), rhythm_family(r.duration_beats)
            if pf is not None and rf is not None:
                families.append(pf == rf)
    return {
        "onset_beat_error": distribution(onsets),
        "written_duration_beat_error": distribution(durations),
        "rhythmic_value_accuracy": ratio(sum(abs(d) <= 1e-7 for d in durations), len(durations)),
        "subdivision_family_accuracy": ratio(sum(families), len(families)),
        "family_evaluated_notes": len(families),
        "duration_evaluated_notes": len(durations),
    }
