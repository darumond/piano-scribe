"""One-to-one simultaneous pitch-set metrics; singleton attacks are included."""

from __future__ import annotations

from typing import Any

from piano_transcriber.evaluation.notes import detection, window_pairs
from piano_transcriber.evaluation.types import Alignment, EvaluationScore


def _groups(
    score: EvaluationScore, alignment: Alignment, tolerance: float
) -> list[tuple[float, frozenset[int]]]:
    events = sorted(
        (alignment.apply(onset), note.pitch)
        for note in score.notes
        if (onset := note.onset(alignment.domain)) is not None
    )
    groups: list[tuple[float, set[int]]] = []
    for onset, pitch in events:
        # Anchor to the first attack; no unbounded chaining of arpeggio notes.
        if not groups or onset - groups[-1][0] > tolerance + 1e-9:
            groups.append((onset, {pitch}))
        else:
            groups[-1][1].add(pitch)
    return [(t, frozenset(pitches)) for t, pitches in groups]


def chord_metrics(
    prediction: EvaluationScore,
    reference: EvaluationScore,
    alignment: Alignment,
    grouping_tolerance: float,
    matching_tolerance: float,
) -> dict[str, Any]:
    pg = _groups(prediction, alignment, grouping_tolerance)
    rg = _groups(reference, Alignment(alignment.domain, "none"), grouping_tolerance)
    exact = window_pairs(
        [g[0] for g in pg],
        [g[0] for g in rg],
        matching_tolerance,
        lambda p, r: pg[p][1] == rg[r][1],
    )
    used_p, used_r = {p for p, _r in exact}, {r for _p, r in exact}
    partial = window_pairs(
        [g[0] for g in pg],
        [g[0] for g in rg],
        matching_tolerance,
        lambda p, r: p not in used_p and r not in used_r and bool(pg[p][1] & rg[r][1]),
    )
    return {
        "predicted_groups": len(pg),
        "reference_groups": len(rg),
        "predicted_multi_note_groups": sum(len(g[1]) > 1 for g in pg),
        "reference_multi_note_groups": sum(len(g[1]) > 1 for g in rg),
        "exact_chord_set_matches": len(exact),
        "partial_chord_matches": len(partial),
        **detection(len(exact), len(pg), len(rg)),
        "grouping_tolerance": grouping_tolerance,
        "domain": alignment.domain,
    }
