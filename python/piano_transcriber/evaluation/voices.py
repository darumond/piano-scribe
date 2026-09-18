"""Staff labels and permutation-invariant voice-track comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from piano_transcriber.evaluation.notes import ratio
from piano_transcriber.evaluation.types import EvaluationNote, EvaluationScore, Match


def assignment_metrics(
    prediction: EvaluationScore, reference: EvaluationScore, matches: tuple[Match, ...], field: str
) -> dict[str, Any]:
    pairs = [
        (
            getattr(prediction.notes[m.prediction], field),
            getattr(reference.notes[m.reference], field),
        )
        for m in matches
    ]
    known = [(p, r) for p, r in pairs if p is not None and r is not None]
    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for (p, r), count in sorted(Counter(known).items()):
        confusion[str(r)][str(p)] = count
    result = {
        "accuracy": ratio(sum(p == r for p, r in known), len(known)),
        "evaluated_notes": len(known),
        "matched_notes": len(matches),
        "confusion_reference_rows_prediction_columns": dict(confusion),
    }
    if field == "staff":
        roles = [
            (prediction.notes[m.prediction].staff_role, reference.notes[m.reference].staff_role)
            for m in matches
        ]
        known_roles = [(p, r) for p, r in roles if p is not None and r is not None]
        matrix: dict[str, dict[str, int]] = defaultdict(dict)
        for (p, r), count in sorted(Counter(known_roles).items()):
            matrix[str(r)][str(p)] = count
        result["treble_bass_confusion"] = dict(matrix)
        result["clef_evaluated_notes"] = len(known_roles)
    return result


def _track(note: EvaluationNote) -> str | None:
    # MusicXML voices are scoped to a part, not to a staff: cross-staff voices stay one track.
    return f"{note.part}:{note.voice}" if note.voice is not None else None


def voice_metrics(
    prediction: EvaluationScore, reference: EvaluationScore, matches: tuple[Match, ...]
) -> dict[str, Any]:
    known: list[tuple[str, str, float]] = []
    for match in matches:
        p, r = prediction.notes[match.prediction], reference.notes[match.reference]
        pt, rt = _track(p), _track(r)
        if pt is not None and rt is not None:
            known.append(
                (
                    pt,
                    rt,
                    r.onset_beats if r.onset_beats is not None else float(r.onset_seconds or 0),
                )
            )
    result: dict[str, Any] = {
        "evaluated_notes": len(known),
        "assignment_accuracy": None,
        "matched_track_consistency": None,
        "fragmentation": None,
        "voice_id_switches": None,
        "track_purity": None,
        "track_mapping": {},
        "per_reference_track": {},
    }
    if not known:
        return result
    pids = sorted({p for p, _r, _t in known})
    rids = sorted({r for _p, r, _t in known})
    pi, ri = {v: i for i, v in enumerate(pids)}, {v: i for i, v in enumerate(rids)}
    counts = np.zeros((len(pids), len(rids)), dtype=int)
    by_reference: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for predicted_track, reference_track, time in known:
        counts[pi[predicted_track], ri[reference_track]] += 1
        by_reference[reference_track].append((time, predicted_track))
    rows, cols = linear_sum_assignment(counts, maximize=True)
    correct = int(counts[rows, cols].sum())
    details: dict[str, Any] = {}
    for track, events in sorted(by_reference.items()):
        groups: dict[float, set[str]] = defaultdict(set)
        for time, owner in events:
            groups[time].add(owner)
        ownership = [owners for _time, owners in sorted(groups.items())]
        details[track] = {
            "predicted_tracks": len({owner for _time, owner in events}),
            "switches": sum(a != b for a, b in pairwise(ownership)),
            "simultaneous_split_groups": sum(len(owners) > 1 for owners in ownership),
        }
    return {
        **result,
        "assignment_accuracy": correct / len(known),
        "matched_track_consistency": float(
            sum(counts[p, r] / counts[:, r].sum() for p, r in zip(rows, cols, strict=True))
            / len(rids)
        ),
        "fragmentation": sum(d["predicted_tracks"] - 1 for d in details.values()),
        "voice_id_switches": sum(d["switches"] for d in details.values()),
        "track_purity": float(counts.max(axis=1).sum() / len(known)),
        "track_mapping": {pids[p]: rids[r] for p, r in zip(rows, cols, strict=True)},
        "per_reference_track": details,
    }
