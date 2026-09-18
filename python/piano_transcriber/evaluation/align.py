"""Deterministic global alignment with an inspectable offset and tempo scale."""

from __future__ import annotations

from collections import Counter

import numpy as np

from piano_transcriber.evaluation.notes import match_notes
from piano_transcriber.evaluation.types import Alignment, Domain, EvaluationConfig, EvaluationScore


def available(score: EvaluationScore, domain: Domain) -> bool:
    return all(note.onset(domain) is not None for note in score.notes)


def choose_domain(
    prediction: EvaluationScore, reference: EvaluationScore, config: EvaluationConfig
) -> Domain | None:
    symbolic = config.stage == "symbolic" or (
        config.stage == "auto" and prediction.stage != "transcription"
    )
    if config.alignment == "symbolic":
        symbolic = True
    if symbolic and available(prediction, "beats") and available(reference, "beats"):
        return "beats"
    if config.alignment == "symbolic":
        raise ValueError("symbolic alignment requires beat positions on both inputs")
    if available(prediction, "seconds") and available(reference, "seconds"):
        return "seconds"
    return None


def align_seconds(
    prediction: EvaluationScore, reference: EvaluationScore, config: EvaluationConfig
) -> Alignment:
    if config.alignment in {"none", "symbolic"} or not prediction.notes or not reference.notes:
        return Alignment("seconds", "none")
    # Bound anchor work independently of piece length; final evaluation uses every note.
    p_indices = np.linspace(0, len(prediction.notes) - 1, min(96, len(prediction.notes)), dtype=int)
    r_indices = np.linspace(0, len(reference.notes) - 1, min(96, len(reference.notes)), dtype=int)
    p_anchors = [prediction.notes[i] for i in p_indices]
    r_anchors = [reference.notes[i] for i in r_indices]
    samples = [
        prediction.notes[i]
        for i in np.linspace(
            0, len(prediction.notes) - 1, min(256, len(prediction.notes)), dtype=int
        )
    ]
    scales = [1.0]
    if config.alignment == "affine":
        scales = sorted(
            set(
                [
                    1.0,
                    *np.linspace(
                        1 - config.max_tempo_scale_deviation,
                        1 + config.max_tempo_scale_deviation,
                        21,
                    ).tolist(),
                ]
            )
        )
    tolerance = max(config.onset_tolerances_ms) / 1000
    best = Alignment("seconds", "offset")
    best_key = (-1, float("-inf"), float("-inf"), float("-inf"))
    for scale in scales:
        votes: Counter[float] = Counter()
        for p in p_anchors:
            for r in r_anchors:
                if (
                    p.pitch == r.pitch
                    and p.onset_seconds is not None
                    and r.onset_seconds is not None
                ):
                    votes[round((r.onset_seconds - scale * p.onset_seconds) / 0.01) * 0.01] += 1
        candidates = [0.0, *(offset for offset, _count in votes.most_common(8))]
        for offset in candidates:
            trial = Alignment(
                "seconds", "affine" if config.alignment == "affine" else "offset", scale, offset
            )
            pairs = match_notes(samples, reference.notes, trial, tolerance)
            error = sum(abs(m.onset_error) for m in pairs)
            key = (len(pairs), -error, -abs(scale - 1), -abs(offset))
            if key > best_key:
                best_key = key
                best = trial
    pairs = match_notes(prediction.notes, reference.notes, best, tolerance)
    if pairs:
        x = np.array([prediction.notes[m.prediction].onset_seconds for m in pairs], dtype=float)
        y = np.array([reference.notes[m.reference].onset_seconds for m in pairs], dtype=float)
        scale = best.scale
        if config.alignment == "affine" and len(pairs) >= 3 and np.ptp(x) > 1:
            # Trim large residuals before a single least-squares refinement.
            residual = np.abs(y - (scale * x + best.offset))
            keep = residual <= np.percentile(residual, 80) + 1e-9
            if np.ptp(x[keep]) > 1:
                fitted = float(np.polyfit(x[keep], y[keep], 1)[0])
                if abs(fitted - 1) <= config.max_tempo_scale_deviation:
                    scale = fitted
        refined = Alignment(
            "seconds", best.strategy, scale, float(np.median(y - scale * x)), len(pairs)
        )
        check = match_notes(prediction.notes, reference.notes, refined, tolerance)
        if (len(check), -sum(abs(m.onset_error) for m in check)) >= (
            len(pairs),
            -sum(abs(m.onset_error) for m in pairs),
        ):
            best = refined
    return Alignment(
        best.domain,
        best.strategy,
        best.scale,
        best.offset,
        len(pairs),
        "Global fit only; repeated patterns, missing introductions and rubato can make "
        "alignment ambiguous. Inspect the transform.",
    )
