"""Voice-aware rest simplification for already reconstructed piano music."""

from __future__ import annotations

from fractions import Fraction

from piano_transcriber.engraving.types import RestDecision
from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.quantize import WRITTEN_DURATIONS
from piano_transcriber.score.types import ReconstructedScore, ScoreRest

_REST_VALUES = tuple(sorted((*WRITTEN_DURATIONS, Fraction(1, 24), Fraction(1, 48)), reverse=True))


def optimize_rests(
    score: ReconstructedScore,
    *,
    minimum_interpretive_rest_beats: Fraction,
) -> tuple[tuple[ScoreRest, ...], tuple[RestDecision, ...], int, int, int]:
    """Remove redundant voice-padding rests and merge adjacent retained gaps."""
    if minimum_interpretive_rest_beats <= 0:
        raise ValueError("minimum interpretive rest must be positive")
    notes_by_stream: dict[tuple[int, int], list[tuple[Fraction, Fraction]]] = {}
    for note in score.notes:
        notes_by_stream.setdefault((note.staff, note.voice), []).append(
            (note.onset_beats, note.offset_beats)
        )
    for intervals in notes_by_stream.values():
        intervals.sort()

    decisions: list[RestDecision] = []
    retained: list[ScoreRest] = []
    for rest in sorted(score.rests, key=_rest_key):
        stream = (rest.staff, rest.voice)
        intervals = notes_by_stream.get(stream, [])
        previous = max(
            (offset for _onset, offset in intervals if offset <= rest.onset_beats),
            default=None,
        )
        following = min(
            (onset for onset, _offset in intervals if onset >= rest.offset_beats),
            default=None,
        )
        boundary_padding = previous is None or following is None
        if rest.voice > 1 and boundary_padding:
            action = "omit"
            reason = "secondary-voice-boundary-padding"
        elif rest.voice > 1 and rest.duration_beats < minimum_interpretive_rest_beats:
            action = "omit"
            reason = "secondary-voice-micro-gap"
        else:
            action = "keep"
            reason = "voice-rhythm"
            retained.append(rest)
        decisions.append(
            RestDecision(
                rest.onset_beats,
                rest.duration_beats,
                rest.staff,
                rest.voice,
                action,
                reason,
            )
        )

    merged: list[ScoreRest] = []
    merged_count = 0
    for rest in retained:
        if merged and _can_merge(merged[-1], rest, score):
            previous_rest = merged[-1]
            merged[-1] = ScoreRest(
                previous_rest.onset_beats,
                previous_rest.duration_beats + rest.duration_beats,
                previous_rest.staff,
                previous_rest.voice,
            )
            merged_count += 1
            decisions.append(
                RestDecision(
                    rest.onset_beats,
                    rest.duration_beats,
                    rest.staff,
                    rest.voice,
                    "merge",
                    "adjacent-compatible-rest",
                )
            )
        else:
            merged.append(rest)

    before = sum(
        _fragment_count(rest.duration_beats, score.minimum_explicit_rest_beats)
        for rest in score.rests
    )
    after = sum(
        _fragment_count(rest.duration_beats, score.minimum_explicit_rest_beats) for rest in merged
    )
    return tuple(merged), tuple(decisions), before, after, merged_count


def _can_merge(left: ScoreRest, right: ScoreRest, score: ReconstructedScore) -> bool:
    if (left.staff, left.voice, left.offset_beats) != (
        right.staff,
        right.voice,
        right.onset_beats,
    ):
        return False
    left_measure, _left_local, _left_length = locate_score_measure(
        left.onset_beats,
        score.time_signature,
        score.pickup_beats,
    )
    right_measure, _right_local, _right_length = locate_score_measure(
        right.onset_beats,
        score.time_signature,
        score.pickup_beats,
    )
    return left_measure == right_measure


def _fragment_count(duration: Fraction, minimum_duration: Fraction) -> int:
    remaining = duration
    count = 0
    while remaining > 0:
        if remaining < minimum_duration:
            break
        candidates = tuple(value for value in _REST_VALUES if value <= remaining)
        if not candidates:
            return count + 1
        remaining -= candidates[0]
        count += 1
    return count


def _rest_key(rest: ScoreRest) -> tuple[int, int, Fraction, Fraction]:
    return rest.staff, rest.voice, rest.onset_beats, rest.duration_beats
