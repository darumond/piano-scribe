"""Exact onset and written-duration quantization."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING

from piano_transcriber.score.types import QuantizationCandidate

if TYPE_CHECKING:
    from piano_transcriber.score.tracking import BeatTrack


class QuantizationGrid(StrEnum):
    QUARTER = "quarter"
    EIGHTH = "eighth"
    EIGHTH_TRIPLET = "eighth-triplet"
    SIXTEENTH = "sixteenth"
    SIXTEENTH_TRIPLET = "sixteenth-triplet"
    THIRTY_SECOND = "thirty-second"

    @property
    def step_beats(self) -> Fraction:
        return {
            self.QUARTER: Fraction(1),
            self.EIGHTH: Fraction(1, 2),
            self.EIGHTH_TRIPLET: Fraction(1, 3),
            self.SIXTEENTH: Fraction(1, 4),
            self.SIXTEENTH_TRIPLET: Fraction(1, 6),
            self.THIRTY_SECOND: Fraction(1, 8),
        }[self]

    @property
    def complexity(self) -> float:
        return {
            self.QUARTER: 0.0,
            self.EIGHTH: 0.12,
            self.EIGHTH_TRIPLET: 0.7,
            self.SIXTEENTH: 0.32,
            self.SIXTEENTH_TRIPLET: 0.95,
            self.THIRTY_SECOND: 0.62,
        }[self]


WRITTEN_DURATIONS: tuple[Fraction, ...] = tuple(
    sorted(
        {
            Fraction(1, 8),
            Fraction(1, 6),
            Fraction(3, 16),
            Fraction(1, 4),
            Fraction(1, 3),
            Fraction(3, 8),
            Fraction(1, 2),
            Fraction(2, 3),
            Fraction(3, 4),
            Fraction(1),
            Fraction(4, 3),
            Fraction(3, 2),
            Fraction(2),
            Fraction(3),
            Fraction(4),
        }
    )
)


def snap_to_grid(position: Fraction, step: Fraction) -> Fraction:
    if position < 0:
        raise ValueError("position must be non-negative")
    if step <= 0:
        raise ValueError("grid step must be positive")
    quotient, remainder = divmod(position, step)
    if remainder * 2 < step:
        return quotient * step
    return (quotient + 1) * step


def snap_written_duration(
    duration: Fraction,
    *,
    maximum: Fraction | None = None,
) -> Fraction:
    if duration <= 0:
        duration = WRITTEN_DURATIONS[0]
    candidates = WRITTEN_DURATIONS
    if maximum is not None:
        bounded = tuple(candidate for candidate in candidates if candidate <= maximum)
        if bounded:
            candidates = bounded
    return min(candidates, key=lambda candidate: (abs(candidate - duration), candidate))


def choose_quantization(
    continuous_position: float,
    raw_seconds: float,
    beat_track: BeatTrack,
    *,
    complexity_cost: float = 0.35,
    tolerance_ms: float = 125.0,
    beat_offset: float = 0.0,
    grids: tuple[QuantizationGrid, ...] = tuple(QuantizationGrid),
) -> tuple[QuantizationCandidate, tuple[QuantizationCandidate, ...]]:
    """Score timing fit and notation complexity, preferring simpler near-equivalent grids."""
    if complexity_cost < 0 or tolerance_ms <= 0:
        raise ValueError("complexity cost must be non-negative and tolerance positive")
    position = Fraction(str(continuous_position))
    candidates: list[QuantizationCandidate] = []
    for grid in grids:
        snapped = snap_to_grid(position, grid.step_beats)
        snapped_seconds = beat_track.beats_to_seconds(float(snapped) - beat_offset)
        timing_error = snapped_seconds - raw_seconds
        penalty = complexity_cost * grid.complexity
        total = abs(timing_error) * 1000 / tolerance_ms + penalty
        candidates.append(QuantizationCandidate(grid.value, snapped, timing_error, penalty, total))
    ordered = tuple(
        sorted(candidates, key=lambda item: (item.total_score, item.complexity_penalty))
    )
    return ordered[0], ordered
