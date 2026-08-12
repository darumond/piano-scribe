"""Exact onset and written-duration quantization."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction


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
