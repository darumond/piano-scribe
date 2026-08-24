"""Beat and measure helpers."""

from __future__ import annotations

from fractions import Fraction

from piano_transcriber.score.types import MeasurePosition, TimeSignature


def locate_measure(position: Fraction, time_signature: TimeSignature) -> MeasurePosition:
    if position < 0:
        raise ValueError("position must be non-negative")
    measure_length = time_signature.measure_beats
    measure_index = int(position // measure_length)
    return MeasurePosition(measure_index + 1, position - measure_index * measure_length)


def measure_position(position: Fraction, time_signature: TimeSignature) -> tuple[int, Fraction]:
    located = locate_measure(position, time_signature)
    return located.measure_number, located.beat_in_measure


def required_measures(end_position: Fraction, time_signature: TimeSignature) -> int:
    if end_position <= 0:
        return 1
    measure_length = time_signature.measure_beats
    ratio = end_position / measure_length
    return max(1, (ratio.numerator + ratio.denominator - 1) // ratio.denominator)
