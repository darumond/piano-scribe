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


def locate_score_measure(
    position: Fraction,
    time_signature: TimeSignature,
    pickup_beats: Fraction = Fraction(0),
) -> tuple[int, Fraction, Fraction]:
    """Return zero-based measure index, local position, and measure duration."""
    if position < 0 or pickup_beats < 0 or pickup_beats >= time_signature.measure_beats:
        raise ValueError("invalid score or pickup position")
    measure_length = time_signature.measure_beats
    if pickup_beats > 0 and position < pickup_beats:
        return 0, position, pickup_beats
    shifted = position - pickup_beats if pickup_beats > 0 else position
    full_index = int(shifted // measure_length)
    index = full_index + (1 if pickup_beats > 0 else 0)
    return index, shifted - full_index * measure_length, measure_length


def required_score_measures(
    end_position: Fraction,
    time_signature: TimeSignature,
    pickup_beats: Fraction = Fraction(0),
) -> int:
    if end_position <= 0:
        return 1
    index, local, _length = locate_score_measure(
        end_position,
        time_signature,
        pickup_beats,
    )
    return max(1, index + (1 if local > 0 else 0))
