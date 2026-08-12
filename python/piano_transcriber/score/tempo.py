"""Tempo strategies for conversion from seconds to quarter-note units."""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from typing import Protocol

from piano_transcriber.transcription.types import TranscriptionResult


class TempoEstimator(Protocol):
    def estimate_bpm(self, result: TranscriptionResult) -> float: ...


class ExplicitTempo:
    def __init__(self, bpm: float) -> None:
        if not math.isfinite(bpm) or bpm <= 0.0:
            raise ValueError("BPM must be finite and positive")
        self.bpm = bpm

    def estimate_bpm(self, result: TranscriptionResult) -> float:
        del result
        return self.bpm


class MedianInterOnsetTempoEstimator:
    """Small baseline estimator; dense polyphony can make its result ambiguous."""

    def estimate_bpm(self, result: TranscriptionResult) -> float:
        unique_onsets = sorted({note.onset_seconds for note in result.notes})
        intervals = [
            later - earlier
            for earlier, later in pairwise(unique_onsets)
            if 0.08 <= later - earlier <= 2.0
        ]
        if not intervals:
            raise ValueError("cannot estimate tempo without usable onset intervals")
        median_interval = statistics.median(intervals)
        candidates = [60.0 / (median_interval * subdivision) for subdivision in (1, 2, 3, 4)]
        in_range = [candidate for candidate in candidates if 40.0 <= candidate <= 200.0]
        if not in_range:
            raise ValueError("automatic tempo estimate is outside the supported range")
        return min(in_range, key=lambda candidate: abs(candidate - 90.0))


def seconds_to_beats(seconds: float, bpm: float) -> Fraction:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("seconds must be finite and non-negative")
    if not math.isfinite(bpm) or bpm <= 0.0:
        raise ValueError("BPM must be finite and positive")
    return Fraction(Decimal(str(seconds)) * Decimal(str(bpm)) / Decimal(60))


def beats_to_seconds(beats: Fraction, bpm: float) -> float:
    if not math.isfinite(bpm) or bpm <= 0.0:
        raise ValueError("BPM must be finite and positive")
    return float(beats) * 60.0 / bpm
