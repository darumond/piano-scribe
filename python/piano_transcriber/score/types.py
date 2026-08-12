"""Exact symbolic score-domain types, separate from acoustic transcription events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("time-signature numerator must be positive")
        if self.denominator <= 0 or self.denominator & (self.denominator - 1):
            raise ValueError("time-signature denominator must be a positive power of two")

    @property
    def measure_beats(self) -> Fraction:
        """Measure length in quarter-note units."""
        return Fraction(self.numerator * 4, self.denominator)

    @classmethod
    def parse(cls, value: str) -> TimeSignature:
        try:
            numerator_text, denominator_text = value.split("/", maxsplit=1)
            return cls(int(numerator_text), int(denominator_text))
        except (TypeError, ValueError) as error:
            raise ValueError("time signature must look like 4/4") from error

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True, slots=True)
class PedalInterval:
    raw_onset_seconds: float
    raw_offset_seconds: float
    onset_beats: Fraction
    offset_beats: Fraction

    def __post_init__(self) -> None:
        if self.raw_onset_seconds < 0.0 or self.raw_offset_seconds <= self.raw_onset_seconds:
            raise ValueError("pedal interval must have a positive duration")
        if self.onset_beats < 0 or self.offset_beats <= self.onset_beats:
            raise ValueError("pedal beat interval must have a positive duration")


@dataclass(frozen=True, slots=True)
class ScoreNote:
    source_index: int
    pitch: int
    velocity: int
    confidence: float
    raw_onset_seconds: float
    raw_offset_seconds: float
    onset_beats: Fraction
    duration_beats: Fraction
    quantization_error_seconds: float
    pedal: bool | None
    suspicious_reasons: tuple[str, ...] = ()
    pedal_duration_shortened: bool = False

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not 0 <= self.pitch <= 127:
            raise ValueError("pitch must be between 0 and 127")
        if not 0 <= self.velocity <= 127:
            raise ValueError("velocity must be between 0 and 127")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.raw_onset_seconds < 0.0 or self.raw_offset_seconds <= self.raw_onset_seconds:
            raise ValueError("raw note interval must have a positive duration")
        if self.onset_beats < 0 or self.duration_beats <= 0:
            raise ValueError("symbolic note interval must have a positive duration")

    @property
    def offset_beats(self) -> Fraction:
        return self.onset_beats + self.duration_beats


@dataclass(frozen=True, slots=True)
class ScoreChord:
    onset_beats: Fraction
    notes: tuple[ScoreNote, ...]

    def __post_init__(self) -> None:
        if not self.notes:
            raise ValueError("a chord must contain at least one note")
        if any(note.onset_beats != self.onset_beats for note in self.notes):
            raise ValueError("all chord notes must share the chord onset")


@dataclass(frozen=True, slots=True)
class EventDiagnostic:
    source_index: int
    pitch: int
    raw_onset_seconds: float
    raw_offset_seconds: float
    quantized_onset_beats: Fraction
    quantization_error_seconds: float
    written_duration_beats: Fraction | None
    action: str
    suspicious_reasons: tuple[str, ...] = ()
    merged_into_source_index: int | None = None
    pedal_duration_shortened: bool = False


@dataclass(frozen=True, slots=True)
class ReconstructedScore:
    bpm: float
    time_signature: TimeSignature
    grid_name: str
    grid_step_beats: Fraction
    notes: tuple[ScoreNote, ...]
    chords: tuple[ScoreChord, ...]
    diagnostics: tuple[EventDiagnostic, ...]
    pedal_intervals: tuple[PedalInterval, ...]
    measure_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.bpm) or self.bpm <= 0.0:
            raise ValueError("BPM must be finite and positive")
        if self.grid_step_beats <= 0:
            raise ValueError("grid step must be positive")
        if self.measure_count <= 0:
            raise ValueError("measure_count must be positive")
