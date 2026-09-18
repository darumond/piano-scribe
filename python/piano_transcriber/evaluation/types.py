"""Common optional-metadata representation for reference and predicted scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Domain = Literal["seconds", "beats"]


@dataclass(frozen=True)
class EvaluationNote:
    pitch: int
    onset_seconds: float | None = None
    duration_seconds: float | None = None
    onset_beats: float | None = None
    duration_beats: float | None = None
    measure: str | None = None
    beat: float | None = None
    staff: str | None = None
    voice: str | None = None
    hand: str | None = None
    part: str = "1"
    subdivision: str | None = None
    staff_role: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pitch, int) or not 0 <= self.pitch <= 127:
            raise ValueError("evaluation pitch must be a MIDI integer in 0..127")
        for name in ("onset_seconds", "onset_beats", "duration_seconds", "duration_beats", "beat"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.duration_seconds == 0 or self.duration_beats == 0:
            raise ValueError("evaluation note durations must be positive")
        if self.onset_seconds is None and self.onset_beats is None:
            raise ValueError("a note needs an onset in seconds or beats")

    def onset(self, domain: Domain) -> float | None:
        return self.onset_seconds if domain == "seconds" else self.onset_beats

    def duration(self, domain: Domain) -> float | None:
        return self.duration_seconds if domain == "seconds" else self.duration_beats


@dataclass(frozen=True)
class EvaluationScore:
    notes: tuple[EvaluationNote, ...]
    source_format: str = "normalized"
    stage: str = "symbolic"
    # Positions and durations are always quarter-note units, including compound meters.
    time_signatures: tuple[tuple[float, int, int], ...] = ()
    tempos: tuple[tuple[float, float], ...] = ()
    pickup_beats: float | None = None
    first_downbeat_beats: float | None = None
    measure_boundaries: tuple[float, ...] | None = None
    beats: tuple[float, ...] | None = None
    downbeats: tuple[float, ...] | None = None
    structure: dict[str, int | None] = field(default_factory=dict)
    meter_hypotheses: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationConfig:
    onset_tolerances_ms: tuple[float, ...] = (25.0, 50.0, 100.0)
    onset_tolerance_beats: float = 0.125
    duration_tolerance_seconds: float | None = None
    duration_tolerance_beats: float | None = None
    chord_group_seconds: float = 0.03
    chord_group_beats: float = 0.0625
    boundary_tolerance_beats: float = 0.125
    alignment: str = "auto"
    max_tempo_scale_deviation: float = 0.05
    stage: str = "auto"
    prediction_voice_scope: str = "part"
    reference_voice_scope: str = "part"

    def __post_init__(self) -> None:
        if self.alignment not in {"auto", "none", "offset", "affine", "symbolic"}:
            raise ValueError("unknown alignment strategy")
        if self.stage not in {"auto", "transcription", "symbolic"}:
            raise ValueError("unknown evaluation stage")
        if self.prediction_voice_scope not in {
            "part",
            "staff",
        } or self.reference_voice_scope not in {"part", "staff"}:
            raise ValueError("voice scope must be part or staff")
        values = (
            *self.onset_tolerances_ms,
            self.onset_tolerance_beats,
            self.chord_group_seconds,
            self.chord_group_beats,
            self.boundary_tolerance_beats,
        )
        if not self.onset_tolerances_ms or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("tolerances must be finite and non-negative")
        for value in (self.duration_tolerance_seconds, self.duration_tolerance_beats):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("duration tolerance must be finite and non-negative")
        if not 0 <= self.max_tempo_scale_deviation < 1:
            raise ValueError("tempo scale deviation must lie in [0, 1)")


@dataclass(frozen=True)
class Alignment:
    domain: Domain
    strategy: str
    scale: float = 1.0
    offset: float = 0.0
    anchor_count: int = 0
    warning: str | None = None

    def apply(self, onset: float) -> float:
        return self.scale * onset + self.offset


@dataclass(frozen=True)
class Match:
    prediction: int
    reference: int
    onset_error: float
