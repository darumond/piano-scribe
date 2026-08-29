"""Exact symbolic score-domain types, separate from acoustic transcription events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING

from piano_transcriber.engraving.types import (
    BeamAnnotation,
    CrossStaffCandidate,
    HandSpanDiagnostic,
    LedgerLineDiagnostic,
    RestDecision,
    TupletAnnotation,
)

if TYPE_CHECKING:
    from piano_transcriber.score.tracking import BeatTrack


class PianoHand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


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
class MeasurePosition:
    measure_number: int
    beat_in_measure: Fraction

    def __post_init__(self) -> None:
        if self.measure_number <= 0 or self.beat_in_measure < 0:
            raise ValueError("measure position must be non-negative and one-indexed")


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
    hand: PianoHand | None = None
    staff: int = 1
    voice: int = 1
    chord_id: int | None = None
    tie_across_measure: bool = False
    hand_assignment_cost: float = 0.0
    hand_assignment_confidence: float = 0.0
    voice_assignment_cost: float = 0.0
    previous_continuity_cost: float = 0.0
    next_continuity_cost: float = 0.0
    voice_duration_adjusted: bool = False
    original_duration_beats: Fraction | None = None
    voice_identity_switched: bool = False
    repeated_pitch_voice_switched: bool = False
    voice_assignment_reason: str | None = None
    extra_voice_reason: str | None = None
    track_previous_pitch: float | None = None
    track_direction: int = 0
    voice_continuity_score: float = 0.0
    duration_change_reason: str | None = None

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
        if self.staff not in {1, 2}:
            raise ValueError("staff must be 1 (treble) or 2 (bass)")
        if self.voice <= 0:
            raise ValueError("voice must be positive")
        if self.chord_id is not None and self.chord_id < 0:
            raise ValueError("chord_id must be non-negative")
        costs = (
            self.hand_assignment_cost,
            self.voice_assignment_cost,
            self.previous_continuity_cost,
            self.next_continuity_cost,
            self.voice_continuity_score,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in costs):
            raise ValueError("assignment costs must be finite and non-negative")
        if not 0.0 <= self.hand_assignment_confidence <= 1.0:
            raise ValueError("hand assignment confidence must be between zero and one")
        if self.original_duration_beats is not None and self.original_duration_beats <= 0:
            raise ValueError("original duration must be positive")
        if self.track_previous_pitch is not None and not 0 <= self.track_previous_pitch <= 127:
            raise ValueError("track previous pitch must be between 0 and 127")
        if self.track_direction not in {-1, 0, 1}:
            raise ValueError("track direction must be -1, 0, or 1")

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
class ScoreRest:
    onset_beats: Fraction
    duration_beats: Fraction
    staff: int
    voice: int

    def __post_init__(self) -> None:
        if self.onset_beats < 0 or self.duration_beats <= 0:
            raise ValueError("rest interval must have a positive duration")
        if self.staff not in {1, 2}:
            raise ValueError("rest staff must be 1 or 2")
        if self.voice <= 0:
            raise ValueError("rest voice must be positive")

    @property
    def offset_beats(self) -> Fraction:
        return self.onset_beats + self.duration_beats


@dataclass(frozen=True, slots=True)
class QuantizationCandidate:
    subdivision: str
    position_beats: Fraction
    timing_error_seconds: float
    complexity_penalty: float
    total_score: float


@dataclass(frozen=True, slots=True)
class DurationCandidate:
    duration_beats: Fraction
    timing_error_seconds: float
    complexity_penalty: float
    requires_tie: bool
    tiny_tie_fragment: bool
    dotted_micro_value: bool
    unusual_short_value: bool
    total_score: float


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
    continuous_onset_beats: float | None = None
    selected_subdivision: str | None = None
    quantization_candidates: tuple[QuantizationCandidate, ...] = ()
    duration_candidates: tuple[DurationCandidate, ...] = ()
    rhythm_group_index: int | None = None
    selected_rhythm_family: str | None = None
    rhythm_metric_position_beats: Fraction | None = None
    rhythm_requires_tie: bool = False
    rhythm_group_timing_error_seconds: float | None = None
    rhythm_complexity_cost: float | None = None
    rhythm_local_cost: float | None = None
    rhythm_transition_cost: float | None = None
    rhythm_cumulative_score: float | None = None
    local_best_subdivision: str | None = None
    local_best_position_beats: Fraction | None = None
    optimizer_selection_reason: str | None = None
    optimizer_changed_local_choice: bool = False
    assigned_hand: str | None = None
    assigned_staff: int | None = None
    assigned_voice: int | None = None
    chord_id: int | None = None
    hand_assignment_cost: float | None = None
    hand_assignment_confidence: float | None = None
    voice_assignment_cost: float | None = None
    previous_continuity_cost: float | None = None
    next_continuity_cost: float | None = None
    voice_duration_adjusted: bool = False
    original_duration_beats: Fraction | None = None
    voice_identity_switched: bool = False
    repeated_pitch_voice_switched: bool = False
    voice_assignment_reason: str | None = None
    extra_voice_reason: str | None = None
    track_previous_pitch: float | None = None
    track_direction: int = 0
    voice_continuity_score: float | None = None
    duration_change_reason: str | None = None


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
    beat_track: BeatTrack | None = None
    pickup_beats: Fraction = Fraction(0)
    first_full_downbeat_beats: Fraction = Fraction(0)
    beat_position_offset: float = 0.0
    rhythm_optimizer: str = "local"
    rhythm_optimizer_seconds: float = 0.0
    rhythm_evaluated_transitions: int = 0
    rests: tuple[ScoreRest, ...] = ()
    piano_layout: str = "none"
    hand_optimizer_seconds: float = 0.0
    hand_evaluated_transitions: int = 0
    voice_optimizer_seconds: float = 0.0
    voice_evaluated_transitions: int = 0
    voice_duration_changes: int = 0
    minimum_explicit_rest_beats: Fraction = Fraction(1, 4)
    engraving_mode: str = "basic"
    beam_annotations: tuple[BeamAnnotation, ...] = ()
    tuplet_annotations: tuple[TupletAnnotation, ...] = ()
    rest_decisions: tuple[RestDecision, ...] = ()
    hand_span_diagnostics: tuple[HandSpanDiagnostic, ...] = ()
    cross_staff_candidates: tuple[CrossStaffCandidate, ...] = ()
    ledger_line_diagnostics: tuple[LedgerLineDiagnostic, ...] = ()
    voice_stability_seconds: float = 0.0
    rest_optimizer_seconds: float = 0.0
    engraving_annotation_seconds: float = 0.0
    engraving_total_seconds: float = 0.0
    rest_fragments_before: int = 0
    rest_fragments_after: int = 0
    merged_rest_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.bpm) or self.bpm <= 0.0:
            raise ValueError("BPM must be finite and positive")
        if self.grid_step_beats <= 0:
            raise ValueError("grid step must be positive")
        if self.measure_count <= 0:
            raise ValueError("measure_count must be positive")
        if self.pickup_beats < 0 or self.pickup_beats >= self.time_signature.measure_beats:
            raise ValueError("pickup must be shorter than one complete measure")
        if self.first_full_downbeat_beats < 0:
            raise ValueError("first full downbeat must be non-negative")
        if self.minimum_explicit_rest_beats <= 0:
            raise ValueError("minimum explicit rest duration must be positive")
