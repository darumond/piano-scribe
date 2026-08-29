"""Engraving annotations kept separate from musical reconstruction decisions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class BeamAnnotation:
    source_index: int
    measure_index: int
    onset_in_measure: Fraction
    staff: int
    voice: int
    group_id: int
    level: int
    value: str


@dataclass(frozen=True, slots=True)
class TupletAnnotation:
    source_index: int
    measure_index: int
    onset_in_measure: Fraction
    staff: int
    voice: int
    group_id: int
    value: str
    actual_notes: int = 3
    normal_notes: int = 2


@dataclass(frozen=True, slots=True)
class RestDecision:
    onset_beats: Fraction
    duration_beats: Fraction
    staff: int
    voice: int
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class HandSpanDiagnostic:
    hand: str
    onset_beats: Fraction
    attack_span_semitones: int
    overlap_span_semitones: int
    threshold_flags: tuple[str, ...]
    cause: str


@dataclass(frozen=True, slots=True)
class CrossStaffCandidate:
    source_index: int
    hand: str
    pitch: int
    onset_beats: Fraction
    reason: str


@dataclass(frozen=True, slots=True)
class LedgerLineDiagnostic:
    source_index: int
    staff: int
    pitch: int
    estimated_ledger_lines: int
