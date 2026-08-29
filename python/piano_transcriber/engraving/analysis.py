"""Non-mutating piano engraving and playability diagnostics."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from piano_transcriber.engraving.types import (
    CrossStaffCandidate,
    HandSpanDiagnostic,
    LedgerLineDiagnostic,
)
from piano_transcriber.score.types import PianoHand, ReconstructedScore


def derive_hand_spans(score: ReconstructedScore) -> tuple[HandSpanDiagnostic, ...]:
    by_onset: dict[tuple[PianoHand, Fraction], list[int]] = defaultdict(list)
    for note in score.notes:
        if note.hand is not None:
            by_onset[(note.hand, note.onset_beats)].append(note.pitch)
    diagnostics: list[HandSpanDiagnostic] = []
    ordered = sorted(by_onset.items(), key=lambda item: (item[0][1], item[0][0]))
    for (hand, onset), attacks in ordered:
        sounding = [
            note.pitch
            for note in score.notes
            if note.hand is hand and note.onset_beats <= onset < note.offset_beats
        ]
        attack_span = _span(attacks)
        overlap_span = _span(sounding)
        flags = tuple(
            label
            for threshold, label in ((12, ">12"), (16, ">16"), (20, ">20"))
            if max(attack_span, overlap_span) > threshold
        )
        if flags:
            if attack_span >= overlap_span:
                cause = "simultaneous-attack"
            elif any(
                note.pedal is True
                for note in score.notes
                if note.hand is hand and note.onset_beats <= onset < note.offset_beats
            ):
                cause = "pedal-related-overlap"
            else:
                cause = "sustained-overlap"
            diagnostics.append(
                HandSpanDiagnostic(
                    hand.value,
                    onset,
                    attack_span,
                    overlap_span,
                    flags,
                    cause,
                )
            )
    return tuple(diagnostics)


def derive_cross_staff_candidates(score: ReconstructedScore) -> tuple[CrossStaffCandidate, ...]:
    candidates: list[CrossStaffCandidate] = []
    for note in score.notes:
        if note.hand is PianoHand.RIGHT and note.pitch <= 55:
            candidates.append(
                CrossStaffCandidate(
                    note.source_index,
                    note.hand.value,
                    note.pitch,
                    note.onset_beats,
                    "right-hand-low-register",
                )
            )
        elif note.hand is PianoHand.LEFT and note.pitch >= 65:
            candidates.append(
                CrossStaffCandidate(
                    note.source_index,
                    note.hand.value,
                    note.pitch,
                    note.onset_beats,
                    "left-hand-high-register",
                )
            )
    return tuple(candidates)


def derive_ledger_lines(score: ReconstructedScore) -> tuple[LedgerLineDiagnostic, ...]:
    result: list[LedgerLineDiagnostic] = []
    for note in score.notes:
        lines = _estimated_ledger_lines(note.pitch, note.staff)
        if lines >= 3:
            result.append(LedgerLineDiagnostic(note.source_index, note.staff, note.pitch, lines))
    return tuple(result)


def _span(pitches: list[int]) -> int:
    return max(pitches) - min(pitches) if pitches else 0


def _estimated_ledger_lines(pitch: int, staff: int) -> int:
    lower, upper = (64, 77) if staff == 1 else (43, 57)
    if lower <= pitch <= upper:
        return 0
    distance = lower - pitch if pitch < lower else pitch - upper
    return max(1, (distance + 2) // 3)
