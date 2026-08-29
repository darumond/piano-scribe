"""Derived engraving pipeline for an already reconstructed symbolic score."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction

from piano_transcriber.engraving.analysis import (
    derive_cross_staff_candidates,
    derive_hand_spans,
    derive_ledger_lines,
)
from piano_transcriber.engraving.beams import derive_beams
from piano_transcriber.engraving.rests import optimize_rests
from piano_transcriber.engraving.tuplets import derive_tuplets
from piano_transcriber.score.types import ReconstructedScore

logger = logging.getLogger(__name__)


class EngravingMode(StrEnum):
    BASIC = "basic"
    REFINED = "refined"


@dataclass(frozen=True, slots=True)
class EngravingConfig:
    mode: EngravingMode = EngravingMode.BASIC
    minimum_interpretive_rest_beats: Fraction = Fraction(1, 2)

    def __post_init__(self) -> None:
        if self.minimum_interpretive_rest_beats <= 0:
            raise ValueError("minimum interpretive rest must be positive")


def apply_engraving(score: ReconstructedScore, config: EngravingConfig) -> ReconstructedScore:
    """Annotate engraving decisions without revising notes, hands, or rhythm."""
    if config.mode is EngravingMode.BASIC:
        return score
    started = time.perf_counter()
    rest_started = time.perf_counter()
    rests, decisions, before, after, merged = optimize_rests(
        score,
        minimum_interpretive_rest_beats=config.minimum_interpretive_rest_beats,
    )
    rest_seconds = time.perf_counter() - rest_started
    annotation_started = time.perf_counter()
    beams = derive_beams(score)
    tuplets = derive_tuplets(score)
    spans = derive_hand_spans(score)
    cross_staff = derive_cross_staff_candidates(score)
    ledger = derive_ledger_lines(score)
    annotation_seconds = time.perf_counter() - annotation_started
    elapsed = time.perf_counter() - started
    logger.info(
        "Engraving refinement completed in %.3f s (rests %.3f s, annotations %.3f s)",
        elapsed,
        rest_seconds,
        annotation_seconds,
    )
    return replace(
        score,
        rests=rests,
        engraving_mode=config.mode.value,
        beam_annotations=beams,
        tuplet_annotations=tuplets,
        rest_decisions=decisions,
        hand_span_diagnostics=spans,
        cross_staff_candidates=cross_staff,
        ledger_line_diagnostics=ledger,
        rest_optimizer_seconds=rest_seconds,
        engraving_annotation_seconds=annotation_seconds,
        engraving_total_seconds=elapsed,
        rest_fragments_before=before,
        rest_fragments_after=after,
        merged_rest_count=merged,
    )
