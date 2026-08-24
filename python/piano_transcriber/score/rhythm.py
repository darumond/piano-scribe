"""Bounded phrase-level optimization of onset and duration interpretations."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction

from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.quantize import (
    WRITTEN_DURATIONS,
    QuantizationGrid,
    choose_quantization,
)
from piano_transcriber.score.tracking import BeatTrack
from piano_transcriber.score.types import DurationCandidate, QuantizationCandidate, TimeSignature
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


class RhythmOptimizerMode(StrEnum):
    LOCAL = "local"
    SEQUENCE = "sequence"


class RhythmFamily(StrEnum):
    QUARTER = "quarter-level"
    EIGHTH_STRAIGHT = "eighth-straight"
    EIGHTH_TRIPLET = "eighth-triplet"
    SIXTEENTH_STRAIGHT = "sixteenth-straight"
    SIXTEENTH_TRIPLET = "sixteenth-triplet"
    THIRTY_SECOND = "thirty-second"

    @property
    def ternary(self) -> bool:
        return self in {self.EIGHTH_TRIPLET, self.SIXTEENTH_TRIPLET}


@dataclass(frozen=True, slots=True)
class RhythmSequenceWeights:
    onset_timing: float = 1.0
    duration_timing: float = 0.6
    notation_complexity: float = 0.65
    family_switch: float = 0.35
    straight_triplet_switch: float = 0.9
    isolated_triplet: float = 0.65
    dotted_micro_value: float = 0.65
    thirty_second_value: float = 0.65
    unusual_short_value: float = 0.4
    tie: float = 0.25
    tiny_tie_fragment: float = 0.8
    metric_accent: float = 0.12
    pickup_plausibility: float = 0.1
    pattern_consistency: float = 0.45
    duration_pattern: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.onset_timing,
            self.duration_timing,
            self.notation_complexity,
            self.family_switch,
            self.straight_triplet_switch,
            self.isolated_triplet,
            self.dotted_micro_value,
            self.thirty_second_value,
            self.unusual_short_value,
            self.tie,
            self.tiny_tie_fragment,
            self.metric_accent,
            self.pickup_plausibility,
            self.pattern_consistency,
            self.duration_pattern,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("rhythm sequence weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RhythmSequenceConfig:
    candidate_limit: int = 5
    candidate_window_ms: float = 55.0
    duration_candidate_limit: int = 4
    duration_candidate_window_ms: float = 200.0
    beam_size: int = 64
    chord_window_ms: float = 45.0
    pattern_similarity_beats: float = 0.15
    weights: RhythmSequenceWeights = field(default_factory=RhythmSequenceWeights)

    def __post_init__(self) -> None:
        if self.candidate_limit <= 0 or self.duration_candidate_limit <= 0 or self.beam_size <= 0:
            raise ValueError("candidate limits and beam size must be positive")
        if self.candidate_window_ms < 0 or self.duration_candidate_window_ms < 0:
            raise ValueError("candidate timing windows must be non-negative")
        if self.chord_window_ms <= 0 or self.pattern_similarity_beats < 0:
            raise ValueError("chord and pattern windows must be valid")


@dataclass(frozen=True, slots=True)
class NoteDurationChoice:
    source_index: int
    selected: DurationCandidate
    candidates: tuple[DurationCandidate, ...]
    pedal_shortened: bool


@dataclass(frozen=True, slots=True)
class RhythmCandidate:
    quantization: QuantizationCandidate
    family: RhythmFamily
    metric_position: Fraction
    duration_choices: tuple[NoteDurationChoice, ...]
    requires_tie: bool
    local_cost: float

    @property
    def representative_duration(self) -> Fraction:
        durations = sorted(choice.selected.duration_beats for choice in self.duration_choices)
        return durations[len(durations) // 2] if durations else Fraction(0)

    @property
    def complexity_cost(self) -> float:
        duration_costs = [choice.selected.complexity_penalty for choice in self.duration_choices]
        duration_cost = statistics.mean(duration_costs) if duration_costs else 0.0
        return self.quantization.complexity_penalty + duration_cost


@dataclass(frozen=True, slots=True)
class RhythmGroup:
    index: int
    source_indices: tuple[int, ...]
    raw_onset_seconds: float
    continuous_onset_beats: float
    accent_strength: float
    candidates: tuple[RhythmCandidate, ...]
    local_best: QuantizationCandidate


@dataclass(frozen=True, slots=True)
class RhythmSelection:
    group_index: int
    source_indices: tuple[int, ...]
    candidate: RhythmCandidate
    candidates: tuple[RhythmCandidate, ...]
    local_best: QuantizationCandidate
    transition_cost: float
    cumulative_score: float
    reason: str

    @property
    def differs_from_local(self) -> bool:
        return (
            self.candidate.quantization.position_beats != self.local_best.position_beats
            or self.candidate.quantization.subdivision != self.local_best.subdivision
        )


@dataclass(frozen=True, slots=True)
class RhythmOptimizationResult:
    selections: tuple[RhythmSelection, ...]
    elapsed_seconds: float
    evaluated_transitions: int

    def by_source_index(self) -> dict[int, RhythmSelection]:
        return {
            source_index: selection
            for selection in self.selections
            for source_index in selection.source_indices
        }


@dataclass(frozen=True, slots=True)
class _IndexedNote:
    source_index: int
    note: NoteEvent
    continuous_onset: float
    continuous_offset: float


@dataclass(frozen=True, slots=True)
class _BeamState:
    score: float
    selections: tuple[RhythmCandidate, ...]
    transition_costs: tuple[float, ...]
    cumulative_scores: tuple[float, ...]
    last_symbolic_gap: Fraction | None
    last_raw_gap: float | None


def optimize_rhythm_sequence(
    transcription: TranscriptionResult,
    beat_track: BeatTrack,
    *,
    beat_offset: float,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
    quantization_complexity_cost: float,
    timing_tolerance_ms: float,
    config: RhythmSequenceConfig | None = None,
) -> RhythmOptimizationResult:
    """Choose one coherent candidate path using a deterministic bounded beam."""
    started = time.perf_counter()
    sequence = config or RhythmSequenceConfig()
    groups = _build_groups(
        transcription,
        beat_track,
        beat_offset,
        time_signature,
        pickup_beats,
        quantization_complexity_cost,
        timing_tolerance_ms,
        sequence,
    )
    if not groups:
        return RhythmOptimizationResult((), time.perf_counter() - started, 0)

    states = [
        _BeamState(
            candidate.local_cost,
            (candidate,),
            (0.0,),
            (candidate.local_cost,),
            None,
            None,
        )
        for candidate in groups[0].candidates
    ]
    evaluated = 0
    for group_index in range(1, len(groups)):
        group = groups[group_index]
        previous_group = groups[group_index - 1]
        expanded: list[_BeamState] = []
        for state in states:
            previous = state.selections[-1]
            for candidate in group.candidates:
                evaluated += 1
                if candidate.quantization.position_beats <= previous.quantization.position_beats:
                    continue
                transition = _transition_cost(
                    state,
                    previous,
                    candidate,
                    previous_group,
                    group,
                    sequence,
                )
                score = state.score + candidate.local_cost + transition
                symbolic_gap = (
                    candidate.quantization.position_beats - previous.quantization.position_beats
                )
                raw_gap = group.continuous_onset_beats - previous_group.continuous_onset_beats
                expanded.append(
                    _BeamState(
                        score,
                        (*state.selections, candidate),
                        (*state.transition_costs, transition),
                        (*state.cumulative_scores, score),
                        symbolic_gap,
                        raw_gap,
                    )
                )
        if not expanded:
            raise ValueError(f"rhythm optimizer found no increasing path at group {group_index}")
        states = sorted(expanded, key=_state_sort_key)[: sequence.beam_size]

    best = min(states, key=_state_sort_key)
    selections = tuple(
        RhythmSelection(
            group.index,
            group.source_indices,
            candidate,
            group.candidates,
            group.local_best,
            transition,
            cumulative,
            _selection_reason(candidate, group.local_best, transition),
        )
        for group, candidate, transition, cumulative in zip(
            groups,
            best.selections,
            best.transition_costs,
            best.cumulative_scores,
            strict=True,
        )
    )
    return RhythmOptimizationResult(selections, time.perf_counter() - started, evaluated)


def _build_groups(
    transcription: TranscriptionResult,
    beat_track: BeatTrack,
    beat_offset: float,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
    quantization_complexity_cost: float,
    timing_tolerance_ms: float,
    config: RhythmSequenceConfig,
) -> tuple[RhythmGroup, ...]:
    indexed = sorted(
        (
            _IndexedNote(
                source_index,
                note,
                max(0.0, beat_track.seconds_to_beats(note.onset_seconds) + beat_offset),
                beat_track.seconds_to_beats(note.offset_seconds) + beat_offset,
            )
            for source_index, note in enumerate(transcription.notes)
        ),
        key=lambda item: (item.note.onset_seconds, item.note.pitch, item.source_index),
    )
    clusters: list[list[_IndexedNote]] = []
    for indexed_note in indexed:
        if (
            not clusters
            or indexed_note.note.onset_seconds - clusters[-1][0].note.onset_seconds
            > config.chord_window_ms / 1000
        ):
            clusters.append([indexed_note])
        else:
            clusters[-1].append(indexed_note)

    next_same_pitch: dict[int, float | None] = {}
    by_pitch: dict[int, list[_IndexedNote]] = {}
    for indexed_note in indexed:
        by_pitch.setdefault(indexed_note.note.pitch, []).append(indexed_note)
    for pitch_notes in by_pitch.values():
        for index, indexed_note in enumerate(pitch_notes):
            next_same_pitch[indexed_note.source_index] = (
                pitch_notes[index + 1].continuous_onset if index + 1 < len(pitch_notes) else None
            )

    groups: list[RhythmGroup] = []
    for group_index, cluster in enumerate(clusters):
        weights = [max(1, item.note.velocity) for item in cluster]
        raw_onset = sum(
            item.note.onset_seconds * weight for item, weight in zip(cluster, weights, strict=True)
        ) / sum(weights)
        continuous = max(0.0, beat_track.seconds_to_beats(raw_onset) + beat_offset)
        local_best, all_quantization = choose_quantization(
            continuous,
            raw_onset,
            beat_track,
            complexity_cost=quantization_complexity_cost,
            tolerance_ms=timing_tolerance_ms,
            beat_offset=beat_offset,
        )
        next_group_seconds = (
            clusters[group_index + 1][0].note.onset_seconds
            if group_index + 1 < len(clusters)
            else None
        )
        accent = min(1.0, len(cluster) / 5 + statistics.mean(weights) / 180)
        candidates = _candidate_interpretations(
            group_index,
            cluster,
            all_quantization,
            local_best,
            next_group_seconds,
            next_same_pitch,
            beat_track,
            beat_offset,
            time_signature,
            pickup_beats,
            timing_tolerance_ms,
            accent,
            config,
        )
        groups.append(
            RhythmGroup(
                group_index,
                tuple(item.source_index for item in cluster),
                raw_onset,
                continuous,
                accent,
                candidates,
                local_best,
            )
        )
    return tuple(groups)


def _candidate_interpretations(
    group_index: int,
    cluster: list[_IndexedNote],
    quantization_candidates: tuple[QuantizationCandidate, ...],
    local_best: QuantizationCandidate,
    next_group_seconds: float | None,
    next_same_pitch: dict[int, float | None],
    beat_track: BeatTrack,
    beat_offset: float,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
    timing_tolerance_ms: float,
    accent_strength: float,
    config: RhythmSequenceConfig,
) -> tuple[RhythmCandidate, ...]:
    best_timing = min(
        abs(candidate.timing_error_seconds) * 1000 for candidate in quantization_candidates
    )
    filtered = [
        candidate
        for candidate in quantization_candidates
        if abs(candidate.timing_error_seconds) * 1000 <= best_timing + config.candidate_window_ms
        or candidate == local_best
    ]
    interpretations: list[RhythmCandidate] = []
    seen: set[tuple[Fraction, RhythmFamily]] = set()
    for quantization in filtered:
        family = family_for_subdivision(quantization.subdivision)
        identity = (quantization.position_beats, family)
        if identity in seen:
            continue
        seen.add(identity)
        duration_choices = tuple(
            _duration_choice(
                indexed_note,
                quantization.position_beats,
                next_group_seconds,
                next_same_pitch[indexed_note.source_index],
                family,
                beat_track,
                beat_offset,
                time_signature,
                pickup_beats,
                timing_tolerance_ms,
                config,
            )
            for indexed_note in cluster
        )
        _measure_index, metric_position, _length = locate_score_measure(
            quantization.position_beats,
            time_signature,
            pickup_beats,
        )
        duration_cost = statistics.mean(choice.selected.total_score for choice in duration_choices)
        onset_cost = (
            config.weights.onset_timing
            * abs(quantization.timing_error_seconds)
            * 1000
            / timing_tolerance_ms
            + config.weights.notation_complexity
            * QuantizationGrid(quantization.subdivision).complexity
        )
        if family.ternary:
            onset_cost += config.weights.isolated_triplet
        if family is RhythmFamily.THIRTY_SECOND:
            onset_cost += config.weights.thirty_second_value
        measure_length = time_signature.measure_beats
        boundary_distance = min(metric_position, measure_length - metric_position)
        metric_reward = max(0.0, 1 - float(boundary_distance) / 0.25)
        metric_cost = -config.weights.metric_accent * accent_strength * metric_reward
        pickup_cost = 0.0
        if group_index == 0 and pickup_beats > 0 and quantization.position_beats > pickup_beats:
            pickup_cost = config.weights.pickup_plausibility * float(
                quantization.position_beats - pickup_beats
            )
        interpretations.append(
            RhythmCandidate(
                quantization,
                family,
                metric_position,
                duration_choices,
                any(choice.selected.requires_tie for choice in duration_choices),
                onset_cost + duration_cost + metric_cost + pickup_cost,
            )
        )
    ordered = sorted(
        interpretations,
        key=lambda item: (
            item.local_cost,
            abs(item.quantization.timing_error_seconds),
            item.quantization.position_beats,
            item.family.value,
        ),
    )
    retained = ordered[: config.candidate_limit]
    local_match = next(
        (
            item
            for item in interpretations
            if item.quantization.position_beats == local_best.position_beats
            and item.quantization.subdivision == local_best.subdivision
        ),
        None,
    )
    if local_match is not None and local_match not in retained:
        retained[-1] = local_match
        retained.sort(key=lambda item: (item.local_cost, item.family.value))
    return tuple(retained)


def _duration_choice(
    indexed_note: _IndexedNote,
    onset_beats: Fraction,
    next_group_seconds: float | None,
    next_same_pitch_beats: float | None,
    family: RhythmFamily,
    beat_track: BeatTrack,
    beat_offset: float,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
    timing_tolerance_ms: float,
    config: RhythmSequenceConfig,
) -> NoteDurationChoice:
    note = indexed_note.note
    target_offset_seconds = note.offset_seconds
    pedal_shortened = False
    if (
        note.pedal is True
        and next_group_seconds is not None
        and next_group_seconds < target_offset_seconds
    ):
        target_offset_seconds = next_group_seconds
        pedal_shortened = True
    maximum = (
        Fraction(str(next_same_pitch_beats)) - onset_beats
        if next_same_pitch_beats is not None
        else None
    )
    available = tuple(
        duration for duration in WRITTEN_DURATIONS if maximum is None or duration <= maximum
    ) or (WRITTEN_DURATIONS[0],)
    candidates: list[DurationCandidate] = []
    for duration in available:
        written_offset_seconds = beat_track.beats_to_seconds(
            float(onset_beats + duration) - beat_offset
        )
        timing_error = written_offset_seconds - target_offset_seconds
        complexity = _duration_complexity(duration)
        ties = _tie_count(onset_beats, duration, time_signature, pickup_beats)
        tiny_tie = _has_tiny_tie_fragment(
            onset_beats,
            duration,
            time_signature,
            pickup_beats,
        )
        dotted_micro = duration in {Fraction(3, 8), Fraction(3, 16)}
        thirty_second = duration <= Fraction(1, 8)
        unusual_short = duration <= Fraction(1, 4)
        duration_is_ternary = duration.denominator in {3, 6, 24}
        family_mismatch = duration_is_ternary != family.ternary
        fine_value_support = 0.25 if family is RhythmFamily.THIRTY_SECOND else 1.0
        total = (
            config.weights.duration_timing * abs(timing_error) * 1000 / timing_tolerance_ms
            + config.weights.notation_complexity * complexity
            + config.weights.dotted_micro_value * dotted_micro
            + config.weights.thirty_second_value * thirty_second * fine_value_support
            + config.weights.unusual_short_value * unusual_short * fine_value_support
            + config.weights.tie * ties
            + config.weights.tiny_tie_fragment * tiny_tie
            + config.weights.straight_triplet_switch * 0.75 * family_mismatch
        )
        candidates.append(
            DurationCandidate(
                duration,
                timing_error,
                complexity,
                ties > 0,
                tiny_tie,
                dotted_micro,
                unusual_short,
                total,
            )
        )
    best_timing = min(abs(candidate.timing_error_seconds) * 1000 for candidate in candidates)
    pruned = [
        candidate
        for candidate in candidates
        if abs(candidate.timing_error_seconds) * 1000
        <= best_timing + config.duration_candidate_window_ms
    ]
    ordered = tuple(
        sorted(
            pruned,
            key=lambda item: (
                item.total_score,
                abs(item.timing_error_seconds),
                -item.duration_beats,
            ),
        )[: config.duration_candidate_limit]
    )
    return NoteDurationChoice(indexed_note.source_index, ordered[0], ordered, pedal_shortened)


def _transition_cost(
    state: _BeamState,
    previous: RhythmCandidate,
    current: RhythmCandidate,
    previous_group: RhythmGroup,
    current_group: RhythmGroup,
    config: RhythmSequenceConfig,
) -> float:
    weights = config.weights
    cost = 0.0
    if current.family != previous.family:
        cost += weights.family_switch
    if current.family.ternary != previous.family.ternary:
        cost += weights.straight_triplet_switch
    elif current.family.ternary and previous.family.ternary:
        cost -= weights.isolated_triplet * 1.15

    symbolic_gap = current.quantization.position_beats - previous.quantization.position_beats
    raw_gap = current_group.continuous_onset_beats - previous_group.continuous_onset_beats
    if (
        state.last_symbolic_gap is not None
        and state.last_raw_gap is not None
        and abs(raw_gap - state.last_raw_gap) <= config.pattern_similarity_beats
    ):
        cost += weights.pattern_consistency * min(
            2.0,
            float(abs(symbolic_gap - state.last_symbolic_gap)) / 0.25,
        )
        cost += weights.duration_pattern * min(
            2.0,
            float(abs(current.representative_duration - previous.representative_duration)) / 0.25,
        )
    return cost


def _selection_reason(
    selected: RhythmCandidate,
    local_best: QuantizationCandidate,
    transition_cost: float,
) -> str:
    if (
        selected.quantization.position_beats == local_best.position_beats
        and selected.quantization.subdivision == local_best.subdivision
    ):
        return "local-best"
    if selected.family.ternary != family_for_subdivision(local_best.subdivision).ternary:
        return "straight-triplet-context"
    if selected.quantization.complexity_penalty < local_best.complexity_penalty:
        return "notation-simplicity"
    if transition_cost != 0:
        return "family-or-pattern-continuity"
    return "global-path-score"


def family_for_subdivision(subdivision: str) -> RhythmFamily:
    return {
        QuantizationGrid.QUARTER.value: RhythmFamily.QUARTER,
        QuantizationGrid.EIGHTH.value: RhythmFamily.EIGHTH_STRAIGHT,
        QuantizationGrid.EIGHTH_TRIPLET.value: RhythmFamily.EIGHTH_TRIPLET,
        QuantizationGrid.SIXTEENTH.value: RhythmFamily.SIXTEENTH_STRAIGHT,
        QuantizationGrid.SIXTEENTH_TRIPLET.value: RhythmFamily.SIXTEENTH_TRIPLET,
        QuantizationGrid.THIRTY_SECOND.value: RhythmFamily.THIRTY_SECOND,
    }[subdivision]


def _duration_complexity(duration: Fraction) -> float:
    return {
        Fraction(1, 8): 0.9,
        Fraction(1, 6): 0.95,
        Fraction(3, 16): 0.9,
        Fraction(1, 4): 0.3,
        Fraction(1, 3): 0.55,
        Fraction(3, 8): 0.75,
        Fraction(1, 2): 0.05,
        Fraction(2, 3): 0.4,
        Fraction(3, 4): 0.15,
        Fraction(1): 0.0,
        Fraction(4, 3): 0.35,
        Fraction(3, 2): 0.08,
        Fraction(2): 0.0,
        Fraction(3): 0.05,
        Fraction(4): 0.0,
    }[duration]


def _tie_count(
    onset: Fraction,
    duration: Fraction,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
) -> int:
    offset = onset + duration
    measure_length = time_signature.measure_beats
    boundary = pickup_beats if pickup_beats > 0 else measure_length
    count = 0
    while boundary < offset:
        if onset < boundary:
            count += 1
        boundary += measure_length
    return count


def _has_tiny_tie_fragment(
    onset: Fraction,
    duration: Fraction,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
) -> bool:
    offset = onset + duration
    measure_length = time_signature.measure_beats
    boundary = pickup_beats if pickup_beats > 0 else measure_length
    previous = onset
    while boundary < offset:
        if onset < boundary and boundary - previous < Fraction(1, 6):
            return True
        previous = boundary
        boundary += measure_length
    return offset - previous < Fraction(1, 6) and previous > onset


def _state_sort_key(state: _BeamState) -> tuple[float, tuple[tuple[Fraction, str], ...]]:
    return (
        state.score,
        tuple(
            (candidate.quantization.position_beats, candidate.family.value)
            for candidate in state.selections
        ),
    )
