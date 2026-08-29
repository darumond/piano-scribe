"""Deterministic piano hand, staff, and voice separation."""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from itertools import groupby

from piano_transcriber.score.chords import group_chords
from piano_transcriber.score.quantize import snap_written_duration
from piano_transcriber.score.tempo import seconds_to_beats
from piano_transcriber.score.types import (
    EventDiagnostic,
    PianoHand,
    ReconstructedScore,
    ScoreNote,
    ScoreRest,
    TimeSignature,
)

logger = logging.getLogger(__name__)


class PianoLayoutMode(StrEnum):
    NONE = "none"
    SEQUENCE = "sequence"


@dataclass(frozen=True, slots=True)
class HandAssignmentWeights:
    """Weights for register, continuity, chord, and crossing evidence."""

    register: float = 0.55
    continuity: float = 0.75
    large_jump: float = 0.85
    hand_switch: float = 0.65
    crossing: float = 1.6
    compact_chord_split: float = 0.8
    wide_span: float = 1.0
    hand_load: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.register,
            self.continuity,
            self.large_jump,
            self.hand_switch,
            self.crossing,
            self.compact_chord_split,
            self.wide_span,
            self.hand_load,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("hand-assignment weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VoiceAssignmentWeights:
    """Weights for within-staff melodic continuity and overlap handling."""

    continuity: float = 0.8
    large_jump: float = 0.55
    overlap: float = 3.0
    crossing: float = 1.2
    voice_switch: float = 0.65
    split_chord: float = 0.2
    secondary_voice: float = 0.35
    additional_voice: float = 3.0
    register_consistency: float = 0.35
    contour: float = 0.3
    repeated_pitch: float = 2.5
    track_switch: float = 1.1
    inactivity: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.continuity,
            self.large_jump,
            self.overlap,
            self.crossing,
            self.voice_switch,
            self.split_chord,
            self.secondary_voice,
            self.additional_voice,
            self.register_consistency,
            self.contour,
            self.repeated_pitch,
            self.track_switch,
            self.inactivity,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("voice-assignment weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PianoSeparationConfig:
    mode: PianoLayoutMode = PianoLayoutMode.NONE
    split_pitch: int = 60
    hand_beam_size: int = 64
    hand_candidate_limit: int = 16
    maximum_hand_span_semitones: int = 14
    large_jump_semitones: int = 12
    compact_split_gap_semitones: int = 7
    continuity_horizon_beats: float = 6.0
    preferred_voices_per_staff: int = 2
    maximum_voices_per_staff: int = 4
    voice_beam_size: int = 48
    voice_duration_refinement: bool = True
    duration_improvement_beats: float = 0.25
    minimum_explicit_rest_beats: Fraction = Fraction(1, 4)
    repeated_pitch_memory_beats: float = 12.0
    hand_weights: HandAssignmentWeights = HandAssignmentWeights()
    voice_weights: VoiceAssignmentWeights = VoiceAssignmentWeights()

    def __post_init__(self) -> None:
        if not 0 <= self.split_pitch <= 127:
            raise ValueError("split pitch must be between 0 and 127")
        if self.hand_beam_size <= 0 or self.hand_candidate_limit <= 0:
            raise ValueError("hand beam and candidate limits must be positive")
        if self.maximum_hand_span_semitones <= 0 or self.large_jump_semitones <= 0:
            raise ValueError("hand span and jump limits must be positive")
        if self.compact_split_gap_semitones <= 0 or self.continuity_horizon_beats <= 0:
            raise ValueError("split gap and continuity horizon must be positive")
        if not 1 <= self.preferred_voices_per_staff <= self.maximum_voices_per_staff:
            raise ValueError("preferred voice count must not exceed maximum voice count")
        if self.voice_beam_size <= 0:
            raise ValueError("voice beam size must be positive")
        if (
            not math.isfinite(self.duration_improvement_beats)
            or self.duration_improvement_beats < 0
        ):
            raise ValueError("duration improvement must be finite and non-negative")
        if self.minimum_explicit_rest_beats <= 0:
            raise ValueError("minimum explicit rest duration must be positive")
        if self.repeated_pitch_memory_beats <= 0:
            raise ValueError("repeated-pitch memory must be positive")


@dataclass(frozen=True, slots=True)
class _OnsetGroup:
    onset: Fraction
    notes: tuple[ScoreNote, ...]


@dataclass(frozen=True, slots=True)
class _HandCandidate:
    assignments: tuple[PianoHand, ...]
    local_cost: float
    left_center: float | None
    right_center: float | None


@dataclass(frozen=True, slots=True)
class _HandSelection:
    group: _OnsetGroup
    candidate: _HandCandidate
    transition_cost: float
    cumulative_cost: float


@dataclass(frozen=True, slots=True)
class _HandState:
    score: float
    selections: tuple[_HandSelection, ...]
    last_left: float | None = None
    last_right: float | None = None
    last_left_onset: Fraction | None = None
    last_right_onset: Fraction | None = None


@dataclass(frozen=True, slots=True)
class _VoiceCandidate:
    assignments: tuple[int, ...]
    local_cost: float


@dataclass(frozen=True, slots=True)
class _VoiceSelection:
    group: _OnsetGroup
    candidate: _VoiceCandidate
    transition_cost: float


@dataclass(frozen=True, slots=True)
class VoiceTrackState:
    """Short-lived musical voice state suitable for replacement by another tracker."""

    voice_id: int
    staff: int = 1
    previous_pitch: float | None = None
    previous_onset: Fraction | None = None
    previous_duration: Fraction = Fraction(0)
    register_center: float | None = None
    recent_direction: int = 0
    active_until: Fraction = Fraction(0)
    recent_chord_participation: bool = False
    active: bool = False


@dataclass(frozen=True, slots=True)
class _VoiceState:
    score: float
    selections: tuple[_VoiceSelection, ...]
    tracks: tuple[VoiceTrackState, ...]
    pitch_owners: tuple[tuple[int, int, Fraction], ...] = ()


@dataclass(frozen=True, slots=True)
class _AssignmentResult:
    notes: tuple[ScoreNote, ...]
    elapsed_seconds: float
    evaluated_transitions: int


def separate_piano_score(
    score: ReconstructedScore,
    config: PianoSeparationConfig,
) -> ReconstructedScore:
    """Add traceable hand, staff, voice, duration, and rest information."""
    if config.mode is PianoLayoutMode.NONE:
        return score

    identified = _with_chord_and_tie_identity(score.notes, score.time_signature, score.pickup_beats)
    hand_result = _assign_hands(identified, config)
    voice_result = _assign_voices(hand_result.notes, config)
    refined, duration_changes = _refine_voice_durations(score, voice_result.notes, config)
    refined = _with_chord_and_tie_identity(refined, score.time_signature, score.pickup_beats)
    rests = _generate_rests(
        refined,
        score.measure_count,
        score.time_signature,
        score.pickup_beats,
        config.minimum_explicit_rest_beats,
    )
    by_source = {note.source_index: note for note in refined}
    diagnostics = tuple(_assignment_diagnostic(item, by_source) for item in score.diagnostics)
    logger.info(
        "Piano hand assignment evaluated %d transitions in %.3f s; voice assignment "
        "evaluated %d transitions in %.3f s",
        hand_result.evaluated_transitions,
        hand_result.elapsed_seconds,
        voice_result.evaluated_transitions,
        voice_result.elapsed_seconds,
    )
    return replace(
        score,
        notes=refined,
        chords=group_chords(refined),
        diagnostics=diagnostics,
        rests=rests,
        piano_layout=config.mode.value,
        hand_optimizer_seconds=hand_result.elapsed_seconds,
        hand_evaluated_transitions=hand_result.evaluated_transitions,
        voice_optimizer_seconds=voice_result.elapsed_seconds,
        voice_stability_seconds=voice_result.elapsed_seconds,
        voice_evaluated_transitions=voice_result.evaluated_transitions,
        voice_duration_changes=duration_changes,
        minimum_explicit_rest_beats=config.minimum_explicit_rest_beats,
    )


def _onset_groups(notes: tuple[ScoreNote, ...]) -> tuple[_OnsetGroup, ...]:
    ordered = sorted(notes, key=lambda note: (note.onset_beats, note.pitch, note.source_index))
    return tuple(
        _OnsetGroup(onset, tuple(group))
        for onset, group in groupby(ordered, key=lambda note: note.onset_beats)
    )


def _with_chord_and_tie_identity(
    notes: tuple[ScoreNote, ...],
    time_signature: TimeSignature,
    pickup_beats: Fraction,
) -> tuple[ScoreNote, ...]:
    result: list[ScoreNote] = []
    for chord_id, group in enumerate(_onset_groups(notes)):
        for note in group.notes:
            result.append(
                replace(
                    note,
                    chord_id=chord_id,
                    tie_across_measure=_crosses_measure(
                        note.onset_beats,
                        note.offset_beats,
                        time_signature.measure_beats,
                        pickup_beats,
                    ),
                )
            )
    return tuple(sorted(result, key=lambda note: (note.onset_beats, note.pitch, note.source_index)))


def _assign_hands(
    notes: tuple[ScoreNote, ...],
    config: PianoSeparationConfig,
) -> _AssignmentResult:
    started = time.perf_counter()
    groups = _onset_groups(notes)
    beam = [_HandState(0.0, ())]
    evaluated = 0
    for group in groups:
        candidates = _hand_candidates(group, config)
        next_beam: list[_HandState] = []
        for state in beam:
            for candidate in candidates:
                transition = _hand_transition_cost(state, group, candidate, config)
                evaluated += 1
                cumulative = state.score + candidate.local_cost + transition
                next_beam.append(
                    _HandState(
                        cumulative,
                        (
                            *state.selections,
                            _HandSelection(group, candidate, transition, cumulative),
                        ),
                        candidate.left_center
                        if candidate.left_center is not None
                        else state.last_left,
                        candidate.right_center
                        if candidate.right_center is not None
                        else state.last_right,
                        group.onset if candidate.left_center is not None else state.last_left_onset,
                        group.onset
                        if candidate.right_center is not None
                        else state.last_right_onset,
                    )
                )
        beam = sorted(next_beam, key=_hand_state_key)[: config.hand_beam_size]
    best = beam[0] if beam else _HandState(0.0, ())
    replacements: dict[int, ScoreNote] = {}
    for selection in best.selections:
        per_note_cost = (selection.candidate.local_cost + selection.transition_cost) / len(
            selection.group.notes
        )
        confidence = 1.0 / (1.0 + max(0.0, per_note_cost))
        for note, hand in zip(
            selection.group.notes,
            selection.candidate.assignments,
            strict=True,
        ):
            replacements[note.source_index] = replace(
                note,
                hand=hand,
                staff=1 if hand is PianoHand.RIGHT else 2,
                hand_assignment_cost=max(0.0, per_note_cost),
                hand_assignment_confidence=confidence,
            )
    assigned = tuple(replacements[note.source_index] for note in notes)
    assigned = _add_hand_continuity_costs(assigned, config)
    return _AssignmentResult(assigned, time.perf_counter() - started, evaluated)


def _hand_candidates(
    group: _OnsetGroup,
    config: PianoSeparationConfig,
) -> tuple[_HandCandidate, ...]:
    notes = group.notes
    count = len(notes)
    patterns: set[tuple[PianoHand, ...]] = {
        (PianoHand.LEFT,) * count,
        (PianoHand.RIGHT,) * count,
    }
    for split in range(1, count):
        patterns.add((PianoHand.LEFT,) * split + (PianoHand.RIGHT,) * (count - split))
        if count <= 6:
            patterns.add((PianoHand.RIGHT,) * split + (PianoHand.LEFT,) * (count - split))

    candidates = [_make_hand_candidate(notes, assignments, config) for assignments in patterns]
    candidates.sort(
        key=lambda item: (
            item.local_cost,
            tuple(hand.value for hand in item.assignments),
        )
    )
    return tuple(candidates[: config.hand_candidate_limit])


def _make_hand_candidate(
    notes: tuple[ScoreNote, ...],
    assignments: tuple[PianoHand, ...],
    config: PianoSeparationConfig,
) -> _HandCandidate:
    left = [
        note.pitch for note, hand in zip(notes, assignments, strict=True) if hand is PianoHand.LEFT
    ]
    right = [
        note.pitch for note, hand in zip(notes, assignments, strict=True) if hand is PianoHand.RIGHT
    ]
    weights = config.hand_weights
    register_cost = (
        weights.register
        * (
            sum(max(0, pitch - config.split_pitch) for pitch in left)
            + sum(max(0, config.split_pitch - pitch) for pitch in right)
        )
        / 12.0
    )
    span_cost = (
        weights.wide_span
        * (
            _span_excess(left, config.maximum_hand_span_semitones)
            + _span_excess(right, config.maximum_hand_span_semitones)
        )
        / 12.0
    )
    load_cost = weights.hand_load * (max(0, len(left) - 5) + max(0, len(right) - 5))
    split_cost = 0.0
    if left and right:
        boundary_gaps = [
            notes[index + 1].pitch - notes[index].pitch
            for index in range(len(notes) - 1)
            if assignments[index] is not assignments[index + 1]
        ]
        strongest_gap = max(boundary_gaps, default=0)
        compactness = max(
            0.0,
            (config.compact_split_gap_semitones - strongest_gap)
            / config.compact_split_gap_semitones,
        )
        total_span = notes[-1].pitch - notes[0].pitch
        if total_span > config.maximum_hand_span_semitones:
            compactness *= 0.25
        split_cost = weights.compact_chord_split * compactness
    crossing_cost = 0.0
    if left and right and max(left) > min(right):
        crossing_cost = weights.crossing * (1.0 + (max(left) - min(right)) / 12.0)
    return _HandCandidate(
        assignments,
        register_cost + span_cost + load_cost + split_cost + crossing_cost,
        statistics.mean(left) if left else None,
        statistics.mean(right) if right else None,
    )


def _span_excess(pitches: list[int], maximum: int) -> int:
    return max(0, max(pitches) - min(pitches) - maximum) if pitches else 0


def _hand_transition_cost(
    state: _HandState,
    group: _OnsetGroup,
    candidate: _HandCandidate,
    config: PianoSeparationConfig,
) -> float:
    cost = 0.0
    cost += _hand_motion_cost(
        state.last_left,
        state.last_left_onset,
        candidate.left_center,
        group.onset,
        config,
    )
    cost += _hand_motion_cost(
        state.last_right,
        state.last_right_onset,
        candidate.right_center,
        group.onset,
        config,
    )
    for current, own, other in (
        (candidate.left_center, state.last_left, state.last_right),
        (candidate.right_center, state.last_right, state.last_left),
    ):
        if current is None or own is None or other is None:
            continue
        own_distance = abs(current - own)
        other_distance = abs(current - other)
        if other_distance + 3 < own_distance:
            cost += config.hand_weights.hand_switch * (own_distance - other_distance - 3) / 12.0
    if (
        candidate.left_center is not None
        and candidate.right_center is not None
        and candidate.left_center > candidate.right_center
    ):
        cost += config.hand_weights.crossing * (
            1.0 + (candidate.left_center - candidate.right_center) / 12.0
        )
    return cost


def _hand_motion_cost(
    previous: float | None,
    previous_onset: Fraction | None,
    current: float | None,
    current_onset: Fraction,
    config: PianoSeparationConfig,
) -> float:
    if previous is None or previous_onset is None or current is None:
        return 0.0
    gap = float(current_onset - previous_onset)
    memory = max(0.2, 1.0 - gap / config.continuity_horizon_beats)
    distance = abs(current - previous)
    cost = config.hand_weights.continuity * distance / 12.0 * memory
    if distance > config.large_jump_semitones:
        excess = (distance - config.large_jump_semitones) / 12.0
        cost += config.hand_weights.large_jump * excess * excess * memory
    return cost


def _hand_state_key(state: _HandState) -> tuple[float, tuple[tuple[str, ...], ...]]:
    return (
        state.score,
        tuple(
            tuple(hand.value for hand in selection.candidate.assignments)
            for selection in state.selections
        ),
    )


def _add_hand_continuity_costs(
    notes: tuple[ScoreNote, ...], config: PianoSeparationConfig
) -> tuple[ScoreNote, ...]:
    replacements: dict[int, ScoreNote] = {}
    for hand in PianoHand:
        hand_notes = tuple(note for note in notes if note.hand is hand)
        groups = _onset_groups(hand_notes)
        centers = [statistics.mean(note.pitch for note in group.notes) for group in groups]
        for index, group in enumerate(groups):
            for note in group.notes:
                previous = (
                    config.hand_weights.continuity * abs(note.pitch - centers[index - 1]) / 12.0
                    if index > 0
                    else 0.0
                )
                following = (
                    config.hand_weights.continuity * abs(note.pitch - centers[index + 1]) / 12.0
                    if index + 1 < len(groups)
                    else 0.0
                )
                replacements[note.source_index] = replace(
                    note,
                    previous_continuity_cost=previous,
                    next_continuity_cost=following,
                )
    return tuple(replacements[note.source_index] for note in notes)


def _assign_voices(
    notes: tuple[ScoreNote, ...], config: PianoSeparationConfig
) -> _AssignmentResult:
    started = time.perf_counter()
    replacements: dict[int, ScoreNote] = {}
    evaluated = 0
    for staff in (1, 2):
        staff_notes = tuple(note for note in notes if note.staff == staff)
        groups = _onset_groups(staff_notes)
        tracks = tuple(
            VoiceTrackState(voice_id, staff=staff)
            for voice_id in range(1, config.maximum_voices_per_staff + 1)
        )
        beam = [_VoiceState(0.0, (), tracks)]
        for group in groups:
            candidates = _voice_candidates(group, config)
            next_beam: list[_VoiceState] = []
            for state in beam:
                for candidate in candidates:
                    transition = _voice_transition_cost(state, group, candidate, config)
                    evaluated += 1
                    updated_tracks, owners = _updated_voice_state(state, group, candidate, config)
                    next_beam.append(
                        _VoiceState(
                            state.score + candidate.local_cost + transition,
                            (*state.selections, _VoiceSelection(group, candidate, transition)),
                            updated_tracks,
                            owners,
                        )
                    )
            beam = sorted(next_beam, key=_voice_state_key)[: config.voice_beam_size]
        best = beam[0]
        pitch_owners: dict[int, tuple[int, Fraction]] = {}
        track_pitches: dict[int, float] = {}
        for selection in best.selections:
            previous_track_pitches = dict(track_pitches)
            per_note_cost = (selection.candidate.local_cost + selection.transition_cost) / len(
                selection.group.notes
            )
            for note, voice in zip(
                selection.group.notes,
                selection.candidate.assignments,
                strict=True,
            ):
                owner = pitch_owners.get(note.pitch)
                owner_is_recent = owner is not None and (
                    float(note.onset_beats - owner[1]) <= config.repeated_pitch_memory_beats
                )
                repeated_switch = owner_is_recent and owner is not None and owner[0] != voice
                previous_pitch = previous_track_pitches.get(voice)
                nearest_track = min(
                    (
                        (abs(note.pitch - pitch), track_voice)
                        for track_voice, pitch in previous_track_pitches.items()
                    ),
                    default=None,
                )
                identity_switch = (
                    previous_pitch is not None
                    and nearest_track is not None
                    and nearest_track[1] != voice
                    and nearest_track[0] + 3 < abs(note.pitch - previous_pitch)
                )
                direction = (
                    _motion_direction(note.pitch - previous_pitch)
                    if previous_pitch is not None
                    else 0
                )
                continuity = (
                    config.voice_weights.continuity * abs(note.pitch - previous_pitch) / 12.0
                    if previous_pitch is not None
                    else 0.0
                )
                reason = "track-continuity"
                if repeated_switch:
                    reason = "repeated-pitch-reassignment"
                elif identity_switch:
                    reason = "neighbor-track-exchange"
                elif owner_is_recent and owner is not None and owner[0] == voice:
                    reason = "repeated-pitch-continuity"
                replacements[note.source_index] = replace(
                    note,
                    voice=voice,
                    voice_assignment_cost=max(0.0, per_note_cost),
                    voice_identity_switched=identity_switch,
                    repeated_pitch_voice_switched=repeated_switch,
                    voice_assignment_reason=reason,
                    track_previous_pitch=previous_pitch,
                    track_direction=direction,
                    voice_continuity_score=continuity,
                )
                pitch_owners[note.pitch] = (voice, note.onset_beats)
            track_pitches.update(_voice_representatives(selection.group, selection.candidate))
    assigned = tuple(replacements[note.source_index] for note in notes)
    assigned = _collapse_unnecessary_extra_voices(
        assigned,
        config.preferred_voices_per_staff,
    )
    assigned = _refresh_voice_diagnostics(assigned, config)
    assigned = _add_extra_voice_reasons(assigned, config.preferred_voices_per_staff)
    return _AssignmentResult(assigned, time.perf_counter() - started, evaluated)


def _voice_candidates(
    group: _OnsetGroup, config: PianoSeparationConfig
) -> tuple[_VoiceCandidate, ...]:
    count = len(group.notes)
    patterns: set[tuple[int, ...]] = {
        (voice,) * count for voice in range(1, config.preferred_voices_per_staff + 1)
    }
    if config.preferred_voices_per_staff >= 2:
        for split in range(1, count):
            patterns.add((2,) * split + (1,) * (count - split))
            patterns.add((1,) * split + (2,) * (count - split))
    for voice in range(
        config.preferred_voices_per_staff + 1,
        config.maximum_voices_per_staff + 1,
    ):
        patterns.add((voice,) * count)
    candidates = []
    for assignments in patterns:
        used = set(assignments)
        local = config.voice_weights.split_chord * max(0, len(used) - 1)
        local += config.voice_weights.additional_voice * sum(
            voice > config.preferred_voices_per_staff for voice in used
        )
        if 1 in used and 2 in used:
            upper = [
                note.pitch
                for note, voice in zip(group.notes, assignments, strict=True)
                if voice == 1
            ]
            lower = [
                note.pitch
                for note, voice in zip(group.notes, assignments, strict=True)
                if voice == 2
            ]
            if upper and lower and min(upper) < max(lower):
                local += config.voice_weights.crossing * (1.0 + (max(lower) - min(upper)) / 12.0)
        candidates.append(_VoiceCandidate(assignments, local))
    return tuple(sorted(candidates, key=lambda item: (item.local_cost, item.assignments)))


def _voice_transition_cost(
    state: _VoiceState,
    group: _OnsetGroup,
    candidate: _VoiceCandidate,
    config: PianoSeparationConfig,
) -> float:
    representatives = _voice_representatives(group, candidate)
    cost = 0.0
    for voice, current in representatives.items():
        index = voice - 1
        track = state.tracks[index]
        previous = track.previous_pitch
        if previous is None and 1 < voice <= config.preferred_voices_per_staff:
            cost += config.voice_weights.secondary_voice * (voice - 1)
        if previous is not None:
            distance = abs(current - previous)
            cost += config.voice_weights.continuity * distance / 12.0
            if track.register_center is not None:
                cost += (
                    config.voice_weights.register_consistency
                    * abs(current - track.register_center)
                    / 12.0
                )
            if distance > config.large_jump_semitones:
                cost += (
                    config.voice_weights.large_jump
                    * (distance - config.large_jump_semitones)
                    / 12.0
                )
            other_distances = [
                abs(current - other.previous_pitch)
                for other_index, other in enumerate(state.tracks)
                if other_index != index and other.previous_pitch is not None
            ]
            if other_distances and min(other_distances) + 3 < distance:
                cost += (
                    config.voice_weights.voice_switch * (distance - min(other_distances) - 3) / 12.0
                )
            direction = _motion_direction(current - previous)
            if (
                direction != 0
                and track.recent_direction != 0
                and direction != track.recent_direction
            ):
                cost += config.voice_weights.contour * min(1.0, distance / 12.0)
            if track.previous_onset is not None:
                gap = float(group.onset - track.previous_onset)
                cost += config.voice_weights.inactivity * min(2.0, gap / 4.0)
        if track.active_until > group.onset and track.previous_onset != group.onset:
            overlap = float(track.active_until - group.onset)
            cost += config.voice_weights.overlap * (4.0 + min(overlap, 2.0) / 2.0)
    if 1 in representatives and 2 in representatives and representatives[1] < representatives[2]:
        cost += config.voice_weights.crossing * (
            1.0 + (representatives[2] - representatives[1]) / 12.0
        )
    if state.selections:
        previous_selection = state.selections[-1]
        previous_assignments = dict(
            zip(
                (note.source_index for note in previous_selection.group.notes),
                previous_selection.candidate.assignments,
                strict=True,
            )
        )
        for note, voice in zip(group.notes, candidate.assignments, strict=True):
            closest = min(
                previous_selection.group.notes,
                key=lambda previous_note: (
                    abs(previous_note.pitch - note.pitch),
                    previous_note.source_index,
                ),
            )
            if previous_assignments[closest.source_index] != voice:
                similarity = max(0.15, 1.0 - abs(closest.pitch - note.pitch) / 12.0)
                cost += config.voice_weights.track_switch * similarity
    owners = {pitch: (voice, onset) for pitch, voice, onset in state.pitch_owners}
    for note, voice in zip(group.notes, candidate.assignments, strict=True):
        owner = owners.get(note.pitch)
        if (
            owner is not None
            and owner[0] != voice
            and float(group.onset - owner[1]) <= config.repeated_pitch_memory_beats
        ):
            cost += config.voice_weights.repeated_pitch
    return cost


def _voice_representatives(group: _OnsetGroup, candidate: _VoiceCandidate) -> dict[int, float]:
    pitches: dict[int, list[int]] = {}
    for note, voice in zip(group.notes, candidate.assignments, strict=True):
        pitches.setdefault(voice, []).append(note.pitch)
    return {voice: statistics.mean(values) for voice, values in pitches.items()}


def _updated_voice_state(
    state: _VoiceState,
    group: _OnsetGroup,
    candidate: _VoiceCandidate,
    config: PianoSeparationConfig,
) -> tuple[tuple[VoiceTrackState, ...], tuple[tuple[int, int, Fraction], ...]]:
    tracks = [replace(track, active=track.active_until > group.onset) for track in state.tracks]
    representatives = _voice_representatives(group, candidate)
    for voice, representative in representatives.items():
        index = voice - 1
        previous = tracks[index]
        assigned_notes = tuple(
            note
            for note, assigned in zip(group.notes, candidate.assignments, strict=True)
            if assigned == voice
        )
        active_until = max(note.offset_beats for note in assigned_notes)
        direction = (
            _motion_direction(representative - previous.previous_pitch)
            if previous.previous_pitch is not None
            else 0
        )
        register = (
            representative
            if previous.register_center is None
            else previous.register_center * 0.7 + representative * 0.3
        )
        tracks[index] = VoiceTrackState(
            voice_id=voice,
            staff=previous.staff,
            previous_pitch=representative,
            previous_onset=group.onset,
            previous_duration=max(note.duration_beats for note in assigned_notes),
            register_center=register,
            recent_direction=direction or previous.recent_direction,
            active_until=active_until,
            recent_chord_participation=len(assigned_notes) > 1,
            active=True,
        )
    owners = {
        pitch: (voice, onset)
        for pitch, voice, onset in state.pitch_owners
        if float(group.onset - onset) <= config.repeated_pitch_memory_beats
    }
    for note, voice in zip(group.notes, candidate.assignments, strict=True):
        owners[note.pitch] = (voice, group.onset)
    encoded = tuple(sorted((pitch, voice, onset) for pitch, (voice, onset) in owners.items()))
    return tuple(tracks), encoded


def _voice_state_key(state: _VoiceState) -> tuple[float, tuple[tuple[int, ...], ...]]:
    return (
        state.score,
        tuple(selection.candidate.assignments for selection in state.selections),
    )


def _motion_direction(delta: float) -> int:
    return 1 if delta > 0 else -1 if delta < 0 else 0


def _collapse_unnecessary_extra_voices(
    notes: tuple[ScoreNote, ...],
    preferred_voice_count: int,
) -> tuple[ScoreNote, ...]:
    replacements = {note.source_index: note for note in notes}
    keys = sorted(
        {
            (note.staff, note.onset_beats, note.voice)
            for note in notes
            if note.voice > preferred_voice_count
        }
    )
    for staff, onset, extra_voice in keys:
        current_notes = tuple(replacements.values())
        group = tuple(
            note
            for note in current_notes
            if note.staff == staff and note.onset_beats == onset and note.voice == extra_voice
        )
        if not group:
            continue
        candidates = [
            voice
            for voice in range(1, preferred_voice_count + 1)
            if not _voice_group_conflicts(group, voice, current_notes)
        ]
        if not candidates:
            continue
        representative = statistics.mean(note.pitch for note in group)
        target = min(
            candidates,
            key=lambda voice: (
                _neighboring_voice_distance(
                    representative,
                    onset,
                    voice,
                    staff,
                    current_notes,
                    excluded={note.source_index for note in group},
                ),
                voice,
            ),
        )
        current_distance = _neighboring_voice_distance(
            representative,
            onset,
            extra_voice,
            staff,
            current_notes,
            excluded={note.source_index for note in group},
        )
        target_distance = _neighboring_voice_distance(
            representative,
            onset,
            target,
            staff,
            current_notes,
            excluded={note.source_index for note in group},
        )
        if target_distance > current_distance + 12.0:
            continue
        for note in group:
            replacements[note.source_index] = replace(
                note,
                voice=target,
                voice_assignment_reason="preferred-voice-collapse",
                extra_voice_reason=None,
            )
    return tuple(replacements[note.source_index] for note in notes)


def _voice_group_conflicts(
    group: tuple[ScoreNote, ...],
    voice: int,
    notes: tuple[ScoreNote, ...],
) -> bool:
    sources = {note.source_index for note in group}
    return any(
        other.source_index not in sources
        and other.staff == group[0].staff
        and other.voice == voice
        and other.onset_beats != group[0].onset_beats
        and any(
            note.onset_beats < other.offset_beats and other.onset_beats < note.offset_beats
            for note in group
        )
        for other in notes
    )


def _neighboring_voice_distance(
    pitch: float,
    onset: Fraction,
    voice: int,
    staff: int,
    notes: tuple[ScoreNote, ...],
    *,
    excluded: set[int],
) -> float:
    candidates = tuple(
        note
        for note in notes
        if note.source_index not in excluded and note.staff == staff and note.voice == voice
    )
    previous = max(
        (note for note in candidates if note.onset_beats < onset),
        key=lambda note: (note.onset_beats, note.source_index),
        default=None,
    )
    following = min(
        (note for note in candidates if note.onset_beats > onset),
        key=lambda note: (note.onset_beats, note.source_index),
        default=None,
    )
    distances = [abs(pitch - note.pitch) for note in (previous, following) if note is not None]
    return statistics.mean(distances) if distances else 0.0


def _refresh_voice_diagnostics(
    notes: tuple[ScoreNote, ...],
    config: PianoSeparationConfig,
) -> tuple[ScoreNote, ...]:
    replacements: dict[int, ScoreNote] = {}
    for staff in (1, 2):
        pitch_owners: dict[int, tuple[int, Fraction]] = {}
        track_pitches: dict[int, float] = {}
        for group in _onset_groups(tuple(note for note in notes if note.staff == staff)):
            previous_track_pitches = dict(track_pitches)
            by_voice: dict[int, list[int]] = {}
            for note in group.notes:
                owner = pitch_owners.get(note.pitch)
                owner_is_recent = owner is not None and (
                    float(note.onset_beats - owner[1]) <= config.repeated_pitch_memory_beats
                )
                repeated_switch = owner_is_recent and owner is not None and owner[0] != note.voice
                previous_pitch = previous_track_pitches.get(note.voice)
                nearest_track = min(
                    (
                        (abs(note.pitch - pitch), track_voice)
                        for track_voice, pitch in previous_track_pitches.items()
                    ),
                    default=None,
                )
                identity_switch = (
                    previous_pitch is not None
                    and nearest_track is not None
                    and nearest_track[1] != note.voice
                    and nearest_track[0] + 3 < abs(note.pitch - previous_pitch)
                )
                reason = note.voice_assignment_reason or "track-continuity"
                if repeated_switch:
                    reason = "repeated-pitch-reassignment"
                elif identity_switch:
                    reason = "neighbor-track-exchange"
                elif owner_is_recent and owner is not None and owner[0] == note.voice:
                    reason = "repeated-pitch-continuity"
                direction = (
                    _motion_direction(note.pitch - previous_pitch)
                    if previous_pitch is not None
                    else 0
                )
                continuity = (
                    config.voice_weights.continuity * abs(note.pitch - previous_pitch) / 12.0
                    if previous_pitch is not None
                    else 0.0
                )
                replacements[note.source_index] = replace(
                    note,
                    voice_identity_switched=identity_switch,
                    repeated_pitch_voice_switched=repeated_switch,
                    voice_assignment_reason=reason,
                    track_previous_pitch=previous_pitch,
                    track_direction=direction,
                    voice_continuity_score=continuity,
                )
                pitch_owners[note.pitch] = (note.voice, note.onset_beats)
                by_voice.setdefault(note.voice, []).append(note.pitch)
            track_pitches.update(
                {voice: statistics.mean(pitches) for voice, pitches in by_voice.items()}
            )
    return tuple(replacements[note.source_index] for note in notes)


def _add_extra_voice_reasons(
    notes: tuple[ScoreNote, ...], preferred_voice_count: int
) -> tuple[ScoreNote, ...]:
    replacements: dict[int, ScoreNote] = {}
    for note in notes:
        if note.voice <= preferred_voice_count:
            replacements[note.source_index] = note
            continue
        preferred = tuple(
            other
            for other in notes
            if other.staff == note.staff and other.voice <= preferred_voice_count
        )
        if any(
            other.onset_beats != note.onset_beats
            and note.onset_beats < other.offset_beats
            and other.onset_beats < note.offset_beats
            for other in preferred
        ):
            reason = "overlap-required"
        elif any(other.onset_beats == note.onset_beats for other in preferred):
            reason = "chord-decomposition"
        else:
            reason = "voice-leading"
        replacements[note.source_index] = replace(note, extra_voice_reason=reason)
    return tuple(replacements[note.source_index] for note in notes)


def _refine_voice_durations(
    score: ReconstructedScore,
    notes: tuple[ScoreNote, ...],
    config: PianoSeparationConfig,
) -> tuple[tuple[ScoreNote, ...], int]:
    if not config.voice_duration_refinement:
        return notes, 0
    diagnostics = {item.source_index: item for item in score.diagnostics}
    by_pitch: dict[int, list[ScoreNote]] = {}
    for note in notes:
        by_pitch.setdefault(note.pitch, []).append(note)
    next_same_pitch: dict[int, Fraction | None] = {}
    for pitch_notes in by_pitch.values():
        ordered = sorted(pitch_notes, key=lambda note: (note.onset_beats, note.source_index))
        for index, note in enumerate(ordered):
            next_same_pitch[note.source_index] = (
                ordered[index + 1].onset_beats if index + 1 < len(ordered) else None
            )

    replacements: dict[int, ScoreNote] = {}
    changed = 0
    for note in notes:
        if note.pedal is True:
            replacements[note.source_index] = note
            continue
        raw_offset = (
            seconds_to_beats(note.raw_offset_seconds, score.bpm)
            if score.beat_track is None
            else Fraction(
                str(
                    score.beat_track.seconds_to_beats(note.raw_offset_seconds)
                    + score.beat_position_offset
                )
            )
        )
        target_duration = raw_offset - note.onset_beats
        if target_duration <= note.duration_beats:
            replacements[note.source_index] = note
            continue
        explained_overlap = any(
            other.staff == note.staff
            and other.voice != note.voice
            and note.onset_beats < other.onset_beats < raw_offset
            for other in notes
        )
        if not explained_overlap:
            replacements[note.source_index] = note
            continue
        maximum = next_same_pitch[note.source_index]
        maximum_duration = maximum - note.onset_beats if maximum is not None else None
        diagnostic = diagnostics.get(note.source_index)
        alternatives = (
            [item.duration_beats for item in diagnostic.duration_candidates]
            if diagnostic is not None and diagnostic.duration_candidates
            else [snap_written_duration(target_duration, maximum=maximum_duration)]
        )
        viable = [
            duration
            for duration in alternatives
            if duration > note.duration_beats
            and (maximum_duration is None or duration <= maximum_duration)
        ]
        if not viable:
            replacements[note.source_index] = note
            continue
        selected = min(viable, key=lambda duration: (abs(duration - target_duration), -duration))
        current_error = abs(note.duration_beats - target_duration)
        selected_error = abs(selected - target_duration)
        if float(current_error - selected_error) < config.duration_improvement_beats:
            replacements[note.source_index] = note
            continue
        replacements[note.source_index] = replace(
            note,
            duration_beats=selected,
            voice_duration_adjusted=True,
            original_duration_beats=note.duration_beats,
            duration_change_reason="raw-release-supported-overlap-extension",
        )
        changed += 1
    return tuple(replacements[note.source_index] for note in notes), changed


def _generate_rests(
    notes: tuple[ScoreNote, ...],
    measure_count: int,
    time_signature: TimeSignature,
    pickup_beats: Fraction,
    minimum_duration: Fraction,
) -> tuple[ScoreRest, ...]:
    rests: list[ScoreRest] = []
    for measure_index in range(measure_count):
        start, end = _measure_bounds(
            measure_index,
            time_signature.measure_beats,
            pickup_beats,
        )
        for staff in (1, 2):
            staff_notes = [
                note
                for note in notes
                if note.staff == staff and note.onset_beats < end and note.offset_beats > start
            ]
            voices = sorted({note.voice for note in staff_notes}) or [1]
            for voice in voices:
                intervals = sorted(
                    (
                        max(start, note.onset_beats),
                        min(end, note.offset_beats),
                    )
                    for note in staff_notes
                    if note.voice == voice
                )
                merged: list[tuple[Fraction, Fraction]] = []
                for onset, offset in intervals:
                    if not merged or onset > merged[-1][1]:
                        merged.append((onset, offset))
                    else:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
                cursor = start
                for onset, offset in merged:
                    if onset > cursor:
                        gap = onset - cursor
                        if gap >= minimum_duration:
                            rests.append(ScoreRest(cursor, gap, staff, voice))
                    cursor = max(cursor, offset)
                if cursor < end:
                    gap = end - cursor
                    if gap >= minimum_duration:
                        rests.append(ScoreRest(cursor, gap, staff, voice))
    return tuple(sorted(rests, key=lambda rest: (rest.onset_beats, rest.staff, rest.voice)))


def _measure_bounds(
    index: int, measure_length: Fraction, pickup_beats: Fraction
) -> tuple[Fraction, Fraction]:
    if pickup_beats > 0:
        if index == 0:
            return Fraction(0), pickup_beats
        start = pickup_beats + (index - 1) * measure_length
    else:
        start = index * measure_length
    return start, start + measure_length


def _crosses_measure(
    onset: Fraction,
    offset: Fraction,
    measure_length: Fraction,
    pickup_beats: Fraction,
) -> bool:
    boundary = pickup_beats if pickup_beats > 0 else measure_length
    while boundary < offset:
        if onset < boundary:
            return True
        boundary += measure_length
    return False


def _assignment_diagnostic(
    diagnostic: EventDiagnostic,
    notes: dict[int, ScoreNote],
) -> EventDiagnostic:
    note = notes.get(diagnostic.source_index)
    if note is None:
        return diagnostic
    return replace(
        diagnostic,
        written_duration_beats=note.duration_beats,
        assigned_hand=note.hand.value if note.hand is not None else None,
        assigned_staff=note.staff,
        assigned_voice=note.voice,
        chord_id=note.chord_id,
        hand_assignment_cost=note.hand_assignment_cost,
        hand_assignment_confidence=note.hand_assignment_confidence,
        voice_assignment_cost=note.voice_assignment_cost,
        previous_continuity_cost=note.previous_continuity_cost,
        next_continuity_cost=note.next_continuity_cost,
        voice_duration_adjusted=note.voice_duration_adjusted,
        original_duration_beats=note.original_duration_beats,
        voice_identity_switched=note.voice_identity_switched,
        repeated_pitch_voice_switched=note.repeated_pitch_voice_switched,
        voice_assignment_reason=note.voice_assignment_reason,
        extra_voice_reason=note.extra_voice_reason,
        track_previous_pitch=note.track_previous_pitch,
        track_direction=note.track_direction,
        voice_continuity_score=note.voice_continuity_score,
        duration_change_reason=note.duration_change_reason,
    )
