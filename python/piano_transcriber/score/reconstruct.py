"""Heuristic reconstruction of exact written events from acoustic note intervals."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction

from piano_transcriber.score.beats import required_score_measures
from piano_transcriber.score.chords import group_chords
from piano_transcriber.score.quantize import (
    QuantizationGrid,
    choose_quantization,
    snap_to_grid,
    snap_written_duration,
)
from piano_transcriber.score.rhythm import (
    NoteDurationChoice,
    RhythmOptimizerMode,
    RhythmSelection,
    RhythmSequenceConfig,
    optimize_rhythm_sequence,
)
from piano_transcriber.score.tempo import beats_to_seconds, seconds_to_beats
from piano_transcriber.score.tracking import BeatTrack
from piano_transcriber.score.types import (
    EventDiagnostic,
    PedalInterval,
    QuantizationCandidate,
    ReconstructedScore,
    ScoreNote,
    TimeSignature,
)
from piano_transcriber.transcription.types import NoteEvent, PedalEvent, TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    bpm: float
    grid: QuantizationGrid = QuantizationGrid.SIXTEENTH
    maximum_quantization_error_ms: float = 125.0
    adaptive_quantization: bool = False
    rhythmic_complexity_cost: float = 0.35
    minimum_note_duration_ms: float | None = None
    suspicious_note_duration_ms: float = 30.0
    merge_same_pitch_at_quantized_onset: bool = True
    time_signature: TimeSignature = field(default_factory=TimeSignature)
    infer_pickup: bool = False
    pickup_beats: Fraction | None = None
    downbeat_position_beats: float | None = None
    rhythm_optimizer: RhythmOptimizerMode = RhythmOptimizerMode.LOCAL
    rhythm_sequence: RhythmSequenceConfig = field(default_factory=RhythmSequenceConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.bpm) or self.bpm <= 0.0:
            raise ValueError("BPM must be finite and positive")
        if self.maximum_quantization_error_ms < 0.0:
            raise ValueError("maximum quantization error must be non-negative")
        if self.rhythmic_complexity_cost < 0.0:
            raise ValueError("rhythmic complexity cost must be non-negative")
        if self.minimum_note_duration_ms is not None and self.minimum_note_duration_ms < 0.0:
            raise ValueError("minimum note duration must be non-negative")
        if self.suspicious_note_duration_ms < 0.0:
            raise ValueError("suspicious note duration must be non-negative")
        if self.pickup_beats is not None and (
            self.pickup_beats < 0 or self.pickup_beats >= self.time_signature.measure_beats
        ):
            raise ValueError("pickup must be non-negative and shorter than a measure")


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_index: int
    note: NoteEvent
    onset_beats: Fraction
    quantization_error_seconds: float
    suspicious_reasons: tuple[str, ...]
    continuous_onset_beats: float
    selected_subdivision: str
    quantization_candidates: tuple[QuantizationCandidate, ...]
    duration_choice: NoteDurationChoice | None = None
    rhythm_selection: RhythmSelection | None = None


def reconstruct_score(
    result: TranscriptionResult,
    config: ReconstructionConfig,
    *,
    pedal_intervals: tuple[PedalEvent, ...] | None = None,
    beat_track: BeatTrack | None = None,
) -> ReconstructedScore:
    """Create a new symbolic score without mutating the transcription result."""
    candidates: list[_Candidate] = []
    diagnostics: dict[int, EventDiagnostic] = {}
    step = (
        min(grid.step_beats for grid in QuantizationGrid)
        if config.adaptive_quantization
        else config.grid.step_beats
    )
    beat_offset = 0.0
    pickup_beats = config.pickup_beats or Fraction(0)
    first_downbeat = Fraction(0)
    if beat_track is not None:
        beat_offset, pickup_beats, first_downbeat = _score_alignment(
            result,
            beat_track,
            config,
        )
    rhythm_result = None
    rhythm_by_source: dict[int, RhythmSelection] = {}
    if config.rhythm_optimizer is RhythmOptimizerMode.SEQUENCE:
        if beat_track is None or not config.adaptive_quantization:
            raise ValueError(
                "sequence rhythm optimization requires adaptive beat-aware quantization"
            )
        rhythm_result = optimize_rhythm_sequence(
            result,
            beat_track,
            beat_offset=beat_offset,
            time_signature=config.time_signature,
            pickup_beats=pickup_beats,
            quantization_complexity_cost=config.rhythmic_complexity_cost,
            timing_tolerance_ms=config.maximum_quantization_error_ms,
            config=config.rhythm_sequence,
        )
        rhythm_by_source = rhythm_result.by_source_index()
        logger.info(
            "Rhythm sequence optimization evaluated %d transitions across %d groups in %.3f s",
            rhythm_result.evaluated_transitions,
            len(rhythm_result.selections),
            rhythm_result.elapsed_seconds,
        )
    for source_index, note in enumerate(result.notes):
        duration_choice: NoteDurationChoice | None = None
        rhythm_selection = rhythm_by_source.get(source_index)
        if beat_track is None:
            continuous_fraction = seconds_to_beats(note.onset_seconds, config.bpm)
            continuous_onset = float(continuous_fraction)
            quantized_onset = snap_to_grid(continuous_fraction, step)
            error_seconds = beats_to_seconds(quantized_onset - continuous_fraction, config.bpm)
            selected_subdivision = config.grid.value
            quantization_candidates: tuple[QuantizationCandidate, ...] = ()
        else:
            continuous_onset = max(
                0.0,
                beat_track.seconds_to_beats(note.onset_seconds) + beat_offset,
            )
            if rhythm_selection is not None:
                selected_rhythm = rhythm_selection.candidate
                quantized_onset = selected_rhythm.quantization.position_beats
                error_seconds = (
                    beat_track.beats_to_seconds(float(quantized_onset) - beat_offset)
                    - note.onset_seconds
                )
                selected_subdivision = selected_rhythm.quantization.subdivision
                quantization_candidates = tuple(
                    QuantizationCandidate(
                        candidate.quantization.subdivision,
                        candidate.quantization.position_beats,
                        beat_track.beats_to_seconds(
                            float(candidate.quantization.position_beats) - beat_offset
                        )
                        - note.onset_seconds,
                        candidate.quantization.complexity_penalty,
                        abs(
                            beat_track.beats_to_seconds(
                                float(candidate.quantization.position_beats) - beat_offset
                            )
                            - note.onset_seconds
                        )
                        * 1000
                        / config.maximum_quantization_error_ms
                        + candidate.quantization.complexity_penalty,
                    )
                    for candidate in rhythm_selection.candidates
                )
                duration_choice = next(
                    choice
                    for choice in selected_rhythm.duration_choices
                    if choice.source_index == source_index
                )
            elif config.adaptive_quantization:
                selected, quantization_candidates = choose_quantization(
                    continuous_onset,
                    note.onset_seconds,
                    beat_track,
                    complexity_cost=config.rhythmic_complexity_cost,
                    tolerance_ms=config.maximum_quantization_error_ms,
                    beat_offset=beat_offset,
                )
                quantized_onset = selected.position_beats
                error_seconds = selected.timing_error_seconds
                selected_subdivision = selected.subdivision
            else:
                quantized_onset = snap_to_grid(Fraction(str(continuous_onset)), step)
                error_seconds = (
                    beat_track.beats_to_seconds(float(quantized_onset) - beat_offset)
                    - note.onset_seconds
                )
                selected_subdivision = config.grid.value
                quantization_candidates = ()
        raw_duration_ms = (note.offset_seconds - note.onset_seconds) * 1000.0
        reasons: list[str] = []
        if raw_duration_ms < config.suspicious_note_duration_ms:
            reasons.append("raw-duration-below-suspicious-threshold")
        if abs(error_seconds) * 1000.0 > config.maximum_quantization_error_ms:
            reasons.append("quantization-error-above-tolerance")
        if (
            config.minimum_note_duration_ms is not None
            and raw_duration_ms < config.minimum_note_duration_ms
        ):
            diagnostics[source_index] = EventDiagnostic(
                source_index,
                note.pitch,
                note.onset_seconds,
                note.offset_seconds,
                quantized_onset,
                error_seconds,
                None,
                "filtered",
                tuple(reasons),
                continuous_onset_beats=continuous_onset,
                selected_subdivision=selected_subdivision,
                quantization_candidates=quantization_candidates,
                duration_candidates=(
                    duration_choice.candidates if duration_choice is not None else ()
                ),
                rhythm_group_index=(
                    rhythm_selection.group_index if rhythm_selection is not None else None
                ),
                selected_rhythm_family=(
                    rhythm_selection.candidate.family.value
                    if rhythm_selection is not None
                    else None
                ),
                rhythm_metric_position_beats=(
                    rhythm_selection.candidate.metric_position
                    if rhythm_selection is not None
                    else None
                ),
                rhythm_requires_tie=(
                    rhythm_selection.candidate.requires_tie
                    if rhythm_selection is not None
                    else False
                ),
                rhythm_group_timing_error_seconds=(
                    rhythm_selection.candidate.quantization.timing_error_seconds
                    if rhythm_selection is not None
                    else None
                ),
                rhythm_complexity_cost=(
                    rhythm_selection.candidate.complexity_cost
                    if rhythm_selection is not None
                    else None
                ),
                rhythm_local_cost=(
                    rhythm_selection.candidate.local_cost if rhythm_selection is not None else None
                ),
                rhythm_transition_cost=(
                    rhythm_selection.transition_cost if rhythm_selection is not None else None
                ),
                rhythm_cumulative_score=(
                    rhythm_selection.cumulative_score if rhythm_selection is not None else None
                ),
                local_best_subdivision=(
                    rhythm_selection.local_best.subdivision
                    if rhythm_selection is not None
                    else None
                ),
                local_best_position_beats=(
                    rhythm_selection.local_best.position_beats
                    if rhythm_selection is not None
                    else None
                ),
                optimizer_selection_reason=(
                    rhythm_selection.reason if rhythm_selection is not None else None
                ),
                optimizer_changed_local_choice=(
                    rhythm_selection.differs_from_local if rhythm_selection is not None else False
                ),
            )
            continue
        candidates.append(
            _Candidate(
                source_index,
                note,
                quantized_onset,
                error_seconds,
                tuple(reasons),
                continuous_onset,
                selected_subdivision,
                quantization_candidates,
                duration_choice,
                rhythm_selection,
            )
        )

    retained: list[_Candidate] = []
    if config.merge_same_pitch_at_quantized_onset:
        grouped: dict[tuple[Fraction, int], list[_Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault((candidate.onset_beats, candidate.note.pitch), []).append(candidate)
        for same_event in grouped.values():
            primary = max(
                same_event,
                key=lambda item: (
                    item.note.confidence,
                    item.note.velocity,
                    item.note.offset_seconds - item.note.onset_seconds,
                    -item.source_index,
                ),
            )
            retained.append(primary)
            for duplicate in same_event:
                if duplicate is primary:
                    continue
                diagnostics[duplicate.source_index] = EventDiagnostic(
                    duplicate.source_index,
                    duplicate.note.pitch,
                    duplicate.note.onset_seconds,
                    duplicate.note.offset_seconds,
                    duplicate.onset_beats,
                    duplicate.quantization_error_seconds,
                    None,
                    "merged",
                    duplicate.suspicious_reasons,
                    primary.source_index,
                    continuous_onset_beats=duplicate.continuous_onset_beats,
                    selected_subdivision=duplicate.selected_subdivision,
                    quantization_candidates=duplicate.quantization_candidates,
                    duration_candidates=(
                        duplicate.duration_choice.candidates
                        if duplicate.duration_choice is not None
                        else ()
                    ),
                    rhythm_group_index=(
                        duplicate.rhythm_selection.group_index
                        if duplicate.rhythm_selection is not None
                        else None
                    ),
                    selected_rhythm_family=(
                        duplicate.rhythm_selection.candidate.family.value
                        if duplicate.rhythm_selection is not None
                        else None
                    ),
                    rhythm_metric_position_beats=(
                        duplicate.rhythm_selection.candidate.metric_position
                        if duplicate.rhythm_selection is not None
                        else None
                    ),
                    rhythm_requires_tie=(
                        duplicate.rhythm_selection.candidate.requires_tie
                        if duplicate.rhythm_selection is not None
                        else False
                    ),
                    rhythm_group_timing_error_seconds=(
                        duplicate.rhythm_selection.candidate.quantization.timing_error_seconds
                        if duplicate.rhythm_selection is not None
                        else None
                    ),
                    rhythm_complexity_cost=(
                        duplicate.rhythm_selection.candidate.complexity_cost
                        if duplicate.rhythm_selection is not None
                        else None
                    ),
                )
    else:
        retained = candidates
    retained.sort(key=lambda item: (item.onset_beats, item.note.pitch, item.source_index))

    distinct_onsets = sorted({candidate.onset_beats for candidate in retained})
    next_onset_by_onset = {
        onset: distinct_onsets[index + 1] if index + 1 < len(distinct_onsets) else None
        for index, onset in enumerate(distinct_onsets)
    }
    by_pitch: dict[int, list[_Candidate]] = {}
    for candidate in retained:
        by_pitch.setdefault(candidate.note.pitch, []).append(candidate)
    next_same_pitch: dict[int, Fraction | None] = {}
    for pitch_candidates in by_pitch.values():
        for index, candidate in enumerate(pitch_candidates):
            next_same_pitch[candidate.source_index] = (
                pitch_candidates[index + 1].onset_beats
                if index + 1 < len(pitch_candidates)
                else None
            )

    score_notes: list[ScoreNote] = []
    for candidate in retained:
        note = candidate.note
        raw_offset_beats = (
            seconds_to_beats(note.offset_seconds, config.bpm)
            if beat_track is None
            else Fraction(str(beat_track.seconds_to_beats(note.offset_seconds) + beat_offset))
        )
        target_offset = raw_offset_beats
        pedal_shortened = False
        next_group = next_onset_by_onset[candidate.onset_beats]
        if note.pedal is True and next_group is not None and next_group < target_offset:
            target_offset = next_group
            pedal_shortened = True
        same_pitch_onset = next_same_pitch[candidate.source_index]
        if same_pitch_onset is not None and same_pitch_onset < target_offset:
            target_offset = same_pitch_onset
        maximum_duration = (
            same_pitch_onset - candidate.onset_beats if same_pitch_onset is not None else None
        )
        if candidate.duration_choice is not None:
            duration_options = candidate.duration_choice.candidates
            if maximum_duration is not None:
                bounded = tuple(
                    item for item in duration_options if item.duration_beats <= maximum_duration
                )
                if bounded:
                    duration_options = bounded
            selected_duration = min(duration_options, key=lambda item: item.total_score)
            written_duration = selected_duration.duration_beats
            pedal_shortened = candidate.duration_choice.pedal_shortened
        else:
            target_duration = target_offset - candidate.onset_beats
            written_duration = snap_written_duration(target_duration, maximum=maximum_duration)
        score_note = ScoreNote(
            source_index=candidate.source_index,
            pitch=note.pitch,
            velocity=note.velocity,
            confidence=note.confidence,
            raw_onset_seconds=note.onset_seconds,
            raw_offset_seconds=note.offset_seconds,
            onset_beats=candidate.onset_beats,
            duration_beats=written_duration,
            quantization_error_seconds=candidate.quantization_error_seconds,
            pedal=note.pedal,
            suspicious_reasons=candidate.suspicious_reasons,
            pedal_duration_shortened=pedal_shortened,
        )
        score_notes.append(score_note)
        diagnostics[candidate.source_index] = EventDiagnostic(
            candidate.source_index,
            note.pitch,
            note.onset_seconds,
            note.offset_seconds,
            candidate.onset_beats,
            candidate.quantization_error_seconds,
            written_duration,
            "quantized",
            candidate.suspicious_reasons,
            pedal_duration_shortened=pedal_shortened,
            continuous_onset_beats=candidate.continuous_onset_beats,
            selected_subdivision=candidate.selected_subdivision,
            quantization_candidates=candidate.quantization_candidates,
            duration_candidates=(
                candidate.duration_choice.candidates
                if candidate.duration_choice is not None
                else ()
            ),
            rhythm_group_index=(
                candidate.rhythm_selection.group_index
                if candidate.rhythm_selection is not None
                else None
            ),
            selected_rhythm_family=(
                candidate.rhythm_selection.candidate.family.value
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_metric_position_beats=(
                candidate.rhythm_selection.candidate.metric_position
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_requires_tie=(
                candidate.rhythm_selection.candidate.requires_tie
                if candidate.rhythm_selection is not None
                else False
            ),
            rhythm_group_timing_error_seconds=(
                candidate.rhythm_selection.candidate.quantization.timing_error_seconds
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_complexity_cost=(
                candidate.rhythm_selection.candidate.complexity_cost
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_local_cost=(
                candidate.rhythm_selection.candidate.local_cost
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_transition_cost=(
                candidate.rhythm_selection.transition_cost
                if candidate.rhythm_selection is not None
                else None
            ),
            rhythm_cumulative_score=(
                candidate.rhythm_selection.cumulative_score
                if candidate.rhythm_selection is not None
                else None
            ),
            local_best_subdivision=(
                candidate.rhythm_selection.local_best.subdivision
                if candidate.rhythm_selection is not None
                else None
            ),
            local_best_position_beats=(
                candidate.rhythm_selection.local_best.position_beats
                if candidate.rhythm_selection is not None
                else None
            ),
            optimizer_selection_reason=(
                candidate.rhythm_selection.reason
                if candidate.rhythm_selection is not None
                else None
            ),
            optimizer_changed_local_choice=(
                candidate.rhythm_selection.differs_from_local
                if candidate.rhythm_selection is not None
                else False
            ),
        )

    score_notes_tuple = tuple(sorted(score_notes, key=lambda note: (note.onset_beats, note.pitch)))
    chords = group_chords(score_notes_tuple)
    end_position = max((note.offset_beats for note in score_notes_tuple), default=Fraction(0))
    raw_pedals = result.pedal_events if pedal_intervals is None else pedal_intervals
    score_pedals_list: list[PedalInterval] = []
    for interval in raw_pedals:
        pedal_onset = (
            seconds_to_beats(interval.onset_seconds, config.bpm)
            if beat_track is None
            else Fraction(str(beat_track.seconds_to_beats(interval.onset_seconds) + beat_offset))
        )
        pedal_offset = (
            seconds_to_beats(interval.offset_seconds, config.bpm)
            if beat_track is None
            else Fraction(str(beat_track.seconds_to_beats(interval.offset_seconds) + beat_offset))
        )
        pedal_onset = max(Fraction(0), pedal_onset)
        if pedal_offset > pedal_onset:
            score_pedals_list.append(
                PedalInterval(
                    interval.onset_seconds,
                    interval.offset_seconds,
                    pedal_onset,
                    pedal_offset,
                )
            )
    score_pedals = tuple(score_pedals_list)
    ordered_diagnostics = tuple(diagnostics[index] for index in sorted(diagnostics))
    return ReconstructedScore(
        bpm=config.bpm,
        time_signature=config.time_signature,
        grid_name="adaptive" if config.adaptive_quantization else config.grid.value,
        grid_step_beats=step,
        notes=score_notes_tuple,
        chords=chords,
        diagnostics=ordered_diagnostics,
        pedal_intervals=score_pedals,
        measure_count=required_score_measures(
            end_position,
            config.time_signature,
            pickup_beats,
        ),
        beat_track=beat_track,
        pickup_beats=pickup_beats,
        first_full_downbeat_beats=first_downbeat,
        beat_position_offset=beat_offset,
        rhythm_optimizer=config.rhythm_optimizer.value,
        rhythm_optimizer_seconds=(
            rhythm_result.elapsed_seconds if rhythm_result is not None else 0.0
        ),
        rhythm_evaluated_transitions=(
            rhythm_result.evaluated_transitions if rhythm_result is not None else 0
        ),
    )


def _score_alignment(
    result: TranscriptionResult,
    beat_track: BeatTrack,
    config: ReconstructionConfig,
) -> tuple[float, Fraction, Fraction]:
    if not config.infer_pickup and config.pickup_beats is None:
        padding = beat_track.measure_padding_beats
        return padding, Fraction(0), Fraction(0)
    first_event = min(
        (beat_track.seconds_to_beats(note.onset_seconds) for note in result.notes),
        default=0.0,
    )
    measure_length = float(config.time_signature.measure_beats)
    phase = (
        config.downbeat_position_beats
        if config.downbeat_position_beats is not None
        else float(beat_track.downbeat_phase)
    )
    cycles = math.ceil((first_event - phase) / measure_length)
    next_downbeat = phase + cycles * measure_length
    if next_downbeat < first_event - 1e-6:
        next_downbeat += measure_length
    if config.pickup_beats is not None:
        pickup = config.pickup_beats
    elif abs(next_downbeat - first_event) <= 1e-6:
        pickup = Fraction(0)
    else:
        raw_pickup = max(0.0, min(measure_length - 0.125, next_downbeat - first_event))
        pickup = snap_to_grid(Fraction(str(raw_pickup)), Fraction(1, 8))
        if pickup >= config.time_signature.measure_beats:
            pickup = Fraction(0)
    origin = next_downbeat - float(pickup)
    return -origin, pickup, pickup


def with_uniform_chord_durations(score: ReconstructedScore) -> ReconstructedScore:
    """Optional presentation helper; not applied by the default reconstruction."""
    replacements = {
        note.source_index: replace(note, duration_beats=chord.notes[0].duration_beats)
        for chord in score.chords
        for note in chord.notes
    }
    notes = tuple(replacements[note.source_index] for note in score.notes)
    return replace(score, notes=notes, chords=group_chords(notes))
