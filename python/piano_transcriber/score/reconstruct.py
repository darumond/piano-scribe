"""Heuristic reconstruction of exact written events from acoustic note intervals."""

from __future__ import annotations

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
    for source_index, note in enumerate(result.notes):
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
            if config.adaptive_quantization:
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
        target_duration = target_offset - candidate.onset_beats
        maximum_duration = (
            same_pitch_onset - candidate.onset_beats if same_pitch_onset is not None else None
        )
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
