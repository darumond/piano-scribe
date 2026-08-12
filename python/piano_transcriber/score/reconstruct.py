"""Heuristic reconstruction of exact written events from acoustic note intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction

from piano_transcriber.score.beats import required_measures
from piano_transcriber.score.chords import group_chords
from piano_transcriber.score.quantize import QuantizationGrid, snap_to_grid, snap_written_duration
from piano_transcriber.score.tempo import beats_to_seconds, seconds_to_beats
from piano_transcriber.score.types import (
    EventDiagnostic,
    PedalInterval,
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
    minimum_note_duration_ms: float | None = None
    suspicious_note_duration_ms: float = 30.0
    merge_same_pitch_at_quantized_onset: bool = True
    time_signature: TimeSignature = field(default_factory=TimeSignature)

    def __post_init__(self) -> None:
        if not math.isfinite(self.bpm) or self.bpm <= 0.0:
            raise ValueError("BPM must be finite and positive")
        if self.maximum_quantization_error_ms < 0.0:
            raise ValueError("maximum quantization error must be non-negative")
        if self.minimum_note_duration_ms is not None and self.minimum_note_duration_ms < 0.0:
            raise ValueError("minimum note duration must be non-negative")
        if self.suspicious_note_duration_ms < 0.0:
            raise ValueError("suspicious note duration must be non-negative")


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_index: int
    note: NoteEvent
    onset_beats: Fraction
    quantization_error_seconds: float
    suspicious_reasons: tuple[str, ...]


def reconstruct_score(
    result: TranscriptionResult,
    config: ReconstructionConfig,
    *,
    pedal_intervals: tuple[PedalEvent, ...] | None = None,
) -> ReconstructedScore:
    """Create a new symbolic score without mutating the transcription result."""
    candidates: list[_Candidate] = []
    diagnostics: dict[int, EventDiagnostic] = {}
    step = config.grid.step_beats
    for source_index, note in enumerate(result.notes):
        continuous_onset = seconds_to_beats(note.onset_seconds, config.bpm)
        quantized_onset = snap_to_grid(continuous_onset, step)
        error_seconds = beats_to_seconds(quantized_onset - continuous_onset, config.bpm)
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
            )
            continue
        candidates.append(
            _Candidate(
                source_index,
                note,
                quantized_onset,
                error_seconds,
                tuple(reasons),
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
        raw_offset_beats = seconds_to_beats(note.offset_seconds, config.bpm)
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
        )

    score_notes_tuple = tuple(sorted(score_notes, key=lambda note: (note.onset_beats, note.pitch)))
    chords = group_chords(score_notes_tuple)
    end_position = max((note.offset_beats for note in score_notes_tuple), default=Fraction(0))
    raw_pedals = result.pedal_events if pedal_intervals is None else pedal_intervals
    score_pedals = tuple(
        PedalInterval(
            interval.onset_seconds,
            interval.offset_seconds,
            seconds_to_beats(interval.onset_seconds, config.bpm),
            seconds_to_beats(interval.offset_seconds, config.bpm),
        )
        for interval in raw_pedals
    )
    ordered_diagnostics = tuple(diagnostics[index] for index in sorted(diagnostics))
    return ReconstructedScore(
        bpm=config.bpm,
        time_signature=config.time_signature,
        grid_name=config.grid.value,
        grid_step_beats=step,
        notes=score_notes_tuple,
        chords=chords,
        diagnostics=ordered_diagnostics,
        pedal_intervals=score_pedals,
        measure_count=required_measures(end_position, config.time_signature),
    )


def with_uniform_chord_durations(score: ReconstructedScore) -> ReconstructedScore:
    """Optional presentation helper; not applied by the default reconstruction."""
    replacements = {
        note.source_index: replace(note, duration_beats=chord.notes[0].duration_beats)
        for chord in score.chords
        for note in chord.notes
    }
    notes = tuple(replacements[note.source_index] for note in score.notes)
    return replace(score, notes=notes, chords=group_chords(notes))
