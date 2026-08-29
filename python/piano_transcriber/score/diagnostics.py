"""JSON/TSV interchange and observability for score reconstruction."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from fractions import Fraction
from itertools import groupby, pairwise
from pathlib import Path
from typing import Any, cast

from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.meter import JointMeterResult, MeterHypothesis
from piano_transcriber.score.types import ReconstructedScore
from piano_transcriber.transcription.types import NoteEvent, PedalEvent, TranscriptionResult

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def duration_name(value: Fraction) -> str:
    return {
        Fraction(1, 24): "sixty-fourth-triplet",
        Fraction(4): "whole",
        Fraction(3): "dotted-half",
        Fraction(2): "half",
        Fraction(3, 2): "dotted-quarter",
        Fraction(4, 3): "half-triplet",
        Fraction(1): "quarter",
        Fraction(3, 4): "dotted-eighth",
        Fraction(2, 3): "quarter-triplet",
        Fraction(1, 2): "eighth",
        Fraction(3, 8): "dotted-sixteenth",
        Fraction(1, 3): "eighth-triplet",
        Fraction(1, 4): "sixteenth",
        Fraction(3, 16): "dotted-thirty-second",
        Fraction(1, 6): "sixteenth-triplet",
        Fraction(1, 8): "thirty-second",
    }.get(value, f"{fraction_text(value)}-beats")


def load_transcription_json(
    path: str | Path,
) -> TranscriptionResult:
    data = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    raw_notes = cast(list[Mapping[str, Any]], data.get("notes", []))
    notes = tuple(
        NoteEvent(
            pitch=int(note["pitch"]),
            onset_seconds=float(note["onset_seconds"]),
            offset_seconds=float(note["offset_seconds"]),
            velocity=int(note.get("velocity", 80)),
            confidence=float(note.get("confidence", 1.0)),
            pedal=cast(bool | None, note.get("pedal")),
        )
        for note in raw_notes
    )
    inferred_duration = max((note.offset_seconds for note in notes), default=0.0)
    duration = float(data.get("audio_duration_seconds", inferred_duration))
    raw_pedals = cast(list[Mapping[str, Any]], data.get("pedal_events", []))
    pedals = tuple(
        PedalEvent(
            float(pedal.get("onset_time", pedal.get("onset_seconds"))),
            float(pedal.get("offset_time", pedal.get("offset_seconds"))),
        )
        for pedal in raw_pedals
    )
    return TranscriptionResult(
        notes,
        str(data.get("model_name", "imported")),
        duration,
        pedals,
    )


def score_diagnostics(score: ReconstructedScore) -> dict[str, object]:
    score_notes = {note.source_index: note for note in score.notes}
    chord_size = {
        note.source_index: len(chord.notes) for chord in score.chords for note in chord.notes
    }
    events: list[dict[str, object]] = []
    for diagnostic in score.diagnostics:
        measure_index, beat, _measure_length = locate_score_measure(
            diagnostic.quantized_onset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        note = score_notes.get(diagnostic.source_index)
        events.append(
            {
                "source_index": diagnostic.source_index,
                "pitch": diagnostic.pitch,
                "note_name": note_name(diagnostic.pitch),
                "raw_onset_seconds": diagnostic.raw_onset_seconds,
                "raw_offset_seconds": diagnostic.raw_offset_seconds,
                "raw_duration_seconds": diagnostic.raw_offset_seconds
                - diagnostic.raw_onset_seconds,
                "quantized_onset_beats": fraction_text(diagnostic.quantized_onset_beats),
                "continuous_onset_beats": diagnostic.continuous_onset_beats,
                "quantization_error_ms": diagnostic.quantization_error_seconds * 1000.0,
                "selected_subdivision": diagnostic.selected_subdivision,
                "quantization_candidates": [
                    {
                        "subdivision": candidate.subdivision,
                        "position_beats": fraction_text(candidate.position_beats),
                        "timing_error_ms": candidate.timing_error_seconds * 1000.0,
                        "complexity_penalty": candidate.complexity_penalty,
                        "total_score": candidate.total_score,
                    }
                    for candidate in diagnostic.quantization_candidates
                ],
                "duration_candidates": [
                    {
                        "duration_beats": fraction_text(candidate.duration_beats),
                        "timing_error_ms": candidate.timing_error_seconds * 1000.0,
                        "complexity_penalty": candidate.complexity_penalty,
                        "requires_tie": candidate.requires_tie,
                        "tiny_tie_fragment": candidate.tiny_tie_fragment,
                        "dotted_micro_value": candidate.dotted_micro_value,
                        "unusual_short_value": candidate.unusual_short_value,
                        "total_score": candidate.total_score,
                    }
                    for candidate in diagnostic.duration_candidates
                ],
                "written_duration_beats": fraction_text(diagnostic.written_duration_beats),
                "written_duration_name": (
                    duration_name(diagnostic.written_duration_beats)
                    if diagnostic.written_duration_beats is not None
                    else None
                ),
                "measure": measure_index if score.pickup_beats > 0 else measure_index + 1,
                "beat_in_measure": fraction_text(beat),
                "chord_size": chord_size.get(diagnostic.source_index),
                "action": diagnostic.action,
                "suspicious_reasons": list(diagnostic.suspicious_reasons),
                "merged_into_source_index": diagnostic.merged_into_source_index,
                "pedal_duration_shortened": diagnostic.pedal_duration_shortened,
                "velocity": note.velocity if note is not None else None,
                "confidence": note.confidence if note is not None else None,
                "rhythm_group_index": diagnostic.rhythm_group_index,
                "selected_rhythm_family": diagnostic.selected_rhythm_family,
                "rhythm_metric_position_beats": fraction_text(
                    diagnostic.rhythm_metric_position_beats
                ),
                "rhythm_requires_tie": diagnostic.rhythm_requires_tie,
                "rhythm_group_timing_error_ms": (
                    diagnostic.rhythm_group_timing_error_seconds * 1000.0
                    if diagnostic.rhythm_group_timing_error_seconds is not None
                    else None
                ),
                "rhythm_complexity_cost": diagnostic.rhythm_complexity_cost,
                "rhythm_local_cost": diagnostic.rhythm_local_cost,
                "rhythm_transition_cost": diagnostic.rhythm_transition_cost,
                "rhythm_cumulative_score": diagnostic.rhythm_cumulative_score,
                "local_best_subdivision": diagnostic.local_best_subdivision,
                "local_best_position_beats": fraction_text(diagnostic.local_best_position_beats),
                "optimizer_selection_reason": diagnostic.optimizer_selection_reason,
                "optimizer_changed_local_choice": diagnostic.optimizer_changed_local_choice,
                "assigned_hand": diagnostic.assigned_hand,
                "assigned_staff": diagnostic.assigned_staff,
                "assigned_voice": diagnostic.assigned_voice,
                "chord_id": diagnostic.chord_id,
                "hand_assignment_cost": diagnostic.hand_assignment_cost,
                "hand_assignment_confidence": diagnostic.hand_assignment_confidence,
                "voice_assignment_cost": diagnostic.voice_assignment_cost,
                "previous_continuity_cost": diagnostic.previous_continuity_cost,
                "next_continuity_cost": diagnostic.next_continuity_cost,
                "voice_duration_adjusted": diagnostic.voice_duration_adjusted,
                "original_duration_beats": fraction_text(diagnostic.original_duration_beats),
                "duration_change_reason": diagnostic.duration_change_reason,
                "voice_identity_switched": diagnostic.voice_identity_switched,
                "repeated_pitch_voice_switched": diagnostic.repeated_pitch_voice_switched,
                "voice_assignment_reason": diagnostic.voice_assignment_reason,
                "extra_voice_reason": diagnostic.extra_voice_reason,
                "track_previous_pitch": diagnostic.track_previous_pitch,
                "track_direction": diagnostic.track_direction,
                "voice_continuity_score": diagnostic.voice_continuity_score,
                "tie_across_measure": note.tie_across_measure if note is not None else False,
            }
        )
    action_counts = Counter(item.action for item in score.diagnostics)
    rhythms = Counter(duration_name(note.duration_beats) for note in score.notes)
    errors = sorted(abs(item.quantization_error_seconds) * 1000.0 for item in score.diagnostics)
    beat_summary: dict[str, object] | None = None
    if score.beat_track is not None:
        minimum_bpm, maximum_bpm = score.beat_track.bpm_range
        beat_summary = {
            "beat_count": len(score.beat_track.beats),
            "median_bpm": score.beat_track.median_bpm,
            "minimum_bpm": minimum_bpm,
            "maximum_bpm": maximum_bpm,
            "downbeat_phase": score.beat_track.downbeat_phase,
            "downbeat_confidence": score.beat_track.downbeat_confidence,
        }
    return {
        "bpm": score.bpm,
        "time_signature": str(score.time_signature),
        "quantization_grid": score.grid_name,
        "grid_step_beats": fraction_text(score.grid_step_beats),
        "written_note_count": len(score.notes),
        "chord_group_count": len(score.chords),
        "multi_note_chord_count": sum(len(chord.notes) > 1 for chord in score.chords),
        "measure_count": score.measure_count,
        "pickup_beats": fraction_text(score.pickup_beats),
        "first_full_downbeat_beats": fraction_text(score.first_full_downbeat_beats),
        "rhythmic_values": dict(sorted(rhythms.items())),
        "actions": dict(sorted(action_counts.items())),
        "pedal_event_count": len(score.pedal_intervals),
        "pedal_extended_durations_shortened": sum(
            note.pedal_duration_shortened for note in score.notes
        ),
        "quantization_error_ms": {
            "minimum": errors[0] if errors else 0.0,
            "median": errors[len(errors) // 2] if errors else 0.0,
            "p95": errors[min(len(errors) - 1, int(len(errors) * 0.95))] if errors else 0.0,
            "maximum": errors[-1] if errors else 0.0,
        },
        "beat_tracking": beat_summary,
        "rhythm_optimization": rhythm_optimization_data(score),
        "piano_layout": piano_layout_data(score),
        "engraving": engraving_data(score),
        "events": events,
    }


def piano_layout_data(score: ReconstructedScore) -> dict[str, object]:
    """Summarize staff, hand, voice, chord, and continuity assignments."""
    hand_counts = Counter(
        note.hand.value if note.hand is not None else "unassigned" for note in score.notes
    )
    staff_counts = Counter(note.staff for note in score.notes)
    voice_counts = {
        str(staff): len({note.voice for note in score.notes if note.staff == staff})
        for staff in (1, 2)
    }
    hand_crossings = 0
    split_chords = 0
    maximum_spans = {"left": 0, "right": 0}
    for chord in score.chords:
        hands = {note.hand for note in chord.notes if note.hand is not None}
        if len(hands) > 1:
            split_chords += 1
        left = [
            note.pitch
            for note in chord.notes
            if note.hand is not None and note.hand.value == "left"
        ]
        right = [
            note.pitch
            for note in chord.notes
            if note.hand is not None and note.hand.value == "right"
        ]
        if left:
            maximum_spans["left"] = max(maximum_spans["left"], max(left) - min(left))
        if right:
            maximum_spans["right"] = max(maximum_spans["right"], max(right) - min(right))
        if left and right and max(left) > min(right):
            hand_crossings += 1

    melodic_intervals: dict[str, float] = {}
    voice_switches = 0
    for staff in (1, 2):
        for voice in sorted({note.voice for note in score.notes if note.staff == staff}):
            representatives = []
            voice_notes = sorted(
                (note for note in score.notes if note.staff == staff and note.voice == voice),
                key=lambda note: (note.onset_beats, note.pitch),
            )
            for _onset, group in groupby(voice_notes, key=lambda note: note.onset_beats):
                representatives.append(statistics.mean(note.pitch for note in group))
            intervals = [abs(later - earlier) for earlier, later in pairwise(representatives)]
            melodic_intervals[f"staff_{staff}_voice_{voice}"] = (
                statistics.mean(intervals) if intervals else 0.0
            )
        by_pitch: dict[int, list[Any]] = {}
        for note in score.notes:
            if note.staff == staff:
                by_pitch.setdefault(note.pitch, []).append(note)
        voice_switches += sum(
            earlier.voice != later.voice
            for pitch_notes in by_pitch.values()
            for earlier, later in pairwise(
                sorted(pitch_notes, key=lambda note: (note.onset_beats, note.source_index))
            )
        )

    staff_crossings = sum(
        (note.hand is not None)
        and (
            (note.hand.value == "right" and note.staff != 1)
            or (note.hand.value == "left" and note.staff != 2)
        )
        for note in score.notes
    )
    return {
        "mode": score.piano_layout,
        "notes_by_hand": dict(sorted(hand_counts.items())),
        "notes_by_staff": {str(key): value for key, value in sorted(staff_counts.items())},
        "voices_per_staff": voice_counts,
        "voice_count": sum(voice_counts.values()),
        "voice_switches": voice_switches,
        "voice_identity_switches": sum(note.voice_identity_switched for note in score.notes),
        "repeated_pitch_voice_switches": sum(
            note.repeated_pitch_voice_switched for note in score.notes
        ),
        "extra_voice_reasons": dict(
            sorted(
                Counter(
                    note.extra_voice_reason
                    for note in score.notes
                    if note.extra_voice_reason is not None
                ).items()
            )
        ),
        "hand_crossings": hand_crossings,
        "staff_crossings": staff_crossings,
        "average_melodic_interval_semitones": melodic_intervals,
        "maximum_simultaneous_span_semitones": maximum_spans,
        "chord_groups_split_between_hands": split_chords,
        "voice_duration_changes": score.voice_duration_changes,
        "explicit_rest_count": len(score.rests),
        "hand_optimizer_seconds": score.hand_optimizer_seconds,
        "hand_evaluated_transitions": score.hand_evaluated_transitions,
        "voice_optimizer_seconds": score.voice_optimizer_seconds,
        "voice_stability_seconds": score.voice_stability_seconds,
        "voice_evaluated_transitions": score.voice_evaluated_transitions,
    }


def engraving_data(score: ReconstructedScore) -> dict[str, object]:
    """Summarize derived rest, beam, tuplet, span, and staff-placement evidence."""
    original_rest_decisions = [item for item in score.rest_decisions if item.action != "merge"]
    fragments_by_stream: Counter[str] = Counter()
    for rest in score.rests:
        measure, _local, _length = locate_score_measure(
            rest.onset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        fragments_by_stream[f"measure_{measure}_staff_{rest.staff}_voice_{rest.voice}"] += 1
    ledger_by_staff = {
        str(staff): {
            "event_count": len(values),
            "maximum_estimated_ledger_lines": max(
                (item.estimated_ledger_lines for item in values), default=0
            ),
        }
        for staff in (1, 2)
        if (values := [item for item in score.ledger_line_diagnostics if item.staff == staff])
    }
    measure_complexity = []
    for measure_index in range(score.measure_count):
        notes = [
            note
            for note in score.notes
            if locate_score_measure(
                note.onset_beats,
                score.time_signature,
                score.pickup_beats,
            )[0]
            == measure_index
        ]
        rests = [
            rest
            for rest in score.rests
            if locate_score_measure(
                rest.onset_beats,
                score.time_signature,
                score.pickup_beats,
            )[0]
            == measure_index
        ]
        beam_count = sum(item.measure_index == measure_index for item in score.beam_annotations)
        tuplet_count = sum(item.measure_index == measure_index for item in score.tuplet_annotations)
        voices = len({(note.staff, note.voice) for note in notes})
        measure_complexity.append(
            {
                "measure_index": measure_index,
                "note_attacks": len(notes),
                "logical_rests": len(rests),
                "beam_annotations": beam_count,
                "tuplet_marks": tuplet_count,
                "active_voice_streams": voices,
                "complexity_score": len(notes)
                + len(rests)
                + beam_count
                + 2 * tuplet_count
                + voices,
            }
        )
    return {
        "mode": score.engraving_mode,
        "timings_seconds": {
            "voice_stability": score.voice_stability_seconds,
            "rest_optimization": score.rest_optimizer_seconds,
            "annotation": score.engraving_annotation_seconds,
            "total_engraving": score.engraving_total_seconds,
        },
        "rests": {
            "logical_before": len(original_rest_decisions),
            "logical_after": len(score.rests),
            "total_duration_beats_before": fraction_text(
                sum((item.duration_beats for item in original_rest_decisions), Fraction(0))
            ),
            "total_duration_beats_after": fraction_text(
                sum((item.duration_beats for item in score.rests), Fraction(0))
            ),
            "fragments_before": score.rest_fragments_before,
            "fragments_after": score.rest_fragments_after,
            "merged": score.merged_rest_count,
            "actions": dict(sorted(Counter(item.action for item in score.rest_decisions).items())),
            "fragments_per_measure_voice": dict(sorted(fragments_by_stream.items())),
        },
        "beam_groups": len({item.group_id for item in score.beam_annotations}),
        "beam_annotations_by_level": dict(
            sorted(Counter(str(item.level) for item in score.beam_annotations).items())
        ),
        "tuplet_groups": len({item.group_id for item in score.tuplet_annotations}),
        "hand_span_flags": dict(
            sorted(
                Counter(
                    flag
                    for diagnostic in score.hand_span_diagnostics
                    for flag in diagnostic.threshold_flags
                ).items()
            )
        ),
        "hand_span_diagnostics": [
            {
                "hand": item.hand,
                "onset_beats": fraction_text(item.onset_beats),
                "attack_span_semitones": item.attack_span_semitones,
                "overlap_span_semitones": item.overlap_span_semitones,
                "threshold_flags": list(item.threshold_flags),
                "cause": item.cause,
            }
            for item in score.hand_span_diagnostics
        ],
        "cross_staff_candidate_count": len(score.cross_staff_candidates),
        "cross_staff_candidates": [
            {
                "source_index": item.source_index,
                "hand": item.hand,
                "pitch": item.pitch,
                "onset_beats": fraction_text(item.onset_beats),
                "reason": item.reason,
            }
            for item in score.cross_staff_candidates
        ],
        "ledger_line_diagnostic_count": len(score.ledger_line_diagnostics),
        "ledger_line_pressure_by_staff": ledger_by_staff,
        "ledger_line_diagnostics": [
            {
                "source_index": item.source_index,
                "staff": item.staff,
                "pitch": item.pitch,
                "estimated_ledger_lines": item.estimated_ledger_lines,
            }
            for item in score.ledger_line_diagnostics
        ],
        "measure_complexity": sorted(
            measure_complexity,
            key=lambda item: (-int(item["complexity_score"]), int(item["measure_index"])),
        ),
    }


def rhythm_optimization_data(score: ReconstructedScore) -> dict[str, object]:
    grouped = {
        item.rhythm_group_index: item
        for item in score.diagnostics
        if item.action == "quantized" and item.rhythm_group_index is not None
    }
    ordered = [grouped[index] for index in sorted(grouped)]
    families = [item.selected_rhythm_family or "" for item in ordered]
    ternary = ["triplet" in family for family in families]
    group_errors = sorted(
        abs(item.rhythm_group_timing_error_seconds) * 1000.0
        for item in ordered
        if item.rhythm_group_timing_error_seconds is not None
    )
    family_switches = sum(later != earlier for earlier, later in pairwise(families))
    ternary_switches = sum(later != earlier for earlier, later in pairwise(ternary))
    isolated_triplets = sum(
        is_triplet
        and (index == 0 or not ternary[index - 1])
        and (index + 1 == len(ternary) or not ternary[index + 1])
        for index, is_triplet in enumerate(ternary)
    )
    tie_count = sum(
        _measure_tie_count(
            note.onset_beats,
            note.duration_beats,
            score.time_signature.measure_beats,
            score.pickup_beats,
        )
        for note in score.notes
    )
    sequence_score = (ordered[-1].rhythm_cumulative_score or 0.0) / len(ordered) if ordered else 0.0
    return {
        "mode": score.rhythm_optimizer,
        "elapsed_seconds": score.rhythm_optimizer_seconds,
        "evaluated_transitions": score.rhythm_evaluated_transitions,
        "group_count": len(ordered),
        "rhythmic_family_switches": family_switches,
        "straight_triplet_switches": ternary_switches,
        "isolated_triplet_events": isolated_triplets,
        "isolated_thirty_second_values": _isolated_duration_groups(score, Fraction(1, 8)),
        "isolated_dotted_sixteenth_values": _isolated_duration_groups(score, Fraction(3, 8)),
        "tie_count": tie_count,
        "notation_complexity_score": _written_notation_complexity(score),
        "sequence_score_per_group": sequence_score,
        "events_changed_from_local": sum(item.optimizer_changed_local_choice for item in ordered),
        "timing_error_ms": {
            "minimum": group_errors[0] if group_errors else 0.0,
            "median": group_errors[len(group_errors) // 2] if group_errors else 0.0,
            "p95": (
                group_errors[min(len(group_errors) - 1, int(len(group_errors) * 0.95))]
                if group_errors
                else 0.0
            ),
            "maximum": group_errors[-1] if group_errors else 0.0,
        },
    }


def _written_notation_complexity(score: ReconstructedScore) -> float:
    costs = {
        "whole": 0.0,
        "dotted-half": 0.03,
        "half": 0.0,
        "dotted-quarter": 0.05,
        "quarter": 0.0,
        "half-triplet": 0.25,
        "dotted-eighth": 0.08,
        "eighth": 0.1,
        "quarter-triplet": 0.3,
        "dotted-sixteenth": 0.7,
        "sixteenth": 0.35,
        "eighth-triplet": 0.45,
        "dotted-thirty-second": 1.0,
        "thirty-second": 0.85,
        "sixteenth-triplet": 0.7,
    }
    values = [costs.get(duration_name(note.duration_beats), 1.0) for note in score.notes]
    return statistics.mean(values) if values else 0.0


def _measure_tie_count(
    onset: Fraction,
    duration: Fraction,
    measure_length: Fraction,
    pickup_beats: Fraction,
) -> int:
    boundary = pickup_beats if pickup_beats > 0 else measure_length
    offset = onset + duration
    count = 0
    while boundary < offset:
        count += onset < boundary
        boundary += measure_length
    return count


def _isolated_duration_groups(score: ReconstructedScore, duration: Fraction) -> int:
    present = [
        any(note.duration_beats == duration for note in chord.notes) for chord in score.chords
    ]
    return sum(
        value
        and (index == 0 or not present[index - 1])
        and (index + 1 == len(present) or not present[index + 1])
        for index, value in enumerate(present)
    )


def write_diagnostics_json(score: ReconstructedScore, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(score_diagnostics(score), indent=2), encoding="utf-8")
    return output


def write_diagnostics_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    events = cast(list[dict[str, object]], score_diagnostics(score)["events"])
    fieldnames = list(events[0]) if events else ["source_index"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["suspicious_reasons"] = ",".join(cast(list[str], row["suspicious_reasons"]))
            row["quantization_candidates"] = json.dumps(row["quantization_candidates"])
            row["duration_candidates"] = json.dumps(row["duration_candidates"])
            writer.writerow(row)
    return output


def write_rhythm_path_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    if score.rhythm_optimizer != "sequence":
        raise ValueError("rhythm path diagnostics require sequence optimization")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[Any]] = {}
    for diagnostic in score.diagnostics:
        if diagnostic.action == "quantized" and diagnostic.rhythm_group_index is not None:
            grouped.setdefault(diagnostic.rhythm_group_index, []).append(diagnostic)
    rows: list[dict[str, object]] = []
    for group_index in sorted(grouped):
        diagnostics = grouped[group_index]
        first = diagnostics[0]
        rows.append(
            {
                "group_index": group_index,
                "source_indices": ",".join(str(item.source_index) for item in diagnostics),
                "raw_onset_seconds": min(item.raw_onset_seconds for item in diagnostics),
                "selected_position_beats": fraction_text(first.quantized_onset_beats),
                "selected_subdivision": first.selected_subdivision,
                "selected_family": first.selected_rhythm_family,
                "metric_position_beats": fraction_text(first.rhythm_metric_position_beats),
                "requires_tie": first.rhythm_requires_tie,
                "timing_error_ms": (
                    first.rhythm_group_timing_error_seconds * 1000.0
                    if first.rhythm_group_timing_error_seconds is not None
                    else None
                ),
                "local_complexity_cost": first.rhythm_complexity_cost,
                "local_path_cost": first.rhythm_local_cost,
                "transition_cost": first.rhythm_transition_cost,
                "cumulative_score": first.rhythm_cumulative_score,
                "local_best_position_beats": fraction_text(first.local_best_position_beats),
                "local_best_subdivision": first.local_best_subdivision,
                "changed_from_local": first.optimizer_changed_local_choice,
                "selection_reason": first.optimizer_selection_reason,
                "selected_durations": ",".join(
                    fraction_text(item.written_duration_beats) or "" for item in diagnostics
                ),
            }
        )
    fieldnames = list(rows[0]) if rows else ["group_index"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_staff_assignment_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    if score.piano_layout == "none":
        raise ValueError("staff assignment diagnostics require piano layout optimization")
    rows: list[dict[str, object]] = [
        {
            "source_index": note.source_index,
            "pitch": note.pitch,
            "onset_beats": fraction_text(note.onset_beats),
            "duration_beats": fraction_text(note.duration_beats),
            "hand": note.hand.value if note.hand is not None else None,
            "staff": note.staff,
            "chord_id": note.chord_id,
            "assignment_cost": note.hand_assignment_cost,
            "assignment_confidence": note.hand_assignment_confidence,
            "previous_continuity_cost": note.previous_continuity_cost,
            "next_continuity_cost": note.next_continuity_cost,
            "tie_across_measure": note.tie_across_measure,
        }
        for note in score.notes
    ]
    return _write_rows_tsv(rows, path, "source_index")


def write_voice_assignment_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    if score.piano_layout == "none":
        raise ValueError("voice assignment diagnostics require piano layout optimization")
    rows: list[dict[str, object]] = [
        {
            "source_index": note.source_index,
            "pitch": note.pitch,
            "onset_beats": fraction_text(note.onset_beats),
            "duration_beats": fraction_text(note.duration_beats),
            "hand": note.hand.value if note.hand is not None else None,
            "staff": note.staff,
            "voice": note.voice,
            "chord_id": note.chord_id,
            "assignment_cost": note.voice_assignment_cost,
            "duration_changed": note.voice_duration_adjusted,
            "original_duration_beats": fraction_text(note.original_duration_beats),
            "tie_across_measure": note.tie_across_measure,
        }
        for note in score.notes
    ]
    return _write_rows_tsv(rows, path, "source_index")


def _write_rows_tsv(rows: list[dict[str, object]], path: str | Path, empty_field: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [empty_field]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_beats_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    if score.beat_track is None:
        raise ValueError("beat diagnostics require a beat-aware score")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("beat_number", "timestamp_seconds", "bpm", "downbeat", "confidence"))
        for beat in score.beat_track.beats:
            writer.writerow(
                (beat.number, beat.timestamp_seconds, beat.bpm, beat.downbeat, beat.confidence)
            )
    return output


def write_tempo_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    if score.beat_track is None:
        raise ValueError("tempo diagnostics require a beat-aware score")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("start_beat", "end_beat", "start_seconds", "end_seconds", "bpm"))
        for segment in score.beat_track.tempo_segments:
            writer.writerow(
                (
                    segment.start_beat,
                    segment.end_beat,
                    segment.start_seconds,
                    segment.end_seconds,
                    segment.bpm,
                )
            )
    return output


def meter_hypothesis_data(hypothesis: MeterHypothesis) -> dict[str, object]:
    return {
        "time_signature": str(hypothesis.time_signature),
        "pulse_factor": hypothesis.pulse_factor,
        "pulse_bpm": hypothesis.pulse_bpm,
        "notated_beat_bpm": hypothesis.notated_beat_bpm,
        "higher_level_bpm": hypothesis.higher_level_bpm,
        "downbeat_phase_beats": fraction_text(hypothesis.downbeat_phase_beats),
        "pickup_beats": fraction_text(hypothesis.pickup_beats),
        "measure_count": hypothesis.measure_count,
        "timing_error_ms": hypothesis.timing_error_ms,
        "rhythmic_complexity_score": hypothesis.rhythmic_complexity_score,
        "tie_count": hypothesis.tie_count,
        "triplet_ratio": hypothesis.triplet_ratio,
        "metric_accent_score": hypothesis.metric_accent_score,
        "tempo_smoothness_score": hypothesis.tempo_smoothness_score,
        "total_score": hypothesis.total_score,
        "normalized_score": hypothesis.normalized_score,
    }


def write_meter_hypotheses_tsv(result: JointMeterResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, hypothesis in enumerate(result.hypotheses, start=1):
        row = {
            "rank": rank,
            **meter_hypothesis_data(hypothesis),
            "relative_score": hypothesis.total_score - result.best.total_score,
            "confidence_margin": result.confidence_margin if rank == 1 else "",
        }
        rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_joint_diagnostics_json(result: JointMeterResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = score_diagnostics(result.score)
    data["meter_inference"] = {
        "confidence_margin": result.confidence_margin,
        "weights": asdict(result.config.weights),
        "best": meter_hypothesis_data(result.best),
        "hypotheses": [meter_hypothesis_data(hypothesis) for hypothesis in result.hypotheses],
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output
