"""JSON/TSV interchange and observability for score reconstruction."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from piano_transcriber.score.beats import measure_position
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
        measure, beat = measure_position(diagnostic.quantized_onset_beats, score.time_signature)
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
                "quantization_error_ms": diagnostic.quantization_error_seconds * 1000.0,
                "written_duration_beats": fraction_text(diagnostic.written_duration_beats),
                "written_duration_name": (
                    duration_name(diagnostic.written_duration_beats)
                    if diagnostic.written_duration_beats is not None
                    else None
                ),
                "measure": measure,
                "beat_in_measure": fraction_text(beat),
                "chord_size": chord_size.get(diagnostic.source_index),
                "action": diagnostic.action,
                "suspicious_reasons": list(diagnostic.suspicious_reasons),
                "merged_into_source_index": diagnostic.merged_into_source_index,
                "pedal_duration_shortened": diagnostic.pedal_duration_shortened,
                "velocity": note.velocity if note is not None else None,
                "confidence": note.confidence if note is not None else None,
            }
        )
    action_counts = Counter(item.action for item in score.diagnostics)
    rhythms = Counter(duration_name(note.duration_beats) for note in score.notes)
    errors = sorted(abs(item.quantization_error_seconds) * 1000.0 for item in score.diagnostics)
    return {
        "bpm": score.bpm,
        "time_signature": str(score.time_signature),
        "quantization_grid": score.grid_name,
        "grid_step_beats": fraction_text(score.grid_step_beats),
        "written_note_count": len(score.notes),
        "chord_group_count": len(score.chords),
        "multi_note_chord_count": sum(len(chord.notes) > 1 for chord in score.chords),
        "measure_count": score.measure_count,
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
        "events": events,
    }


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
            writer.writerow(row)
    return output
