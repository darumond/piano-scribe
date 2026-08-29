"""Human-readable diagnostics for voice stability and engraving decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from piano_transcriber.score.diagnostics import engraving_data, fraction_text, note_name
from piano_transcriber.score.types import ReconstructedScore


def write_voice_stability_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    rows: list[dict[str, object]] = [
        {
            "source_index": note.source_index,
            "pitch": note.pitch,
            "note_name": note_name(note.pitch),
            "onset_beats": fraction_text(note.onset_beats),
            "duration_beats": fraction_text(note.duration_beats),
            "staff": note.staff,
            "voice": note.voice,
            "track_previous_pitch": note.track_previous_pitch,
            "track_direction": note.track_direction,
            "continuity_score": note.voice_continuity_score,
            "identity_switched": note.voice_identity_switched,
            "repeated_pitch_switched": note.repeated_pitch_voice_switched,
            "assignment_reason": note.voice_assignment_reason,
            "extra_voice_reason": note.extra_voice_reason,
            "duration_changed": note.voice_duration_adjusted,
            "duration_change_reason": note.duration_change_reason,
        }
        for note in score.notes
    ]
    return _write_rows(rows, path, "source_index")


def write_rests_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    rows: list[dict[str, object]] = [
        {
            "onset_beats": fraction_text(item.onset_beats),
            "duration_beats": fraction_text(item.duration_beats),
            "staff": item.staff,
            "voice": item.voice,
            "action": item.action,
            "reason": item.reason,
        }
        for item in score.rest_decisions
    ]
    return _write_rows(rows, path, "onset_beats")


def write_beams_tsv(score: ReconstructedScore, path: str | Path) -> Path:
    rows: list[dict[str, object]] = [
        {
            "group_id": item.group_id,
            "source_index": item.source_index,
            "measure_index": item.measure_index,
            "onset_in_measure": fraction_text(item.onset_in_measure),
            "staff": item.staff,
            "voice": item.voice,
            "level": item.level,
            "value": item.value,
        }
        for item in score.beam_annotations
    ]
    return _write_rows(rows, path, "group_id")


def write_engraving_diagnostics_json(
    score: ReconstructedScore,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(engraving_data(score), indent=2), encoding="utf-8")
    return output


def _write_rows(
    rows: list[dict[str, object]],
    path: str | Path,
    empty_field: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [empty_field]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return output
