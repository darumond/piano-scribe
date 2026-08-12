"""Dependency-light MusicXML export for normalized note events."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path

from piano_transcriber.score.types import ReconstructedScore, ScoreNote
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult

_PITCHES: tuple[tuple[str, int], ...] = (
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
)


def _append_pitch(parent: ET.Element, note: NoteEvent | ScoreNote) -> None:
    step, alter = _PITCHES[note.pitch % 12]
    pitch = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(note.pitch // 12 - 1)


def write_musicxml(
    result: TranscriptionResult,
    path: str | Path,
    *,
    tempo_bpm: int = 120,
    divisions: int = 480,
) -> Path:
    if tempo_bpm <= 0 or divisions <= 0:
        raise ValueError("tempo_bpm and divisions must be positive")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = "Piano Transcription"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")
    measure = ET.SubElement(part, "measure", number="1")
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = "4"
    ET.SubElement(time, "beat-type").text = "4"
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"
    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(tempo_bpm)
    sound = ET.SubElement(direction, "sound")
    sound.set("tempo", str(tempo_bpm))

    ticks_per_second = divisions * tempo_bpm / 60.0
    cursor_seconds = 0.0
    for event in sorted(result.notes):
        if event.onset_seconds > cursor_seconds:
            rest_ticks = max(1, round((event.onset_seconds - cursor_seconds) * ticks_per_second))
            rest = ET.SubElement(measure, "note")
            ET.SubElement(rest, "rest")
            ET.SubElement(rest, "duration").text = str(rest_ticks)
        note = ET.SubElement(measure, "note")
        _append_pitch(note, event)
        duration = max(1, round((event.offset_seconds - event.onset_seconds) * ticks_per_second))
        ET.SubElement(note, "duration").text = str(duration)
        ET.SubElement(note, "voice").text = "1"
        ET.SubElement(note, "type").text = "quarter"
        cursor_seconds = max(cursor_seconds, event.offset_seconds)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


@dataclass(frozen=True, slots=True)
class _NoteSegment:
    note: ScoreNote
    measure_index: int
    onset_in_measure: Fraction
    duration: Fraction
    tie_stop: bool
    tie_start: bool


def _segments(score: ReconstructedScore) -> tuple[_NoteSegment, ...]:
    measure_length = score.time_signature.measure_beats
    segments: list[_NoteSegment] = []
    for note in score.notes:
        position = note.onset_beats
        remaining = note.duration_beats
        first = True
        while remaining > 0:
            measure_index = int(position // measure_length)
            onset = position - measure_index * measure_length
            duration = min(remaining, measure_length - onset)
            remaining -= duration
            segments.append(
                _NoteSegment(
                    note,
                    measure_index,
                    onset,
                    duration,
                    tie_stop=not first,
                    tie_start=remaining > 0,
                )
            )
            position += duration
            first = False
    return tuple(segments)


def _duration_notation(duration: Fraction) -> tuple[str, int, tuple[int, int] | None]:
    notation = {
        Fraction(4): ("whole", 0, None),
        Fraction(3): ("half", 1, None),
        Fraction(2): ("half", 0, None),
        Fraction(3, 2): ("quarter", 1, None),
        Fraction(4, 3): ("half", 0, (3, 2)),
        Fraction(1): ("quarter", 0, None),
        Fraction(3, 4): ("eighth", 1, None),
        Fraction(2, 3): ("quarter", 0, (3, 2)),
        Fraction(1, 2): ("eighth", 0, None),
        Fraction(3, 8): ("16th", 1, None),
        Fraction(1, 3): ("eighth", 0, (3, 2)),
        Fraction(1, 4): ("16th", 0, None),
        Fraction(3, 16): ("32nd", 1, None),
        Fraction(1, 6): ("16th", 0, (3, 2)),
        Fraction(1, 8): ("32nd", 0, None),
    }
    if duration not in notation:
        raise ValueError(f"unsupported written duration: {duration}")
    return notation[duration]


def _append_score_note(
    measure: ET.Element,
    segment: _NoteSegment,
    *,
    divisions: int,
    chord: bool,
) -> None:
    element = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(element, "chord")
    _append_pitch(element, segment.note)
    ET.SubElement(element, "duration").text = str(int(segment.duration * divisions))
    ET.SubElement(element, "voice").text = "1"
    note_type, dots, time_modification = _duration_notation(segment.duration)
    ET.SubElement(element, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(element, "dot")
    if time_modification is not None:
        actual, normal = time_modification
        modification = ET.SubElement(element, "time-modification")
        ET.SubElement(modification, "actual-notes").text = str(actual)
        ET.SubElement(modification, "normal-notes").text = str(normal)
    if segment.tie_stop:
        ET.SubElement(element, "tie", type="stop")
    if segment.tie_start:
        ET.SubElement(element, "tie", type="start")
    if segment.tie_start or segment.tie_stop:
        notations = ET.SubElement(element, "notations")
        if segment.tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if segment.tie_start:
            ET.SubElement(notations, "tied", type="start")


def write_score_musicxml(score: ReconstructedScore, path: str | Path) -> Path:
    """Serialize an exact reconstructed score with measures and chord semantics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = _segments(score)
    divisions = lcm(
        24,
        *(segment.onset_in_measure.denominator for segment in segments),
        *(segment.duration.denominator for segment in segments),
    )
    measure_length_ticks = int(score.time_signature.measure_beats * divisions)

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = "PianoScribe Reconstructed Score"
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")

    by_measure: dict[int, list[_NoteSegment]] = {}
    for segment in segments:
        by_measure.setdefault(segment.measure_index, []).append(segment)
    for measure_index in range(score.measure_count):
        measure = ET.SubElement(part, "measure", number=str(measure_index + 1))
        if measure_index == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(score.time_signature.numerator)
            ET.SubElement(time, "beat-type").text = str(score.time_signature.denominator)
            staves = ET.SubElement(attributes, "staves")
            staves.text = "1"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = f"{score.bpm:g}"
            ET.SubElement(direction, "sound", tempo=f"{score.bpm:g}")

        measure_segments = by_measure.get(measure_index, [])
        grouped: dict[Fraction, list[_NoteSegment]] = {}
        for segment in measure_segments:
            grouped.setdefault(segment.onset_in_measure, []).append(segment)
        cursor_ticks = 0
        for onset, chord_segments in sorted(grouped.items()):
            onset_ticks = int(onset * divisions)
            movement = onset_ticks - cursor_ticks
            if movement > 0:
                forward = ET.SubElement(measure, "forward")
                ET.SubElement(forward, "duration").text = str(movement)
            elif movement < 0:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(-movement)
            ordered = sorted(chord_segments, key=lambda item: (-item.duration, item.note.pitch))
            for index, segment in enumerate(ordered):
                _append_score_note(measure, segment, divisions=divisions, chord=index > 0)
            cursor_ticks = onset_ticks + int(ordered[0].duration * divisions)
        if cursor_ticks < measure_length_ticks:
            forward = ET.SubElement(measure, "forward")
            ET.SubElement(forward, "duration").text = str(measure_length_ticks - cursor_ticks)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path
