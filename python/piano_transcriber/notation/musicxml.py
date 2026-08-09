"""Dependency-light MusicXML export for normalized note events."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

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


def _append_pitch(parent: ET.Element, note: NoteEvent) -> None:
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
