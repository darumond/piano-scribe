"""Dependency-light MusicXML export for normalized note events."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path
from time import perf_counter

from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.quantize import WRITTEN_DURATIONS
from piano_transcriber.score.types import ReconstructedScore, ScoreNote, ScoreRest
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult

_SPLIT_DURATIONS = tuple(sorted((*WRITTEN_DURATIONS, Fraction(1, 24), Fraction(1, 48))))

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class _RestSegment:
    rest: ScoreRest
    measure_index: int
    onset_in_measure: Fraction
    duration: Fraction


def _segments(score: ReconstructedScore) -> tuple[_NoteSegment, ...]:
    segments: list[_NoteSegment] = []
    for note in score.notes:
        position = note.onset_beats
        remaining = note.duration_beats
        first = True
        while remaining > 0:
            measure_index, onset, measure_length = locate_score_measure(
                position,
                score.time_signature,
                score.pickup_beats,
            )
            measure_remaining = min(remaining, measure_length - onset)
            while measure_remaining > 0:
                conventional = max(
                    duration for duration in _SPLIT_DURATIONS if duration <= measure_remaining
                )
                remaining -= conventional
                measure_remaining -= conventional
                segments.append(
                    _NoteSegment(
                        note,
                        measure_index,
                        onset,
                        conventional,
                        tie_stop=not first,
                        tie_start=remaining > 0,
                    )
                )
                position += conventional
                onset += conventional
                first = False
    return tuple(segments)


def _rest_segments(score: ReconstructedScore) -> tuple[_RestSegment, ...]:
    segments: list[_RestSegment] = []
    for rest in score.rests:
        position = rest.onset_beats
        remaining = rest.duration_beats
        while remaining > 0:
            measure_index, onset, measure_length = locate_score_measure(
                position,
                score.time_signature,
                score.pickup_beats,
            )
            measure_remaining = min(remaining, measure_length - onset)
            while measure_remaining > 0:
                if measure_remaining < score.minimum_explicit_rest_beats:
                    position += measure_remaining
                    remaining -= measure_remaining
                    break
                conventional = max(
                    duration for duration in _SPLIT_DURATIONS if duration <= measure_remaining
                )
                remaining -= conventional
                measure_remaining -= conventional
                segments.append(_RestSegment(rest, measure_index, onset, conventional))
                position += conventional
                onset += conventional
    return tuple(segments)


def _duration_notation(duration: Fraction) -> tuple[str, int, tuple[int, int] | None]:
    notation = {
        Fraction(1, 48): ("128th", 0, (3, 2)),
        Fraction(1, 24): ("64th", 0, (3, 2)),
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
    beams: tuple[tuple[int, str], ...] = (),
    tuplets: tuple[tuple[int, str], ...] = (),
) -> None:
    element = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(element, "chord")
    _append_pitch(element, segment.note)
    ET.SubElement(element, "duration").text = str(int(segment.duration * divisions))
    ET.SubElement(element, "voice").text = str(segment.note.voice)
    note_type, dots, time_modification = _duration_notation(segment.duration)
    ET.SubElement(element, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(element, "dot")
    if time_modification is not None:
        actual, normal = time_modification
        modification = ET.SubElement(element, "time-modification")
        ET.SubElement(modification, "actual-notes").text = str(actual)
        ET.SubElement(modification, "normal-notes").text = str(normal)
    for level, value in beams:
        ET.SubElement(element, "beam", number=str(level)).text = value
    if segment.tie_stop:
        ET.SubElement(element, "tie", type="stop")
    if segment.tie_start:
        ET.SubElement(element, "tie", type="start")
    if segment.tie_start or segment.tie_stop or tuplets:
        notations = ET.SubElement(element, "notations")
        if segment.tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if segment.tie_start:
            ET.SubElement(notations, "tied", type="start")
        for number, value in tuplets:
            ET.SubElement(notations, "tuplet", number=str(number), type=value)
    ET.SubElement(element, "staff").text = str(segment.note.staff)


def _append_score_rest(
    measure: ET.Element,
    segment: _RestSegment,
    *,
    divisions: int,
) -> None:
    element = ET.SubElement(measure, "note")
    ET.SubElement(element, "rest")
    ET.SubElement(element, "duration").text = str(int(segment.duration * divisions))
    ET.SubElement(element, "voice").text = str(segment.rest.voice)
    note_type, dots, time_modification = _duration_notation(segment.duration)
    ET.SubElement(element, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(element, "dot")
    if time_modification is not None:
        actual, normal = time_modification
        modification = ET.SubElement(element, "time-modification")
        ET.SubElement(modification, "actual-notes").text = str(actual)
        ET.SubElement(modification, "normal-notes").text = str(normal)
    ET.SubElement(element, "staff").text = str(segment.rest.staff)


def write_score_musicxml(score: ReconstructedScore, path: str | Path) -> Path:
    """Serialize an exact reconstructed score with measures and chord semantics."""
    started = perf_counter()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = _segments(score)
    rest_segments = _rest_segments(score)
    divisions = lcm(
        24,
        *(segment.onset_in_measure.denominator for segment in segments),
        *(segment.duration.denominator for segment in segments),
        *(segment.onset_in_measure.denominator for segment in rest_segments),
        *(segment.duration.denominator for segment in rest_segments),
    )

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
    rests_by_measure: dict[int, list[_RestSegment]] = {}
    for rest_segment in rest_segments:
        rests_by_measure.setdefault(rest_segment.measure_index, []).append(rest_segment)
    beam_lookup: dict[tuple[int, Fraction, int, int], list[tuple[int, str]]] = {}
    for beam_annotation in score.beam_annotations:
        beam_key = (
            beam_annotation.measure_index,
            beam_annotation.onset_in_measure,
            beam_annotation.staff,
            beam_annotation.voice,
        )
        beam_lookup.setdefault(beam_key, []).append((beam_annotation.level, beam_annotation.value))
    tuplet_lookup: dict[tuple[int, Fraction, int, int, int], list[tuple[int, str]]] = {}
    for tuplet_annotation in score.tuplet_annotations:
        tuplet_annotation_key = (
            tuplet_annotation.measure_index,
            tuplet_annotation.onset_in_measure,
            tuplet_annotation.staff,
            tuplet_annotation.voice,
            tuplet_annotation.source_index,
        )
        tuplet_lookup.setdefault(tuplet_annotation_key, []).append(
            (tuplet_annotation.group_id, tuplet_annotation.value)
        )
    staff_count = (
        2 if score.piano_layout != "none" else max((note.staff for note in score.notes), default=1)
    )
    for measure_index in range(score.measure_count):
        is_pickup = score.pickup_beats > 0 and measure_index == 0
        measure_number = (
            "0"
            if is_pickup
            else str(measure_index if score.pickup_beats > 0 else measure_index + 1)
        )
        measure = ET.SubElement(part, "measure", number=measure_number)
        if is_pickup:
            measure.set("implicit", "yes")
        if measure_index == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key_signature = ET.SubElement(attributes, "key")
            ET.SubElement(key_signature, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(score.time_signature.numerator)
            ET.SubElement(time, "beat-type").text = str(score.time_signature.denominator)
            staves = ET.SubElement(attributes, "staves")
            staves.text = str(staff_count)
            treble_clef = ET.SubElement(attributes, "clef", number="1")
            ET.SubElement(treble_clef, "sign").text = "G"
            ET.SubElement(treble_clef, "line").text = "2"
            if staff_count == 2:
                bass_clef = ET.SubElement(attributes, "clef", number="2")
                ET.SubElement(bass_clef, "sign").text = "F"
                ET.SubElement(bass_clef, "line").text = "4"
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = f"{score.bpm:g}"
            ET.SubElement(direction, "sound", tempo=f"{score.bpm:g}")
        elif score.beat_track is not None:
            if score.pickup_beats > 0:
                score_beat = float(
                    score.pickup_beats + (measure_index - 1) * score.time_signature.measure_beats
                )
            else:
                score_beat = measure_index * float(score.time_signature.measure_beats)
            track_beat = score_beat - score.beat_position_offset
            nearest = min(
                score.beat_track.beats,
                key=lambda beat: abs(beat.number - track_beat),
            )
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = f"{nearest.bpm:g}"
            ET.SubElement(direction, "sound", tempo=f"{nearest.bpm:g}")

        _append_pedal_directions(measure, score, measure_index, divisions)

        measure_segments = by_measure.get(measure_index, [])
        current_measure_beats = (
            score.pickup_beats if is_pickup else score.time_signature.measure_beats
        )
        measure_length_ticks = int(current_measure_beats * divisions)
        grouped: dict[tuple[int, int], dict[Fraction, list[_NoteSegment]]] = {}
        for note_segment in measure_segments:
            assignment = (note_segment.note.staff, note_segment.note.voice)
            grouped.setdefault(assignment, {}).setdefault(note_segment.onset_in_measure, []).append(
                note_segment
            )
        grouped_rests: dict[tuple[int, int], dict[Fraction, list[_RestSegment]]] = {}
        for rest_segment in rests_by_measure.get(measure_index, []):
            assignment = (rest_segment.rest.staff, rest_segment.rest.voice)
            grouped_rests.setdefault(assignment, {}).setdefault(
                rest_segment.onset_in_measure, []
            ).append(rest_segment)
        stream_keys = sorted(set(grouped) | set(grouped_rests))
        for stream_index, stream_key in enumerate(stream_keys):
            if stream_index > 0:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(measure_length_ticks)
            note_onsets = grouped.get(stream_key, {})
            rest_onsets = grouped_rests.get(stream_key, {})
            cursor_ticks = 0
            for onset in sorted(set(note_onsets) | set(rest_onsets)):
                onset_ticks = int(onset * divisions)
                if onset_ticks > cursor_ticks:
                    forward = ET.SubElement(measure, "forward")
                    ET.SubElement(forward, "duration").text = str(onset_ticks - cursor_ticks)
                for rest_segment in rest_onsets.get(onset, []):
                    _append_score_rest(measure, rest_segment, divisions=divisions)
                    cursor_ticks = max(
                        cursor_ticks,
                        onset_ticks + int(rest_segment.duration * divisions),
                    )
                chord_segments = note_onsets.get(onset, [])
                ordered = sorted(
                    chord_segments,
                    key=lambda item: (-item.duration, item.note.pitch),
                )
                for index, note_segment in enumerate(ordered):
                    annotation_key = (
                        measure_index,
                        note_segment.onset_in_measure,
                        note_segment.note.staff,
                        note_segment.note.voice,
                    )
                    tuplet_key = (*annotation_key, note_segment.note.source_index)
                    _append_score_note(
                        measure,
                        note_segment,
                        divisions=divisions,
                        chord=index > 0,
                        beams=tuple(sorted(beam_lookup.get(annotation_key, ()))),
                        tuplets=tuple(sorted(tuplet_lookup.get(tuplet_key, ()))),
                    )
                if ordered:
                    cursor_ticks = max(
                        cursor_ticks,
                        onset_ticks + int(ordered[0].duration * divisions),
                    )
            if cursor_ticks < measure_length_ticks:
                forward = ET.SubElement(measure, "forward")
                ET.SubElement(forward, "duration").text = str(measure_length_ticks - cursor_ticks)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info(
        "Wrote reconstructed MusicXML in %.3f s to %s",
        perf_counter() - started,
        output_path,
    )
    return output_path


def _append_pedal_directions(
    measure: ET.Element,
    score: ReconstructedScore,
    measure_index: int,
    divisions: int,
) -> None:
    events: list[tuple[Fraction, str]] = []
    for interval in score.pedal_intervals:
        start_measure, start_local, _start_length = locate_score_measure(
            interval.onset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        stop_measure, stop_local, _stop_length = locate_score_measure(
            interval.offset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        if start_measure == measure_index:
            events.append((start_local, "start"))
        if stop_measure == measure_index:
            events.append((stop_local, "stop"))
    for offset, value in sorted(events, key=lambda item: (item[0], item[1] == "start")):
        direction = ET.SubElement(measure, "direction", placement="below")
        direction_type = ET.SubElement(direction, "direction-type")
        ET.SubElement(direction_type, "pedal", type=value, line="yes")
        if offset:
            ET.SubElement(direction, "offset").text = str(int(offset * divisions))
        ET.SubElement(direction, "staff").text = "2" if score.piano_layout != "none" else "1"
