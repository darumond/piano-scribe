"""Reference readers and adapters. No inference modules are called by readers."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

import mido

from piano_transcriber.evaluation.types import EvaluationNote, EvaluationScore

if TYPE_CHECKING:
    from piano_transcriber.score.types import ReconstructedScore
    from piano_transcriber.transcription.types import TranscriptionResult


class TempoMap:
    def __init__(self, changes: tuple[tuple[float, float], ...]) -> None:
        self.positions: list[float] = []
        self.bpms: list[float] = []
        self.seconds: list[float] = []
        for beat, bpm in sorted(dict(changes).items()):
            if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(beat) or beat < 0:
                raise ValueError("invalid tempo map")
            elapsed = (
                0.0
                if not self.positions
                else self.seconds[-1] + (beat - self.positions[-1]) * 60 / self.bpms[-1]
            )
            self.positions.append(beat)
            self.bpms.append(bpm)
            self.seconds.append(elapsed)

    def at(self, beat: float) -> float | None:
        index = bisect_right(self.positions, beat + 1e-10) - 1
        if index < 0 or not self.positions or self.positions[0] != 0:
            return None
        return self.seconds[index] + (beat - self.positions[index]) * 60 / self.bpms[index]


def _with_seconds(
    notes: list[EvaluationNote], tempos: tuple[tuple[float, float], ...]
) -> tuple[EvaluationNote, ...]:
    mapping = TempoMap(tempos)
    result: list[EvaluationNote] = []
    for note in notes:
        start = mapping.at(note.onset_beats) if note.onset_beats is not None else None
        end = (
            mapping.at(note.onset_beats + note.duration_beats)
            if note.onset_beats is not None and note.duration_beats is not None
            else None
        )
        result.append(
            replace(
                note,
                onset_seconds=start,
                duration_seconds=end - start if start is not None and end is not None else None,
            )
        )
    return tuple(result)


def _number(value: Any) -> float | None:
    return float(Fraction(str(value))) if value is not None else None


def _label(value: Any) -> str | None:
    return str(value) if value is not None else None


def _ordered(score: EvaluationScore) -> EvaluationScore:
    return replace(
        score,
        notes=tuple(
            sorted(
                score.notes,
                key=lambda n: (
                    n.onset_beats if n.onset_beats is not None else float(n.onset_seconds or 0),
                    n.pitch,
                    n.part,
                    n.voice or "",
                ),
            )
        ),
    )


def load_score(path: str | Path, *, voice_scope: str = "part") -> EvaluationScore:
    source = Path(path)
    suffix = source.suffix.lower()
    try:
        if suffix in {".mid", ".midi"}:
            return _ordered(read_midi(source))
        if suffix in {".musicxml", ".xml", ".mxl"}:
            if voice_scope not in {"part", "staff"}:
                raise ValueError("voice scope must be part or staff")
            return _ordered(read_musicxml(source, voice_scope=voice_scope))
        if suffix == ".json":
            return _ordered(read_json(source))
    except (OSError, ET.ParseError, KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"cannot read evaluation input {source}: {error}") from error
    raise ValueError(f"unsupported reference/prediction format: {suffix}")


def read_midi(path: Path) -> EvaluationScore:
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("evaluation supports synchronous MIDI types 0/1 with PPQ timing")
    events: list[tuple[int, int, int, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            events.append((tick, track_index, order, message))
    events.sort(key=lambda event: event[:3])
    tempo: dict[float, float] = {0.0: 120.0}
    signatures: dict[float, tuple[int, int]] = {}
    active: dict[tuple[int, int, int], deque[float]] = defaultdict(deque)
    notes: list[EvaluationNote] = []
    explicit_tempo = False
    for tick, track, _order, message in events:
        beat = tick / midi.ticks_per_beat
        if message.type == "set_tempo":
            tempo[beat] = float(mido.tempo2bpm(message.tempo))
            explicit_tempo = True
        elif message.type == "time_signature":
            signatures[beat] = (message.numerator, message.denominator)
        elif message.type in {"note_on", "note_off"}:
            key = (track, message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                active[key].append(beat)
            elif active[key]:
                onset = active[key].popleft()
                if beat <= onset:
                    raise ValueError("MIDI contains a non-positive note duration")
                notes.append(
                    EvaluationNote(message.note, onset_beats=onset, duration_beats=beat - onset)
                )
    if any(active.values()):
        raise ValueError("MIDI contains unterminated notes")
    tempos = tuple(sorted(tempo.items()))
    # MIDI bar positions follow the file's explicit meter map; pickup is still unknown.
    boundaries: list[float] = []
    beat_grid: list[float] = []
    if 0.0 in signatures:
        end = max(
            (float(n.onset_beats or 0) + float(n.duration_beats or 0) for n in notes), default=0.0
        )
        changes = sorted(signatures)
        for index, position in enumerate(changes):
            numerator, denominator = signatures[position]
            segment_end = changes[index + 1] if index + 1 < len(changes) else end
            length = numerator * 4 / denominator
            boundary = position
            while boundary < segment_end - 1e-9:
                boundaries.append(boundary)
                boundary += length
            beat = position
            while beat < segment_end - 1e-9:
                beat_grid.append(beat)
                beat += 4 / denominator
        notes = [
            replace(
                n,
                measure=str(bisect_right(boundaries, float(n.onset_beats or 0))),
                beat=float(n.onset_beats or 0)
                - boundaries[max(0, bisect_right(boundaries, float(n.onset_beats or 0)) - 1)],
            )
            for n in notes
        ]
    warnings = ["MIDI channels/tracks do not imply staff, physical hand, voice, ties, or pickup."]
    if not explicit_tempo:
        warnings.append(
            "MIDI seconds use the format default of 120 BPM; no explicit tempo was present."
        )
    return EvaluationScore(
        _with_seconds(notes, tempos),
        "midi",
        time_signatures=tuple((b, *sig) for b, sig in sorted(signatures.items())),
        tempos=tempos,
        measure_boundaries=tuple(boundaries) if boundaries else None,
        beats=tuple(beat_grid) if beat_grid else None,
        structure={"measure_count": len(boundaries) if boundaries else None},
        warnings=tuple(warnings),
    )


def _xml_root(path: Path) -> ET.Element:
    if path.suffix.lower() == ".mxl":
        with ZipFile(path) as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            entry = next(
                (
                    e.attrib["full-path"]
                    for e in container.iter()
                    if e.tag.rsplit("}", 1)[-1] == "rootfile"
                ),
                None,
            )
            if entry is None or archive.getinfo(entry).file_size > 50_000_000:
                raise ValueError("invalid or oversized compressed MusicXML root")
            root = ET.fromstring(archive.read(entry))
    else:
        root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "score-partwise":
        raise ValueError("only score-partwise MusicXML is supported")
    return root


def read_musicxml(path: Path, *, voice_scope: str = "part") -> EvaluationScore:
    root = _xml_root(path)
    notes: list[EvaluationNote] = []
    tempos: dict[float, float] = {}
    signatures: dict[float, tuple[int, int]] = {}
    boundaries: list[float] = []
    grid: list[float] = []
    full_downbeats: list[float] = []
    pickup: float | None = None
    warnings: set[str] = set()
    ties: dict[tuple[str, str | None, int], deque[int]] = defaultdict(deque)
    tie_count = rest_count = beam_count = tuplet_count = 0
    measures = 0
    declared_staves: dict[str, int] = {}
    for part_index, part in enumerate(root.findall("part")):
        part_id = str(part_index + 1)
        measure_start = Fraction(0)
        divisions = Fraction(1)
        signature: tuple[int, int] | None = None
        transpose = 0
        clefs: dict[str, str | None] = {}
        for measure_index, measure in enumerate(part.findall("measure")):
            cursor = Fraction(0)
            extent = Fraction(0)
            last_attack = Fraction(0)
            for element in measure:
                if element.tag == "attributes":
                    for clef in element.findall("clef"):
                        clefs[clef.get("number", "1")] = {
                            ("G", "2"): "treble",
                            ("F", "4"): "bass",
                        }.get((clef.findtext("sign", ""), clef.findtext("line", "")))
                    divisions = Fraction(element.findtext("divisions", str(divisions)))
                    if divisions <= 0:
                        raise ValueError("MusicXML divisions must be positive")
                    staves = element.findtext("staves")
                    if staves is not None:
                        declared_staves[part_id] = max(declared_staves.get(part_id, 0), int(staves))
                    time = element.find("time")
                    if time is not None and time.find("senza-misura") is None:
                        numerator = sum(int(x) for x in time.findtext("beats", "").split("+"))
                        denominator = int(time.findtext("beat-type", "0"))
                        if numerator <= 0 or denominator <= 0:
                            raise ValueError("invalid MusicXML meter")
                        changed = signature != (numerator, denominator)
                        signature = numerator, denominator
                        if part_index == 0 and changed:
                            signatures[float(measure_start)] = signature
                    trans = element.find("transpose")
                    if trans is not None:
                        transpose = int(trans.findtext("chromatic", "0")) + 12 * int(
                            trans.findtext("octave-change", "0")
                        )
                elif element.tag in {"backup", "forward"}:
                    delta = Fraction(element.findtext("duration", "0")) / divisions
                    cursor += delta if element.tag == "forward" else -delta
                    if cursor < 0:
                        raise ValueError("MusicXML backup moves before the measure start")
                    extent = max(extent, cursor)
                elif element.tag == "direction":
                    position = (
                        measure_start
                        + cursor
                        + Fraction(element.findtext("offset", "0")) / divisions
                    )
                    sound = element.find("sound")
                    metronome = element.find("direction-type/metronome")
                    if sound is not None and "tempo" in sound.attrib:
                        tempos[float(position)] = float(sound.attrib["tempo"])
                    elif metronome is not None and metronome.findtext("per-minute"):
                        units = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "16th": 0.25}
                        unit = units.get(metronome.findtext("beat-unit", ""))
                        if unit is not None:
                            dots = len(metronome.findall("beat-unit-dot"))
                            tempos[float(position)] = (
                                float(metronome.findtext("per-minute", "0"))
                                * unit
                                * (2 - 0.5**dots)
                            )
                elif element.tag == "note":
                    if element.find("grace") is not None:
                        warnings.add("Grace notes are omitted from timed note metrics.")
                        continue
                    duration = Fraction(element.findtext("duration", "0")) / divisions
                    if duration <= 0:
                        raise ValueError(
                            "MusicXML contains a non-positive timed note/rest duration"
                        )
                    chord = element.find("chord") is not None
                    onset = last_attack if chord else cursor
                    if not chord:
                        last_attack = onset
                        cursor += duration
                    extent = max(extent, onset + duration)
                    rest_count += int(element.find("rest") is not None)
                    beam_count += sum(
                        e.text == "begin" and e.get("number", "1") == "1"
                        for e in element.findall("beam")
                    )
                    tuplet_count += sum(
                        e.get("type") == "start" for e in element.findall("notations/tuplet")
                    )
                    pitch = element.find("pitch")
                    if pitch is None:
                        if element.find("rest") is None:
                            warnings.add("Unpitched events are omitted from piano note metrics.")
                        continue
                    alter = float(pitch.findtext("alter", "0"))
                    if not alter.is_integer():
                        raise ValueError(
                            "microtonal MusicXML pitches cannot be reduced to MIDI integers"
                        )
                    midi_pitch = (
                        (int(pitch.findtext("octave", "4")) + 1) * 12
                        + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[
                            pitch.findtext("step", "C")
                        ]
                        + int(alter)
                        + transpose
                    )
                    voice = element.findtext("voice")
                    staff = element.findtext("staff")
                    if voice_scope == "staff" and voice is not None and staff is not None:
                        voice = f"{staff}:{voice}"
                    note = EvaluationNote(
                        midi_pitch,
                        onset_beats=float(measure_start + onset),
                        duration_beats=float(duration),
                        measure=measure.get("number", str(measure_index + 1)),
                        beat=float(onset),
                        staff=staff,
                        voice=voice,
                        part=part_id,
                        staff_role=clefs.get(staff) if staff is not None else None,
                    )
                    types = {
                        e.get("type")
                        for e in [*element.findall("tie"), *element.findall("notations/tied")]
                    }
                    key = (part_id, voice, midi_pitch)
                    index = len(notes)
                    if "stop" in types and ties[key]:
                        index = ties[key].popleft()
                        previous = notes[index]
                        if (
                            abs(
                                float(previous.onset_beats or 0)
                                + float(previous.duration_beats or 0)
                                - float(note.onset_beats or 0)
                            )
                            > 1e-7
                        ):
                            raise ValueError("noncontiguous MusicXML tie")
                        notes[index] = replace(
                            previous,
                            duration_beats=float(previous.duration_beats or 0) + float(duration),
                        )
                    else:
                        notes.append(note)
                        if "stop" in types:
                            warnings.add("A tie ends without a start inside this excerpt.")
                    if "start" in types:
                        ties[key].append(index)
                        tie_count += 1
            length = Fraction(signature[0] * 4, signature[1]) if signature is not None else extent
            implicit = measure.get("implicit") == "yes" or measure.get("number") == "0"
            actual = extent if implicit and extent > 0 else max(length, extent)
            if part_index == 0:
                measures += 1
                boundaries.append(float(measure_start))
                if measure_index == 0 and signature is not None:
                    pickup = float(extent) if implicit and 0 < extent < length else 0.0
                if not (measure_index == 0 and pickup and pickup > 0):
                    full_downbeats.append(float(measure_start))
                if signature is not None:
                    beat_unit = Fraction(4, signature[1])
                    position = Fraction(0)
                    while position < actual:
                        grid.append(float(measure_start + position))
                        position += beat_unit
            measure_start += actual
    if any(ties.values()):
        warnings.add("A tie starts without ending inside this excerpt.")
    if root.find(".//repeat") is not None:
        warnings.add(
            "Repeats are evaluated in written order; provide an unfolded reference "
            "for performance comparison."
        )
    if not tempos or 0.0 not in tempos:
        warnings.add(
            "No initial explicit tempo: seconds metrics unavailable; symbolic beats remain usable."
        )
    tempo_changes = tuple(sorted(tempos.items()))
    return EvaluationScore(
        _with_seconds(notes, tempo_changes),
        "musicxml",
        time_signatures=tuple((b, *sig) for b, sig in sorted(signatures.items())),
        tempos=tempo_changes,
        pickup_beats=pickup,
        first_downbeat_beats=full_downbeats[0] if full_downbeats and signatures else None,
        measure_boundaries=tuple(boundaries),
        beats=tuple(grid) if signatures else None,
        downbeats=tuple(full_downbeats) if signatures else None,
        structure={
            "measure_count": measures,
            "staff_count": sum(declared_staves.values())
            if len(declared_staves) == len(root.findall("part")) and declared_staves
            else None,
            "tie_count": tie_count,
            "rest_count": rest_count,
            "beam_groups": beam_count,
            "tuplet_groups": tuplet_count,
        },
        warnings=tuple(sorted(warnings)),
    )


def read_json(path: Path) -> EvaluationScore:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "notes" in data:
        notes = []
        for item in data["notes"]:
            onset = _number(item.get("onset_seconds"))
            duration = _number(item.get("duration_seconds"))
            if duration is None and onset is not None and item.get("offset_seconds") is not None:
                duration = float(item["offset_seconds"]) - onset
            notes.append(
                EvaluationNote(
                    int(item["pitch"]),
                    onset,
                    duration,
                    _number(item.get("onset_beats")),
                    _number(item.get("duration_beats")),
                    _label(item.get("measure")),
                    _number(item.get("beat")),
                    _label(item.get("staff")),
                    _label(item.get("voice")),
                    _label(item.get("hand")),
                    str(item.get("part", "1")),
                    staff_role=item.get("staff_role"),
                )
            )
        return EvaluationScore(
            tuple(notes),
            "json",
            stage=data.get("stage", "transcription" if "model_name" in data else "symbolic"),
            time_signatures=tuple(tuple(x) for x in data.get("time_signatures", [])),
            tempos=tuple(tuple(x) for x in data.get("tempos", [])),
            pickup_beats=_number(data.get("pickup_beats")),
            first_downbeat_beats=_number(data.get("first_downbeat_beats")),
            measure_boundaries=_optional_grid(data, "measure_boundaries"),
            beats=_optional_grid(data, "beats"),
            downbeats=_optional_grid(data, "downbeats"),
            structure=data.get("structure", {}),
            meter_hypotheses=tuple(data.get("meter_hypotheses", [])),
        )
    if "events" in data:
        notes = []
        for item in data["events"]:
            if (
                item.get("action") in {"filtered", "merged"}
                or item.get("written_duration_beats") is None
            ):
                continue
            staff, voice = item.get("assigned_staff"), item.get("assigned_voice")
            # Production voice numbers are local to staff; scope the track explicitly.
            track = f"{staff}:{voice}" if voice is not None else None
            notes.append(
                EvaluationNote(
                    int(item["pitch"]),
                    float(item["raw_onset_seconds"]),
                    float(item["raw_offset_seconds"]) - float(item["raw_onset_seconds"]),
                    _number(item["quantized_onset_beats"]),
                    _number(item["written_duration_beats"]),
                    _label(item.get("measure")),
                    _number(item.get("beat_in_measure")),
                    _label(staff),
                    track,
                    item.get("assigned_hand"),
                    subdivision=item.get("selected_subdivision"),
                )
            )
        numerator, denominator = (int(x) for x in data["time_signature"].split("/"))
        pickup = float(_number(data.get("pickup_beats")) or 0)
        count = int(data["measure_count"])
        length = numerator * 4 / denominator
        boundaries = tuple(
            [0.0]
            + [
                pickup + i * length if pickup else (i + 1) * length
                for i in range(max(0, count - 1))
            ]
        )
        return EvaluationScore(
            tuple(notes),
            "score-diagnostics",
            time_signatures=((0.0, numerator, denominator),),
            tempos=((0.0, float(data["bpm"])),),
            pickup_beats=pickup,
            first_downbeat_beats=_number(data.get("first_full_downbeat_beats")),
            measure_boundaries=boundaries,
            downbeats=boundaries[1:] if pickup else boundaries,
            structure={"measure_count": count},
            meter_hypotheses=tuple(data.get("meter_inference", {}).get("hypotheses", [])),
            warnings=(
                "Diagnostic seconds preserve acoustic offsets; beats represent written durations.",
            ),
        )
    raise ValueError("JSON must contain normalized notes or pipeline score diagnostic events")


def _optional_grid(data: dict[str, Any], key: str) -> tuple[float, ...] | None:
    return tuple(float(x) for x in data[key]) if data.get(key) is not None else None


def from_transcription(result: TranscriptionResult) -> EvaluationScore:
    return EvaluationScore(
        tuple(
            EvaluationNote(n.pitch, n.onset_seconds, n.offset_seconds - n.onset_seconds)
            for n in result.notes
        ),
        "transcription-result",
        "transcription",
    )


def from_reconstructed(score: ReconstructedScore) -> EvaluationScore:
    notes = tuple(
        EvaluationNote(
            n.pitch,
            n.raw_onset_seconds,
            n.raw_offset_seconds - n.raw_onset_seconds,
            float(n.onset_beats),
            float(n.duration_beats),
            staff=str(n.staff),
            voice=f"{n.staff}:{n.voice}",
            hand=n.hand.value if n.hand is not None else None,
        )
        for n in score.notes
    )
    length, pickup = float(score.time_signature.measure_beats), float(score.pickup_beats)
    boundaries = tuple(
        [0.0]
        + [
            pickup + i * length if pickup else (i + 1) * length
            for i in range(score.measure_count - 1)
        ]
    )
    return EvaluationScore(
        notes,
        "reconstructed-score",
        time_signatures=((0.0, score.time_signature.numerator, score.time_signature.denominator),),
        tempos=((0.0, score.bpm),),
        pickup_beats=pickup,
        first_downbeat_beats=float(score.first_full_downbeat_beats),
        measure_boundaries=boundaries,
        downbeats=boundaries[1:] if pickup else boundaries,
        structure={
            "measure_count": score.measure_count,
            "rest_count": len(score.rests),
            "beam_groups": len({x.group_id for x in score.beam_annotations}),
            "tuplet_groups": len({x.group_id for x in score.tuplet_annotations}),
        },
    )
