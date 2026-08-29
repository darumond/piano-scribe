"""Meter-aware beam grouping for score-note segments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

from piano_transcriber.engraving.types import BeamAnnotation
from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.types import ReconstructedScore, ScoreNote, TimeSignature


@dataclass(frozen=True, slots=True)
class _Beamable:
    note: ScoreNote
    measure_index: int
    onset: Fraction
    duration: Fraction
    levels: int


def derive_beams(score: ReconstructedScore) -> tuple[BeamAnnotation, ...]:
    """Group beamable attacks inside the time signature's primary pulse hierarchy."""
    streams: dict[tuple[int, int, int, int], list[_Beamable]] = defaultdict(list)
    pulse = _primary_pulse(score.time_signature)
    for note in score.notes:
        measure_index, local, measure_length = locate_score_measure(
            note.onset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        duration = min(note.duration_beats, measure_length - local)
        levels = _beam_levels(duration)
        if levels == 0:
            continue
        bucket = int(local // pulse)
        streams[(measure_index, note.staff, note.voice, bucket)].append(
            _Beamable(note, measure_index, local, duration, levels)
        )

    annotations: list[BeamAnnotation] = []
    group_id = 0
    for key in sorted(streams):
        attacks = _distinct_attacks(streams[key])
        for run in _contiguous_runs(attacks):
            if len(run) < 2:
                continue
            group_id += 1
            for level in range(1, max(item.levels for item in run) + 1):
                level_members = [item for item in run if item.levels >= level]
                if not level_members:
                    continue
                if len(level_members) == 1:
                    item = level_members[0]
                    value = _hook_value(item, run)
                    annotations.append(_annotation(item, group_id, level, value))
                    continue
                for index, item in enumerate(level_members):
                    value = (
                        "begin"
                        if index == 0
                        else "end"
                        if index == len(level_members) - 1
                        else "continue"
                    )
                    annotations.append(_annotation(item, group_id, level, value))
    return tuple(sorted(annotations, key=_annotation_key))


def _primary_pulse(time_signature: TimeSignature) -> Fraction:
    if time_signature.numerator in {6, 9, 12} and time_signature.denominator == 8:
        return Fraction(3, 2)
    if time_signature.numerator in {6, 9, 12} and time_signature.denominator == 4:
        return Fraction(3)
    return Fraction(4, time_signature.denominator)


def _beam_levels(duration: Fraction) -> int:
    if duration <= Fraction(1, 8):
        return 3
    if duration <= Fraction(1, 4):
        return 2
    if duration <= Fraction(1, 2):
        return 1
    return 0


def _distinct_attacks(items: list[_Beamable]) -> list[_Beamable]:
    by_onset: dict[Fraction, _Beamable] = {}
    for item in sorted(items, key=lambda value: (value.onset, value.note.pitch)):
        current = by_onset.get(item.onset)
        if current is None or item.levels > current.levels:
            by_onset[item.onset] = item
    return [by_onset[onset] for onset in sorted(by_onset)]


def _contiguous_runs(items: list[_Beamable]) -> tuple[tuple[_Beamable, ...], ...]:
    if not items:
        return ()
    runs: list[list[_Beamable]] = [[items[0]]]
    for item in items[1:]:
        previous = runs[-1][-1]
        if item.onset <= previous.onset + previous.duration:
            runs[-1].append(item)
        else:
            runs.append([item])
    return tuple(tuple(run) for run in runs)


def _hook_value(item: _Beamable, run: tuple[_Beamable, ...]) -> str:
    index = run.index(item)
    return "backward hook" if index == len(run) - 1 else "forward hook"


def _annotation(
    item: _Beamable,
    group_id: int,
    level: int,
    value: str,
) -> BeamAnnotation:
    return BeamAnnotation(
        item.note.source_index,
        item.measure_index,
        item.onset,
        item.note.staff,
        item.note.voice,
        group_id,
        level,
        value,
    )


def _annotation_key(item: BeamAnnotation) -> tuple[int, int, int, Fraction, int, int]:
    return (
        item.measure_index,
        item.staff,
        item.voice,
        item.onset_in_measure,
        item.source_index,
        item.level,
    )
