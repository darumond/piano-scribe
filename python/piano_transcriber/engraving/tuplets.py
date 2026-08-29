"""Coherent triplet-bracket grouping from existing quantization evidence."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from piano_transcriber.engraving.types import TupletAnnotation
from piano_transcriber.score.beats import locate_score_measure
from piano_transcriber.score.types import ReconstructedScore, ScoreNote


def derive_tuplets(score: ReconstructedScore) -> tuple[TupletAnnotation, ...]:
    diagnostics = {item.source_index: item for item in score.diagnostics}
    streams: dict[tuple[int, int, int], list[tuple[Fraction, ScoreNote]]] = defaultdict(list)
    for note in score.notes:
        diagnostic = diagnostics.get(note.source_index)
        subdivision = diagnostic.selected_subdivision if diagnostic is not None else None
        is_triplet = subdivision in {"eighth-triplet", "sixteenth-triplet"} or (
            note.duration_beats in {Fraction(1, 3), Fraction(1, 6)}
        )
        if not is_triplet:
            continue
        measure, onset, _length = locate_score_measure(
            note.onset_beats,
            score.time_signature,
            score.pickup_beats,
        )
        streams[(measure, note.staff, note.voice)].append((onset, note))

    annotations: list[TupletAnnotation] = []
    group_id = 0
    for (measure, staff, voice), values in sorted(streams.items()):
        representatives = {
            onset: min(
                (note for candidate, note in values if candidate == onset),
                key=lambda note: note.source_index,
            )
            for onset in {candidate for candidate, _note in values}
        }
        attacks = sorted(representatives)
        index = 0
        while index + 2 < len(attacks):
            first, second, third = attacks[index : index + 3]
            step = second - first
            if step in {Fraction(1, 3), Fraction(1, 6)} and third - second == step:
                group_id += 1
                annotations.extend(
                    (
                        TupletAnnotation(
                            representatives[first].source_index,
                            measure,
                            first,
                            staff,
                            voice,
                            group_id,
                            "start",
                        ),
                        TupletAnnotation(
                            representatives[third].source_index,
                            measure,
                            third,
                            staff,
                            voice,
                            group_id,
                            "stop",
                        ),
                    )
                )
                index += 3
            else:
                index += 1
    return tuple(annotations)
