"""Tiny deterministic symbolic examples generated locally without external data."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from piano_transcriber.evaluation.report import write_json
from piano_transcriber.evaluation.types import EvaluationNote, EvaluationScore


def synthetic_corpus() -> dict[str, EvaluationScore]:
    examples: dict[str, tuple[list[tuple[int, float, float, str, str]], tuple[int, int], float]] = {
        "scale": (
            [(60 + p, i * 0.5, 0.5, "1", "1") for i, p in enumerate((0, 2, 4, 5, 7, 9, 11, 12))],
            (4, 4),
            0,
        ),
        "blocked-chords": (
            [
                (p + shift, float(i), 1, "1", "1")
                for i, shift in enumerate((0, 5, 7, 0))
                for p in (60, 64, 67)
            ],
            (4, 4),
            0,
        ),
        "arpeggios": (
            [(p, i * 0.25, 0.25, "1", "1") for i, p in enumerate((60, 64, 67, 72, 67, 64, 60, 55))],
            (4, 4),
            0,
        ),
        "melody-bass": (
            [(72 + i, float(i), 1.0, "1", "1") for i in range(4)]
            + [(48, 0.0, 2.0, "2", "2"), (43, 2.0, 2.0, "2", "2")],
            (4, 4),
            0,
        ),
        "two-voices": (
            [(72, 0, 4, "1", "1")] + [(60 + i, float(i), 1, "1", "2") for i in range(4)],
            (4, 4),
            0,
        ),
        "triplets": ([(60 + i, i / 3, 1 / 3, "1", "1") for i in range(6)], (4, 4), 0),
        "pickup": ([(60 + i, float(i), 1, "1", "1") for i in range(5)], (4, 4), 1),
    }
    for numerator, denominator in ((3, 4), (4, 4), (6, 8), (6, 4)):
        examples[f"meter-{numerator}-{denominator}"] = (
            [
                (60 + i % 5, i * 4 / denominator, 4 / denominator, "1", "1")
                for i in range(numerator * 2)
            ],
            (numerator, denominator),
            0,
        )
    corpus = {}
    for name, (events, meter, pickup) in examples.items():
        notes = tuple(
            EvaluationNote(
                pitch, onset * 0.5, duration * 0.5, onset, duration, staff=staff, voice=voice
            )
            for pitch, onset, duration, staff, voice in events
        )
        end = max(float(n.onset_beats or 0) + float(n.duration_beats or 0) for n in notes)
        length = meter[0] * 4 / meter[1]
        count = math.ceil((end - pickup) / length) + int(pickup > 0)
        boundaries = tuple(
            [0.0] + [pickup + i * length if pickup else (i + 1) * length for i in range(count - 1)]
        )
        corpus[name] = EvaluationScore(
            tuple(sorted(notes, key=lambda n: (float(n.onset_beats or 0), n.pitch))),
            "synthetic",
            time_signatures=((0, *meter),),
            tempos=((0, 120),),
            pickup_beats=pickup,
            first_downbeat_beats=pickup,
            measure_boundaries=boundaries,
            downbeats=boundaries[1:] if pickup else boundaries,
            beats=tuple(i * 4 / meter[1] for i in range(math.ceil(end * meter[1] / 4))),
            structure={
                "measure_count": count,
                "rest_count": 0,
                "tie_count": 0,
                "beam_groups": 0,
                "tuplet_groups": 2 if name == "triplets" else 0,
            },
        )
    return corpus


def write_corpus(directory: Path) -> Path:
    pieces = []
    for name, score in synthetic_corpus().items():
        write_json(asdict(score), directory / f"{name}.json")
        pieces.append({"id": name, "prediction": f"{name}.json", "reference": f"{name}.json"})
    manifest = directory / "manifest.json"
    write_json({"pieces": pieces}, manifest)
    return manifest
