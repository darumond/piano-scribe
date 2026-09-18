"""Offline sparse matching benchmark; writes only to an explicit output path."""

from __future__ import annotations

import argparse
import platform
import statistics
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from piano_transcriber.evaluation.report import write_json
from piano_transcriber.evaluation.score import evaluate
from piano_transcriber.evaluation.synthetic import write_corpus
from piano_transcriber.evaluation.types import EvaluationConfig, EvaluationNote, EvaluationScore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, default=8000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", type=Path, default=Path("outputs/evaluation/benchmark.json"))
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.notes < 1 or args.runs < 1:
        parser.error("notes and runs must be positive")
    reference = EvaluationScore(
        tuple(
            EvaluationNote(
                48 + i % 36,
                i * 0.125,
                0.11,
                i * 0.25,
                0.22,
                staff=str(1 + (i % 36 < 12)),
                voice=str(1 + i % 2),
            )
            for i in range(args.notes)
        )
    )
    prediction = replace(
        reference,
        notes=tuple(
            replace(
                n,
                pitch=n.pitch + 12 if i % 97 == 0 else n.pitch,
                onset_seconds=float(n.onset_seconds or 0) + 0.008,
                onset_beats=float(n.onset_beats or 0) + 0.016,
            )
            for i, n in enumerate(reference.notes)
        ),
    )
    timings = []
    result = {}
    for _ in range(args.runs):
        start = perf_counter()
        result = evaluate(prediction, reference, EvaluationConfig())
        timings.append(perf_counter() - start)
    report = {
        "notes_per_input": args.notes,
        "runs": args.runs,
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "note_f1": result["metrics"]["notes"]["f1"],
        "scope": "full metric computation including alignment; no file loading or inference",
    }
    write_json(report, args.json)
    print(report)
    if args.corpus:
        print(f"Synthetic corpus: {write_corpus(args.corpus)}")


if __name__ == "__main__":
    main()
