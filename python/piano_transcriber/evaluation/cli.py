"""Evaluation commands registered by the application CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from piano_transcriber.evaluation.report import compare, terminal_summary, write_json, write_table
from piano_transcriber.evaluation.suite import evaluate_files, evaluate_suite
from piano_transcriber.evaluation.types import EvaluationConfig


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", type=Path, default=Path("outputs/evaluation/metrics.json"))
    parser.add_argument("--table", type=Path, help="optional .tsv or .csv metric table")
    parser.add_argument("--onset-tolerances-ms", type=float, nargs="+", default=[25, 50, 100])
    parser.add_argument("--onset-tolerance-beats", type=float, default=0.125)
    parser.add_argument("--duration-tolerance-seconds", type=float)
    parser.add_argument("--duration-tolerance-beats", type=float)
    parser.add_argument("--chord-group-seconds", type=float, default=0.03)
    parser.add_argument("--chord-group-beats", type=float, default=0.0625)
    parser.add_argument("--boundary-tolerance-beats", type=float, default=0.125)
    parser.add_argument(
        "--alignment", choices=["auto", "none", "symbolic", "offset", "affine"], default="auto"
    )
    parser.add_argument("--max-tempo-scale-deviation", type=float, default=0.05)
    parser.add_argument("--stage", choices=["auto", "transcription", "symbolic"], default="auto")
    parser.add_argument("--prediction-voice-scope", choices=["part", "staff"], default="part")
    parser.add_argument("--reference-voice-scope", choices=["part", "staff"], default="part")


def register_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    single = subparsers.add_parser(
        "evaluate", help="compare a cached prediction to reference MIDI/MusicXML"
    )
    single.add_argument("--prediction", type=Path, required=True)
    single.add_argument("--reference", type=Path, required=True)
    _options(single)
    single.set_defaults(handler=_run)
    suite = subparsers.add_parser(
        "evaluate-suite", help="evaluate a YAML/JSON manifest without running inference"
    )
    suite.add_argument("manifest", type=Path)
    _options(suite)
    suite.set_defaults(handler=_run)
    comparison = subparsers.add_parser(
        "compare-evaluations", help="report separate metric regressions and improvements"
    )
    comparison.add_argument("old", type=Path)
    comparison.add_argument("new", type=Path)
    comparison.add_argument("--json", type=Path, default=Path("outputs/evaluation/comparison.json"))
    comparison.set_defaults(handler=_compare)


def _run(args: argparse.Namespace) -> int:
    config = EvaluationConfig(
        onset_tolerances_ms=tuple(args.onset_tolerances_ms),
        onset_tolerance_beats=args.onset_tolerance_beats,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        duration_tolerance_beats=args.duration_tolerance_beats,
        chord_group_seconds=args.chord_group_seconds,
        chord_group_beats=args.chord_group_beats,
        boundary_tolerance_beats=args.boundary_tolerance_beats,
        alignment=args.alignment,
        max_tempo_scale_deviation=args.max_tempo_scale_deviation,
        stage=args.stage,
        prediction_voice_scope=args.prediction_voice_scope,
        reference_voice_scope=args.reference_voice_scope,
    )
    try:
        report = (
            evaluate_suite(args.manifest, config)
            if args.command == "evaluate-suite"
            else evaluate_files(args.prediction, args.reference, config)
        )
        write_json(report, args.json)
        if args.table:
            write_table(report, args.table)
    except OSError as error:
        raise ValueError(f"evaluation file error: {error}") from error
    print(terminal_summary(report))
    print(f"Metrics: {args.json}")
    return 2 if report.get("errors") or report.get("pending") else 0


def _compare(args: argparse.Namespace) -> int:
    try:
        report = compare(
            json.loads(args.old.read_text(encoding="utf-8")),
            json.loads(args.new.read_text(encoding="utf-8")),
        )
        write_json(report, args.json)
    except OSError as error:
        raise ValueError(f"comparison file error: {error}") from error
    print(json.dumps(report, indent=2))
    return 0
