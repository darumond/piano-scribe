"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from piano_transcriber.audio.loader import AudioLoadError, load_audio
from piano_transcriber.audio.preprocessing import calculate_rms
from piano_transcriber.config import ModelName, PipelineConfig
from piano_transcriber.midi.writer import write_score_midi
from piano_transcriber.models.base import MissingModelDependencyError, ModelCheckpointError
from piano_transcriber.notation.musicxml import write_score_musicxml
from piano_transcriber.score.diagnostics import (
    load_transcription_json,
    score_diagnostics,
    write_diagnostics_json,
    write_diagnostics_tsv,
)
from piano_transcriber.score.quantize import QuantizationGrid
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.tempo import (
    ExplicitTempo,
    MedianInterOnsetTempoEstimator,
    TempoEstimator,
)
from piano_transcriber.score.types import ReconstructedScore, TimeSignature
from piano_transcriber.transcription.pipeline import TranscriptionPipeline
from piano_transcriber.transcription.types import TranscriptionResult

logger = logging.getLogger(__name__)


def _add_score_options(parser: argparse.ArgumentParser) -> None:
    tempo = parser.add_mutually_exclusive_group()
    tempo.add_argument("--bpm", type=float, help="authoritative tempo for score reconstruction")
    tempo.add_argument(
        "--estimate-bpm",
        action="store_true",
        help="use the simple onset-based tempo baseline",
    )
    parser.add_argument(
        "--time-signature",
        default="4/4",
        help="score time signature (default: 4/4)",
    )
    parser.add_argument(
        "--quantization",
        choices=[item.value for item in QuantizationGrid],
        default=QuantizationGrid.SIXTEENTH.value,
        help="onset grid used when --bpm is supplied (default: sixteenth)",
    )
    parser.add_argument(
        "--max-quantization-error-ms",
        type=float,
        default=125.0,
        help="flag onset snaps beyond this tolerance (default: 125)",
    )
    parser.add_argument(
        "--min-note-duration-ms",
        type=float,
        help="opt-in removal threshold recorded in score diagnostics",
    )
    parser.add_argument("--diagnostics-json", type=Path, help="score diagnostics JSON path")
    parser.add_argument("--diagnostics-tsv", type=Path, help="score diagnostics TSV path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piano-scribe", description="PianoScribe automatic piano music transcription"
    )
    parser.add_argument("--verbose", action="store_true", help="enable detailed logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="transcribe audio and export notation")
    transcribe.add_argument("input", type=Path)
    transcribe.add_argument("--model", choices=[item.value for item in ModelName], default="mock")
    transcribe.add_argument("--midi", type=Path, help="output MIDI path")
    transcribe.add_argument("--musicxml", type=Path, help="output MusicXML path")
    transcribe.add_argument("--sample-rate", type=int, default=16_000)
    transcribe.add_argument("--minimum-confidence", type=float, default=0.0)
    transcribe.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device for supported models (default: auto)",
    )
    transcribe.add_argument(
        "--checkpoint",
        type=Path,
        help="explicit checkpoint path for the piano-transcription model",
    )
    _add_score_options(transcribe)
    transcribe.set_defaults(handler=_run_transcribe)

    analyze = subparsers.add_parser(
        "analyze-score",
        help="reconstruct a score from normalized transcription JSON",
    )
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--midi", type=Path, help="quantized MIDI path")
    analyze.add_argument("--musicxml", type=Path, help="quantized MusicXML path")
    _add_score_options(analyze)
    analyze.set_defaults(handler=_run_analyze_score)

    inspect = subparsers.add_parser("inspect", help="show audio metadata and levels")
    inspect.add_argument("input", type=Path)
    inspect.set_defaults(handler=_run_inspect)
    return parser


def _run_transcribe(args: argparse.Namespace) -> int:
    if args.midi is None and args.musicxml is None:
        raise ValueError("specify at least one output: --midi or --musicxml")
    config = PipelineConfig(
        model=ModelName(args.model),
        target_sample_rate=args.sample_rate,
        minimum_confidence=args.minimum_confidence,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if args.bpm is None and not args.estimate_bpm:
        if args.diagnostics_json is not None or args.diagnostics_tsv is not None:
            raise ValueError("score diagnostics require --bpm")
        output = TranscriptionPipeline(config).run(
            args.input, midi_path=args.midi, musicxml_path=args.musicxml
        )
        print(f"Transcribed {len(output.result.notes)} notes with {output.result.model_name}")
        return 0

    output = TranscriptionPipeline(config).run(args.input)
    score = reconstruct_score(output.result, _score_config(args, output.result))
    _write_score_outputs(score, args)
    print(f"Transcribed {len(output.result.notes)} notes with {output.result.model_name}")
    print(
        f"Reconstructed {len(score.notes)} written notes in {score.measure_count} measures "
        f"at {score.bpm:g} BPM"
    )
    return 0


def _score_config(args: argparse.Namespace, result: TranscriptionResult) -> ReconstructionConfig:
    tempo: TempoEstimator
    if args.bpm is not None:
        tempo = ExplicitTempo(args.bpm)
    elif args.estimate_bpm:
        tempo = MedianInterOnsetTempoEstimator()
    else:
        raise ValueError("score reconstruction requires --bpm or --estimate-bpm")
    return ReconstructionConfig(
        bpm=tempo.estimate_bpm(result),
        grid=QuantizationGrid(args.quantization),
        maximum_quantization_error_ms=args.max_quantization_error_ms,
        minimum_note_duration_ms=args.min_note_duration_ms,
        time_signature=TimeSignature.parse(args.time_signature),
    )


def _write_score_outputs(score: ReconstructedScore, args: argparse.Namespace) -> None:
    if args.midi is not None:
        write_score_midi(score, args.midi)
        logger.info("Wrote reconstructed MIDI to %s", args.midi)
    if args.musicxml is not None:
        write_score_musicxml(score, args.musicxml)
        logger.info("Wrote reconstructed MusicXML to %s", args.musicxml)
    if args.diagnostics_json is not None:
        write_diagnostics_json(score, args.diagnostics_json)
        logger.info("Wrote score diagnostics JSON to %s", args.diagnostics_json)
    if args.diagnostics_tsv is not None:
        write_diagnostics_tsv(score, args.diagnostics_tsv)
        logger.info("Wrote score diagnostics TSV to %s", args.diagnostics_tsv)


def _run_analyze_score(args: argparse.Namespace) -> int:
    result = load_transcription_json(args.input)
    score = reconstruct_score(
        result,
        _score_config(args, result),
    )
    _write_score_outputs(score, args)
    summary = score_diagnostics(score)
    print(f"BPM: {score.bpm:g}")
    print(f"Grid: {score.grid_name} ({score.grid_step_beats} quarter-note beats)")
    print(f"Written notes: {len(score.notes)}")
    print(f"Chord groups: {len(score.chords)}")
    print(f"Measures: {score.measure_count}")
    print(f"Actions: {summary['actions']}")
    print(f"Rhythms: {summary['rhythmic_values']}")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    audio = load_audio(args.input)
    flat = np.asarray(audio.samples.reshape(-1), dtype=np.float32)
    print(f"Sample rate: {audio.sample_rate} Hz")
    print(f"Duration: {audio.duration_seconds:.3f} s")
    print(f"Channels: {audio.channels}")
    print(f"RMS: {calculate_rms(flat):.6f}")
    print(f"Peak amplitude: {float(np.max(np.abs(flat))):.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except (
        AudioLoadError,
        MissingModelDependencyError,
        ModelCheckpointError,
        ValueError,
        NotImplementedError,
    ) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
