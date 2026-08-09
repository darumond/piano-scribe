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
from piano_transcriber.models.base import MissingModelDependencyError, ModelCheckpointError
from piano_transcriber.transcription.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piano-transcriber", description="Automatic piano music transcription"
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
    transcribe.set_defaults(handler=_run_transcribe)

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
    output = TranscriptionPipeline(config).run(
        args.input, midi_path=args.midi, musicxml_path=args.musicxml
    )
    print(f"Transcribed {len(output.result.notes)} notes with {output.result.model_name}")
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
