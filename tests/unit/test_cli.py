from __future__ import annotations

from pathlib import Path

from piano_transcriber.cli import _parser, main


def test_inspect_prints_metadata(synthetic_wav: Path, capsys: object) -> None:
    assert main(["inspect", str(synthetic_wav)]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Sample rate: 16000 Hz" in output
    assert "Channels: 1" in output
    assert "RMS:" in output


def test_transcribe_requires_output(synthetic_wav: Path) -> None:
    assert main(["transcribe", str(synthetic_wav)]) == 2


def test_transcribe_parser_accepts_piano_runtime_options() -> None:
    args = _parser().parse_args(
        [
            "transcribe",
            "piano.wav",
            "--model",
            "piano-transcription",
            "--device",
            "cpu",
            "--checkpoint",
            "weights/model.pth",
            "--midi",
            "piano.mid",
        ]
    )
    assert args.model == "piano-transcription"
    assert args.device == "cpu"
    assert args.checkpoint == Path("weights/model.pth")
