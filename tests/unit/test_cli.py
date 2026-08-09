from __future__ import annotations

from pathlib import Path

from piano_transcriber.cli import main


def test_inspect_prints_metadata(synthetic_wav: Path, capsys: object) -> None:
    assert main(["inspect", str(synthetic_wav)]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Sample rate: 16000 Hz" in output
    assert "Channels: 1" in output
    assert "RMS:" in output


def test_transcribe_requires_output(synthetic_wav: Path) -> None:
    assert main(["transcribe", str(synthetic_wav)]) == 2
