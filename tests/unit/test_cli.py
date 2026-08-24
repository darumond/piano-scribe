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
    parser = _parser()
    args = parser.parse_args(
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
    assert parser.prog == "piano-scribe"
    assert args.model == "piano-transcription"
    assert args.device == "cpu"
    assert args.checkpoint == Path("weights/model.pth")


def test_transcribe_parser_accepts_score_reconstruction_options() -> None:
    args = _parser().parse_args(
        [
            "transcribe",
            "piano.wav",
            "--midi",
            "piano.mid",
            "--bpm",
            "60",
            "--time-signature",
            "4/4",
            "--quantization",
            "eighth-triplet",
            "--min-note-duration-ms",
            "30",
        ]
    )
    assert args.bpm == 60.0
    assert args.time_signature == "4/4"
    assert args.quantization == "eighth-triplet"
    assert args.min_note_duration_ms == 30.0


def test_analyze_parser_accepts_local_tracking_and_diagnostics() -> None:
    args = _parser().parse_args(
        [
            "analyze-score",
            "transcription.json",
            "--track-beats",
            "--first-downbeat",
            "1.23",
            "--minimum-bpm",
            "40",
            "--maximum-bpm",
            "180",
            "--rhythmic-complexity-cost",
            "0.5",
            "--beats-tsv",
            "beats.tsv",
            "--tempo-tsv",
            "tempo.tsv",
            "--quantization-tsv",
            "quantization.tsv",
        ]
    )
    assert args.track_beats
    assert args.first_downbeat == 1.23
    assert args.minimum_bpm == 40.0
    assert args.maximum_bpm == 180.0
    assert args.rhythmic_complexity_cost == 0.5
    assert args.beats_tsv == Path("beats.tsv")


def test_analyze_parser_accepts_joint_meter_and_pickup_options() -> None:
    args = _parser().parse_args(
        [
            "analyze-score",
            "transcription.json",
            "--infer-meter",
            "--pickup-beats",
            "3/2",
            "--meter-hypotheses-tsv",
            "meter-hypotheses.tsv",
            "--meter-accent-weight",
            "1.25",
        ]
    )
    assert args.infer_meter
    assert args.pickup_beats == "3/2"
    assert args.meter_hypotheses_tsv == Path("meter-hypotheses.tsv")
    assert args.meter_accent_weight == 1.25


def test_analyze_parser_accepts_sequence_rhythm_options() -> None:
    args = _parser().parse_args(
        [
            "analyze-score",
            "transcription.json",
            "--infer-meter",
            "--rhythm-optimizer",
            "sequence",
            "--rhythm-path-tsv",
            "rhythm-path.tsv",
            "--rhythm-beam-size",
            "32",
            "--rhythm-triplet-switch-weight",
            "1.2",
        ]
    )
    assert args.rhythm_optimizer == "sequence"
    assert args.rhythm_path_tsv == Path("rhythm-path.tsv")
    assert args.rhythm_beam_size == 32
    assert args.rhythm_triplet_switch_weight == 1.2
