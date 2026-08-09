from __future__ import annotations

from pathlib import Path

import pytest
from piano_transcriber.config import ModelName, PipelineConfig, create_model
from piano_transcriber.models.mock import MockTranscriptionModel
from piano_transcriber.models.piano_transcription import PianoTranscriptionModel


def test_select_mock_model() -> None:
    assert isinstance(create_model(ModelName.MOCK), MockTranscriptionModel)


def test_unknown_model_has_helpful_error() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        create_model("unknown")


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PipelineConfig(minimum_confidence=2.0)


def test_select_piano_model_with_runtime_options(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    model = create_model(ModelName.PIANO_TRANSCRIPTION, checkpoint_path=checkpoint, device="cpu")
    assert isinstance(model, PianoTranscriptionModel)
    assert model.checkpoint_path == checkpoint
    assert model.device == "cpu"


def test_checkpoint_is_rejected_for_unrelated_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only valid"):
        PipelineConfig(checkpoint_path=tmp_path / "model.pth")
