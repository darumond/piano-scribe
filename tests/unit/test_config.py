from __future__ import annotations

import numpy as np
import pytest
from piano_transcriber.config import ModelName, PipelineConfig, create_model
from piano_transcriber.models.base import MissingModelDependencyError
from piano_transcriber.models.mock import MockTranscriptionModel


def test_select_mock_model() -> None:
    assert isinstance(create_model(ModelName.MOCK), MockTranscriptionModel)


def test_unknown_model_has_helpful_error() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        create_model("unknown")


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PipelineConfig(minimum_confidence=2.0)


def test_missing_optional_dependency_fails_gracefully() -> None:
    model = create_model(ModelName.PIANO_TRANSCRIPTION)
    with pytest.raises(MissingModelDependencyError, match="optional runtime"):
        model.transcribe(np.zeros(10, dtype=np.float32), 16_000)
