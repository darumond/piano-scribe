from __future__ import annotations

import numpy as np
from piano_transcriber.models.mock import MockTranscriptionModel


def test_mock_model_is_deterministic() -> None:
    audio = np.zeros(16_000, dtype=np.float32)
    model = MockTranscriptionModel()
    first = model.transcribe(audio, 16_000)
    second = model.transcribe(audio, 16_000)
    assert first == second
    assert [note.pitch for note in first.notes] == [60, 64, 67]
    assert first.model_name == "mock"


def test_mock_model_handles_empty_audio() -> None:
    result = MockTranscriptionModel().transcribe(np.array([], dtype=np.float32), 16_000)
    assert result.notes == ()
