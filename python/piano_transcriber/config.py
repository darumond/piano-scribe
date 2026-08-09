"""Configuration and model construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from piano_transcriber.models.base import TranscriptionModel


class ModelName(StrEnum):
    MOCK = "mock"
    BASIC_PITCH = "basic-pitch"
    PIANO_TRANSCRIPTION = "piano-transcription"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    model: ModelName = ModelName.MOCK
    target_sample_rate: int | None = 16_000
    normalize_audio: bool = True
    minimum_confidence: float = 0.0
    checkpoint_path: Path | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.target_sample_rate is not None and self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.checkpoint_path is not None and self.model is not ModelName.PIANO_TRANSCRIPTION:
            raise ValueError("checkpoint_path is only valid for the piano-transcription model")


def create_model(
    name: ModelName | str,
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> TranscriptionModel:
    """Construct a backend without importing optional dependencies eagerly."""
    try:
        model_name = ModelName(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in ModelName)
        raise ValueError(f"unknown model {name!r}; choose one of: {choices}") from error

    if model_name is ModelName.MOCK:
        from piano_transcriber.models.mock import MockTranscriptionModel

        return MockTranscriptionModel()
    if model_name is ModelName.BASIC_PITCH:
        from piano_transcriber.models.basic_pitch import BasicPitchModel

        return BasicPitchModel()

    from piano_transcriber.models.piano_transcription import PianoTranscriptionModel

    return PianoTranscriptionModel(checkpoint_path=checkpoint_path, device=device)
