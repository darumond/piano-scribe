"""Placeholder adapter for ByteDance-style piano transcription models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from piano_transcriber.models.base import MissingModelDependencyError, TranscriptionModel
from piano_transcriber.transcription.types import TranscriptionResult


class PianoTranscriptionModel(TranscriptionModel):
    @property
    def name(self) -> str:
        return "piano-transcription"

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TranscriptionResult:
        try:
            import piano_transcription_inference  # noqa: F401
        except ImportError as error:
            raise MissingModelDependencyError(
                "The piano-specific backend requires a compatible pretrained model package and "
                'weights. Install the optional runtime with: pip install -e ".[pytorch]"; then '
                "install piano-transcription-inference and configure weights explicitly."
            ) from error
        raise NotImplementedError(
            "The piano-transcription adapter is scaffolded, but weight loading and backend API "
            "mapping must be configured explicitly; no weights are downloaded automatically."
        )
