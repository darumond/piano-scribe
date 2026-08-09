"""Backend contract shared by every transcription model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from piano_transcriber.transcription.types import TranscriptionResult


class MissingModelDependencyError(RuntimeError):
    """Raised when a selected optional backend has not been installed."""


class TranscriptionModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier."""

    @abstractmethod
    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TranscriptionResult:
        """Return normalized note events for mono floating-point audio."""
