"""Validated audio-file loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf


class AudioLoadError(ValueError):
    """Raised when an input cannot be decoded as supported audio."""


@dataclass(frozen=True, slots=True)
class AudioData:
    samples: npt.NDArray[np.float32]
    sample_rate: int
    channels: int

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[0]) / self.sample_rate


def load_audio(path: str | Path) -> AudioData:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioLoadError(f"audio file does not exist: {audio_path}")
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as error:
        raise AudioLoadError(f"cannot decode audio file {audio_path}: {error}") from error
    if sample_rate <= 0 or samples.shape[0] == 0:
        raise AudioLoadError(f"audio file is empty or malformed: {audio_path}")
    if not np.isfinite(samples).all():
        raise AudioLoadError(f"audio file contains non-finite samples: {audio_path}")
    channels = int(samples.shape[1])
    if channels < 1:
        raise AudioLoadError(f"audio file contains no channels: {audio_path}")
    return AudioData(np.asarray(samples, dtype=np.float32), int(sample_rate), channels)
