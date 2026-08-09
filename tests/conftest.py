from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf


def synthesize_piano_like(
    *, sample_rate: int = 16_000, duration_seconds: float = 1.2
) -> npt.NDArray[np.float32]:
    """Generate deterministic decaying C-major sine partials for tests."""
    times = np.arange(round(sample_rate * duration_seconds), dtype=np.float64) / sample_rate
    envelope = np.exp(-2.5 * times)
    signal = sum(np.sin(2.0 * np.pi * frequency * times) for frequency in (261.63, 329.63, 392.0))
    return np.asarray(0.2 * envelope * signal / 3.0, dtype=np.float32)


@pytest.fixture
def synthetic_wav(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-piano.wav"
    sf.write(path, synthesize_piano_like(), 16_000, subtype="FLOAT")
    return path
