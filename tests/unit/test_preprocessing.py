from __future__ import annotations

import numpy as np
from piano_transcriber.audio.preprocessing import (
    calculate_rms,
    peak_normalize,
    preprocess_audio,
    resample_audio,
    to_mono,
)


def test_stereo_to_mono() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.0, 0.375])


def test_peak_normalize_and_rms() -> None:
    normalized = peak_normalize(np.array([-0.5, 0.25], dtype=np.float32))
    np.testing.assert_allclose(normalized, [-1.0, 0.5])
    assert calculate_rms(np.array([1.0, -1.0], dtype=np.float32)) == 1.0


def test_resample_preserves_approximate_duration() -> None:
    samples = np.arange(8_000, dtype=np.float32)
    result = resample_audio(samples, 8_000, 16_000)
    assert result.shape == (16_000,)


def test_preprocess_converts_normalizes_and_resamples() -> None:
    stereo = np.tile(np.array([[0.25, 0.5]], dtype=np.float32), (8_000, 1))
    output, sample_rate = preprocess_audio(stereo, 8_000, target_sample_rate=16_000, normalize=True)
    assert output.ndim == 1
    assert output.shape == (16_000,)
    assert sample_rate == 16_000
    assert np.max(np.abs(output)) == 1.0
