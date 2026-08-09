"""Small, deterministic preprocessing operations."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from piano_transcriber import _native
else:
    try:
        from piano_transcriber import _native
    except ImportError:  # pragma: no cover - source-tree fallback
        _native = None


def to_mono(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if samples.ndim == 1:
        return np.ascontiguousarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] == 0:
        raise ValueError("samples must have shape (frames,) or (frames, channels)")
    contiguous = np.ascontiguousarray(samples, dtype=np.float32)
    if _native is not None:
        return np.asarray(
            _native.interleaved_to_mono(contiguous.reshape(-1), contiguous.shape[1]),
            dtype=np.float32,
        )
    logger.debug("native extension unavailable; using NumPy mono conversion")
    return np.asarray(contiguous.mean(axis=1), dtype=np.float32)


def peak_normalize(
    samples: npt.NDArray[np.float32], target_peak: float = 1.0
) -> npt.NDArray[np.float32]:
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be in the interval (0, 1]")
    contiguous = np.ascontiguousarray(samples, dtype=np.float32)
    if _native is not None:
        return np.asarray(_native.peak_normalize(contiguous, target_peak), dtype=np.float32)
    peak = float(np.max(np.abs(contiguous), initial=0.0))
    if peak == 0.0:
        return contiguous.copy()
    return np.asarray(contiguous * (target_peak / peak), dtype=np.float32)


def calculate_rms(samples: npt.NDArray[np.float32]) -> float:
    contiguous = np.ascontiguousarray(samples.reshape(-1), dtype=np.float32)
    if _native is not None:
        return float(_native.rms(contiguous))
    if contiguous.size == 0:
        return 0.0
    values = contiguous.astype(np.float64)
    return float(np.sqrt(np.mean(values * values)))


def resample_audio(
    samples: npt.NDArray[np.float32], source_rate: int, target_rate: int
) -> npt.NDArray[np.float32]:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or samples.size == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(samples, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def preprocess_audio(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
    *,
    target_sample_rate: int | None,
    normalize: bool,
) -> tuple[npt.NDArray[np.float32], int]:
    mono = to_mono(samples)
    if normalize:
        mono = peak_normalize(mono)
    if target_sample_rate is not None and target_sample_rate != sample_rate:
        mono = resample_audio(mono, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
        if normalize:
            # Polyphase filters can ring slightly above the pre-resampling peak.
            mono = peak_normalize(mono)
    return mono, sample_rate
