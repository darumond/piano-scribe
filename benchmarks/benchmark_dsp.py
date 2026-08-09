"""Microbenchmark NumPy and native DSP implementations without external frameworks."""

from __future__ import annotations

import timeit

import numpy as np
from piano_transcriber import _native


def main() -> None:
    positions = np.arange(48_000 * 30, dtype=np.float32) / 48_000
    samples = np.asarray(
        0.3 * np.sin(2 * np.pi * 220.0 * positions) + 0.2 * np.sin(2 * np.pi * 440.0 * positions),
        dtype=np.float32,
    )
    iterations = 20
    python_time = timeit.timeit(lambda: samples / np.max(np.abs(samples)), number=iterations)
    native_time = timeit.timeit(lambda: _native.peak_normalize(samples), number=iterations)
    print(f"NumPy peak normalization: {python_time / iterations:.6f} s/run")
    print(f"C++ peak normalization:   {native_time / iterations:.6f} s/run")


if __name__ == "__main__":
    main()
