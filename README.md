# Piano Transcriber

Piano Transcriber is a production-oriented foundation for automatic piano music transcription. It
loads WAV or MP3 audio, prepares a mono signal, delegates note inference to a replaceable backend,
and exports normalized note events as MIDI and MusicXML. Version 0.1 deliberately ships with a
deterministic mock backend; optional pretrained-model adapters never download weights during setup
or tests.

## Architecture

The project uses a hybrid Python/C++ design:

- Python owns audio I/O, polyphase resampling/preprocessing orchestration, model adapters, domain validation,
  postprocessing, MIDI/MusicXML output, the CLI, and tests.
- A small C++17 library owns reusable DSP primitives: interleaved-to-mono conversion, RMS, peak
  normalization, and framing. `pybind11` exposes these functions as `piano_transcriber._native`.
- `TranscriptionModel` is the only contract the pipeline knows. Backends return an immutable
  `TranscriptionResult` containing validated `NoteEvent` values, so export code never depends on a
  framework or model-specific representation.

The included backends are:

- `mock`: deterministic C-major notes for tests and end-to-end demonstrations.
- `basic-pitch`: a lazy Spotify Basic Pitch adapter. Its dependency is optional.
- `piano-transcription`: an explicit integration seam for a ByteDance-style PyTorch piano model.
  It reports the missing runtime/configuration without fetching weights.

MusicXML output is intentionally basic: it creates a readable single-part piano score but does not
yet perform beat tracking, voice allocation, or sophisticated rhythmic engraving.

## Requirements and installation

- Python 3.11 or newer
- A C++17 compiler
- CMake 3.20 or newer (installed by the `dev` extra if needed)
- Ninja or another CMake generator

Create and activate a virtual environment, then install an editable development build:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`scikit-build-core` invokes CMake and installs the compiled extension with the Python package.
Nothing in the default or development installation downloads model weights.

Optional integrations:

```bash
python -m pip install -e ".[basic-pitch]"
python -m pip install -e ".[pytorch]"
```

The PyTorch extra provides a base runtime only. Install a compatible piano transcription package
and configure its checkpoint explicitly before completing that adapter. This keeps weight choice,
storage, and licensing visible to the operator.

## CLI

Transcribe a file using the offline deterministic backend:

```bash
piano-transcriber transcribe input.wav --model mock --midi output.mid \
  --musicxml output.musicxml
```

Select another backend with `--model basic-pitch` or `--model piano-transcription`. Missing optional
dependencies produce a concise install hint and a non-zero exit code.

Inspect decoded audio:

```bash
piano-transcriber inspect input.wav
```

The command prints sample rate, duration, channel count, RMS, and peak amplitude. Use `--verbose`
before the subcommand for detailed logs.

## Development and testing

Run all Python checks:

```bash
ruff format --check .
ruff check .
mypy
pytest
```

Build and exercise the native library directly:

```bash
cmake -S . -B build/native -G Ninja \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

On multi-configuration generators, pass `--config Release` to the build and CTest commands.

The test suite generates a small decaying C-major signal at runtime. Tests require neither network
access, external fixtures, a GPU, nor model weights. To run the initial DSP comparison:

```bash
python benchmarks/benchmark_dsp.py
```

## Repository layout

```text
cpp/                         C++ DSP library, bindings, and native tests
python/piano_transcriber/    Installable Python package
  audio/                     Loading and preprocessing
  models/                    Backend interface and adapters
  transcription/             Domain types, postprocessing, pipeline
  midi/                      MIDI serialization
  notation/                  MusicXML serialization
tests/unit/                  Focused Python tests
tests/integration/           Synthetic-audio end-to-end test
benchmarks/                  Python/native DSP comparison scaffold
```

## Roadmap

1. Audio → mock transcription → MIDI/MusicXML (this bootstrap).
2. Integrate and validate a pretrained piano transcription model.
3. Add tempo estimation and rhythm quantization.
4. Separate left- and right-hand voices.
5. Fine-tune on MAESTRO or another licensed piano dataset.
6. Experiment with a purpose-built PyTorch or JAX architecture.
