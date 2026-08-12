# PianoScribe

PianoScribe is a production-oriented foundation for automatic piano music transcription. It
loads WAV or MP3 audio, prepares a mono signal, delegates note inference to a replaceable backend,
and exports normalized note events as MIDI and MusicXML. Optional pretrained-model adapters never
download weights during setup or automated tests.

## Architecture

The project uses a hybrid Python/C++ design:

- Python owns audio I/O, polyphase resampling/preprocessing orchestration, model adapters, domain
  validation, symbolic score reconstruction, MIDI/MusicXML output, the CLI, and tests.
- A small C++17 library owns reusable DSP primitives: interleaved-to-mono conversion, RMS, peak
  normalization, and framing. `pybind11` exposes these functions as `piano_transcriber._native`.
- `TranscriptionModel` is the only contract the pipeline knows. Backends return an immutable
  `TranscriptionResult` containing validated `NoteEvent` values, so export code never depends on a
  framework or model-specific representation.

The included backends are:

- `mock`: deterministic C-major notes for tests and end-to-end demonstrations.
- `basic-pitch`: a lazy Spotify Basic Pitch adapter. Its dependency is optional.
- `piano-transcription`: the pretrained ByteDance high-resolution piano CRNN, including note
  onset, offset, velocity, and sustain-pedal predictions. It supports CPU and CUDA execution.

Raw MusicXML output remains available when no tempo is supplied. When score reconstruction is
enabled, acoustic events are converted to exact quarter-note units, quantized, grouped into chords,
and written into measures with conventional note values. The first reconstruction layer does not
yet perform robust beat tracking, voice allocation, staff separation, or sophisticated engraving.

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

The `pytorch` extra installs PyTorch and
[`piano-transcription-inference`](https://github.com/qiuqiangkong/piano_transcription_inference).
It installs code only; model weights are resolved when that backend is first used.

### Piano transcription checkpoint and devices

By default, the first real transcription downloads the official 171,966,578-byte checkpoint from
[Zenodo record 4034264](https://zenodo.org/records/4034264). The download is written atomically and
validated against the size and MD5 checksum published by Zenodo. Subsequent runs reuse it from the
platform cache:

- Linux: `${XDG_CACHE_HOME:-~/.cache}/piano-scribe/models/`
- macOS: `~/Library/Caches/piano-scribe/models/`
- Windows: `%LOCALAPPDATA%\piano-scribe\models\`

Set `PIANO_SCRIBE_CACHE_DIR` to override the cache root. To prevent any download, pass an
existing checkpoint explicitly with `--checkpoint /path/to/model.pth`.

`--device auto` uses CUDA when `torch.cuda.is_available()` is true and otherwise uses CPU. Use
`--device cpu` for predictable development runs or `--device cuda` to require CUDA. CUDA is never a
base requirement.

The upstream ByteDance training repository is archived and documents Python 3.7/PyTorch 1.4. Its
inference package 0.0.6 was published in 2025 but still describes Windows as untested. This project
isolates that legacy API behind the adapter and never invokes its built-in `wget` downloader. The
optional dependency range targets current PyTorch 2.x, but operators should test their specific
OS, PyTorch build, and audio workload before production deployment.

## CLI

Transcribe a file using the offline deterministic backend:

```bash
piano-scribe transcribe input.wav --model mock --midi output.mid \
  --musicxml output.musicxml
```

Select another backend with `--model basic-pitch` or `--model piano-transcription`. Missing optional
dependencies produce a concise install hint and a non-zero exit code.

Manually run the real piano backend on CPU:

```bash
piano-scribe --verbose transcribe piano.wav \
  --model piano-transcription \
  --device cpu \
  --midi piano.mid \
  --musicxml piano.musicxml
```

Reconstruct rhythm at an authoritative tempo while transcribing:

```bash
piano-scribe transcribe piano.wav \
  --model piano-transcription \
  --device cpu \
  --bpm 60 \
  --time-signature 4/4 \
  --quantization sixteenth \
  --midi piano.mid \
  --musicxml piano.musicxml \
  --diagnostics-json score-diagnostics.json \
  --diagnostics-tsv score-notes.tsv
```

`--bpm` is authoritative. `--estimate-bpm` selects a deliberately small onset-interval baseline;
dense polyphonic passages can make that estimate ambiguous. Supported grids are `quarter`,
`eighth`, `eighth-triplet`, `sixteenth`, `sixteenth-triplet`, and `thirty-second`.

Very short acoustic events are retained by default and receive a suspicious-event diagnostic.
Use `--min-note-duration-ms 30` to opt into filtering; every filtered or merged event remains in the
JSON/TSV diagnostics. The current pedal heuristic prevents a pedal-overlapped acoustic offset from
automatically becoming a written duration by capping it at the next quantized onset. Both raw times
and the shortening decision remain observable.

An existing normalized transcription JSON can be reconstructed without rerunning a model:

```bash
piano-scribe analyze-score transcription.json \
  --bpm 60 \
  --time-signature 4/4 \
  --quantization sixteenth-triplet \
  --midi score.mid \
  --musicxml score.musicxml \
  --diagnostics-json diagnostics.json \
  --diagnostics-tsv notes.tsv
```

The diagnostics report raw and quantized onsets, timing error, raw and written durations, measure
position, chord size, suspicious classifications, filtering, merging, and pedal-aware shortening.

Use a local checkpoint and automatic CUDA selection when available:

```bash
piano-scribe --verbose transcribe piano.wav \
  --model piano-transcription \
  --device auto \
  --checkpoint /path/to/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth \
  --midi piano.mid \
  --musicxml piano.musicxml
```

The model expects 16 kHz mono audio; the default pipeline preprocessing performs that conversion.
Model loading and inference durations are logged. A manual real-piano smoke recording has exercised
the complete backend, but that is not a representative quality evaluation corpus.

Inspect decoded audio:

```bash
piano-scribe inspect input.wav
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
access, external fixtures, a GPU, nor model weights. The real adapter is tested through injected
runtime and checkpoint collaborators. To run the initial DSP comparison:

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
  score/                     Exact beat timing, quantization, chords, and diagnostics
  midi/                      MIDI serialization
  notation/                  MusicXML serialization
tests/unit/                  Focused Python tests
tests/integration/           Synthetic-audio end-to-end test
benchmarks/                  Python/native DSP comparison scaffold
```

## Roadmap

1. Audio → mock transcription → MIDI/MusicXML.
2. Integrate a pretrained piano transcription model and validate it on a representative corpus.
3. Add tempo estimation and rhythm quantization.
4. Separate left- and right-hand voices.
5. Fine-tune on MAESTRO or another licensed piano dataset.
6. Experiment with a purpose-built PyTorch or JAX architecture.
