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
and written into measures with conventional note values. An optional post-rhythm stage assigns
piano hands, treble/bass staves, and within-staff voices before producing grand-staff MusicXML.

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

For local beat, downbeat, and tempo tracking from symbolic note attacks, use `--track-beats`:

```bash
piano-scribe analyze-score transcription.json \
  --track-beats \
  --minimum-bpm 30 \
  --maximum-bpm 220 \
  --time-signature 4/4 \
  --rhythmic-complexity-cost 0.35 \
  --midi score.mid \
  --musicxml score.musicxml \
  --beats-tsv beats.tsv \
  --tempo-tsv tempo.tsv \
  --quantization-tsv quantization.tsv \
  --diagnostics-json diagnostics.json
```

The local tracker clusters near-simultaneous chord attacks, weights velocity, bass motion, chord
changes, and onset density, then follows a bounded pulse with robust local-period smoothing. It
reports confidence for every beat and for the inferred downbeat phase. Supported manual meters are
2/4, 3/4, 4/4, 6/4, 6/8, 9/8, and 12/8. Use `--first-beat` and `--first-downbeat` (timestamps in
seconds) for manual alignment. The same overrides work with explicit `--bpm`, preserving a
deterministic constant-tempo mode. `--pickup-beats` accepts quarter-note units, including fractions
such as `3/2`; pickup measures are emitted as implicit MusicXML measure 0 without front padding.

Beat-aware quantization scores all supported subdivisions using timing error plus a configurable
notation-complexity penalty. This makes straight eighths preferable to triplet or finer grids when
their timing fits are close. Candidate timing errors, penalties, and the selected subdivision are
included per note in JSON and TSV diagnostics. Local tempo changes are emitted as MIDI tempo events
and MusicXML measure-level tempo directions.

For automatic joint pulse, meter, downbeat, and pickup selection, use `--infer-meter`:

```bash
piano-scribe analyze-score transcription.json \
  --infer-meter \
  --minimum-bpm 30 \
  --maximum-bpm 220 \
  --midi score.mid \
  --musicxml score.musicxml \
  --beats-tsv beats.tsv \
  --tempo-tsv tempo.tsv \
  --meter-hypotheses-tsv meter-hypotheses.tsv \
  --quantization-tsv quantization.tsv \
  --diagnostics-json diagnostics.json
```

The global evaluator reconstructs every candidate path across the full excerpt. It tests pulse
factors 0.5, 2/3, 1, 1.5, and 2 against all supported meters and half-beat downbeat phases. A single
pulse level is held for the entire hypothesis, preventing frame-to-frame tempo-level switching.
Each path combines median timing fit, local-tempo smoothness, metric-accent recurrence, rhythmic
complexity, ties, pickup length, and distance from the tracker's initial pulse. Compound meters also
score their dotted-quarter grouping and report notated eighth-note and higher-level BPM values. The
report retains every
hypothesis, normalized scores, relative scores, and the best-versus-runner-up confidence margin;
a small margin should be treated as ambiguity, not a confident meter decision.

The default objective weights are timing `1.0`, tempo smoothness `0.2`, metric accent `0.8`, rhythm
complexity `0.55`, ties `0.12`, pickup `0.12`, and tempo-level distance `0.08`. Override them with
`--meter-timing-weight`, `--meter-tempo-smoothness-weight`, `--meter-accent-weight`,
`--meter-rhythm-complexity-weight`, `--meter-tie-weight`, `--meter-pickup-weight`, and
`--meter-tempo-level-weight`. These generic defaults are not fitted to a particular composition.

### Phrase-level rhythm optimization

Local event-by-event quantization remains the default for reproducible comparison. Select the
bounded phrase-level optimizer with `--rhythm-optimizer sequence`:

```bash
piano-scribe analyze-score transcription.json \
  --infer-meter \
  --rhythm-optimizer sequence \
  --midi score.mid \
  --musicxml score.musicxml \
  --meter-hypotheses-tsv meter-hypotheses.tsv \
  --rhythm-path-tsv rhythm-path.tsv \
  --quantization-tsv quantization.tsv \
  --diagnostics-json diagnostics.json
```

The existing quantizer still generates quarter, straight and triplet eighth, straight and triplet
sixteenth, and thirty-second candidates. Notes attacked within 45 ms form one onset group and must
share an onset. Candidates more than 55 ms beyond the best timing fit are pruned, at most five onset
candidates and four duration candidates per group are retained, and a deterministic beam of 64
paths carries state across measure boundaries. Duration pruning keeps alternatives within 200 ms of
the best acoustic-offset fit. These limits are configurable with `--rhythm-candidate-limit`,
`--rhythm-candidate-window-ms`, `--rhythm-duration-candidate-limit`,
`--rhythm-duration-window-ms`, and `--rhythm-beam-size`.

The sequence cost combines onset timing (`1.0`), duration timing (`0.6`), notation complexity
(`0.65`), family switches (`0.35`), straight/triplet switches (`0.9`), isolated triplets (`0.65`),
dotted micro-values (`0.65`), thirty-seconds (`0.65`), unusual short values (`0.4`), ties (`0.25`),
tiny tie fragments (`0.8`), metric accents (`0.12`), pickup plausibility (`0.1`), repeated spacing
patterns (`0.45`), and duration-pattern consistency (`0.2`). Every value has a corresponding
`--rhythm-*-weight` CLI option and is also available through `RhythmSequenceWeights`. Fine values
and triplets remain reachable when sustained timing evidence supports them; the penalties are
priors rather than prohibitions.

`rhythm-path.tsv` reports the selected family, timing and complexity costs, transition and
cumulative costs, the rejected local best candidate, and the reason for each changed decision.
JSON diagnostics add family-switch, isolated-value, tie, complexity, timing, changed-event, search
size, and optimizer-runtime aggregates. During `--infer-meter`, meter selection intentionally uses
the local quantizer first; sequence optimization is then applied to the selected hypothesis so a
local-versus-sequence comparison does not silently change the meter experiment.

### Piano hand, staff, and voice separation

Hand and voice inference is a separate post-rhythm stage and is disabled by default. Enable its
bounded deterministic search with `--piano-layout sequence`:

```bash
piano-scribe analyze-score transcription.json \
  --infer-meter \
  --rhythm-optimizer sequence \
  --piano-layout sequence \
  --midi score.mid \
  --musicxml score.musicxml \
  --staff-assignment-tsv staff-assignment.tsv \
  --voice-assignment-tsv voice-assignment.tsv \
  --diagnostics-json diagnostics.json
```

Each onset group retains joint hand candidates: all-left, all-right, register-ordered chord splits,
and bounded crossing alternatives. A beam of 64 paths combines the configurable middle-C register
prior with melodic continuity, large-jump, rapid hand-switch, crossing, compact-chord-split,
wide-span, and hand-load costs. Register is only a prior: continuity can keep a right-hand line below
the split pitch or a left-hand line above it. Configure search bounds with `--hand-beam-size`,
`--hand-candidate-limit`, and the `--hand-*-weight` options.

Default hand weights are register `0.55`, continuity `0.75`, large jump `0.85`, hand switch `0.65`,
crossing `1.6`, compact-chord split `0.8`, wide span `1.0`, and excess hand load `0.25`. Default
voice weights are continuity `0.8`, large jump `0.55`, overlap `3.0`, crossing `1.2`, identity
switch `0.65`, chord split `0.2`, first use of the secondary voice `0.35`, and additional voices
`1.1`. Each is configurable through its corresponding `--hand-*-weight` or `--voice-*-weight`
option; they are generic priors rather than composition-specific settings.

Within each staff, a second beam assigns coherent voices using pitch continuity, active written
durations, chord membership, overlap, crossing, voice-switch, and additional-voice costs. Two voices
per staff are preferred; up to four are available when overlapping durations require them. A
post-assignment refinement may restore a longer written duration when another voice explains the
overlap, but pedal-marked acoustic sustain is not used for that extension. Disable this pass with
`--no-voice-duration-refinement`.

Layout-aware MusicXML declares two staves with fixed treble and bass clefs, emits staff and voice
numbers on every event, splits chords within the appropriate stream, and uses explicit rests plus
`backup` semantics so every emitted voice is rhythmically complete inside its measure. Dynamic clef
changes, cross-staff beaming, fingering, and advanced engraving remain outside this stage.
Gaps shorter than the configurable `--minimum-explicit-rest-beats` default of `1/4` are represented
with cursor movement instead of tiny engraved rests.

`staff-assignment.tsv` and `voice-assignment.tsv` preserve raw source identity, exact onset and
duration, chord identity, assignments, costs, confidence, continuity evidence, tie boundaries, and
voice-aware duration changes. JSON diagnostics also summarize hand/staff counts, voice counts and
switches, crossings, melodic intervals, hand spans, split chords, explicit rests, search sizes, and
optimizer runtimes.

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
Beat-aware reports additionally include continuous beat position, subdivision candidates, beat
confidence, downbeats, and piecewise tempo segments. The onset-based tracker is intentionally a
replaceable baseline; complex rubato and metrically ambiguous music can still require manual
alignment.

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
  score/                     Rhythm, piano layout, chords, voices, rests, and diagnostics
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
4. Separate piano hands, staves, and voices with traceable sequence heuristics.
5. Improve engraving, key spelling, and layout using evaluated score corpora.
6. Fine-tune on MAESTRO or another licensed piano dataset.
7. Experiment with a purpose-built PyTorch or JAX architecture.
