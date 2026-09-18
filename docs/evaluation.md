# Symbolic score evaluation

Evaluation lives in `python/piano_transcriber/evaluation/`. It reads reference and cached
prediction files without calling the model or reconstruction algorithms. Its normalized
`EvaluationScore` and `EvaluationNote` types are independent of production decisions.
`from_transcription()` and `from_reconstructed()` are optional in-memory adapters.

Good transcription F1 does not imply good notation. Structural simplicity does not imply
score correctness. Reports deliberately have no combined quality score.

## Install and run

Install the project normally (`pip install -e ".[dev]"` for development). Evaluation uses the
existing NumPy, SciPy, and mido dependencies plus PyYAML for safe YAML manifest loading.
It does not require the optional model dependencies, a checkpoint, a GPU, or network access.

```bash
piano-scribe evaluate --prediction outputs/example/score.musicxml \
  --reference references/example.musicxml \
  --prediction-voice-scope staff \
  --json outputs/evaluation/example.json --table outputs/evaluation/example.tsv

piano-scribe evaluate --prediction outputs/example/transcription.json \
  --reference references/example-performance.mid --stage transcription \
  --alignment offset --onset-tolerances-ms 25 50 100 \
  --json outputs/evaluation/example-raw.json

piano-scribe evaluate-suite evaluation.example.yaml \
  --json outputs/evaluation/suite.json --table outputs/evaluation/suite.csv

piano-scribe compare-evaluations outputs/evaluation/old.json \
  outputs/evaluation/new.json --json outputs/evaluation/comparison.json
```

`--json` defaults to `outputs/evaluation/metrics.json`; choose distinct names for saved versions.
The optional `--table` accepts `.tsv` or `.csv`. Numeric metric tables use long format, with
piece, metric, and value columns. JSON also retains unavailable fields as `null`, input SHA-256
fingerprints, configuration, alignment, warnings, coverage counts, and matched note indices.
Indices refer to the normalized input order (onset, pitch, part, voice), after tie merging.
CLI errors and incomplete suites return exit code 2. Completed evaluation returns 0.

## Inputs and metadata

Supported inputs are synchronous PPQ MIDI type 0/1, score-partwise MusicXML (including `.mxl`),
normalized note JSON, and existing pipeline diagnostic JSON. MusicXML supports chords,
backup/forward cursors, variable divisions, tempo/meter changes, explicit pickups, transposition,
staff/voice tags, beams, tuplets, and ties. Tied fragments become a single note for matching;
the structural tie count remains separate. Sounding MIDI pitch avoids enharmonic penalties.

MIDI tempo changes are integrated exactly in quarter-note units. A missing MIDI tempo uses the
file-format default of 120 BPM and produces a warning. MIDI track/channel numbers are never
interpreted as staff, hand, or voice. Explicit MIDI meter allows nominal bar/beat positions,
but MIDI cannot identify a pickup: pickup and downbeat metrics remain unavailable. MIDI
measure boundaries assume a bar begins at tick zero; use MusicXML when pickup matters.

MusicXML seconds require an explicit initial tempo. Without it, beat-domain evaluation still
works and seconds metrics are unavailable. Empty metadata is not scored as an error. Notes
without a shared onset domain produce unavailable matching metrics, retaining structure counts.
Rates with a zero denominator are `null`; two empty inputs do not imply perfect detection. Notes
with partial staff or voice metadata are evaluated only where both sides are annotated; the
report includes the number evaluated. Physical hand is independent of staff and requires an
explicit `hand` field in normalized JSON or the reconstructed-score adapter. Ordinary
MusicXML does not establish which hand played a note. A treble/bass confusion matrix is
available where both inputs carry explicit staff and G2/F4 clef information.

MusicXML voice IDs normally belong to a part and may cross staves. The default voice scope
is therefore `part`. Current PianoScribe exports use local voice numbers on each staff;
pass `--prediction-voice-scope staff` for these exports. Reference scope has its own option,
`--reference-voice-scope`, and should follow the reference producer's convention. Pipeline
diagnostic/in-memory adapters already scope local voice IDs to their staff. No pitch-based
staff or voice assumptions are introduced by the readers.

Normalized JSON may contain:

```json
{
  "stage": "symbolic",
  "notes": [{
    "pitch": 60, "onset_beats": 0, "duration_beats": 1,
    "onset_seconds": 0, "duration_seconds": 0.5,
    "measure": "1", "beat": 0, "staff": "1", "voice": "upper",
    "part": "1", "hand": "right", "staff_role": "treble"
  }],
  "time_signatures": [[0, 4, 4]],
  "tempos": [[0, 120]],
  "pickup_beats": 0,
  "first_downbeat_beats": 0,
  "measure_boundaries": [0, 4],
  "downbeats": [0, 4],
  "beats": [0, 1, 2, 3, 4]
}
```

This is a synthetic schema example, not a reference for a real recording. Only pitch and
an onset domain are required per note. Durations, other metadata, and global grids are
optional. Raw transcription JSON also accepts `offset_seconds`; existing `model_name` JSON
is recognized as the transcription stage. All beat fields use quarter-note units, including
6/8 and 6/4. Measures retain their source labels. The `beat` field is zero-based within a bar.

## Matching and alignment

Exact-pitch note matching indexes reference events by pitch and sorted onset. Binary searches
construct only edges inside the tolerance window. A sparse minimum-cost bipartite assignment
maximizes the number of one-to-one matches first and minimizes total absolute onset error
second. Private dummy columns represent unmatched predictions. One prediction cannot satisfy
two references, including duplicated same-pitch notes. This avoids the missed matches possible
with greedy nearest-neighbor selection and avoids a dense whole-piece note cost matrix.

Tolerance boundaries are inclusive, with a numerical epsilon of 1e-9. Defaults are 25, 50,
and 100 ms for seconds and 0.125 quarter-note beats for symbolic correspondence. Optional
`--duration-tolerance-seconds` / `--duration-tolerance-beats` add a duration gate; the default
uses pitch and onset alone so acoustic offset discrepancies do not hide onset successes.

`--alignment auto` uses unchanged symbolic beat origins when both symbolic inputs have beats.
Otherwise it estimates a single global seconds offset. The seconds section has its own
reported offset even when the primary evaluation uses beats. `--alignment none` disables
time fitting. `symbolic` requires beats and disables seconds fitting. `offset` fits only a
seconds offset; `affine` additionally permits a bounded tempo scale (default +/-5%, adjustable
with `--max-tempo-scale-deviation`). Fit transforms always map prediction into reference time:
`reference_time = scale * prediction_time + offset`.

The seconds estimator votes on same-pitch anchors, tests bounded offset candidates, and
optionally refines a bounded scale using trimmed residuals. Sampling and tie breaks are
deterministic. Candidate fitting uses up to 96 anchors per input and 256 sampled predictions;
final scoring uses every note. No dynamic time warping is performed. The report records the
transform and anchor count; repeated passages and large missing sections may create ambiguity.
Inspect alignment before interpreting an unexpectedly good or bad result. Symbolic beat
origins are never translated or rescaled, so meter and pickup errors cannot be fitted away.

## Metric meanings and stages

| Dimension | Meaning and applicability |
| --- | --- |
| Note detection | TP, FP, FN, precision, recall and F1 for exact-pitch/onset matches; primary domain is recorded. Raw audio evaluation uses seconds, written scores prefer beats. |
| Onsets | Signed error mean/range and absolute MAE, median and P95 for matched notes. Errors are prediction minus reference, after the reported time transform. |
| Offsets/durations | Separate offset MAE and duration MAE, median absolute and P95 absolute errors. Available only for pairs with durations. Seconds are acoustic or serialized timing depending on the input; beat durations are written rhythm. |
| Pitch | A separate onset-only assignment, with pitch distance as a small tie breaker, reports exact pitch and pitch-class accuracy, nonzero octave-multiple errors, and signed semitone counts. Two or more identical nonzero shifts are listed as recurring errors. Exact-note matching separately supplies FP/FN pitch histograms. |
| Rhythm | Matched-note onset beat error and written-duration beat error, exact rhythmic-value accuracy (epsilon 1e-7), and duration-family accuracy for recognized straight, dotted, or triplet values. Arbitrary tied sums have no family. |
| Meter | Full meter-map equality, signed pickup/first-full-downbeat errors, downbeat and measure-boundary P/R/F1, and beat-grid alignment. Grid matching is one-to-one within 0.125 beats by default. |
| Boundary error | Every reference boundary's nearest predicted-boundary signed distance, summarized as MAE/median/P95. This is separate from one-to-one boundary detection, so missing and extra boundaries remain visible. |
| Staff/hand | Accuracy and reference-row/prediction-column confusion matrix on annotated matched notes. Staff IDs are compared directly; hand is scored only if explicit. Clef roles are reported separately when available. |
| Voice | Permutation-invariant assignment, consistency, fragmentation, switches and purity; definitions below. |
| Chords | Anchor-based onset groups include singletons. Exact pitch-set one-to-one matches give P/R/F1. Remaining overlapping sets are separately reported as partial matches and do not inflate exact F1. |
| Structure | Note, measure, staff, voice, tie, rest, primary-beam-group and tuplet-group counts, separately for reference and prediction. These are diagnostics without a preferred direction. |

Chord grouping defaults to 30 ms or 0.0625 beats; set `--chord-group-seconds` or
`--chord-group-beats`. Events are compared with the group's first onset, so grouping cannot
chain an entire arpeggio. Pitch sets deliberately ignore unison multiplicity; note detection
still penalizes duplicate notes. Chord matching uses the primary note onset tolerance.

Seconds tables are not proof of acoustic transcription accuracy when the input is an exported
score. A published written score also does not specify the exact rubato and pedal offsets of
a performance. Use a performance-aligned note reference for raw audio evaluation. Written
duration errors should be assessed in the rhythm section independently of acoustic offsets.
Likewise the meter grids currently evaluate notated quarter-note positions, not annotated
audio downbeat timestamps. MusicXML beat grids follow the denominator unit, including eighth
positions in compound meter; they are not perceptual dotted-pulse tracking scores.

## Voice evaluation

Only exact-pitch/onset matched notes with voice metadata on both sides participate. An overlap
matrix counts how many such notes each predicted track shares with each reference track.
Hungarian assignment maximizes this overlap with a one-to-one track mapping. It never assumes
that voice 1 means the same thing in both files.

- Assignment accuracy is mapped overlap divided by all eligible matched notes.
- Matched-track consistency averages mapped coverage per reference track; an unmapped
  reference track contributes zero.
- Fragmentation sums `number of distinct predicted tracks - 1` for each reference track.
- Voice-ID switches count changes in predicted ownership between successive annotated onset
  groups along each reference track. Simultaneous split groups are counted separately, so
  arbitrary within-chord ordering cannot create switches.
- Track purity is the sum of each predicted track's largest reference overlap divided by
  eligible matched notes. A fragmented but pure voice can have purity 1 and poor assignment
  accuracy; both dimensions matter.

These are conditional on note correspondence. Missed or extra notes remain in detection
metrics; they do not silently enter the assignment denominator. Track mapping and per-reference
details are saved for inspection. Highly ambiguous repeated unisons can make identity uncertain.

## Suites and regression comparisons

```yaml
pieces:
  - id: excerpt-one
    audio: test_audio/excerpt-one.wav
    reference: references/excerpt-one.musicxml
    prediction_voice_scope: staff
    predictions:
      transcription: outputs/excerpt-one/transcription.json
      reconstruction: outputs/excerpt-one/diagnostics.json
      final-score: outputs/excerpt-one/score.musicxml

  - id: excerpt-two
    reference: references/excerpt-two.mid
    prediction: outputs/excerpt-two/score.mid
```

All paths are relative to the manifest location. `audio` records provenance; suite evaluation
never implicitly launches inference or downloads weights. Audio-only entries are pending until
you generate a cached output with the production CLI. `reference: null` is also pending, not
perfect, zero, or invented ground truth. Missing/malformed files are reported per piece without
preventing valid examples from being evaluated. Duplicate piece IDs are rejected.

Macro mean and median are calculated across available per-piece metrics, with an availability
count. Micro note precision/recall/F1 pool TP/FP/FN. Aggregation is separated by prediction label,
stage, and matching domain so multiple pipeline versions are not counted as independent pieces.
Signed pickup errors aggregate as absolute magnitude. No confidence intervals or statistical
significance claims are made from these small suites.

Regression comparison requires identical schema, configuration, reference fingerprints, stage,
and matching domain; suites additionally need identical completed piece IDs. Changed reference
material or tolerances therefore cannot masquerade as a regression. Each objective metric is
classified as improved, regressed, or unchanged. Structural/count/histogram changes are descriptive,
and newly available/unavailable metrics are reported separately. No single score ranks versions.

## Add an example and prepare Liebestraum

1. Place a legally obtained excerpt reference under ignored `references/` and its recording under
   ignored `test_audio/`. Use matching excerpt boundaries and unfold repeats when needed.
2. Generate/cache production outputs with the existing CLI. Preserve raw transcription separately
   from reconstructed score and diagnostics so stage differences remain visible.
3. Add manifest paths, a stable piece ID, and the correct voice scope. Run the suite and inspect
   its alignment, missing-metadata warnings, and note correspondence before using scores as targets.
4. Save JSON reports under `outputs/evaluation/` and compare versions with identical references.

The local `outputs/liebestraum/evaluation-template.yaml` refers to the existing raw transcription,
staff-aware diagnostics, and engraved MusicXML, with `reference: null`. Replace that field only
after supplying a legally obtained, excerpt-matched reference. No reference score is fetched.
The accompanying local reference-field notes describe which metadata unlocks each metric.
MusicXML will usually provide richer notation evaluation than MIDI. Neither format automatically
establishes what was played in the recording.

## Offline corpus and performance

```bash
python benchmarks/benchmark_evaluation.py --notes 8000 --runs 3 \
  --json outputs/evaluation/benchmark.json --corpus outputs/evaluation/synthetic
piano-scribe evaluate-suite outputs/evaluation/synthetic/manifest.json \
  --json outputs/evaluation/synthetic-suite.json
```

The deterministic corpus covers scales, blocked chords, arpeggios, melody/bass, independent
voices, triplets, a pickup, and 3/4, 4/4, 6/8, 6/4. It generates small normalized symbolic
references locally. Self-comparisons test mechanics, not transcription quality. Unit tests
add deliberate note, rhythm, meter, assignment and timing errors, plus independent MIDI/XML
reader cases. Tests do not access recordings, network services, model weights or external data.

The benchmark includes full metric computation and alignment on thousands of notes; it excludes
file loading and inference. Timing varies by host. Matching memory follows candidate edges
inside pitch/onset windows. Pathological thousands-of-notes-at-one-onset inputs can still form
dense candidate windows. Voice assignment is dense only over track counts, normally a handful.

## Current limits

Grace/unpitched events are omitted with warnings. Repeats are not unfolded automatically;
score-timewise XML, asynchronous/SMPTE MIDI, and microtonal pitches are rejected. The first
MusicXML part defines the global measure grid. Staff roles can change with clefs and are not
physical hands. Meter hypotheses are retained from pipeline diagnostic JSON or supplied normalized
metadata; ordinary MIDI/XML do not encode those alternatives. General rubato alignment, excerpt
search, pedal correctness, key spelling, visual layout and perceptual engraving scores are not
implemented. None of these limitations are converted into false claims of score correctness.
