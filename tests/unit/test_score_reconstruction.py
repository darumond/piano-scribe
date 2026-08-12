from __future__ import annotations

from fractions import Fraction

from piano_transcriber.score.beats import measure_position
from piano_transcriber.score.quantize import QuantizationGrid, snap_to_grid
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.tempo import MedianInterOnsetTempoEstimator, seconds_to_beats
from piano_transcriber.score.types import TimeSignature
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def result(*notes: NoteEvent, duration: float = 8.0) -> TranscriptionResult:
    return TranscriptionResult(tuple(notes), "test", duration)


def test_seconds_to_beats_is_exact() -> None:
    assert seconds_to_beats(0.5, 60.0) == Fraction(1, 2)
    assert seconds_to_beats(1.5, 120.0) == Fraction(3)


def test_automatic_tempo_is_an_independent_strategy() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=0.2),
        NoteEvent(pitch=62, onset_seconds=0.5, offset_seconds=0.7),
        NoteEvent(pitch=64, onset_seconds=1.0, offset_seconds=1.2),
    )
    assert MedianInterOnsetTempoEstimator().estimate_bpm(raw) == 120.0


def test_onsets_quantize_to_eighth_notes_without_mutating_raw_result() -> None:
    raw = result(
        *(
            NoteEvent(pitch=60 + index, onset_seconds=onset, offset_seconds=onset + 0.3)
            for index, onset in enumerate((0.01, 0.49, 1.02, 1.51))
        )
    )
    original_notes = raw.notes
    score = reconstruct_score(
        raw,
        ReconstructionConfig(bpm=60.0, grid=QuantizationGrid.EIGHTH),
    )
    assert [note.onset_beats for note in score.notes] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(3, 2),
    ]
    assert raw.notes is original_notes
    assert [note.raw_onset_seconds for note in score.notes] == [0.01, 0.49, 1.02, 1.51]


def test_supported_regular_and_triplet_grids() -> None:
    assert snap_to_grid(Fraction(49, 100), QuantizationGrid.QUARTER.step_beats) == 0
    assert snap_to_grid(Fraction(49, 100), QuantizationGrid.SIXTEENTH.step_beats) == Fraction(1, 2)
    assert snap_to_grid(Fraction(34, 100), QuantizationGrid.EIGHTH_TRIPLET.step_beats) == Fraction(
        1, 3
    )
    assert snap_to_grid(
        Fraction(17, 100), QuantizationGrid.SIXTEENTH_TRIPLET.step_beats
    ) == Fraction(1, 6)
    assert snap_to_grid(Fraction(13, 100), QuantizationGrid.THIRTY_SECOND.step_beats) == Fraction(
        1, 8
    )


def test_written_durations_include_dotted_and_triplet_values() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=1.49),
        NoteEvent(pitch=64, onset_seconds=2.0, offset_seconds=2.66),
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert [note.duration_beats for note in score.notes] == [Fraction(3, 2), Fraction(2, 3)]


def test_slightly_different_onsets_become_a_chord() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.01, offset_seconds=0.5),
        NoteEvent(pitch=64, onset_seconds=0.02, offset_seconds=0.5),
    )
    score = reconstruct_score(
        raw,
        ReconstructionConfig(bpm=60.0, grid=QuantizationGrid.EIGHTH),
    )
    assert len(score.chords) == 1
    assert [note.pitch for note in score.chords[0].notes] == [60, 64]


def test_very_short_notes_are_retained_by_default_and_filtering_is_observable() -> None:
    raw = result(NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=0.01))
    retained = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert len(retained.notes) == 1
    assert retained.notes[0].duration_beats == Fraction(1, 8)
    assert retained.diagnostics[0].action == "quantized"
    assert retained.diagnostics[0].suspicious_reasons

    filtered = reconstruct_score(
        raw,
        ReconstructionConfig(bpm=60.0, minimum_note_duration_ms=30.0),
    )
    assert filtered.notes == ()
    assert filtered.diagnostics[0].action == "filtered"


def test_same_pitch_events_on_one_grid_position_are_merged_observably() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.01, offset_seconds=0.4, velocity=50),
        NoteEvent(pitch=60, onset_seconds=0.02, offset_seconds=0.5, velocity=80),
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert len(score.notes) == 1
    assert score.notes[0].velocity == 80
    assert sorted(item.action for item in score.diagnostics) == ["merged", "quantized"]
    merged = next(item for item in score.diagnostics if item.action == "merged")
    assert merged.merged_into_source_index == score.notes[0].source_index


def test_pedal_extended_offset_is_shortened_at_next_quantized_onset() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=2.0, pedal=True),
        NoteEvent(pitch=64, onset_seconds=0.51, offset_seconds=1.0, pedal=False),
    )
    score = reconstruct_score(
        raw,
        ReconstructionConfig(bpm=60.0, grid=QuantizationGrid.EIGHTH),
    )
    first = next(note for note in score.notes if note.pitch == 60)
    assert first.duration_beats == Fraction(1, 2)
    assert first.raw_offset_seconds == 2.0
    assert first.pedal_duration_shortened


def test_overlapping_notes_remain_independent_symbolic_events() -> None:
    raw = result(
        NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=2.0),
        NoteEvent(pitch=67, onset_seconds=0.5, offset_seconds=1.0),
    )
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert len(score.notes) == 2
    assert score.notes[0].offset_beats > score.notes[1].onset_beats


def test_measure_positions_and_boundaries() -> None:
    signature = TimeSignature.parse("4/4")
    assert measure_position(Fraction(15, 4), signature) == (1, Fraction(15, 4))
    assert measure_position(Fraction(4), signature) == (2, Fraction(0))
    raw = result(NoteEvent(pitch=60, onset_seconds=3.75, offset_seconds=4.25))
    score = reconstruct_score(raw, ReconstructionConfig(bpm=60.0))
    assert score.measure_count == 2
