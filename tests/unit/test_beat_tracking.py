from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from piano_transcriber.score.diagnostics import (
    score_diagnostics,
    write_beats_tsv,
    write_tempo_tsv,
)
from piano_transcriber.score.quantize import QuantizationGrid, choose_quantization
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.tracking import (
    BeatPoint,
    BeatTrack,
    SymbolicBeatTrackerConfig,
    SymbolicOnsetBeatTracker,
    TempoSegment,
    cluster_onsets,
    fixed_beat_track,
)
from piano_transcriber.score.types import TimeSignature
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def symbolic_result(
    onsets: list[float],
    *,
    chord_size: int = 1,
    duration: float | None = None,
) -> TranscriptionResult:
    notes = tuple(
        NoteEvent(
            pitch=48 + voice * 12,
            onset_seconds=onset + voice * 0.004,
            offset_seconds=onset + 0.25,
            velocity=90 - voice * 5,
        )
        for onset in onsets
        for voice in range(chord_size)
    )
    end = duration if duration is not None else max(note.offset_seconds for note in notes) + 0.25
    return TranscriptionResult(notes, "synthetic", end)


def test_chord_onsets_are_clustered_with_accent_strength() -> None:
    groups = cluster_onsets(symbolic_result([0.0, 0.5], chord_size=3))
    assert len(groups) == 2
    assert groups[0].pitches == (48, 60, 72)
    assert groups[0].strength > 1.0


def test_constant_tempo_with_jitter_and_extra_offbeats() -> None:
    beats = [0.01, 0.49, 1.02, 1.50, 2.01, 2.49, 3.02, 3.50]
    offbeats = [0.25, 1.75, 2.75]
    result = symbolic_result(sorted(beats + offbeats))
    tracker = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, minimum_bpm=80, maximum_bpm=160)
    )
    track = tracker.track(result)
    assert track.median_bpm == pytest.approx(120, abs=8)
    assert track.seconds_to_beats(2.0) == pytest.approx(4.0, abs=0.2)


@pytest.mark.parametrize(
    ("periods", "direction"),
    [
        ([0.7, 0.67, 0.64, 0.61, 0.58, 0.55, 0.52], "accelerando"),
        ([0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.63], "ritardando"),
    ],
)
def test_gradual_local_tempo_change(periods: list[float], direction: str) -> None:
    onsets = [0.0]
    for period in periods:
        onsets.append(onsets[-1] + period)
    track = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, smoothing=0.5)
    ).track(symbolic_result(onsets))
    early = track.tempo_segments[1].bpm
    late = track.tempo_segments[-2].bpm
    if direction == "accelerando":
        assert late > early
    else:
        assert late < early


def test_missing_beat_attack_is_interpolated_with_lower_confidence() -> None:
    track = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, minimum_bpm=100, maximum_bpm=140)
    ).track(symbolic_result([0.0, 0.5, 1.0, 2.0, 2.5, 3.0]))
    missing = min(track.beats, key=lambda beat: abs(beat.timestamp_seconds - 1.5))
    assert missing.timestamp_seconds == pytest.approx(1.5, abs=0.15)
    assert missing.confidence <= 0.45


def test_manual_downbeat_alignment_has_explicit_confidence() -> None:
    track = fixed_beat_track(5.0, 120.0, first_downbeat_seconds=1.0)
    downbeat = min(track.beats, key=lambda beat: abs(beat.timestamp_seconds - 1.0))
    assert downbeat.downbeat
    assert track.downbeat_confidence == 1.0
    assert track.downbeat_phase == 2


def test_accented_bass_chords_provide_downbeat_evidence() -> None:
    notes: list[NoteEvent] = []
    for index in range(12):
        chord = (36, 60, 67) if index % 4 == 0 else (60,)
        notes.extend(
            NoteEvent(
                pitch=pitch,
                onset_seconds=index * 0.5 + voice * 0.003,
                offset_seconds=index * 0.5 + 0.25,
                velocity=100 if index % 4 == 0 else 55,
            )
            for voice, pitch in enumerate(chord)
        )
    result = TranscriptionResult(tuple(notes), "synthetic", 6.0)
    track = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, minimum_bpm=100, maximum_bpm=140)
    ).track(result)
    assert track.beats[track.downbeat_phase].downbeat
    assert track.downbeat_confidence > 0.1


def test_beat_seconds_mapping_round_trip_with_tempo_changes() -> None:
    beats = (
        BeatPoint(0, 0.0, 120.0, 1.0, True),
        BeatPoint(1, 0.5, 100.0, 1.0, False),
        BeatPoint(2, 1.1, 75.0, 1.0, False),
        BeatPoint(3, 1.9, 75.0, 1.0, False),
    )
    segments = tuple(
        TempoSegment(
            index,
            index + 1,
            beats[index].timestamp_seconds,
            beats[index + 1].timestamp_seconds,
            bpm,
        )
        for index, bpm in enumerate((120.0, 100.0, 75.0))
    )
    track = BeatTrack(beats, segments, TimeSignature(), 0, 0.5)
    for seconds in (0.0, 0.2, 0.8, 1.5, 2.1):
        assert track.beats_to_seconds(track.seconds_to_beats(seconds)) == pytest.approx(seconds)


def test_imperfect_straight_eighths_do_not_overfit_triplets() -> None:
    track = fixed_beat_track(3.0, 60.0)
    for onset in (0.48, 1.03, 1.49):
        selected, candidates = choose_quantization(
            track.seconds_to_beats(onset), onset, track, complexity_cost=0.35
        )
        assert selected.subdivision in {
            QuantizationGrid.QUARTER.value,
            QuantizationGrid.EIGHTH.value,
        }
        assert "triplet" not in selected.subdivision
        assert len(candidates) == len(QuantizationGrid)


def test_triplets_and_sixteenths_are_selected_when_supported_by_timing() -> None:
    track = fixed_beat_track(3.0, 60.0)
    triplet, _ = choose_quantization(1 / 3, 1 / 3, track, complexity_cost=0.35)
    sixteenth, _ = choose_quantization(0.25, 0.25, track, complexity_cost=0.35)
    assert triplet.subdivision == QuantizationGrid.EIGHTH_TRIPLET.value
    assert sixteenth.subdivision == QuantizationGrid.SIXTEENTH.value


def test_note_spanning_tempo_change_uses_beat_mapping_for_duration() -> None:
    beats = (
        BeatPoint(0, 0.0, 120.0, 1.0, True),
        BeatPoint(1, 0.5, 60.0, 1.0, False),
        BeatPoint(2, 1.5, 60.0, 1.0, False),
    )
    segments = (
        TempoSegment(0, 1, 0.0, 0.5, 120.0),
        TempoSegment(1, 2, 0.5, 1.5, 60.0),
    )
    track = BeatTrack(beats, segments, TimeSignature(), 0, 1.0)
    result = TranscriptionResult(
        (NoteEvent(pitch=60, onset_seconds=0.0, offset_seconds=1.5),),
        "synthetic",
        1.5,
    )
    score = reconstruct_score(
        result,
        ReconstructionConfig(bpm=80.0, adaptive_quantization=True),
        beat_track=track,
    )
    assert score.notes[0].onset_beats == Fraction(0)
    assert score.notes[0].duration_beats == Fraction(2)
    assert score.grid_name == "adaptive"
    assert score.diagnostics[0].quantization_candidates
    assert score.diagnostics[0].selected_subdivision == "quarter"


def test_beat_tempo_and_candidate_diagnostics_are_exportable(tmp_path: Path) -> None:
    result = symbolic_result([0.0, 0.5, 1.0, 1.5])
    track = fixed_beat_track(2.0, 120.0)
    score = reconstruct_score(
        result,
        ReconstructionConfig(bpm=120.0, adaptive_quantization=True),
        beat_track=track,
    )
    beats_path = write_beats_tsv(score, tmp_path / "beats.tsv")
    tempo_path = write_tempo_tsv(score, tmp_path / "tempo.tsv")
    diagnostics = score_diagnostics(score)
    first = diagnostics["events"][0]  # type: ignore[index]
    assert "beat_number\ttimestamp_seconds\tbpm" in beats_path.read_text()
    assert "start_beat\tend_beat" in tempo_path.read_text()
    assert first["continuous_onset_beats"] == pytest.approx(0.0)  # type: ignore[index]
    assert first["quantization_candidates"]  # type: ignore[index]
