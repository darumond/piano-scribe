from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from piano_transcriber.score.diagnostics import write_meter_hypotheses_tsv
from piano_transcriber.score.meter import (
    SUPPORTED_METERS,
    JointMeterConfig,
    infer_joint_meter_score,
    rescale_pulse_track,
)
from piano_transcriber.score.reconstruct import ReconstructionConfig
from piano_transcriber.score.tracking import (
    SymbolicBeatTrackerConfig,
    SymbolicOnsetBeatTracker,
    fixed_beat_track,
)
from piano_transcriber.score.types import TimeSignature
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def accented_pattern(
    measure_beats: int,
    *,
    measure_count: int = 5,
    missing: set[int] | None = None,
    jitter: float = 0.0,
) -> TranscriptionResult:
    omitted = missing or set()
    notes: list[NoteEvent] = []
    total_beats = measure_beats * measure_count
    for beat in range(total_beats):
        if beat in omitted:
            continue
        onset = beat * 0.5 + (jitter if beat % 2 else -jitter)
        downbeat = beat % measure_beats == 0
        pitches = (36, 60, 67) if downbeat else (60,)
        for voice, pitch in enumerate(pitches):
            notes.append(
                NoteEvent(
                    pitch=pitch,
                    onset_seconds=max(0.0, onset + voice * 0.003),
                    offset_seconds=onset + 0.3,
                    velocity=105 if downbeat else 52,
                )
            )
    return TranscriptionResult(tuple(notes), "synthetic", total_beats * 0.5)


def test_required_simple_and_compound_meters_are_supported() -> None:
    assert tuple(map(str, SUPPORTED_METERS)) == (
        "2/4",
        "3/4",
        "4/4",
        "6/4",
        "6/8",
        "9/8",
        "12/8",
    )
    assert TimeSignature.parse("6/8").measure_beats == Fraction(3)
    assert TimeSignature.parse("9/8").measure_beats == Fraction(9, 2)


def test_half_and_double_time_pulse_hypotheses_are_explicit() -> None:
    base = fixed_beat_track(8.0, 120.0)
    assert rescale_pulse_track(base, 0.5).median_bpm == pytest.approx(60.0)
    assert rescale_pulse_track(base, 2.0).median_bpm == pytest.approx(240.0)


@pytest.mark.parametrize("meter", ["3/4", "4/4", "6/4", "6/8"])
def test_each_meter_can_be_scored_with_missing_beats_and_expressive_timing(
    meter: str,
) -> None:
    raw = accented_pattern(6 if meter == "6/4" else 3, missing={4}, jitter=0.015)
    track = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, minimum_bpm=100, maximum_bpm=140)
    ).track(raw)
    signature = TimeSignature.parse(meter)
    result = infer_joint_meter_score(
        raw,
        track,
        ReconstructionConfig(bpm=120.0, adaptive_quantization=True),
        JointMeterConfig(meters=(signature,), pulse_factors=(1.0,)),
    )
    assert result.best.time_signature == signature
    assert result.score.notes
    assert result.hypotheses


def test_competing_meter_hypotheses_preserve_scores_and_favor_accent_periodicity(
    tmp_path: Path,
) -> None:
    raw = accented_pattern(6)
    track = SymbolicOnsetBeatTracker(
        SymbolicBeatTrackerConfig(first_beat_seconds=0.0, minimum_bpm=100, maximum_bpm=140)
    ).track(raw)
    result = infer_joint_meter_score(
        raw,
        track,
        ReconstructionConfig(bpm=120.0, adaptive_quantization=True),
        JointMeterConfig(
            meters=tuple(TimeSignature.parse(value) for value in ("3/4", "4/4", "6/4")),
            pulse_factors=(0.5, 1.0, 2.0),
            maximum_bpm=250.0,
        ),
    )
    assert result.best.time_signature == TimeSignature.parse("6/4")
    assert {hypothesis.pulse_factor for hypothesis in result.hypotheses} == {0.5, 1.0, 2.0}
    assert sum(item.normalized_score for item in result.hypotheses) == pytest.approx(1.0)
    assert 0.0 <= result.confidence_margin <= 1.0
    output = write_meter_hypotheses_tsv(result, tmp_path / "meter-hypotheses.tsv")
    report = output.read_text(encoding="utf-8")
    assert (
        "rank\ttime_signature\tpulse_factor\tpulse_bpm\tnotated_beat_bpm\thigher_level_bpm"
        in report
    )
    assert "relative_score\tconfidence_margin" in report.splitlines()[0]
    assert "6/4" in report


def test_notation_complexity_can_outweigh_a_small_timing_advantage() -> None:
    raw = accented_pattern(4, jitter=0.035)
    track = fixed_beat_track(raw.audio_duration_seconds, 120.0)
    result = infer_joint_meter_score(
        raw,
        track,
        ReconstructionConfig(bpm=120.0, adaptive_quantization=True),
        JointMeterConfig(
            meters=(TimeSignature.parse("4/4"),),
            pulse_factors=(2 / 3, 1.0, 1.5),
        ),
    )
    lowest_timing = min(result.hypotheses, key=lambda item: item.timing_error_ms)
    assert result.best.total_score <= lowest_timing.total_score
    assert any(
        candidate.timing_error_ms < result.best.timing_error_ms
        and candidate.rhythmic_complexity_score > result.best.rhythmic_complexity_score
        for candidate in result.hypotheses
    )


def test_manual_downbeat_and_pickup_constrain_joint_alignment() -> None:
    raw = accented_pattern(3, missing={0})
    track = fixed_beat_track(raw.audio_duration_seconds, 120.0)
    result = infer_joint_meter_score(
        raw,
        track,
        ReconstructionConfig(
            bpm=120.0,
            adaptive_quantization=True,
            pickup_beats=Fraction(1),
        ),
        JointMeterConfig(
            meters=(TimeSignature.parse("3/4"),),
            pulse_factors=(1.0,),
            first_downbeat_seconds=1.0,
        ),
    )
    assert {item.downbeat_phase_beats for item in result.hypotheses} == {Fraction(2)}
    assert result.best.pickup_beats == Fraction(1)
