"""Joint global pulse-level, meter, downbeat, and pickup inference."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise

from piano_transcriber.score.quantize import QuantizationGrid
from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.tracking import BeatPoint, BeatTrack, TempoSegment
from piano_transcriber.score.types import ReconstructedScore, TimeSignature
from piano_transcriber.transcription.types import TranscriptionResult

SUPPORTED_METERS: tuple[TimeSignature, ...] = tuple(
    TimeSignature.parse(value) for value in ("2/4", "3/4", "4/4", "6/4", "6/8", "9/8", "12/8")
)


@dataclass(frozen=True, slots=True)
class JointMeterWeights:
    timing_fit: float = 1.0
    tempo_smoothness: float = 0.2
    metric_accent: float = 0.8
    rhythmic_complexity: float = 0.55
    tie_complexity: float = 0.12
    pickup_penalty: float = 0.12
    tempo_level_distance: float = 0.08

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.timing_fit,
                self.tempo_smoothness,
                self.metric_accent,
                self.rhythmic_complexity,
                self.tie_complexity,
                self.pickup_penalty,
                self.tempo_level_distance,
            )
        ):
            raise ValueError("joint meter weights must be non-negative")


@dataclass(frozen=True, slots=True)
class JointMeterConfig:
    meters: tuple[TimeSignature, ...] = SUPPORTED_METERS
    pulse_factors: tuple[float, ...] = (0.5, 2 / 3, 1.0, 1.5, 2.0)
    minimum_bpm: float = 30.0
    maximum_bpm: float = 220.0
    phase_resolution_beats: Fraction = Fraction(1, 2)
    first_downbeat_seconds: float | None = None
    weights: JointMeterWeights = field(default_factory=JointMeterWeights)

    def __post_init__(self) -> None:
        if not self.meters or not self.pulse_factors:
            raise ValueError("joint meter inference requires meter and pulse hypotheses")
        if not 0 < self.minimum_bpm < self.maximum_bpm:
            raise ValueError("tempo bounds must be positive and increasing")
        if self.phase_resolution_beats <= 0:
            raise ValueError("phase resolution must be positive")
        if self.first_downbeat_seconds is not None and self.first_downbeat_seconds < 0:
            raise ValueError("first downbeat must be non-negative")


@dataclass(frozen=True, slots=True)
class MeterHypothesis:
    time_signature: TimeSignature
    pulse_factor: float
    pulse_bpm: float
    notated_beat_bpm: float
    higher_level_bpm: float | None
    downbeat_phase_beats: Fraction
    pickup_beats: Fraction
    measure_count: int
    timing_error_ms: float
    rhythmic_complexity_score: float
    tie_count: int
    triplet_ratio: float
    metric_accent_score: float
    tempo_smoothness_score: float
    total_score: float
    normalized_score: float = 0.0


@dataclass(frozen=True, slots=True)
class JointMeterResult:
    score: ReconstructedScore
    best: MeterHypothesis
    hypotheses: tuple[MeterHypothesis, ...]
    confidence_margin: float
    config: JointMeterConfig


def infer_joint_meter_score(
    transcription: TranscriptionResult,
    base_track: BeatTrack,
    reconstruction: ReconstructionConfig,
    config: JointMeterConfig | None = None,
) -> JointMeterResult:
    """Evaluate globally coherent pulse/meter paths and retain competing explanations."""
    joint = config or JointMeterConfig()
    evaluated: list[tuple[MeterHypothesis, ReconstructedScore]] = []
    for factor in joint.pulse_factors:
        pulse_track = rescale_pulse_track(base_track, factor)
        pulse_bpm = pulse_track.median_bpm
        for signature in joint.meters:
            phases = _candidate_phases(pulse_track, signature, joint)
            for phase in phases:
                candidate_track = align_track_meter(pulse_track, signature, phase)
                candidate_config = replace(
                    reconstruction,
                    bpm=pulse_bpm,
                    time_signature=signature,
                    adaptive_quantization=True,
                    infer_pickup=True,
                    pickup_beats=reconstruction.pickup_beats,
                    downbeat_position_beats=float(phase),
                )
                score = reconstruct_score(
                    transcription,
                    candidate_config,
                    beat_track=candidate_track,
                )
                hypothesis = _score_hypothesis(
                    score,
                    factor,
                    phase,
                    joint,
                )
                evaluated.append((hypothesis, score))
    if not evaluated:
        raise ValueError("no pulse/meter hypotheses were generated")
    evaluated.sort(key=lambda item: item[0].total_score)
    raw_scores = [
        math.exp(-(item[0].total_score - evaluated[0][0].total_score)) for item in evaluated
    ]
    total_probability = sum(raw_scores)
    normalized = [value / total_probability for value in raw_scores]
    hypotheses = tuple(
        replace(item[0], normalized_score=probability)
        for item, probability in zip(evaluated, normalized, strict=True)
    )
    margin = normalized[0] - normalized[1] if len(normalized) > 1 else 1.0
    return JointMeterResult(evaluated[0][1], hypotheses[0], hypotheses, margin, joint)


def _candidate_phases(
    track: BeatTrack,
    signature: TimeSignature,
    config: JointMeterConfig,
) -> tuple[Fraction, ...]:
    measure_length = signature.measure_beats
    if config.first_downbeat_seconds is not None:
        position = track.seconds_to_beats(config.first_downbeat_seconds)
        phase = Fraction(str(position % float(measure_length))).limit_denominator(48)
        return (phase,)
    phases: list[Fraction] = []
    phase = Fraction(0)
    while phase < measure_length:
        phases.append(phase)
        phase += config.phase_resolution_beats
    return tuple(phases)


def rescale_pulse_track(track: BeatTrack, factor: float) -> BeatTrack:
    if factor <= 0:
        raise ValueError("pulse factor must be positive")
    maximum_base_position = track.beats[-1].number
    count = max(2, math.floor(maximum_base_position * factor) + 1)
    timestamps = [track.beats_to_seconds(index / factor) for index in range(count)]
    points = tuple(
        BeatPoint(
            index,
            timestamp,
            60 / (timestamps[min(index + 1, count - 1)] - timestamp)
            if index + 1 < count
            else 60 / (timestamp - timestamps[index - 1]),
            min(track.beats, key=lambda beat: abs(beat.timestamp_seconds - timestamp)).confidence,
            False,
        )
        for index, timestamp in enumerate(timestamps)
    )
    segments = tuple(
        TempoSegment(index, index + 1, earlier, later, 60 / (later - earlier))
        for index, (earlier, later) in enumerate(pairwise(timestamps))
    )
    return BeatTrack(
        points,
        segments,
        track.time_signature,
        0,
        track.downbeat_confidence,
        track.onset_groups,
    )


def align_track_meter(
    track: BeatTrack,
    signature: TimeSignature,
    phase: Fraction,
) -> BeatTrack:
    measure_length = float(signature.measure_beats)
    phase_value = float(phase)
    points = tuple(
        replace(
            beat,
            downbeat=_cyclic_distance(beat.number, phase_value, measure_length) < 1e-6,
        )
        for beat in track.beats
    )
    return replace(
        track,
        beats=points,
        time_signature=signature,
        downbeat_phase=phase_value,
    )


def _score_hypothesis(
    score: ReconstructedScore,
    pulse_factor: float,
    phase: Fraction,
    config: JointMeterConfig,
) -> MeterHypothesis:
    quantized = [item for item in score.diagnostics if item.action == "quantized"]
    timing = (
        statistics.median(abs(item.quantization_error_seconds) * 1000 for item in quantized)
        if quantized
        else 0.0
    )
    subdivision_counts = Counter(item.selected_subdivision for item in quantized)
    triplet_count = sum(
        count for name, count in subdivision_counts.items() if name and "triplet" in name
    )
    complexity_values = []
    for item in quantized:
        selected = next(
            (
                candidate
                for candidate in item.quantization_candidates
                if candidate.subdivision == item.selected_subdivision
            ),
            None,
        )
        complexity_values.append(selected.complexity_penalty if selected is not None else 0.0)
    fine_notes = sum(
        count
        for name, count in subdivision_counts.items()
        if name in {QuantizationGrid.THIRTY_SECOND.value, QuantizationGrid.SIXTEENTH_TRIPLET.value}
    )
    rhythmic_complexity = (
        statistics.mean(complexity_values)
        + triplet_count / max(1, len(quantized))
        + 0.5 * fine_notes / max(1, len(quantized))
    )
    ties = _tie_count(score)
    metric_accent = _metric_accent(score, phase)
    if score.beat_track is None:
        raise AssertionError("joint meter candidates require a beat track")
    periods = [
        segment.end_seconds - segment.start_seconds for segment in score.beat_track.tempo_segments
    ]
    tempo_smoothness = (
        statistics.mean(abs(later - earlier) for earlier, later in pairwise(periods))
        / statistics.mean(periods)
        if len(periods) > 1
        else 0.0
    )
    pickup_fraction = float(score.pickup_beats / score.time_signature.measure_beats)
    weights = config.weights
    tempo_bounds_distance = max(
        0.0,
        math.log2(config.minimum_bpm / score.bpm),
        math.log2(score.bpm / config.maximum_bpm),
    )
    total = (
        weights.timing_fit * timing / 125
        + weights.tempo_smoothness * tempo_smoothness
        - weights.metric_accent * metric_accent
        + weights.rhythmic_complexity * rhythmic_complexity
        + weights.tie_complexity * ties / max(1, len(score.notes))
        + weights.pickup_penalty * pickup_fraction
        + weights.tempo_level_distance * (abs(math.log2(pulse_factor)) + 2 * tempo_bounds_distance)
    )
    compound = score.time_signature.denominator == 8 and score.time_signature.numerator >= 6
    notated_beat_bpm = score.bpm * score.time_signature.denominator / 4
    higher_level = score.bpm / 1.5 if compound else None
    return MeterHypothesis(
        score.time_signature,
        pulse_factor,
        score.bpm,
        notated_beat_bpm,
        higher_level,
        phase,
        score.pickup_beats,
        score.measure_count,
        timing,
        rhythmic_complexity,
        ties,
        triplet_count / max(1, len(quantized)),
        metric_accent,
        tempo_smoothness,
        total,
    )


def _metric_accent(score: ReconstructedScore, phase: Fraction) -> float:
    track = score.beat_track
    if track is None or not track.onset_groups:
        return 0.0
    measure_length = float(score.time_signature.measure_beats)
    primary = _periodic_accent(track, float(phase), measure_length)
    if score.time_signature.denominator == 8:
        return primary + 0.2 * _periodic_accent(track, float(phase), 1.5)
    if score.time_signature.numerator == 6:
        grouping_by_two = _periodic_accent(track, float(phase), 2.0)
        grouping_by_three = _periodic_accent(track, float(phase), 3.0)
        return primary + 0.2 * max(grouping_by_two, grouping_by_three)
    return primary


def _periodic_accent(track: BeatTrack, phase: float, period: float) -> float:
    positions = [track.seconds_to_beats(group.timestamp_seconds) for group in track.onset_groups]
    strengths = [group.strength for group in track.onset_groups]
    global_mean = statistics.mean(strengths)
    first = min(positions)
    last = max(positions)
    boundary = phase + math.ceil((first - phase) / period) * period
    boundary_strengths: list[float] = []
    while boundary <= last + 1e-6:
        nearest = min(
            zip(positions, strengths, strict=True),
            key=lambda item: abs(item[0] - boundary),
        )
        distance = abs(nearest[0] - boundary)
        boundary_strengths.append(nearest[1] * max(0.0, 1 - distance / 0.25))
        boundary += period
    if not boundary_strengths or global_mean == 0:
        return 0.0
    return statistics.mean(boundary_strengths) / global_mean - 1.0


def _tie_count(score: ReconstructedScore) -> int:
    measure_length = score.time_signature.measure_beats
    pickup = score.pickup_beats
    count = 0
    for note in score.notes:
        boundaries: list[Fraction] = []
        boundary = pickup if pickup > 0 else measure_length
        while boundary < note.offset_beats:
            boundaries.append(boundary)
            boundary += measure_length
        count += sum(note.onset_beats < boundary < note.offset_beats for boundary in boundaries)
    return count


def _cyclic_distance(position: float, phase: float, period: float) -> float:
    remainder = (position - phase) % period
    return min(remainder, period - remainder)
