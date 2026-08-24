"""Replaceable symbolic beat, downbeat, and local-tempo tracking."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

from piano_transcriber.score.types import TimeSignature
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


@dataclass(frozen=True, slots=True)
class OnsetGroup:
    timestamp_seconds: float
    strength: float
    pitches: tuple[int, ...]
    mean_velocity: float
    bass_pitch: int


@dataclass(frozen=True, slots=True)
class BeatPoint:
    number: int
    timestamp_seconds: float
    bpm: float
    confidence: float
    downbeat: bool


@dataclass(frozen=True, slots=True)
class TempoSegment:
    start_beat: float
    end_beat: float
    start_seconds: float
    end_seconds: float
    bpm: float


@dataclass(frozen=True, slots=True)
class BeatTrack:
    beats: tuple[BeatPoint, ...]
    tempo_segments: tuple[TempoSegment, ...]
    time_signature: TimeSignature
    downbeat_phase: float
    downbeat_confidence: float
    onset_groups: tuple[OnsetGroup, ...] = ()

    def __post_init__(self) -> None:
        if len(self.beats) < 2:
            raise ValueError("a beat track requires at least two beats")
        if any(
            later.timestamp_seconds <= earlier.timestamp_seconds
            for earlier, later in pairwise(self.beats)
        ):
            raise ValueError("beat timestamps must increase")
        if not 0 <= self.downbeat_confidence <= 1:
            raise ValueError("downbeat confidence must be between zero and one")

    @property
    def median_bpm(self) -> float:
        return statistics.median(beat.bpm for beat in self.beats)

    @property
    def bpm_range(self) -> tuple[float, float]:
        values = [beat.bpm for beat in self.beats]
        return min(values), max(values)

    @property
    def measure_padding_beats(self) -> float:
        beats_per_measure = float(self.time_signature.measure_beats)
        return (-float(self.downbeat_phase)) % beats_per_measure

    def seconds_to_beats(self, seconds: float) -> float:
        if not math.isfinite(seconds):
            raise ValueError("seconds must be finite")
        first, last = self.beats[0], self.beats[-1]
        if seconds <= first.timestamp_seconds:
            period = self.beats[1].timestamp_seconds - first.timestamp_seconds
            return first.number + (seconds - first.timestamp_seconds) / period
        if seconds >= last.timestamp_seconds:
            period = last.timestamp_seconds - self.beats[-2].timestamp_seconds
            return last.number + (seconds - last.timestamp_seconds) / period
        for earlier, later in pairwise(self.beats):
            if earlier.timestamp_seconds <= seconds <= later.timestamp_seconds:
                fraction = (seconds - earlier.timestamp_seconds) / (
                    later.timestamp_seconds - earlier.timestamp_seconds
                )
                return earlier.number + fraction
        raise AssertionError("unreachable beat interpolation")

    def beats_to_seconds(self, beats: float) -> float:
        if not math.isfinite(beats):
            raise ValueError("beats must be finite")
        first, last = self.beats[0], self.beats[-1]
        if beats <= first.number:
            period = self.beats[1].timestamp_seconds - first.timestamp_seconds
            return first.timestamp_seconds + (beats - first.number) * period
        if beats >= last.number:
            period = last.timestamp_seconds - self.beats[-2].timestamp_seconds
            return last.timestamp_seconds + (beats - last.number) * period
        index = math.floor(beats) - first.number
        earlier = self.beats[index]
        later = self.beats[index + 1]
        return earlier.timestamp_seconds + (beats - earlier.number) * (
            later.timestamp_seconds - earlier.timestamp_seconds
        )


class BeatTracker(Protocol):
    def track(self, transcription: TranscriptionResult) -> BeatTrack: ...


@dataclass(frozen=True, slots=True)
class SymbolicBeatTrackerConfig:
    minimum_bpm: float = 30.0
    maximum_bpm: float = 220.0
    onset_cluster_ms: float = 45.0
    smoothing: float = 0.25
    time_signature: TimeSignature = field(default_factory=TimeSignature)
    first_beat_seconds: float | None = None
    first_downbeat_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.minimum_bpm < self.maximum_bpm:
            raise ValueError("tempo bounds must be positive and increasing")
        if self.onset_cluster_ms <= 0:
            raise ValueError("onset cluster width must be positive")
        if not 0 <= self.smoothing <= 1:
            raise ValueError("smoothing must be between zero and one")


def cluster_onsets(
    transcription: TranscriptionResult,
    *,
    window_ms: float = 45.0,
) -> tuple[OnsetGroup, ...]:
    """Collapse chord attacks and compute symbolic accent strength."""
    ordered = sorted(transcription.notes, key=lambda note: note.onset_seconds)
    clusters: list[list[NoteEvent]] = []
    for note in ordered:
        if not clusters or note.onset_seconds - clusters[-1][0].onset_seconds > window_ms / 1000:
            clusters.append([note])
        else:
            clusters[-1].append(note)
    groups: list[OnsetGroup] = []
    previous_bass: int | None = None
    previous_pitches: set[int] | None = None
    for cluster in clusters:
        weights = [max(1, note.velocity) for note in cluster]
        timestamp = sum(
            note.onset_seconds * weight for note, weight in zip(cluster, weights, strict=True)
        ) / sum(weights)
        pitches = tuple(sorted(note.pitch for note in cluster))
        bass = pitches[0]
        velocity = statistics.mean(note.velocity for note in cluster)
        chord_strength = min(len(cluster), 6) / 6
        velocity_strength = velocity / 127
        bass_strength = max(0.0, (60 - bass) / 36)
        bass_change = 0.3 if previous_bass is not None and bass != previous_bass else 0.0
        pitch_set = set(pitches)
        chord_change = (
            0.2
            if previous_pitches is not None
            and pitch_set.isdisjoint(previous_pitches - {previous_bass})
            else 0.0
        )
        local_density = sum(abs(other[0].onset_seconds - timestamp) <= 0.3 for other in clusters)
        density_strength = min(local_density, 4) * 0.04
        strength = (
            0.25
            + 0.9 * chord_strength
            + 0.7 * velocity_strength
            + 0.45 * bass_strength
            + bass_change
            + chord_change
            + density_strength
        )
        groups.append(OnsetGroup(timestamp, strength, pitches, velocity, bass))
        previous_bass = bass
        previous_pitches = pitch_set
    return tuple(groups)


class SymbolicOnsetBeatTracker:
    """Robust onset-cluster pulse tracker with smoothly varying local periods."""

    def __init__(self, config: SymbolicBeatTrackerConfig | None = None) -> None:
        self.config = config or SymbolicBeatTrackerConfig()

    def track(self, transcription: TranscriptionResult) -> BeatTrack:
        groups = cluster_onsets(transcription, window_ms=self.config.onset_cluster_ms)
        if len(groups) < 2:
            raise ValueError("beat tracking requires at least two onset groups")
        period, anchor = self._initial_pulse(groups)
        if self.config.first_beat_seconds is not None:
            anchor = self.config.first_beat_seconds
        timestamps, confidences = self._follow_pulse(
            groups, period, anchor, transcription.audio_duration_seconds
        )
        periods = [later - earlier for earlier, later in pairwise(timestamps)]
        smoothed = self._smooth_periods(periods)
        bpms = [60 / smoothed[min(index, len(smoothed) - 1)] for index in range(len(timestamps))]
        phase, downbeat_confidence = self._downbeat_phase(groups, timestamps)
        if self.config.first_downbeat_seconds is not None:
            manual_downbeat = self.config.first_downbeat_seconds
            nearest = min(
                range(len(timestamps)),
                key=lambda index: abs(timestamps[index] - manual_downbeat),
            )
            phase = nearest % float(self.config.time_signature.measure_beats)
            downbeat_confidence = 1.0
        measure_length = float(self.config.time_signature.measure_beats)
        beats = tuple(
            BeatPoint(
                number=index,
                timestamp_seconds=timestamp,
                bpm=bpms[index],
                confidence=confidences[index],
                downbeat=_cyclic_distance(index, phase, measure_length) < 1e-6,
            )
            for index, timestamp in enumerate(timestamps)
        )
        segments = tuple(
            TempoSegment(index, index + 1, earlier, later, 60 / (later - earlier))
            for index, (earlier, later) in enumerate(pairwise(timestamps))
        )
        return BeatTrack(
            beats,
            segments,
            self.config.time_signature,
            phase,
            downbeat_confidence,
            groups,
        )

    def _initial_pulse(self, groups: tuple[OnsetGroup, ...]) -> tuple[float, float]:
        minimum_period = 60 / self.config.maximum_bpm
        maximum_period = 60 / self.config.minimum_bpm
        differences = [
            later.timestamp_seconds - earlier.timestamp_seconds
            for earlier, later in pairwise(groups)
            if minimum_period / 4
            <= later.timestamp_seconds - earlier.timestamp_seconds
            <= maximum_period
        ]
        if not differences:
            raise ValueError("onset groups do not contain a plausible pulse")
        candidates: set[float] = set()
        for difference in differences:
            for multiplier in (1, 2, 3, 4):
                candidate = difference * multiplier
                if minimum_period <= candidate <= maximum_period:
                    candidates.add(round(candidate, 4))
        anchors = sorted(groups, key=lambda group: group.strength, reverse=True)[:8]
        best_score = -math.inf
        best = (statistics.median(differences), groups[0].timestamp_seconds)
        for period in candidates:
            for anchor_group in anchors:
                alignment = 0.0
                coverage: set[int] = set()
                for group in groups:
                    coordinate = (group.timestamp_seconds - anchor_group.timestamp_seconds) / period
                    distance = abs(coordinate - round(coordinate))
                    if distance <= 0.22:
                        alignment += group.strength * (1 - distance / 0.22)
                        coverage.add(round(coordinate))
                score = alignment + 0.2 * len(coverage) - 0.12 * (period / minimum_period)
                if score > best_score:
                    best_score = score
                    best = period, anchor_group.timestamp_seconds
        return best

    def _follow_pulse(
        self,
        groups: tuple[OnsetGroup, ...],
        initial_period: float,
        anchor: float,
        duration: float,
    ) -> tuple[list[float], list[float]]:
        period = initial_period
        while anchor - period >= 0:
            anchor -= period
        timestamps = [anchor]
        confidences = [0.8]
        used: set[int] = set()
        while timestamps[-1] + period <= duration:
            predicted = timestamps[-1] + period
            window = 0.28 * period
            options = [
                (index, group)
                for index, group in enumerate(groups)
                if index not in used and abs(group.timestamp_seconds - predicted) <= window
            ]
            if options:
                index, group = max(
                    options,
                    key=lambda item: (
                        item[1].strength * (1 - abs(item[1].timestamp_seconds - predicted) / window)
                    ),
                )
                used.add(index)
                observed = group.timestamp_seconds - timestamps[-1]
                period = (1 - self.config.smoothing) * period + self.config.smoothing * observed
                period = min(
                    60 / self.config.minimum_bpm, max(60 / self.config.maximum_bpm, period)
                )
                timestamp = timestamps[-1] + period
                confidence = min(1.0, 0.45 + 0.2 * group.strength)
            else:
                timestamp = predicted
                confidence = 0.25
            timestamps.append(timestamp)
            confidences.append(confidence)
        return timestamps, confidences

    @staticmethod
    def _smooth_periods(periods: list[float]) -> list[float]:
        if len(periods) < 3:
            return periods
        return [
            statistics.median(periods[max(0, index - 2) : min(len(periods), index + 3)])
            for index in range(len(periods))
        ]

    def _downbeat_phase(
        self,
        groups: tuple[OnsetGroup, ...],
        timestamps: list[float],
    ) -> tuple[float, float]:
        measure_length = float(self.config.time_signature.measure_beats)
        phase_count = max(2, round(measure_length * 2))
        phases = [index / 2 for index in range(phase_count)]
        scores = [0.0] * len(phases)
        for phase_index, phase in enumerate(phases):
            for index, timestamp in enumerate(timestamps):
                if _cyclic_distance(index, phase, measure_length) > 1e-6:
                    continue
                nearby = [
                    group for group in groups if abs(group.timestamp_seconds - timestamp) <= 0.15
                ]
                if nearby:
                    strongest = max(nearby, key=lambda group: group.strength)
                    scores[phase_index] += (
                        strongest.strength + max(0, 55 - strongest.bass_pitch) / 30
                    )
        ranking = sorted(range(len(phases)), key=scores.__getitem__, reverse=True)
        best, second = scores[ranking[0]], scores[ranking[1]]
        confidence = max(0.0, min(1.0, (best - second) / best)) if best else 0.0
        selected = phases[ranking[0]]
        return int(selected) if selected.is_integer() else selected, confidence


def fixed_beat_track(
    duration_seconds: float,
    bpm: float,
    *,
    time_signature: TimeSignature | None = None,
    first_beat_seconds: float = 0.0,
    first_downbeat_seconds: float | None = None,
) -> BeatTrack:
    """Build the manual constant-tempo mode through the same mapping interface."""
    if bpm <= 0 or duration_seconds < 0:
        raise ValueError("BPM must be positive and duration non-negative")
    signature = time_signature or TimeSignature()
    period = 60 / bpm
    start = first_beat_seconds
    while start - period >= 0:
        start -= period
    count = max(2, math.floor((duration_seconds - start) / period) + 1)
    timestamps = [start + index * period for index in range(count)]
    phase = 0.0
    confidence = 1.0
    if first_downbeat_seconds is not None:
        phase = min(
            range(count), key=lambda index: abs(timestamps[index] - first_downbeat_seconds)
        ) % float(signature.measure_beats)
    measure_length = float(signature.measure_beats)
    beats = tuple(
        BeatPoint(
            index,
            timestamp,
            bpm,
            1.0,
            _cyclic_distance(index, phase, measure_length) < 1e-6,
        )
        for index, timestamp in enumerate(timestamps)
    )
    segments = tuple(
        TempoSegment(index, index + 1, earlier, later, bpm)
        for index, (earlier, later) in enumerate(pairwise(timestamps))
    )
    return BeatTrack(beats, segments, signature, phase, confidence)


def _cyclic_distance(position: float, phase: float, period: float) -> float:
    remainder = (position - phase) % period
    return min(remainder, period - remainder)
