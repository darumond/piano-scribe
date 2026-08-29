from __future__ import annotations

from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
from piano_transcriber.score.separation import PianoLayoutMode, PianoSeparationConfig
from piano_transcriber.score.types import ReconstructedScore
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


def _score(events: list[tuple[int, float, float]]) -> ReconstructedScore:
    notes = tuple(
        NoteEvent(onset_seconds=onset, pitch=pitch, offset_seconds=onset + duration)
        for pitch, onset, duration in events
    )
    return reconstruct_score(
        TranscriptionResult(notes, "synthetic", max(note.offset_seconds for note in notes) + 0.5),
        ReconstructionConfig(
            bpm=60.0,
            piano_separation=PianoSeparationConfig(mode=PianoLayoutMode.SEQUENCE),
        ),
    )


def test_repeated_melody_pitch_keeps_its_voice_with_changing_accompaniment() -> None:
    score = _score(
        [
            (72, 0.0, 0.75),
            (60, 0.0, 0.75),
            (72, 1.0, 0.75),
            (62, 1.0, 0.75),
            (72, 2.0, 0.75),
            (64, 2.0, 0.75),
            (72, 3.0, 0.75),
            (65, 3.0, 0.75),
        ]
    )
    melody = [note for note in score.notes if note.pitch == 72]
    assert len({note.voice for note in melody}) == 1
    assert not any(note.repeated_pitch_voice_switched for note in melody)
    assert {note.voice_assignment_reason for note in melody[1:]} == {"repeated-pitch-continuity"}


def test_sustained_upper_melody_and_inner_motion_keep_separate_tracks() -> None:
    score = _score([(72, 0, 3.0), (64, 0, 0.75), (65, 1, 0.75), (67, 2, 0.75)])
    sustained = next(note for note in score.notes if note.pitch == 72)
    inner = [note for note in score.notes if note.pitch in {64, 65, 67}]
    assert sustained.voice == 1
    assert {note.voice for note in inner} == {2}
    assert sustained.duration_beats == 3


def test_two_interleaved_independent_lines_keep_stable_voice_ids() -> None:
    score = _score([(80, 0, 1.5), (74, 0.5, 1.5), (81, 2, 1.5), (75, 2.5, 1.5)])
    upper = [note for note in score.notes if note.pitch >= 80]
    lower = [note for note in score.notes if note.pitch < 80]
    assert {note.voice for note in upper} == {1}
    assert {note.voice for note in lower} == {2}


def test_legitimate_crossing_stays_within_two_persistent_voices() -> None:
    score = _score([(67, 0, 2.0), (72, 0, 2.0), (74, 1, 0.75)])
    crossing = next(note for note in score.notes if note.pitch == 74)
    assert crossing.voice == 2
    assert max(note.voice for note in score.notes) == 2


def test_true_three_voice_overlap_records_why_extra_voice_is_needed() -> None:
    score = _score([(76, 0, 4.0), (72, 1, 3.0), (68, 2, 1.0)])
    assert len({note.voice for note in score.notes}) == 3
    third = next(note for note in score.notes if note.voice == 3)
    assert third.extra_voice_reason == "overlap-required"


def test_voice_aware_duration_change_has_a_traceable_reason() -> None:
    score = _score([(72, 0, 3.0), (64, 0, 0.75), (65, 1, 0.75), (67, 2, 0.75)])
    assert all(
        not note.voice_duration_adjusted or note.duration_change_reason is not None
        for note in score.notes
    )
