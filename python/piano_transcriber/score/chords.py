"""Group notes by exact quantized onset rather than acoustic microtiming."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from piano_transcriber.score.types import ScoreChord, ScoreNote


def group_chords(notes: tuple[ScoreNote, ...]) -> tuple[ScoreChord, ...]:
    grouped: dict[Fraction, list[ScoreNote]] = defaultdict(list)
    for note in notes:
        grouped[note.onset_beats].append(note)
    return tuple(
        ScoreChord(onset, tuple(sorted(onset_notes, key=lambda note: note.pitch)))
        for onset, onset_notes in sorted(grouped.items())
    )
