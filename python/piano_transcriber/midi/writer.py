"""Write normalized events as a standard single-track MIDI file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mido

from piano_transcriber.transcription.types import TranscriptionResult

if TYPE_CHECKING:
    from piano_transcriber.score.types import ReconstructedScore


def write_midi(
    result: TranscriptionResult,
    path: str | Path,
    *,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> Path:
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tempo = mido.bpm2tempo(tempo_bpm)
    events: list[tuple[float, int, mido.Message]] = []
    for note in result.notes:
        events.append(
            (
                note.onset_seconds,
                1,
                mido.Message("note_on", note=note.pitch, velocity=note.velocity),
            )
        )
        events.append(
            (note.offset_seconds, 0, mido.Message("note_off", note=note.pitch, velocity=0))
        )
    events.sort(key=lambda item: (item[0], item[1]))

    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Piano transcription", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous_seconds = 0.0
    for event_seconds, _priority, message in events:
        delta_seconds = max(0.0, event_seconds - previous_seconds)
        message.time = round(mido.second2tick(delta_seconds, ticks_per_beat, tempo))
        track.append(message)
        previous_seconds = event_seconds
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output_path)
    return output_path


def write_score_midi(
    score: ReconstructedScore,
    path: str | Path,
    *,
    ticks_per_beat: int = 480,
) -> Path:
    """Write exact beat-domain score notes and available pedal intervals."""
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tempo = mido.bpm2tempo(score.bpm)
    events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
    for note in score.notes:
        onset_ticks = round(float(note.onset_beats) * ticks_per_beat)
        offset_ticks = round(float(note.offset_beats) * ticks_per_beat)
        events.append(
            (onset_ticks, 2, mido.Message("note_on", note=note.pitch, velocity=note.velocity))
        )
        events.append((offset_ticks, 0, mido.Message("note_off", note=note.pitch, velocity=0)))
    for pedal in score.pedal_intervals:
        events.append(
            (
                round(float(pedal.onset_beats) * ticks_per_beat),
                1,
                mido.Message("control_change", control=64, value=127),
            )
        )
        events.append(
            (
                round(float(pedal.offset_beats) * ticks_per_beat),
                1,
                mido.Message("control_change", control=64, value=0),
            )
        )
    if score.beat_track is not None:
        for beat in score.beat_track.beats:
            beat_ticks = round((beat.number + score.beat_position_offset) * ticks_per_beat)
            if beat_ticks >= 0:
                events.append(
                    (
                        beat_ticks,
                        -1,
                        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(beat.bpm)),
                    )
                )
    events.sort(key=lambda item: (item[0], item[1]))

    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="PianoScribe reconstructed score", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous_ticks = 0
    for absolute_ticks, _priority, message in events:
        message.time = max(0, absolute_ticks - previous_ticks)
        track.append(message)
        previous_ticks = absolute_ticks
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output_path)
    return output_path
