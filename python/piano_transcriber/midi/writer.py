"""Write normalized events as a standard single-track MIDI file."""

from __future__ import annotations

from pathlib import Path

import mido

from piano_transcriber.transcription.types import TranscriptionResult


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
