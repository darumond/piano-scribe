"""Optional Spotify Basic Pitch adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from piano_transcriber.models.base import MissingModelDependencyError, TranscriptionModel
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult


class BasicPitchModel(TranscriptionModel):
    @property
    def name(self) -> str:
        return "basic-pitch"

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TranscriptionResult:
        try:
            from basic_pitch.inference import predict
        except ImportError as error:
            raise MissingModelDependencyError(
                'Basic Pitch is optional. Install it with: pip install -e ".[basic-pitch]"'
            ) from error

        with tempfile.TemporaryDirectory(prefix="piano-scribe-") as directory:
            audio_path = Path(directory) / "input.wav"
            sf.write(audio_path, audio, sample_rate)
            _model_output, _midi_data, raw_notes = predict(str(audio_path))
        notes = tuple(self._convert_note(item) for item in raw_notes)
        return TranscriptionResult(
            notes=notes,
            model_name=self.name,
            audio_duration_seconds=float(audio.size) / sample_rate,
        )

    @staticmethod
    def _convert_note(raw: Any) -> NoteEvent:
        if len(raw) < 4:
            raise ValueError("Basic Pitch returned a malformed note event")
        return NoteEvent(
            pitch=int(raw[2]),
            onset_seconds=float(raw[0]),
            offset_seconds=float(raw[1]),
            velocity=max(1, min(127, round(float(raw[3]) * 127))),
            confidence=1.0,
        )
