"""End-to-end orchestration independent from any one ML implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from piano_transcriber.audio.loader import load_audio
from piano_transcriber.audio.preprocessing import preprocess_audio
from piano_transcriber.config import PipelineConfig, create_model
from piano_transcriber.midi.writer import write_midi
from piano_transcriber.models.base import TranscriptionModel
from piano_transcriber.notation.musicxml import write_musicxml
from piano_transcriber.transcription.postprocess import postprocess_result
from piano_transcriber.transcription.types import TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    result: TranscriptionResult
    midi_path: Path | None = None
    musicxml_path: Path | None = None


class TranscriptionPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: TranscriptionModel | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.model = model or create_model(self.config.model)

    def run(
        self,
        input_path: str | Path,
        *,
        midi_path: str | Path | None = None,
        musicxml_path: str | Path | None = None,
    ) -> PipelineOutput:
        logger.info("Loading audio from %s", input_path)
        loaded = load_audio(input_path)
        audio, sample_rate = preprocess_audio(
            loaded.samples,
            loaded.sample_rate,
            target_sample_rate=self.config.target_sample_rate,
            normalize=self.config.normalize_audio,
        )
        logger.info("Transcribing %.2f seconds with %s", loaded.duration_seconds, self.model.name)
        raw_result = self.model.transcribe(audio, sample_rate)
        result = postprocess_result(raw_result, minimum_confidence=self.config.minimum_confidence)

        final_midi_path = Path(midi_path) if midi_path is not None else None
        final_musicxml_path = Path(musicxml_path) if musicxml_path is not None else None
        if final_midi_path is not None:
            write_midi(result, final_midi_path)
            logger.info("Wrote MIDI to %s", final_midi_path)
        if final_musicxml_path is not None:
            write_musicxml(result, final_musicxml_path)
            logger.info("Wrote MusicXML to %s", final_musicxml_path)
        return PipelineOutput(result, final_midi_path, final_musicxml_path)
