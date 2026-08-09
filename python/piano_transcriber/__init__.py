"""Automatic piano transcription with pluggable model backends."""

from piano_transcriber.transcription.pipeline import TranscriptionPipeline
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult

__all__ = ["NoteEvent", "TranscriptionPipeline", "TranscriptionResult"]
__version__ = "0.1.0"
