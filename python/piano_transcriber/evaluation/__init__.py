"""Offline evaluation, independent of inference and reconstruction decisions."""

from piano_transcriber.evaluation.score import evaluate
from piano_transcriber.evaluation.types import EvaluationConfig, EvaluationNote, EvaluationScore

__all__ = ["EvaluationConfig", "EvaluationNote", "EvaluationScore", "evaluate"]
