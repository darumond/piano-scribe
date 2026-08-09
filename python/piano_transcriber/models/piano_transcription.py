"""Adapter for ByteDance's high-resolution piano transcription model."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from piano_transcriber.models.base import (
    MissingModelDependencyError,
    ModelCheckpointError,
    TranscriptionModel,
)
from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16_000
DEFAULT_CHECKPOINT = "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
DEFAULT_CHECKPOINT_URL = (
    "https://zenodo.org/api/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth/content"
)


class PianoTranscriptionRuntime(Protocol):
    """Narrow view of the optional upstream inference object."""

    def transcribe(
        self, audio: npt.NDArray[np.float32], midi_path: str | None
    ) -> Mapping[str, object]: ...


RuntimeFactory = Callable[[str, Path], PianoTranscriptionRuntime]
DependencyChecker = Callable[[], None]
CudaProbe = Callable[[], bool]
Clock = Callable[[], float]
CheckpointFetcher = Callable[[str, Path], None]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    filename: str
    url: str
    size_bytes: int
    md5: str


OFFICIAL_CHECKPOINT = CheckpointSpec(
    filename=DEFAULT_CHECKPOINT,
    url=DEFAULT_CHECKPOINT_URL,
    size_bytes=171_966_578,
    md5="22b961b77c1878239fec963362097045",
)


def default_model_cache_dir() -> Path:
    """Return a platform-appropriate, overrideable model cache directory."""
    override = os.environ.get("PIANO_SCRIBE_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "models"
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "piano-scribe" / "models"


def _fetch_checkpoint(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as error:
        raise ModelCheckpointError(f"failed to download piano checkpoint: {error}") from error


class CheckpointManager:
    """Resolve explicit checkpoints or atomically populate the shared cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        spec: CheckpointSpec = OFFICIAL_CHECKPOINT,
        fetcher: CheckpointFetcher = _fetch_checkpoint,
    ) -> None:
        self.cache_dir = cache_dir or default_model_cache_dir()
        self.spec = spec
        self._fetcher = fetcher

    def resolve(self, explicit_path: Path | None = None) -> Path:
        if explicit_path is not None:
            checkpoint = explicit_path.expanduser()
            if not checkpoint.is_file():
                raise ModelCheckpointError(f"checkpoint file does not exist: {checkpoint}")
            if checkpoint.stat().st_size == 0:
                raise ModelCheckpointError(f"checkpoint file is empty: {checkpoint}")
            return checkpoint

        checkpoint = self.cache_dir / self.spec.filename
        if checkpoint.is_file() and self._is_valid(checkpoint):
            logger.debug("Using cached piano checkpoint at %s", checkpoint)
            return checkpoint
        if checkpoint.exists():
            logger.warning(
                "Cached piano checkpoint is invalid and will be replaced: %s", checkpoint
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_name(f"{checkpoint.name}.part")
        partial.unlink(missing_ok=True)
        logger.info(
            "Downloading piano checkpoint (%.1f MiB) to %s",
            self.spec.size_bytes / (1024 * 1024),
            checkpoint,
        )
        try:
            self._fetcher(self.spec.url, partial)
            if not self._is_valid(partial):
                raise ModelCheckpointError(
                    "downloaded piano checkpoint failed size or checksum validation"
                )
            partial.replace(checkpoint)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return checkpoint

    def _is_valid(self, path: Path) -> bool:
        if path.stat().st_size != self.spec.size_bytes:
            return False
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == self.spec.md5


def _check_optional_dependencies() -> None:
    try:
        importlib.import_module("torch")
        module = importlib.import_module("piano_transcription_inference")
        if "PianoTranscription" not in vars(module):
            raise AttributeError("PianoTranscription is not exported")
    except (AttributeError, ImportError) as error:
        raise MissingModelDependencyError(
            "The piano-transcription optional runtime is not installed. Install it with: "
            'pip install -e ".[pytorch]"'
        ) from error


def _cuda_is_available() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise MissingModelDependencyError(
            'PyTorch is not installed. Install it with: pip install -e ".[pytorch]"'
        ) from error
    return bool(torch.cuda.is_available())


def _create_upstream_runtime(device: str, checkpoint_path: Path) -> PianoTranscriptionRuntime:
    module = importlib.import_module("piano_transcription_inference")
    runtime_type = vars(module)["PianoTranscription"]
    return cast(
        PianoTranscriptionRuntime,
        runtime_type(device=device, checkpoint_path=str(checkpoint_path)),
    )


class PianoTranscriptionModel(TranscriptionModel):
    """Run the pretrained ByteDance CRNN and normalize its event dictionaries."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        cache_dir: str | Path | None = None,
        runtime_factory: RuntimeFactory | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        dependency_checker: DependencyChecker | None = None,
        cuda_probe: CudaProbe | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.device = device
        resolved_cache = Path(cache_dir) if cache_dir is not None else None
        self._checkpoint_manager = checkpoint_manager or CheckpointManager(resolved_cache)
        self._runtime_factory = runtime_factory or _create_upstream_runtime
        self._dependency_checker = dependency_checker or (
            (lambda: None) if runtime_factory is not None else _check_optional_dependencies
        )
        self._cuda_probe = cuda_probe or _cuda_is_available
        self._clock = clock
        self._runtime: PianoTranscriptionRuntime | None = None
        self._resolved_device: str | None = None

    @property
    def name(self) -> str:
        return "piano-transcription"

    def transcribe(self, audio: npt.NDArray[np.float32], sample_rate: int) -> TranscriptionResult:
        if sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError(
                f"piano-transcription requires {MODEL_SAMPLE_RATE} Hz audio; got {sample_rate} Hz"
            )
        if audio.ndim != 1:
            raise ValueError("piano-transcription requires mono one-dimensional audio")
        if not np.isfinite(audio).all():
            raise ValueError("audio contains non-finite samples")
        duration = float(audio.size) / sample_rate
        if duration == 0.0:
            return TranscriptionResult((), self.name, 0.0)

        runtime = self._load_runtime()
        started = self._clock()
        raw_result = runtime.transcribe(np.ascontiguousarray(audio, dtype=np.float32), None)
        elapsed = self._clock() - started
        realtime = duration / elapsed if elapsed > 0.0 else math.inf
        logger.info(
            "Piano transcription inference completed on %s in %.2f s (%.2fx realtime)",
            self._resolved_device,
            elapsed,
            realtime,
        )
        return self._convert_result(raw_result, duration)

    def _load_runtime(self) -> PianoTranscriptionRuntime:
        if self._runtime is not None:
            return self._runtime
        started = self._clock()
        self._dependency_checker()
        device = self._select_device()
        checkpoint = self._checkpoint_manager.resolve(self.checkpoint_path)
        try:
            self._runtime = self._runtime_factory(device, checkpoint)
        except MissingModelDependencyError:
            raise
        except Exception as error:
            raise ModelCheckpointError(
                f"failed to load piano transcription model: {error}"
            ) from error
        self._resolved_device = device
        logger.info(
            "Loaded piano transcription model on %s from %s in %.2f s",
            device,
            checkpoint,
            self._clock() - started,
        )
        return self._runtime

    def _select_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        cuda_available = self._cuda_probe()
        if self.device == "cuda" and not cuda_available:
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
        return "cuda" if cuda_available else "cpu"

    def _convert_result(
        self, raw_result: Mapping[str, object], duration: float
    ) -> TranscriptionResult:
        raw_notes = self._event_sequence(raw_result, "est_note_events")
        raw_pedals = self._event_sequence(raw_result, "est_pedal_events", required=False)
        pedals = tuple(self._pedal_interval(event) for event in raw_pedals)
        notes: list[NoteEvent] = []
        for raw_note in raw_notes:
            try:
                onset = max(0.0, self._number(raw_note["onset_time"]))
                offset = min(duration, self._number(raw_note["offset_time"]))
                pitch = round(self._number(raw_note["midi_note"]))
                velocity = max(0, min(127, round(self._number(raw_note["velocity"]))))
                confidence = self._number(raw_note.get("confidence", 1.0))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"piano backend returned a malformed note event: {raw_note}"
                ) from error
            if onset >= duration:
                logger.debug("Discarding note beyond audio duration: %s", raw_note)
                continue
            if offset <= onset:
                raise ValueError(f"piano backend returned a non-positive note duration: {raw_note}")
            pedal = any(pedal_on < offset and pedal_off > onset for pedal_on, pedal_off in pedals)
            notes.append(
                NoteEvent(
                    pitch=pitch,
                    onset_seconds=onset,
                    offset_seconds=offset,
                    velocity=velocity,
                    confidence=confidence,
                    pedal=pedal if pedals else None,
                )
            )
        return TranscriptionResult(tuple(sorted(notes)), self.name, duration)

    @staticmethod
    def _event_sequence(
        result: Mapping[str, object], key: str, *, required: bool = True
    ) -> Sequence[Mapping[str, object]]:
        value = result.get(key)
        if value is None and not required:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"piano backend result field {key!r} must be a sequence")
        if not all(isinstance(event, Mapping) for event in value):
            raise ValueError(f"piano backend result field {key!r} contains malformed events")
        return cast(Sequence[Mapping[str, object]], value)

    @staticmethod
    def _pedal_interval(event: Mapping[str, object]) -> tuple[float, float]:
        try:
            onset = PianoTranscriptionModel._number(event["onset_time"])
            offset = PianoTranscriptionModel._number(event["offset_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"piano backend returned a malformed pedal event: {event}") from error
        if not math.isfinite(onset) or not math.isfinite(offset) or onset < 0.0 or offset <= onset:
            raise ValueError(f"piano backend returned a malformed pedal interval: {event}")
        return onset, offset

    @staticmethod
    def _number(value: object) -> float:
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise TypeError("value must be numeric")
        return float(value)
