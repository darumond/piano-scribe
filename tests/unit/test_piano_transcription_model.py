from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import piano_transcriber.models.piano_transcription as adapter
import pytest
from piano_transcriber.models.base import MissingModelDependencyError, ModelCheckpointError
from piano_transcriber.models.piano_transcription import (
    CheckpointManager,
    CheckpointSpec,
    PianoTranscriptionModel,
)


class FakeRuntime:
    def __init__(self, result: Mapping[str, object]) -> None:
        self.result = result
        self.calls = 0
        self.midi_path: str | None = "not-called"

    def transcribe(
        self, audio: np.ndarray[tuple[int], np.dtype[np.float32]], midi_path: str | None
    ) -> Mapping[str, object]:
        self.calls += 1
        self.midi_path = midi_path
        assert audio.dtype == np.float32
        return self.result


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test checkpoint")
    return checkpoint


def test_adapter_converts_notes_pedals_and_logs_timings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runtime = FakeRuntime(
        {
            "est_note_events": [
                {
                    "onset_time": 0.1,
                    "offset_time": 0.4,
                    "midi_note": 60,
                    "velocity": 90,
                },
                {
                    "onset_time": 0.7,
                    "offset_time": 1.2,
                    "midi_note": 64,
                    "velocity": 200,
                    "confidence": 0.8,
                },
            ],
            "est_pedal_events": [{"onset_time": 0.5, "offset_time": 0.9}],
        }
    )
    factory_calls: list[tuple[str, Path]] = []

    def factory(device: str, checkpoint: Path) -> FakeRuntime:
        factory_calls.append((device, checkpoint))
        return runtime

    times = iter((0.0, 1.0, 2.0, 4.0))
    model = PianoTranscriptionModel(
        checkpoint_path=_checkpoint(tmp_path),
        device="cpu",
        runtime_factory=factory,
        clock=lambda: next(times),
    )
    with caplog.at_level(logging.INFO):
        result = model.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)

    assert factory_calls == [("cpu", tmp_path / "model.pth")]
    assert runtime.midi_path is None
    assert [(note.pitch, note.velocity, note.pedal) for note in result.notes] == [
        (60, 90, False),
        (64, 127, True),
    ]
    assert result.notes[1].offset_seconds == 1.0
    assert result.notes[1].confidence == 0.8
    assert [(event.onset_seconds, event.offset_seconds) for event in result.pedal_events] == [
        (0.5, 0.9)
    ]
    assert "Loaded piano transcription model on cpu" in caplog.text
    assert "inference completed on cpu in 2.00 s" in caplog.text


def test_runtime_is_loaded_once(tmp_path: Path) -> None:
    runtime = FakeRuntime({"est_note_events": [], "est_pedal_events": []})
    loads = 0

    def factory(_device: str, _checkpoint: Path) -> FakeRuntime:
        nonlocal loads
        loads += 1
        return runtime

    model = PianoTranscriptionModel(
        checkpoint_path=_checkpoint(tmp_path), device="cpu", runtime_factory=factory
    )
    audio = np.zeros(16_000, dtype=np.float32)
    model.transcribe(audio, 16_000)
    model.transcribe(audio, 16_000)
    assert loads == 1
    assert runtime.calls == 2


def test_auto_device_selects_cuda_when_available(tmp_path: Path) -> None:
    selected: list[str] = []
    runtime = FakeRuntime({"est_note_events": []})

    def factory(device: str, _checkpoint: Path) -> FakeRuntime:
        selected.append(device)
        return runtime

    model = PianoTranscriptionModel(
        checkpoint_path=_checkpoint(tmp_path),
        runtime_factory=factory,
        cuda_probe=lambda: True,
    )
    model.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)
    assert selected == ["cuda"]


def test_requested_unavailable_cuda_fails_before_checkpoint_resolution(tmp_path: Path) -> None:
    model = PianoTranscriptionModel(
        checkpoint_path=tmp_path / "missing.pth",
        device="cuda",
        runtime_factory=lambda _device, _path: FakeRuntime({"est_note_events": []}),
        cuda_probe=lambda: False,
    )
    with pytest.raises(ValueError, match="CUDA was requested"):
        model.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)


def test_adapter_rejects_wrong_sample_rate_without_loading_model() -> None:
    model = PianoTranscriptionModel(
        runtime_factory=lambda _device, _path: FakeRuntime({"est_note_events": []})
    )
    with pytest.raises(ValueError, match="requires 16000 Hz"):
        model.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)


def test_adapter_rejects_malformed_backend_output(tmp_path: Path) -> None:
    runtime = FakeRuntime({"est_note_events": [{"midi_note": 60}]})
    model = PianoTranscriptionModel(
        checkpoint_path=_checkpoint(tmp_path),
        device="cpu",
        runtime_factory=lambda _device, _path: runtime,
    )
    with pytest.raises(ValueError, match="malformed note event"):
        model.transcribe(np.zeros(16_000, dtype=np.float32), 16_000)


def test_checkpoint_manager_downloads_validates_and_reuses_cache(tmp_path: Path) -> None:
    payload = b"small deterministic checkpoint"
    spec = CheckpointSpec(
        filename="test.pth",
        url="https://example.invalid/test.pth",
        size_bytes=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )
    fetches: list[str] = []

    def fetcher(url: str, destination: Path) -> None:
        fetches.append(url)
        destination.write_bytes(payload)

    manager = CheckpointManager(tmp_path, spec=spec, fetcher=fetcher)
    first = manager.resolve()
    second = manager.resolve()
    assert first == second == tmp_path / "test.pth"
    assert first.read_bytes() == payload
    assert fetches == [spec.url]
    assert not (tmp_path / "test.pth.part").exists()


def test_explicit_checkpoint_never_invokes_downloader(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)

    def unexpected_fetch(_url: str, _destination: Path) -> None:
        raise AssertionError("downloader must not be called for an explicit checkpoint")

    manager = CheckpointManager(tmp_path / "cache", fetcher=unexpected_fetch)
    assert manager.resolve(checkpoint) == checkpoint


def test_invalid_download_is_removed(tmp_path: Path) -> None:
    spec = CheckpointSpec("bad.pth", "https://example.invalid", 4, "deadbeef")
    manager = CheckpointManager(
        tmp_path, spec=spec, fetcher=lambda _url, path: path.write_bytes(b"bad")
    )
    with pytest.raises(ModelCheckpointError, match="validation"):
        manager.resolve()
    assert not (tmp_path / "bad.pth.part").exists()


def test_missing_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(_name: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(adapter.importlib, "import_module", missing_import)
    with pytest.raises(MissingModelDependencyError, match=r'pip install -e ".\[pytorch\]"'):
        adapter._check_optional_dependencies()
