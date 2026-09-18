"""Manifest evaluation of cached outputs with explicit pending and error states."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from piano_transcriber.evaluation.io import load_score
from piano_transcriber.evaluation.report import aggregate, fingerprint
from piano_transcriber.evaluation.score import evaluate
from piano_transcriber.evaluation.types import EvaluationConfig


def evaluate_files(prediction: Path, reference: Path, config: EvaluationConfig) -> dict[str, Any]:
    report = evaluate(
        load_score(prediction, voice_scope=config.prediction_voice_scope),
        load_score(reference, voice_scope=config.reference_voice_scope),
        config,
    )
    report.update(
        prediction=str(prediction),
        reference=str(reference),
        prediction_sha256=fingerprint(prediction),
        reference_sha256=fingerprint(reference),
    )
    return report


def evaluate_suite(path: Path, config: EvaluationConfig) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"invalid evaluation manifest: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("pieces"), list):
        raise ValueError("manifest requires a pieces list")
    pieces: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for piece in data["pieces"]:
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str) or not piece["id"]:
            raise ValueError("every manifest piece requires a non-empty string id")
        piece_id = piece["id"]
        if piece_id in seen:
            raise ValueError(f"duplicate piece id: {piece_id}")
        seen.add(piece_id)
        if not piece.get("reference"):
            pending.append(
                {"id": piece_id, "reason": "reference missing; no ground truth supplied"}
            )
            continue
        predictions = piece.get("predictions") or (
            {"output": piece["prediction"]} if piece.get("prediction") else {}
        )
        if not isinstance(predictions, dict):
            raise ValueError("predictions must map stage names to paths")
        if not predictions:
            pending.append(
                {
                    "id": piece_id,
                    "reason": "cached prediction missing; run the production CLI on audio first",
                }
            )
            continue
        for name, prediction in predictions.items():
            run_id = f"{piece_id}/{name}"
            try:
                stage = (
                    "transcription" if name == "transcription" else piece.get("stage", config.stage)
                )
                report = evaluate_files(
                    (path.parent / str(prediction)).resolve(),
                    (path.parent / str(piece["reference"])).resolve(),
                    replace(
                        config,
                        stage=stage,
                        prediction_voice_scope=piece.get(
                            "prediction_voice_scope", config.prediction_voice_scope
                        ),
                        reference_voice_scope=piece.get(
                            "reference_voice_scope", config.reference_voice_scope
                        ),
                    ),
                )
                pieces.append({"id": run_id, "piece_id": piece_id, "report": report})
            except (ValueError, OSError) as error:
                errors.append({"id": run_id, "reason": str(error)})
    # Avoid pooling raw and written durations, seconds and beats, or multiple versions as pieces.
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in pieces:
        report = item["report"]
        name = item["id"].rsplit("/", 1)[-1]
        key = f"{name}:{report['stage']}:{report['alignment']['domain']}"
        groups.setdefault(key, []).append(report)
    return {
        "schema_version": 1,
        "kind": "suite",
        "config": asdict(config),
        "manifest": str(path),
        "completed": len(pieces),
        "pieces": pieces,
        "pending": pending,
        "errors": errors,
        "aggregates": {key: aggregate(reports) for key, reports in sorted(groups.items())},
    }
