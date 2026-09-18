"""Portable reports, dimension-wise aggregation and comparable regression deltas."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from piano_transcriber.evaluation.notes import detection


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (float, int, bool)) and math.isfinite(value):
            result[name] = float(value)
    return result


def direction(metric: str) -> int:
    """+1 higher is better; -1 lower is better; 0 is descriptive only."""
    if (
        "structure." in metric
        or "pitches." in metric
        or "semitone_errors." in metric
        or "confusion_" in metric
        or "per_reference_track." in metric
    ):
        return 0
    leaf = metric.rsplit(".", 1)[-1]
    if leaf in {
        "precision",
        "recall",
        "f1",
        "accuracy",
        "exact_pitch_accuracy",
        "pitch_class_accuracy",
        "rhythmic_value_accuracy",
        "subdivision_family_accuracy",
        "assignment_accuracy",
        "matched_track_consistency",
        "track_purity",
        "correct_meter",
        "true_positives",
    }:
        return 1
    if leaf in {
        "mae",
        "median_absolute",
        "p95_absolute",
        "fragmentation",
        "voice_id_switches",
        "octave_errors",
        "false_positives",
        "false_negatives",
    }:
        return -1
    if leaf in {"pickup_error_beats", "first_full_downbeat_error_beats"}:
        return -1
    return 0


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, list[float]] = {}
    for report in reports:
        for key, value in flatten(report["metrics"]).items():
            if direction(key):
                series.setdefault(key, []).append(
                    abs(value) if key.endswith("error_beats") else value
                )
    macro = {
        key: {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "available_pieces": len(values),
        }
        for key, values in sorted(series.items())
    }
    micro: dict[str, Any] = {}
    # Count pooling, including every missed and extra note, not an average of F1s.
    keys = {
        "notes",
        *(
            f"transcription.{t}"
            for report in reports
            for t in (report["metrics"].get("transcription") or {})
        ),
    }
    for key in sorted(keys):
        candidates = []
        for report in reports:
            data = report["metrics"]
            for part in key.split("."):
                data = data.get(part) if isinstance(data, dict) else None
            if isinstance(data, dict) and data.get("true_positives") is not None:
                candidates.append(data)
        if candidates:
            tp = sum(x["true_positives"] for x in candidates)
            micro[key] = detection(
                tp,
                tp + sum(x["false_positives"] for x in candidates),
                tp + sum(x["false_negatives"] for x in candidates),
            )
    return {
        "piece_count": len(reports),
        "macro": macro,
        "micro": micro,
        "statistical_note": "Descriptive summaries only; no confidence or significance claims.",
    }


def terminal_summary(report: dict[str, Any]) -> str:
    if report.get("kind") == "suite":
        lines = [
            f"Evaluation suite: {report['completed']} completed, "
            f"{len(report['pending'])} pending, {len(report['errors'])} errors"
        ]
        for piece in report["pieces"]:
            lines.append(f"{piece['id']}:\n{terminal_summary(piece['report'])}")
        for pending in report["pending"]:
            lines.append(f"Pending {pending['id']}: {pending['reason']}")
        for error in report["errors"]:
            lines.append(f"Error {error['id']}: {error['reason']}")
        for group, summary in report["aggregates"].items():
            averages = summary["macro"].get("notes.f1", {})
            pooled = summary["micro"].get("notes", {})
            lines.append(
                f"Aggregate {group}: macro F1={averages.get('mean')}, "
                f"median F1={averages.get('median')}, micro F1={pooled.get('f1')}"
            )
        return "\n".join(lines)
    metrics = report["metrics"]

    def show(value: Any) -> str:
        return (
            "unavailable"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value)
        )

    lines = [
        f"Notes ({report['alignment']['domain']}): P={show(metrics['notes']['precision'])} "
        f"R={show(metrics['notes']['recall'])} F1={show(metrics['notes']['f1'])}"
    ]
    for tolerance, values in (metrics.get("transcription") or {}).items():
        lines.append(
            f"Transcription/seconds {tolerance}: F1={show(values['f1'])}, "
            f"onset MAE={show(values['onset_error']['mae'])} s"
        )
    for label, key, field in (
        ("Rhythm", "rhythm", "rhythmic_value_accuracy"),
        ("Meter", "meter", "correct_meter"),
        ("Staff", "staff", "accuracy"),
        ("Hand", "hand", "accuracy"),
        ("Voice", "voice", "assignment_accuracy"),
        ("Chords", "chords", "f1"),
    ):
        lines.append(f"{label}: {field}={show((metrics.get(key) or {}).get(field))}")
    lines.append(f"Structure (prediction): {metrics['structure']['prediction']}")
    lines.append(f"Alignment: {report['alignment']}")
    if report.get("seconds_alignment") and report["alignment"]["domain"] != "seconds":
        lines.append(f"Seconds alignment: {report['seconds_alignment']}")
    lines.extend(f"Warning: {warning}" for warning in dict.fromkeys(report["warnings"]))
    if metrics.get("transcription") is None:
        lines.append("Seconds metrics unavailable: initial tempo or note timestamps are missing.")
    return "\n".join(lines)


def write_table(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = (
        report["pieces"]
        if report.get("kind") == "suite"
        else [{"id": report.get("id", "piece"), "report": report}]
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        writer.writerow(["piece", "metric", "value"])
        for piece in pieces:
            for key, value in sorted(flatten(piece["report"]["metrics"]).items()):
                writer.writerow([piece["id"], key, value])


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for key in ("schema_version", "kind", "config", "reference_sha256", "stage"):
        if old.get(key) != new.get(key):
            raise ValueError(f"incomparable evaluations: {key} differs")
    if old.get("kind") == "suite":
        before = {p["id"]: p["report"] for p in old["pieces"]}
        after = {p["id"]: p["report"] for p in new["pieces"]}
        if (
            set(before) != set(after)
            or old["pending"]
            or new["pending"]
            or old["errors"]
            or new["errors"]
        ):
            raise ValueError(
                "suite comparisons require the same completed pieces and no pending/errors"
            )
        return {
            "kind": "suite-comparison",
            "pieces": {key: compare(before[key], after[key]) for key in sorted(before)},
        }
    if not old.get("reference_sha256"):
        raise ValueError("comparison requires reference fingerprints from saved CLI reports")
    if old.get("alignment", {}).get("domain") != new.get("alignment", {}).get("domain"):
        raise ValueError("comparison requires matching evaluation domains")
    previous, current = flatten(old["metrics"]), flatten(new["metrics"])
    result: dict[str, Any] = {
        "improved": [],
        "regressed": [],
        "unchanged": [],
        "diagnostic_changes": [],
        "availability_changes": [],
    }
    for key in sorted(previous.keys() | current.keys()):
        if key not in previous or key not in current:
            result["availability_changes"].append(
                {"metric": key, "old": previous.get(key), "new": current.get(key)}
            )
            continue
        a, b = previous[key], current[key]
        delta = b - a
        objective = direction(key)
        difference = abs(b) - abs(a) if key.endswith("error_beats") else delta
        category = (
            "unchanged"
            if abs(delta) <= 1e-9
            else "diagnostic_changes"
            if not objective
            else "improved"
            if difference * objective > 0
            else "regressed"
            if difference * objective < 0
            else "unchanged"
        )
        result[category].append({"metric": key, "old": a, "new": b, "delta": delta})
    return result
