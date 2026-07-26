"""Freeze and verify label-blind cohorts for a new confirmation study."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_ids(frame: pd.DataFrame) -> list[str]:
    required = {"event_id", "time_to_tca"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if "y" in frame.columns:
        raise ValueError("Study features must be label-blind")
    if frame.empty:
        raise ValueError("Study cohort must contain at least one feature row")
    if frame.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    if frame["event_id"].isna().any():
        raise ValueError("event_id must not contain missing values")
    return sorted(frame["event_id"].astype(str).unique().tolist())


def _cohort_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_parquet(path)
    event_ids = _event_ids(frame)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "events": len(event_ids),
        "event_ids": event_ids,
    }


def freeze_study(
    calibration_features: str | Path,
    evaluation_features: str | Path,
    preregistration: str | Path,
    preregistration_lock: str | Path,
    output: str | Path,
    lock: str | Path,
) -> dict[str, Any]:
    """Freeze disjoint feature cohorts before event outcomes are opened."""
    output, lock = Path(output), Path(lock)
    if output.exists() or lock.exists():
        raise FileExistsError("Study manifest and lock are one-shot outputs")
    preregistration, preregistration_lock = Path(preregistration), Path(preregistration_lock)
    prereg_hash = file_sha256(preregistration)
    prereg_lock_payload = json.loads(preregistration_lock.read_text(encoding="utf-8"))
    if prereg_lock_payload.get("preregistration_sha256") != prereg_hash:
        raise ValueError("Preregistration does not match its lock")

    calibration = _cohort_record(calibration_features)
    evaluation = _cohort_record(evaluation_features)
    overlap = set(calibration["event_ids"]).intersection(evaluation["event_ids"])
    if overlap:
        raise ValueError(f"Calibration and evaluation cohorts overlap by {len(overlap)} event_id values")
    manifest = {
        "schema_version": 1,
        "status": "frozen-before-outcome-access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcomes_accessed": False,
        "preregistration": {
            "path": str(preregistration.resolve()),
            "sha256": prereg_hash,
            "lock_path": str(preregistration_lock.resolve()),
            "lock_sha256": file_sha256(preregistration_lock),
        },
        "cohorts": {"calibration": calibration, "evaluation": evaluation},
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(payload)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"study_manifest_sha256": manifest_hash}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def read_locked_study(manifest_path: str | Path, lock_path: str | Path) -> tuple[dict[str, Any], str]:
    manifest_path, lock_path = Path(manifest_path), Path(lock_path)
    digest = file_sha256(manifest_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("study_manifest_sha256") != digest:
        raise ValueError("Study manifest does not match its lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen-before-outcome-access" or manifest.get("outcomes_accessed") is not False:
        raise ValueError("Study manifest is not a label-blind frozen study")
    return manifest, digest


def validate_feature_cohort(
    features_path: str | Path,
    manifest_path: str | Path,
    lock_path: str | Path,
    cohort: str,
) -> str:
    """Verify the exact feature file and event roster against a frozen cohort."""
    manifest, digest = read_locked_study(manifest_path, lock_path)
    if cohort not in {"calibration", "evaluation"}:
        raise ValueError("cohort must be 'calibration' or 'evaluation'")
    expected = manifest["cohorts"][cohort]
    if file_sha256(features_path) != expected["sha256"]:
        raise ValueError(f"{cohort} feature file does not match the frozen study")
    actual_ids = _event_ids(pd.read_parquet(features_path))
    if actual_ids != expected["event_ids"]:
        raise ValueError(f"{cohort} event roster does not match the frozen study")
    return digest


def validate_label_roster(labels: pd.DataFrame, manifest: dict[str, Any], cohort: str) -> None:
    if cohort not in {"calibration", "evaluation"}:
        raise ValueError("cohort must be 'calibration' or 'evaluation'")
    if not {"event_id", "y"}.issubset(labels.columns):
        raise ValueError("Labels must contain event_id and y")
    roster = labels.loc[:, ["event_id", "y"]].drop_duplicates()
    if roster["event_id"].astype(str).duplicated().any():
        raise ValueError("Labels must contain one row per event_id")
    actual = sorted(roster["event_id"].astype(str).tolist())
    if actual != manifest["cohorts"][cohort]["event_ids"]:
        raise ValueError(f"{cohort} labels do not match the frozen event roster")
