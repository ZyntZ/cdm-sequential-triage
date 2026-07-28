"""Freeze and verify label-blind cohorts for a new confirmation study."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


OUTCOME_COLUMN_NAMES = frozenset({"y", "label", "target", "outcome"})


def _outcome_columns(columns: pd.Index) -> list[str]:
    """Return columns that explicitly expose a final event outcome."""
    forbidden = []
    for column in columns:
        normalized = str(column).strip().lower()
        tokens = normalized.replace("-", "_").split("_")
        if normalized in OUTCOME_COLUMN_NAMES or "final" in tokens:
            forbidden.append(str(column))
    return sorted(forbidden)


def _event_ids(
    frame: pd.DataFrame,
    min_days_to_tca: float | None = None,
) -> list[str]:
    required = {"event_id", "time_to_tca"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    outcome_columns = _outcome_columns(frame.columns)
    if outcome_columns:
        raise ValueError(
            "Study features must be outcome-blind; forbidden columns: "
            f"{outcome_columns}"
        )
    if frame.empty:
        raise ValueError("Study cohort must contain at least one feature row")
    if frame["event_id"].isna().any():
        raise ValueError("event_id must not contain missing values")
    tca = pd.to_numeric(frame["time_to_tca"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tca).all() or (tca < 0).any():
        raise ValueError("time_to_tca must be finite and non-negative")
    if frame.assign(__time_to_tca=tca).duplicated(["event_id", "__time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    if min_days_to_tca is not None and (tca < min_days_to_tca).any():
        count = int((tca < min_days_to_tca).sum())
        raise ValueError(
            f"Study features contain {count} post-window rows below "
            f"{min_days_to_tca:g} days to TCA; these rows can expose the final outcome"
        )
    return sorted(frame["event_id"].astype(str).unique().tolist())


def _decision_window(preregistration: dict[str, Any]) -> tuple[float, float] | None:
    window = preregistration.get("candidate", {}).get("decision_window_days")
    if window is None:
        return None
    if not isinstance(window, list) or len(window) != 2:
        raise ValueError("Preregistration has an invalid decision window")
    minimum, maximum = map(float, window)
    if not 0 <= minimum < maximum:
        raise ValueError("Preregistration has an invalid decision window")
    return minimum, maximum


def _cohort_record(
    path: str | Path,
    decision_window: tuple[float, float] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_parquet(path)
    minimum = None if decision_window is None else decision_window[0]
    event_ids = _event_ids(frame, min_days_to_tca=minimum)
    tca = pd.to_numeric(frame["time_to_tca"], errors="raise")
    record = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "events": len(event_ids),
        "event_ids": event_ids,
        "time_to_tca_min": float(tca.min()),
        "time_to_tca_max": float(tca.max()),
        "columns": sorted(str(column) for column in frame.columns),
    }
    if decision_window is not None:
        window_rows = tca.between(*decision_window, inclusive="both")
        record["decision_window_rows"] = int(window_rows.sum())
        record["decision_window_events"] = int(frame.loc[window_rows, "event_id"].nunique())
    return record


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
    preregistration_payload = json.loads(preregistration.read_text(encoding="utf-8"))
    decision_window = _decision_window(preregistration_payload)

    calibration = _cohort_record(calibration_features, decision_window)
    evaluation = _cohort_record(evaluation_features, decision_window)
    overlap = set(calibration["event_ids"]).intersection(evaluation["event_ids"])
    if overlap:
        raise ValueError(f"Calibration and evaluation cohorts overlap by {len(overlap)} event_id values")
    manifest = {
        "schema_version": 2,
        "status": "frozen-before-outcome-access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcomes_accessed": False,
        "preregistration": {
            "path": str(preregistration.resolve()),
            "sha256": prereg_hash,
            "lock_path": str(preregistration_lock.resolve()),
            "lock_sha256": file_sha256(preregistration_lock),
        },
        "outcome_firewall": {
            "decision_window_days": None if decision_window is None else list(decision_window),
            "post_window_rows_forbidden": decision_window is not None,
            "explicit_outcome_columns_forbidden": True,
        },
        "cohorts": {"calibration": calibration, "evaluation": evaluation},
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    lock_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        lock_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"study_manifest_sha256": manifest_hash}, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
        lock_created = False
    except Exception:
        if lock_created:
            lock.unlink(missing_ok=True)
        raise
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
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
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported study manifest schema")
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
    firewall = manifest.get("outcome_firewall", {})
    window = firewall.get("decision_window_days")
    minimum = None if window is None else float(window[0])
    actual_ids = _event_ids(
        pd.read_parquet(features_path),
        min_days_to_tca=minimum if firewall.get("post_window_rows_forbidden") else None,
    )
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
