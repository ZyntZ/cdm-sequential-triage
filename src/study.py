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


def _read_denominator_roster(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("event_ids") if isinstance(payload, dict) else payload
        roster = pd.DataFrame({"event_id": values})
    else:
        roster = pd.read_parquet(path)
    if "event_id" not in roster.columns:
        raise ValueError("Denominator roster must contain event_id")
    if roster["event_id"].isna().any():
        raise ValueError("Denominator roster event_id must not contain missing values")
    values = roster["event_id"].astype(str)
    if values.duplicated().any():
        raise ValueError("Denominator roster must contain one row per event_id")
    if values.empty:
        raise ValueError("Denominator roster must contain at least one event")
    return sorted(values.tolist())


def _cohort_record(
    path: str | Path,
    decision_window: tuple[float, float] | None = None,
    denominator_roster: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(path)
    frame = pd.read_parquet(path)
    minimum = None if decision_window is None else decision_window[0]
    feature_event_ids = _event_ids(frame, min_days_to_tca=minimum)
    event_ids = (
        feature_event_ids
        if denominator_roster is None
        else _read_denominator_roster(denominator_roster)
    )
    absent = set(feature_event_ids).difference(event_ids)
    if absent:
        raise ValueError(
            f"Feature file contains {len(absent)} event_id values outside the denominator roster"
        )
    tca = pd.to_numeric(frame["time_to_tca"], errors="raise")
    record = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "events": len(event_ids),
        "event_ids": event_ids,
        "feature_events": len(feature_event_ids),
        "feature_event_ids": feature_event_ids,
        "no_feature_events": len(event_ids) - len(feature_event_ids),
        "time_to_tca_min": float(tca.min()),
        "time_to_tca_max": float(tca.max()),
        "columns": sorted(str(column) for column in frame.columns),
        "denominator_roster": None if denominator_roster is None else {
            "path": str(Path(denominator_roster).resolve()),
            "sha256": file_sha256(denominator_roster),
        },
    }
    if decision_window is not None:
        window_rows = tca.between(*decision_window, inclusive="both")
        decision_event_ids = sorted(
            frame.loc[window_rows, "event_id"].astype(str).unique().tolist()
        )
        record["decision_window_rows"] = int(window_rows.sum())
        record["decision_window_events"] = len(decision_event_ids)
        record["decision_window_event_ids"] = decision_event_ids
    return record


def _prospective_cohort(event_id: str, seed: int, calibration_fraction: float) -> str:
    if not 0 < calibration_fraction < 1:
        raise ValueError("Allocation calibration_fraction must lie in (0, 1)")
    digest = hashlib.sha256(f"{int(seed)}:{event_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "calibration" if value < calibration_fraction else "evaluation"


def _assignment_digest(assignments: dict[str, str]) -> str:
    raw = json.dumps(
        assignments, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def freeze_study(
    calibration_features: str | Path,
    evaluation_features: str | Path,
    preregistration: str | Path,
    preregistration_lock: str | Path,
    output: str | Path,
    lock: str | Path,
    calibration_roster: str | Path | None = None,
    evaluation_roster: str | Path | None = None,
    allocation_manifest: str | Path | None = None,
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

    calibration = _cohort_record(
        calibration_features, decision_window, calibration_roster
    )
    evaluation = _cohort_record(
        evaluation_features, decision_window, evaluation_roster
    )
    overlap = set(calibration["event_ids"]).intersection(evaluation["event_ids"])
    if overlap:
        raise ValueError(f"Calibration and evaluation cohorts overlap by {len(overlap)} event_id values")

    allocation_record = None
    if allocation_manifest is not None:
        allocation_path = Path(allocation_manifest)
        allocation_payload = json.loads(allocation_path.read_text(encoding="utf-8"))
        if allocation_payload.get("status") != "assigned-before-outcome-access":
            raise ValueError("Allocation manifest is not frozen before outcome access")
        if allocation_payload.get("outcomes_accessed") is not False:
            raise ValueError("Allocation manifest has accessed outcomes")
        if allocation_payload.get("collection_status") != "sealed":
            raise ValueError("Allocation manifest must come from a sealed collection")
        expected_counts = {
            "calibration": len(calibration["event_ids"]),
            "evaluation": len(evaluation["event_ids"]),
        }
        for cohort, count in expected_counts.items():
            if int(allocation_payload.get(f"{cohort}_events", -1)) != count:
                raise ValueError(
                    f"Allocation manifest {cohort} roster does not match frozen study"
                )
        seed = int(allocation_payload.get("seed"))
        fraction = float(allocation_payload.get("calibration_fraction"))
        assignments = {
            event_id: "calibration" for event_id in calibration["event_ids"]
        }
        assignments.update({
            event_id: "evaluation" for event_id in evaluation["event_ids"]
        })
        expected_assignments = {
            event_id: _prospective_cohort(event_id, seed, fraction)
            for event_id in assignments
        }
        if assignments != expected_assignments:
            raise ValueError(
                "Frozen cohort rosters do not match the label-blind allocation rule"
            )
        if allocation_payload.get("assignments_sha256") != _assignment_digest(assignments):
            raise ValueError("Allocation manifest assignment digest is invalid")
        outputs = allocation_payload.get("outputs", {})
        expected_paths = {
            "calibration_features": calibration_features,
            "evaluation_features": evaluation_features,
            "calibration_roster": calibration_roster,
            "evaluation_roster": evaluation_roster,
        }
        for name, expected_path in expected_paths.items():
            record = outputs.get(name)
            if expected_path is None or not isinstance(record, dict):
                raise ValueError(f"Allocation manifest is missing {name}")
            if file_sha256(expected_path) != record.get("sha256"):
                raise ValueError(f"Allocation manifest {name} SHA-256 mismatch")
        allocation_record = {
            "path": str(allocation_path.resolve()),
            "sha256": file_sha256(allocation_path),
            "status": allocation_payload["status"],
            "outcomes_accessed": False,
            "collection_status": allocation_payload["collection_status"],
            "ledger_sha256": allocation_payload.get("ledger_sha256"),
            "assignments_sha256": allocation_payload.get("assignments_sha256"),
            "rule": allocation_payload.get("rule"),
            "seed": allocation_payload.get("seed"),
            "calibration_fraction": allocation_payload.get("calibration_fraction"),
        }
    manifest = {
        "schema_version": 4,
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
        "allocation": allocation_record,
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
    if manifest.get("schema_version") not in {1, 2, 3, 4}:
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
    expected_feature_ids = expected.get("feature_event_ids", expected["event_ids"])
    if actual_ids != expected_feature_ids:
        raise ValueError(f"{cohort} feature event roster does not match the frozen study")
    roster_record = expected.get("denominator_roster")
    if roster_record is not None and file_sha256(roster_record["path"]) != roster_record["sha256"]:
        raise ValueError(f"{cohort} denominator roster does not match the frozen study")
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


def validate_scored_cohort_roster(
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    cohort: str,
) -> None:
    """Require scores for every frozen event that enters the decision window."""
    if cohort not in {"calibration", "evaluation"}:
        raise ValueError("cohort must be 'calibration' or 'evaluation'")
    if "event_id" not in scores.columns:
        raise ValueError("Scores must contain event_id")
    if scores["event_id"].isna().any():
        raise ValueError("Score event_id must not contain missing values")
    expected = manifest["cohorts"][cohort].get("decision_window_event_ids")
    if expected is None:
        return
    actual = sorted(scores["event_id"].astype(str).unique().tolist())
    if actual != expected:
        missing = len(set(expected).difference(actual))
        extra = len(set(actual).difference(expected))
        raise ValueError(
            f"{cohort} score roster does not match the frozen decision-window "
            f"roster: {missing} missing and {extra} extra event_id values"
        )
