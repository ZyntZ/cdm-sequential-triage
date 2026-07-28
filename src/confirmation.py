"""Frozen calibration and one-shot confirmation utilities."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from policy import calibrate_positive_threshold, cp_upper, first_safe_decision_table, history_gated_event_table
from shift_gate import ConformalShiftGate

POLICY = {
    "score_column": "catboost_snapshot",
    "alpha": 0.10,
    "confidence": 0.95,
    "calibration_mode": "pac",
    "minimum_history": 3,
    "min_days_to_tca": 2.0,
    "max_days_to_tca": 7.0,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_id_digest(event_ids: pd.Series) -> str:
    values = sorted(str(value) for value in event_ids.drop_duplicates())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


REQUIRED_POLICY_KEYS = {
    "score_column", "alpha", "confidence", "calibration_mode",
    "minimum_history", "min_days_to_tca", "max_days_to_tca",
}


def validate_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(policy, dict) or set(policy) != REQUIRED_POLICY_KEYS:
        raise ValueError("Policy fields do not match the confirmation schema")
    result = dict(policy)
    if not isinstance(result["score_column"], str) or not result["score_column"]:
        raise ValueError("score_column must be a non-empty string")
    if result["calibration_mode"] not in {"marginal", "pac"}:
        raise ValueError("Unsupported calibration_mode")
    if not 0 < float(result["alpha"]) < 1 or not 0 < float(result["confidence"]) < 1:
        raise ValueError("alpha and confidence must lie in (0, 1)")
    if int(result["minimum_history"]) < 1:
        raise ValueError("minimum_history must be at least one")
    if not 0 <= float(result["min_days_to_tca"]) < float(result["max_days_to_tca"]):
        raise ValueError("Invalid decision window")
    return result


def policy_from_model_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    candidate = manifest.get("candidate", {})
    window = candidate.get("decision_window_days", [])
    if len(window) != 2:
        raise ValueError("Model manifest has no valid decision window")
    return validate_policy({
        "score_column": manifest.get("score_column"),
        "alpha": candidate.get("alpha"),
        "confidence": candidate.get("calibration_confidence"),
        "calibration_mode": candidate.get("calibration_mode"),
        "minimum_history": candidate.get("minimum_history"),
        "min_days_to_tca": window[0],
        "max_days_to_tca": window[1],
    })


def prepare_prefix_scores(
    frame: pd.DataFrame, policy: dict[str, Any] | None = None
) -> pd.DataFrame:
    policy = POLICY if policy is None else policy
    score_column = policy["score_column"]
    required = {"event_id", "time_to_tca", "y", score_column, "model_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = frame.loc[
        frame["time_to_tca"].between(
            policy["min_days_to_tca"], policy["max_days_to_tca"], inclusive="both"
        )
    ].copy()
    if selected.empty:
        raise ValueError("No rows fall inside the frozen decision window")
    if selected.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    if not selected["y"].isin([0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    if (selected.groupby("event_id")["y"].nunique() != 1).any():
        raise ValueError("Each event_id must have one event-level label")
    model_hashes = selected["model_sha256"].dropna().astype(str).unique()
    if len(model_hashes) != 1 or selected["model_sha256"].isna().any():
        raise ValueError("Scores must contain exactly one model_sha256")
    selected = selected.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    )
    selected["eligible_history_count"] = (
        selected.groupby("event_id", sort=False).cumcount() + 1
    )
    return selected




def validate_event_labels(event_labels: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "y"}
    missing = required.difference(event_labels.columns)
    if missing:
        raise ValueError(f"Missing event label columns: {sorted(missing)}")
    labels = event_labels.loc[:, ["event_id", "y"]].drop_duplicates()
    if labels["event_id"].duplicated().any():
        raise ValueError("Event labels must contain one row per event_id")
    if not labels["y"].isin([0, 1]).all():
        raise ValueError("Event labels must contain only 0 and 1")
    return labels

def attach_event_labels(
    prefix_scores: pd.DataFrame, event_labels: pd.DataFrame
) -> pd.DataFrame:
    if "y" in prefix_scores.columns:
        raise ValueError("Evaluation scores must be label-blind")
    labels = validate_event_labels(event_labels)
    scored_ids = set(prefix_scores["event_id"].astype(str).unique())
    label_ids = set(labels["event_id"].astype(str).unique())
    if scored_ids != label_ids:
        missing_labels = len(scored_ids.difference(label_ids))
        extra_labels = len(label_ids.difference(scored_ids))
        raise ValueError(
            f"Evaluation label mismatch: {missing_labels} missing and "
            f"{extra_labels} extra event_id values"
        )
    merged = prefix_scores.merge(labels, on="event_id", how="left", validate="many_to_one")
    return merged

def calibrate(
    calibration_prefixes: pd.DataFrame,
    calibration_labels: pd.DataFrame | None = None,
    shift_gate: ConformalShiftGate | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = POLICY.copy() if policy is None else validate_policy(policy)
    if calibration_labels is None:
        prepared_input = calibration_prefixes
    else:
        labels = validate_event_labels(calibration_labels)
        scored_ids = set(calibration_prefixes["event_id"].astype(str).unique())
        label_ids = set(labels["event_id"].astype(str).unique())
        if not scored_ids.issubset(label_ids):
            raise ValueError("Calibration scores contain event_id values without labels")
        score_frame = calibration_prefixes.drop(columns="y", errors="ignore")
        prepared_input = score_frame.merge(
            labels, on="event_id", how="left", validate="many_to_one"
        )
    prepared = prepare_prefix_scores(prepared_input, policy)
    if shift_gate is None:
        scored_events = history_gated_event_table(
            prepared,
            policy["score_column"],
            minimum_history=policy["minimum_history"],
        )
    else:
        decisions = first_safe_decision_table(
            prepared,
            policy["score_column"],
            threshold=float("inf"),
            minimum_history=policy["minimum_history"],
            shift_gate=shift_gate,
        )
        scored_events = decisions.loc[:, ["event_id", "y"]].copy()
        permitted = prepared.loc[
            prepared["eligible_history_count"] >= policy["minimum_history"]
        ].copy()
        permitted = permitted.loc[shift_gate.allows_safe_exclude(permitted)]
        minima = permitted.groupby("event_id", as_index=False).agg(
            min_score=(policy["score_column"], "min")
        )
        scored_events = scored_events.merge(
            minima, on="event_id", how="left", validate="one_to_one"
        )
        scored_events["min_score"] = scored_events["min_score"].fillna(float("inf"))
    if calibration_labels is None:
        labels = scored_events.loc[:, ["event_id", "y"]]
    else:
        scored_events = scored_events.drop(columns="y")
        scored_events = labels.merge(
            scored_events, on="event_id", how="left", validate="one_to_one"
        )
        scored_events["min_score"] = scored_events["min_score"].fillna(float("inf"))
    positives = scored_events.loc[scored_events["y"] == 1, "min_score"]
    rule = calibrate_positive_threshold(
        positives,
        alpha=policy["alpha"],
        mode=policy["calibration_mode"],
        confidence=policy["confidence"],
    )
    if rule["rank"] == 0:
        raise ValueError("Too few positive calibration events for the frozen PAC level")
    return {
        "policy": policy.copy(),
        "calibration": rule,
        "calibration_events": int(scored_events.shape[0]),
        "calibration_event_ids": sorted(str(value) for value in labels["event_id"]),
        "calibration_event_ids_sha256": event_id_digest(labels["event_id"]),
        "model_sha256": str(prepared["model_sha256"].iloc[0]),
        "shift_gate_sha256": None if shift_gate is None else shift_gate.fingerprint(),
    }

def evaluate(
    evaluation_prefixes: pd.DataFrame,
    calibration_artifact: dict[str, Any],
    evaluation_labels: pd.DataFrame | None = None,
    shift_gate: ConformalShiftGate | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_policy = validate_policy(calibration_artifact.get("policy"))
    expected_policy = POLICY if policy is None else validate_policy(policy)
    if artifact_policy != expected_policy:
        raise ValueError("Calibration artifact does not match the frozen policy")
    policy = artifact_policy
    threshold = calibration_artifact.get("calibration", {}).get("threshold")
    if threshold is None:
        raise ValueError("Calibration artifact has no threshold")
    expected_gate_hash = calibration_artifact.get("shift_gate_sha256")
    supplied_gate_hash = None if shift_gate is None else shift_gate.fingerprint()
    if expected_gate_hash != supplied_gate_hash:
        raise ValueError("Calibration artifact and supplied shift gate do not match")
    prepared = prepare_prefix_scores(evaluation_prefixes, policy)
    evaluation_model_sha256 = str(prepared["model_sha256"].iloc[0])
    if calibration_artifact.get("model_sha256") != evaluation_model_sha256:
        raise ValueError("Calibration and evaluation scores use different models")
    if evaluation_labels is None:
        labels = prepared.loc[:, ["event_id", "y"]].drop_duplicates()
    else:
        labels = validate_event_labels(evaluation_labels)
        scored_ids = set(prepared["event_id"].astype(str).unique())
        label_ids = set(labels["event_id"].astype(str).unique())
        if not scored_ids.issubset(label_ids):
            raise ValueError("Evaluation scores contain event_id values without labels")
    calibration_event_ids = calibration_artifact.get("calibration_event_ids")
    calibration_event_ids_sha256 = calibration_artifact.get(
        "calibration_event_ids_sha256"
    )
    if not isinstance(calibration_event_ids, list) or not calibration_event_ids:
        raise ValueError("Calibration artifact has no event roster")
    if any(not isinstance(value, str) for value in calibration_event_ids):
        raise ValueError("Calibration event roster must contain strings")
    if len(calibration_event_ids) != len(set(calibration_event_ids)):
        raise ValueError("Calibration event roster contains duplicate event_id values")
    actual_calibration_digest = event_id_digest(pd.Series(calibration_event_ids))
    if calibration_event_ids_sha256 != actual_calibration_digest:
        raise ValueError("Calibration event roster does not match its SHA-256")
    calibration_ids = set(calibration_event_ids)
    evaluation_ids = set(labels["event_id"].astype(str).unique())
    overlap = calibration_ids.intersection(evaluation_ids)
    if overlap:
        raise ValueError(
            f"Calibration and evaluation overlap by {len(overlap)} event_id values"
        )
    decisions = first_safe_decision_table(
        prepared,
        policy["score_column"],
        float(threshold),
        policy["minimum_history"],
        shift_gate=shift_gate,
    ).drop(columns="y")
    events = labels.merge(decisions, on="event_id", how="left", validate="one_to_one")
    events["safe_exclude"] = events["safe_exclude"].eq(True)
    positive = events["y"] == 1
    negative = ~positive
    danger_k = int((events["safe_exclude"] & positive).sum())
    danger_n = int(positive.sum())
    negative_n = int(negative.sum())
    safe_negative = events["safe_exclude"] & negative
    first_safe = events.loc[safe_negative, "first_safe_tca"]
    metrics = {
        "threshold": float(threshold),
        "minimum_history": policy["minimum_history"],
        "danger_k": danger_k,
        "danger_n": danger_n,
        "danger_rate": danger_k / danger_n,
        "danger_ucb": cp_upper(danger_k, danger_n, policy["confidence"]),
        "safe_negative": int(safe_negative.sum()),
        "negative_n": negative_n,
        "safe_negative_rate": float(safe_negative.sum() / negative_n),
        "shift_gate_blocked_events": int(events["shift_gate_blocked"].eq(True).sum()),
        "shift_gate_blocked_positive": int(
            (events["shift_gate_blocked"].eq(True) & positive).sum()
        ),
        "shift_gate_blocked_negative": int(
            (events["shift_gate_blocked"].eq(True) & negative).sum()
        ),
        "median_first_safe_tca": None if first_safe.empty else float(first_safe.median()),
    }
    return {
        "policy": policy.copy(),
        "evaluation": metrics,
        "evaluation_events": int(labels.shape[0]),
        "evaluation_event_ids_sha256": event_id_digest(labels["event_id"]),
    }

def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n").encode("utf-8")
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def acquire_confirmation_lock(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"Confirmation lock already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
