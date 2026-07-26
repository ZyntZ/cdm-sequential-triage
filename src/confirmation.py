"""Frozen calibration and one-shot confirmation utilities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from policy import calibrate_positive_threshold, cp_upper, first_safe_decision_table, history_gated_event_table

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


def prepare_prefix_scores(frame: pd.DataFrame) -> pd.DataFrame:
    score_column = POLICY["score_column"]
    required = {"event_id", "time_to_tca", "y", score_column, "model_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = frame.loc[
        frame["time_to_tca"].between(
            POLICY["min_days_to_tca"], POLICY["max_days_to_tca"], inclusive="both"
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
) -> dict[str, Any]:
    prepared = prepare_prefix_scores(calibration_prefixes)
    scored_events = history_gated_event_table(
        prepared,
        POLICY["score_column"],
        minimum_history=POLICY["minimum_history"],
    )
    if calibration_labels is None:
        labels = scored_events.loc[:, ["event_id", "y"]]
    else:
        labels = validate_event_labels(calibration_labels)
        scored_ids = set(scored_events["event_id"].astype(str))
        label_ids = set(labels["event_id"].astype(str))
        if not scored_ids.issubset(label_ids):
            raise ValueError("Calibration scores contain event_id values without labels")
        scored_events = scored_events.drop(columns="y")
        scored_events = labels.merge(
            scored_events, on="event_id", how="left", validate="one_to_one"
        )
        scored_events["min_score"] = scored_events["min_score"].fillna(float("inf"))
    positives = scored_events.loc[scored_events["y"] == 1, "min_score"]
    rule = calibrate_positive_threshold(
        positives,
        alpha=POLICY["alpha"],
        mode=POLICY["calibration_mode"],
        confidence=POLICY["confidence"],
    )
    if rule["rank"] == 0:
        raise ValueError("Too few positive calibration events for the frozen PAC level")
    return {
        "policy": POLICY.copy(),
        "calibration": rule,
        "calibration_events": int(scored_events.shape[0]),
        "calibration_event_ids": sorted(str(value) for value in labels["event_id"]),
        "calibration_event_ids_sha256": event_id_digest(labels["event_id"]),
        "model_sha256": str(prepared["model_sha256"].iloc[0]),
    }

def evaluate(
    evaluation_prefixes: pd.DataFrame,
    calibration_artifact: dict[str, Any],
    evaluation_labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if calibration_artifact.get("policy") != POLICY:
        raise ValueError("Calibration artifact does not match the frozen policy")
    threshold = calibration_artifact.get("calibration", {}).get("threshold")
    if threshold is None:
        raise ValueError("Calibration artifact has no threshold")
    prepared = prepare_prefix_scores(evaluation_prefixes)
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
    calibration_ids = set(calibration_artifact.get("calibration_event_ids", []))
    evaluation_ids = set(labels["event_id"].astype(str).unique())
    overlap = calibration_ids.intersection(evaluation_ids)
    if overlap:
        raise ValueError(
            f"Calibration and evaluation overlap by {len(overlap)} event_id values"
        )
    decisions = first_safe_decision_table(
        prepared, POLICY["score_column"], float(threshold), POLICY["minimum_history"]
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
        "minimum_history": POLICY["minimum_history"],
        "danger_k": danger_k,
        "danger_n": danger_n,
        "danger_rate": danger_k / danger_n,
        "danger_ucb": cp_upper(danger_k, danger_n, POLICY["confidence"]),
        "safe_negative": int(safe_negative.sum()),
        "negative_n": negative_n,
        "safe_negative_rate": float(safe_negative.sum() / negative_n),
        "median_first_safe_tca": None if first_safe.empty else float(first_safe.median()),
    }
    return {
        "policy": POLICY.copy(),
        "evaluation": metrics,
        "evaluation_events": int(labels.shape[0]),
        "evaluation_event_ids_sha256": event_id_digest(labels["event_id"]),
    }

def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def acquire_confirmation_lock(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"Confirmation lock already exists: {target}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
