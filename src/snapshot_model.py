"""Training and scoring for the frozen CatBoost snapshot model."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

NUMERIC_FEATURES = (
    "time_to_tca",
    "risk",
    "max_risk_estimate",
    "max_risk_scaling",
    "miss_distance",
    "relative_speed",
    "mahalanobis_distance",
    "t_position_covariance_det",
    "c_position_covariance_det",
    "t_obs_available",
    "t_obs_used",
    "t_weighted_rms",
    "c_obs_available",
    "c_obs_used",
    "c_weighted_rms",
)
CATEGORICAL_FEATURES = ("mission_id", "c_object_type")
MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "random_seed": 24072026,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
}
MIN_DAYS_TO_TCA = 2.0
MAX_DAYS_TO_TCA = 7.0


def _event_ids(frame: pd.DataFrame) -> set[str]:
    return set(frame["event_id"].astype(str).unique())


def assert_disjoint_splits(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for index, left_name in enumerate(names):
        left = _event_ids(splits[left_name])
        for right_name in names[index + 1 :]:
            overlap = left.intersection(_event_ids(splits[right_name]))
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} overlap by "
                    f"{len(overlap)} event_id values"
                )


def prepare_snapshot_frame(
    frame: pd.DataFrame, require_labels: bool = True
) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    if require_labels:
        required.add("y")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = frame.loc[
        frame["time_to_tca"].between(
            MIN_DAYS_TO_TCA, MAX_DAYS_TO_TCA, inclusive="both"
        )
    ].copy()
    if selected.empty:
        raise ValueError("No rows fall inside the 2--7 day decision window")
    if selected.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    if require_labels:
        if not selected["y"].isin([0, 1]).all():
            raise ValueError("y must contain only 0 and 1")
        if (selected.groupby("event_id")["y"].nunique() != 1).any():
            raise ValueError("Each event_id must have one event-level label")
    for column in NUMERIC_FEATURES:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        values = selected[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{column} contains infinite values")
    for column in CATEGORICAL_FEATURES:
        selected[column] = selected[column].astype("string").fillna("__MISSING__")
    return selected.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).reset_index(drop=True)


def event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id")["event_id"].transform("size")
    return (1.0 / counts).to_numpy(dtype=float)


def fit_snapshot_model(
    training: pd.DataFrame, model_params: dict[str, Any] | None = None
) -> CatBoostClassifier:
    prepared = prepare_snapshot_frame(training)
    if prepared["y"].nunique() != 2:
        raise ValueError("Training data must contain both classes")
    features = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    params = MODEL_PARAMS.copy()
    if model_params is not None:
        params.update(model_params)
    model = CatBoostClassifier(**params)
    model.fit(
        prepared[features],
        prepared["y"].astype(int),
        cat_features=list(CATEGORICAL_FEATURES),
        sample_weight=event_equal_weights(prepared),
    )
    return model


def score_snapshot_model(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    include_labels: bool = True,
    passthrough_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    prepared = prepare_snapshot_frame(frame, require_labels=include_labels)
    passthrough = tuple(passthrough_columns)
    unknown = set(passthrough).difference(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown passthrough columns: {sorted(unknown)}")
    if len(set(passthrough)) != len(passthrough):
        raise ValueError("passthrough_columns must be unique")
    features = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    scores = model.predict_proba(prepared[features])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores")
    columns = ["event_id", "time_to_tca"]
    if include_labels:
        columns.append("y")
    columns.extend(column for column in passthrough if column not in columns)
    return prepared.loc[:, columns].assign(catboost_snapshot=scores)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
