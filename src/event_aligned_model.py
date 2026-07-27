"""Development models aligned with event-level sequential exclusion."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from prefix_features import DYNAMIC_COLUMNS, build_prefix_features, eligible_prefixes
from snapshot_model import CATEGORICAL_FEATURES, MODEL_PARAMS, NUMERIC_FEATURES

PREFIX_FEATURES = tuple(
    f"{column}_{suffix}"
    for column in DYNAMIC_COLUMNS
    for suffix in ("first", "min", "max", "range", "delta_first", "delta_prev")
) + ("n_cdm_so_far", "dt_prev", "history_span_days", "eligible_history_count")
DYNAMIC_NUMERIC_FEATURES = tuple(dict.fromkeys(NUMERIC_FEATURES + PREFIX_FEATURES))
DYNAMIC_FEATURES = DYNAMIC_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_dynamic_frame(
    frame: pd.DataFrame, require_labels: bool = True
) -> pd.DataFrame:
    """Build causal, window-local prefix features for training or scoring."""
    featured = eligible_prefixes(build_prefix_features(frame))
    required = {"event_id", "time_to_tca", *DYNAMIC_FEATURES}
    if require_labels:
        required.add("y")
    missing = required.difference(featured.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if require_labels:
        if not featured["y"].isin([0, 1]).all():
            raise ValueError("y must contain only 0 and 1")
        if (featured.groupby("event_id")["y"].nunique() != 1).any():
            raise ValueError("Each event_id must have one event-level label")
    for column in DYNAMIC_NUMERIC_FEATURES:
        featured[column] = pd.to_numeric(featured[column], errors="coerce")
        if np.isinf(featured[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values")
    for column in CATEGORICAL_FEATURES:
        featured[column] = featured[column].astype("string").fillna("__MISSING__")
    return featured.reset_index(drop=True)


def event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id")["event_id"].transform("size")
    return (1.0 / counts).to_numpy(dtype=float)


def positive_tail_weights(
    frame: pd.DataFrame,
    base_scores: pd.Series | np.ndarray,
    hard_fraction: float = 0.25,
    hard_mass: float = 0.50,
) -> np.ndarray:
    """Concentrate part of each positive event's weight on its lowest scores.

    Every event keeps total weight one. Negative events remain uniform. For a
    positive event, ``hard_mass`` is spread over the lowest-scored
    ``hard_fraction`` of prefixes and the remaining mass stays uniform.
    ``base_scores`` must be out-of-fold with respect to every row supplied.
    """
    if not 0 < hard_fraction <= 1:
        raise ValueError("hard_fraction must lie in (0, 1]")
    if not 0 <= hard_mass <= 1:
        raise ValueError("hard_mass must lie in [0, 1]")
    scores = np.asarray(base_scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("base_scores must be one finite value per row")
    if not frame["y"].isin([0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    if (frame.groupby("event_id")["y"].nunique() != 1).any():
        raise ValueError("Each event_id must have one event-level label")

    weights = event_equal_weights(frame)
    work = pd.DataFrame({
        "event_id": frame["event_id"].to_numpy(),
        "y": frame["y"].to_numpy(),
        "score": scores,
        "position": np.arange(len(frame)),
    })
    for _, event in work.loc[work["y"] == 1].groupby("event_id", sort=False):
        positions = event["position"].to_numpy(dtype=int)
        weights[positions] *= 1.0 - hard_mass
        hard_count = max(1, int(np.ceil(hard_fraction * len(event))))
        hard_positions = event.nsmallest(hard_count, ["score", "position"])["position"]
        weights[hard_positions.to_numpy(dtype=int)] += hard_mass / hard_count
    totals = pd.Series(weights).groupby(frame["event_id"].reset_index(drop=True)).sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0):
        raise RuntimeError("Event weights do not sum to one")
    return weights


def fit_dynamic_model(
    prepared: pd.DataFrame,
    sample_weight: np.ndarray,
    model_params: dict[str, Any] | None = None,
) -> CatBoostClassifier:
    if len(prepared) != len(sample_weight):
        raise ValueError("sample_weight length does not match training rows")
    params = MODEL_PARAMS.copy()
    if model_params is not None:
        params.update(model_params)
    model = CatBoostClassifier(**params)
    model.fit(
        prepared[list(DYNAMIC_FEATURES)],
        prepared["y"].astype(int),
        cat_features=list(CATEGORICAL_FEATURES),
        sample_weight=np.asarray(sample_weight, dtype=float),
    )
    return model


def score_dynamic_model(model: CatBoostClassifier, prepared: pd.DataFrame) -> np.ndarray:
    scores = model.predict_proba(prepared[list(DYNAMIC_FEATURES)])[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores")
    return scores


def score_dynamic_frame(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    score_column: str = "catboost_tail_aligned",
    passthrough_columns: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Score raw, label-blind CDM histories and retain selected model features."""
    if "y" in frame.columns:
        raise ValueError("Scoring input must be label-blind; provide labels separately")
    if not isinstance(score_column, str) or not score_column:
        raise ValueError("score_column must be a non-empty string")
    passthrough = tuple(passthrough_columns)
    if len(set(passthrough)) != len(passthrough):
        raise ValueError("passthrough_columns must be unique")
    reserved = {"event_id", "time_to_tca", "eligible_history_count", score_column}
    overlap = reserved.intersection(passthrough)
    if overlap:
        raise ValueError(f"Reserved passthrough columns: {sorted(overlap)}")
    unknown = set(passthrough).difference(DYNAMIC_FEATURES)
    if unknown:
        raise ValueError(f"Unknown passthrough columns: {sorted(unknown)}")

    prepared = prepare_dynamic_frame(frame, require_labels=False)
    scores = score_dynamic_model(model, prepared)
    columns = ["event_id", "time_to_tca", "eligible_history_count", *passthrough]
    return prepared.loc[:, columns].assign(**{score_column: scores})
