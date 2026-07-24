"""Causal prefix features for sequential CDM triage.

Rows must contain one event per event_id and time_to_tca in days. Larger
values occur earlier. Features at a row only use that row and earlier rows.
"""
from __future__ import annotations
import pandas as pd

DYNAMIC_COLUMNS = ("risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance")


def build_prefix_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", *DYNAMIC_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")

    out = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).copy()
    g = out.groupby("event_id", sort=False)
    for col in DYNAMIC_COLUMNS:
        out[f"{col}_first"] = g[col].transform("first")
        out[f"{col}_min"] = g[col].cummin()
        out[f"{col}_max"] = g[col].cummax()
        out[f"{col}_range"] = out[f"{col}_max"] - out[f"{col}_min"]
        out[f"{col}_delta_first"] = out[col] - out[f"{col}_first"]
        out[f"{col}_delta_prev"] = g[col].diff()
    out["n_cdm_so_far"] = g.cumcount() + 1
    out["dt_prev"] = g["time_to_tca"].diff().abs()
    out["history_span_days"] = g["time_to_tca"].transform("first") - out["time_to_tca"]
    return out


def eligible_prefixes(features: pd.DataFrame, min_days: float = 2.0, max_days: float = 7.0) -> pd.DataFrame:
    if not 0 <= min_days < max_days:
        raise ValueError("Require 0 <= min_days < max_days")
    return features.loc[
        features["time_to_tca"].between(min_days, max_days, inclusive="both")
    ].copy()
