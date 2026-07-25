"""Subgroup diagnostics for event-level SAFE-EXCLUDE decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from policy import cp_upper


def assign_quantile_groups(values: pd.Series, labels: tuple[str, ...]) -> pd.Series:
    """Assign stable quantile groups, reducing duplicate cut points when needed."""
    if values.isna().any():
        raise ValueError("Grouping values must be complete")
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def subgroup_metrics(
    events: pd.DataFrame,
    group_col: str,
    decision_col: str = "safe_exclude",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize dangerous exclusions and useful coverage by subgroup.

    A subgroup with no positive or no negative events keeps the corresponding
    rate and bound as missing rather than silently reporting zero.
    """
    required = {"event_id", "y", group_col, decision_col}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("Expected one row per event_id")

    rows = []
    for group, frame in events.groupby(group_col, observed=True, dropna=False):
        positive = frame["y"] == 1
        negative = ~positive
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        danger_k = int((frame[decision_col] & positive).sum())
        safe_negative = int((frame[decision_col] & negative).sum())
        rows.append({
            group_col: group,
            "events": int(len(frame)),
            "positive_events": n_positive,
            "negative_events": n_negative,
            "danger_k": danger_k,
            "danger_rate": danger_k / n_positive if n_positive else np.nan,
            "danger_ucb": cp_upper(danger_k, n_positive, confidence) if n_positive else np.nan,
            "safe_negative": safe_negative,
            "safe_negative_rate": safe_negative / n_negative if n_negative else np.nan,
        })
    return pd.DataFrame(rows)
