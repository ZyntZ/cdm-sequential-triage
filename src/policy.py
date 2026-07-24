"""Event-level sequential-policy utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import beta


def cp_upper(k: int, n: int, confidence: float = 0.95) -> float:
    if not (0 <= k <= n) or n <= 0:
        raise ValueError("Require 0 <= k <= n and n > 0")
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def event_policy_table(prefix_scores: pd.DataFrame, score_col: str) -> pd.DataFrame:
    required = {"event_id", "y", score_col, "time_to_tca"}
    missing = required.difference(prefix_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return prefix_scores.groupby("event_id", as_index=False).agg(
        y=("y", "first"),
        min_score=(score_col, "min"),
        first_available_tca=("time_to_tca", "max"),
        last_available_tca=("time_to_tca", "min"),
    )


def evaluate_threshold(events: pd.DataFrame, threshold: float, confidence: float = 0.95) -> dict:
    safe = events["min_score"] <= threshold
    pos = events["y"] == 1
    neg = ~pos
    k, n_pos = int((safe & pos).sum()), int(pos.sum())
    return {
        "threshold": float(threshold),
        "danger_k": k,
        "danger_n": n_pos,
        "danger_rate": k / n_pos,
        "danger_ucb": cp_upper(k, n_pos, confidence),
        "safe_negative_rate": float((safe & neg).sum() / neg.sum()),
        "safe_all_rate": float(safe.mean()),
    }
