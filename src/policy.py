"""Event-level sequential-policy and calibration utilities."""
from __future__ import annotations

import math
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
    label_counts = prefix_scores.groupby("event_id")["y"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Each event_id must have one event-level label")
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
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Evaluation requires positive and negative events")
    k = int((safe & pos).sum())
    return {
        "threshold": float(threshold),
        "danger_k": k,
        "danger_n": n_pos,
        "danger_rate": k / n_pos,
        "danger_ucb": cp_upper(k, n_pos, confidence),
        "safe_negative_rate": float((safe & neg).sum() / n_neg),
        "safe_all_rate": float(safe.mean()),
    }


def calibration_rank(n_positive: int, alpha: float, mode: str = "marginal", confidence: float = 0.95) -> int:
    """Return the positive order-statistic rank used for a strict safe threshold.

    ``marginal`` controls risk averaged over the random calibration sample:
    rank/(n_positive + 1) <= alpha.

    ``pac`` chooses the largest rank whose one-sided tolerance bound is at
    most alpha with the requested confidence. This is conditional-on-calibration
    risk control with probability at least ``confidence``.
    """
    if n_positive <= 0:
        raise ValueError("n_positive must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if mode == "marginal":
        return min(n_positive, math.floor(alpha * (n_positive + 1)))
    if mode == "pac":
        valid = [
            rank for rank in range(1, n_positive + 1)
            if beta.ppf(confidence, rank, n_positive + 1 - rank) <= alpha
        ]
        return max(valid, default=0)
    raise ValueError("mode must be 'marginal' or 'pac'")


def calibrate_positive_threshold(
    positive_event_scores: pd.Series | np.ndarray,
    alpha: float,
    mode: str = "marginal",
    confidence: float = 0.95,
) -> dict:
    """Calibrate SAFE-EXCLUDE from positive event-level minimum scores.

    A lower score is considered safer. SAFE-EXCLUDE is issued when a new
    event minimum is less than or equal to the returned threshold. The
    threshold is one floating-point step below the selected order statistic,
    which makes the rank rule strict and conservative in the presence of ties.
    """
    scores = np.asarray(positive_event_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("At least one finite positive-event score is required")
    rank = calibration_rank(scores.size, alpha, mode, confidence)
    if rank == 0:
        threshold = -np.inf
    else:
        order_statistic = np.sort(scores)[rank - 1]
        threshold = np.nextafter(order_statistic, -np.inf)
    marginal_bound = rank / (scores.size + 1)
    pac_bound = 0.0 if rank == 0 else float(
        beta.ppf(confidence, rank, scores.size + 1 - rank)
    )
    return {
        "threshold": float(threshold),
        "rank": int(rank),
        "n_positive": int(scores.size),
        "alpha": float(alpha),
        "mode": mode,
        "confidence": float(confidence),
        "marginal_bound": float(marginal_bound),
        "pac_bound": float(pac_bound),
    }
