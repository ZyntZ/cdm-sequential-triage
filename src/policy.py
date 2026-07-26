"""Event-level sequential-policy and calibration utilities."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.stats import beta

from shift_gate import ConformalShiftGate


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



def history_gated_event_table(
    prefix_scores: pd.DataFrame,
    score_col: str,
    minimum_history: int,
    history_col: str = "eligible_history_count",
) -> pd.DataFrame:
    """Aggregate scores after a minimum-history gate without dropping events.

    Events with no eligible prefix receive ``min_score = +inf`` and therefore
    cannot be assigned SAFE-EXCLUDE by a finite lower-tail threshold. Keeping
    these events is required for an event-level calibration denominator.
    """
    if minimum_history < 1:
        raise ValueError("minimum_history must be at least one")
    required = {"event_id", "y", score_col, "time_to_tca", history_col}
    missing = required.difference(prefix_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    labels = prefix_scores.groupby("event_id", as_index=False).agg(y=("y", "first"))
    eligible = prefix_scores.loc[prefix_scores[history_col] >= minimum_history]
    if eligible.empty:
        result = labels.copy()
        result["min_score"] = np.inf
        result["first_available_tca"] = np.nan
        result["last_available_tca"] = np.nan
        return result
    aggregated = event_policy_table(eligible, score_col)
    result = labels.merge(aggregated, on=["event_id", "y"], how="left", validate="one_to_one")
    result["min_score"] = result["min_score"].fillna(np.inf)
    return result

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
    if scores.size == 0:
        raise ValueError("At least one positive-event score is required")
    if np.isnan(scores).any():
        raise ValueError("Positive-event scores must not contain NaN")
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

def first_safe_decision_table(
    prefix_scores: pd.DataFrame,
    score_col: str,
    threshold: float,
    minimum_history: int = 1,
    history_col: str = "eligible_history_count",
    shift_gate: ConformalShiftGate | None = None,
) -> pd.DataFrame:
    """Return one event-level row with the first permitted SAFE-EXCLUDE time.

    Input rows must already be restricted to the decision window. The history
    counter is therefore interpreted within that window, matching the runtime
    policy. If supplied, ``shift_gate`` must have been fitted and calibrated on
    data independent of the rows being evaluated. A threshold crossing blocked
    by the gate is retained as an audit field but is not a SAFE-EXCLUDE.
    Events that never receive SAFE-EXCLUDE are retained.
    """
    if minimum_history < 1:
        raise ValueError("minimum_history must be at least one")
    required = {"event_id", "y", score_col, "time_to_tca", history_col}
    missing = required.difference(prefix_scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if prefix_scores.empty:
        raise ValueError("prefix_scores must contain at least one row")
    if prefix_scores.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    label_counts = prefix_scores.groupby("event_id")["y"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Each event_id must have one event-level label")

    ordered = prefix_scores.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()
    threshold_crossing = (
        (ordered[history_col] >= minimum_history)
        & (ordered[score_col] <= threshold)
    )
    if shift_gate is None:
        gate_allowed = pd.Series(True, index=ordered.index, dtype=bool)
    else:
        gate_allowed = shift_gate.allows_safe_exclude(ordered)
        if not gate_allowed.index.equals(ordered.index):
            raise ValueError("Shift-gate result index does not match prefix rows")
        gate_allowed = gate_allowed.astype(bool)

    safe_rows = ordered.loc[threshold_crossing & gate_allowed].drop_duplicates(
        "event_id", keep="first"
    )
    blocked_rows = ordered.loc[threshold_crossing & ~gate_allowed].drop_duplicates(
        "event_id", keep="first"
    )
    labels = ordered.groupby("event_id", as_index=False).agg(y=("y", "first"))
    decisions = safe_rows.loc[:, ["event_id", "time_to_tca", score_col]].rename(
        columns={
            "time_to_tca": "first_safe_tca",
            score_col: "first_safe_score",
        }
    )
    blocked = blocked_rows.loc[:, ["event_id", "time_to_tca", score_col]].rename(
        columns={
            "time_to_tca": "first_blocked_safe_tca",
            score_col: "first_blocked_safe_score",
        }
    )
    result = labels.merge(decisions, on="event_id", how="left", validate="one_to_one")
    result = result.merge(blocked, on="event_id", how="left", validate="one_to_one")
    result["safe_exclude"] = result["first_safe_tca"].notna()
    result["shift_gate_blocked"] = result["first_blocked_safe_tca"].notna()
    return result


def evaluate_sequential_policy(
    prefix_scores: pd.DataFrame,
    score_col: str,
    threshold: float,
    minimum_history: int = 1,
    history_col: str = "eligible_history_count",
    confidence: float = 0.95,
    shift_gate: ConformalShiftGate | None = None,
) -> dict:
    """Evaluate event-level safety, automation and first-decision timing."""
    decisions = first_safe_decision_table(
        prefix_scores=prefix_scores,
        score_col=score_col,
        threshold=threshold,
        minimum_history=minimum_history,
        history_col=history_col,
        shift_gate=shift_gate,
    )
    positive = decisions["y"] == 1
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Evaluation requires positive and negative events")
    danger_k = int((decisions["safe_exclude"] & positive).sum())
    safe_negative = decisions["safe_exclude"] & negative
    first_safe = decisions.loc[safe_negative, "first_safe_tca"]
    return {
        "threshold": float(threshold),
        "minimum_history": int(minimum_history),
        "danger_k": danger_k,
        "danger_n": n_positive,
        "danger_rate": danger_k / n_positive,
        "danger_ucb": cp_upper(danger_k, n_positive, confidence),
        "safe_negative": int(safe_negative.sum()),
        "negative_n": n_negative,
        "safe_negative_rate": float(safe_negative.sum() / n_negative),
        "shift_gate_blocked_events": int(decisions["shift_gate_blocked"].sum()),
        "shift_gate_blocked_positive": int(
            (decisions["shift_gate_blocked"] & positive).sum()
        ),
        "shift_gate_blocked_negative": int(
            (decisions["shift_gate_blocked"] & negative).sum()
        ),
        "median_first_safe_tca": (
            None if first_safe.empty else float(first_safe.median())
        ),
    }

