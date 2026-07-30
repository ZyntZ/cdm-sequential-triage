"""Analytical planning for a genuinely new event-level validation study."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from scipy.stats import beta, binom


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cp_upper(k: int, n: int, confidence: float = 0.95) -> float:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Require n > 0 and 0 <= k <= n")
    return 1.0 if k == n else float(beta.ppf(confidence, k + 1, n - k))


def maximum_passing_failures(
    n_positive: int, alpha: float = 0.10, confidence: float = 0.95
) -> int:
    if n_positive <= 0:
        raise ValueError("n_positive must be positive")
    passing = [
        k for k in range(n_positive + 1)
        if cp_upper(k, n_positive, confidence) <= alpha
    ]
    return max(passing, default=-1)


def minimum_positive_events(
    failures: int, alpha: float = 0.10, confidence: float = 0.95
) -> int:
    if failures < 0:
        raise ValueError("failures must be non-negative")
    for n_positive in range(max(1, failures), 100000):
        if cp_upper(failures, n_positive, confidence) <= alpha:
            return n_positive
    raise RuntimeError("Search limit exceeded")


def pass_probability(
    n_positive: int,
    true_danger_rate: float,
    alpha: float = 0.10,
    confidence: float = 0.95,
) -> float:
    if not 0 <= true_danger_rate <= 1:
        raise ValueError("true_danger_rate must lie in [0, 1]")
    k_max = maximum_passing_failures(n_positive, alpha, confidence)
    return 0.0 if k_max < 0 else float(binom.cdf(k_max, n_positive, true_danger_rate))



def calibration_design(
    n_positive: int,
    alpha: float = 0.10,
    confidence: float = 0.95,
) -> dict:
    """Return finite-sample operating characteristics for PAC calibration.

    The selected rank is the largest positive-event order statistic whose
    one-sided beta tolerance bound does not exceed ``alpha``. A zero rank
    means that no finite SAFE-EXCLUDE threshold can be certified.
    """
    if n_positive <= 0:
        raise ValueError("n_positive must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")

    valid = [
        rank for rank in range(1, n_positive + 1)
        if beta.ppf(confidence, rank, n_positive + 1 - rank) <= alpha
    ]
    rank = max(valid, default=0)
    pac_bound = 0.0 if rank == 0 else float(
        beta.ppf(confidence, rank, n_positive + 1 - rank)
    )
    return {
        "positive_events": int(n_positive),
        "rank": int(rank),
        "finite_threshold_available": rank > 0,
        "marginal_bound": float(rank / (n_positive + 1)),
        "pac_bound": pac_bound,
        "alpha": float(alpha),
        "confidence": float(confidence),
    }


def validation_design_summary(
    calibration_positive_events: int,
    evaluation_positive_events: int,
    *,
    alpha: float = 0.10,
    calibration_confidence: float = 0.95,
    evaluation_confidence: float = 0.95,
    assumed_true_danger_rate: float = 0.05,
) -> dict:
    """Compute calibration and evaluation claims at their frozen confidence levels."""
    calibration = calibration_design(
        calibration_positive_events,
        alpha=alpha,
        confidence=calibration_confidence,
    )
    maximum_failures = maximum_passing_failures(
        evaluation_positive_events,
        alpha=alpha,
        confidence=evaluation_confidence,
    )
    return {
        "calibration": calibration,
        "evaluation": {
            "positive_events": int(evaluation_positive_events),
            "maximum_passing_dangerous_exclusions": int(maximum_failures),
            "upper_bound_at_maximum": (
                None if maximum_failures < 0
                else cp_upper(
                    maximum_failures,
                    evaluation_positive_events,
                    evaluation_confidence,
                )
            ),
            "assumed_true_danger_rate": float(assumed_true_danger_rate),
            "pass_probability_at_assumed_rate": pass_probability(
                evaluation_positive_events,
                assumed_true_danger_rate,
                alpha=alpha,
                confidence=evaluation_confidence,
            ),
            "alpha": float(alpha),
            "confidence": float(evaluation_confidence),
        },
    }

def evaluation_planning_table(
    positive_counts: tuple[int, ...] = (73, 89, 100, 120, 150, 200, 250, 300),
    assumed_rates: tuple[float, ...] = (0.04, 0.05, 0.06),
    alpha: float = 0.10,
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for n_positive in positive_counts:
        k_max = maximum_passing_failures(n_positive, alpha, confidence)
        row = {
            "positive_events": n_positive,
            "maximum_passing_failures": k_max,
            "upper_bound_at_maximum": (
                None if k_max < 0 else cp_upper(k_max, n_positive, confidence)
            ),
        }
        for rate in assumed_rates:
            row[f"pass_probability_if_true_rate_{rate:.2f}"] = pass_probability(
                n_positive, rate, alpha, confidence
            )
        rows.append(row)
    return pd.DataFrame(rows)
