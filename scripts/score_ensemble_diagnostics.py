"""Development-only score ensemble diagnostics without model retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from event_aligned_robustness import attach_candidate_scores, crossfit_decisions
from policy import cp_upper
from score_ensemble import SCORE_METHODS, combine_scores


def decision_metrics(decisions: pd.DataFrame, confidence: float = 0.95) -> dict:
    positive = decisions["y"] == 1
    negative = ~positive
    danger = decisions["safe_exclude"] & positive
    safe_negative = decisions["safe_exclude"] & negative
    timing = decisions.loc[safe_negative, "first_safe_tca"]
    return {
        "danger_k": int(danger.sum()),
        "danger_n": int(positive.sum()),
        "danger_rate": float(danger.sum() / positive.sum()),
        "danger_ucb": cp_upper(int(danger.sum()), int(positive.sum()), confidence),
        "safe_negative": int(safe_negative.sum()),
        "negative_n": int(negative.sum()),
        "safe_negative_rate": float(safe_negative.sum() / negative.sum()),
        "median_first_safe_tca_days": float(timing.median()),
    }


def paired_bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    replicates: int = 20000,
    seed: int = 24072026,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = rng.choice(values, size=values.size, replace=True).mean()
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def mark_pareto_frontier(summary: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Mark safety-feasible methods not dominated on danger and coverage."""
    result = summary.copy()
    feasible = result["danger_ucb"] <= alpha
    frontier = []
    for index, row in result.iterrows():
        if not feasible.loc[index]:
            frontier.append(False)
            continue
        dominated = (
            feasible
            & (result["danger_k"] <= row["danger_k"])
            & (result["safe_negative"] >= row["safe_negative"])
            & (
                (result["danger_k"] < row["danger_k"])
                | (result["safe_negative"] > row["safe_negative"])
            )
        ).any()
        frontier.append(not bool(dominated))
    result["safety_feasible"] = feasible.to_numpy(dtype=bool)
    result["pareto_frontier"] = frontier
    return result


def paired_change(
    baseline: pd.DataFrame, candidate: pd.DataFrame, confidence: float = 0.95
) -> dict:
    paired = baseline.loc[:, ["event_id", "y", "safe_exclude"]].merge(
        candidate.loc[:, ["event_id", "y", "safe_exclude"]],
        on=["event_id", "y"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    negative = paired["y"] == 0
    positive = ~negative
    gained = int((
        negative & ~paired["safe_exclude_baseline"] & paired["safe_exclude_candidate"]
    ).sum())
    lost = int((
        negative & paired["safe_exclude_baseline"] & ~paired["safe_exclude_candidate"]
    ).sum())
    danger_gained = int((
        positive & ~paired["safe_exclude_baseline"] & paired["safe_exclude_candidate"]
    ).sum())
    danger_lost = int((
        positive & paired["safe_exclude_baseline"] & ~paired["safe_exclude_candidate"]
    ).sum())
    negative_delta = (
        paired.loc[negative, "safe_exclude_candidate"].astype(int).to_numpy()
        - paired.loc[negative, "safe_exclude_baseline"].astype(int).to_numpy()
    )
    ci_low, ci_high = paired_bootstrap_ci(
        negative_delta, confidence=confidence
    )
    return {
        "coverage_gained_events": gained,
        "coverage_lost_events": lost,
        "coverage_net_events": gained - lost,
        "coverage_delta": float(negative_delta.mean()),
        "coverage_delta_ci_low": ci_low,
        "coverage_delta_ci_high": ci_high,
        "coverage_mcnemar_p": float(
            binomtest(min(gained, lost), gained + lost, p=0.5).pvalue
        ) if gained + lost else 1.0,
        "danger_gained_events": danger_gained,
        "danger_lost_events": danger_lost,
        "danger_net_events": danger_gained - danger_lost,
        "danger_mcnemar_p": float(
            binomtest(
                min(danger_gained, danger_lost), danger_gained + danger_lost, p=0.5
            ).pvalue
        ) if danger_gained + danger_lost else 1.0,
    }


def evaluate_ensembles(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    minimum_history: int = 3,
    alpha: float = 0.10,
    mode: str = "pac",
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = combine_scores(attach_candidate_scores(baseline_scores, candidate_scores))
    methods = ["catboost_snapshot", "catboost_tail_aligned", *SCORE_METHODS]
    decisions = {
        method: crossfit_decisions(
            scores, method, minimum_history, alpha, mode, confidence
        ) for method in methods
    }
    baseline = decisions["catboost_snapshot"]
    summary_rows = []
    fold_rows = []
    decision_frames = []
    for method, frame in decisions.items():
        metrics = decision_metrics(frame, confidence)
        changes = paired_change(baseline, frame, confidence)
        summary_rows.append({"method": method, **metrics, **changes})
        stored = frame.copy()
        stored.insert(0, "method", method)
        decision_frames.append(stored)
        for fold, fold_frame in frame.groupby("fold"):
            fold_metrics = decision_metrics(fold_frame, confidence)
            base_fold = baseline.loc[baseline["fold"] == fold]
            fold_rows.append({
                "method": method,
                "fold": int(fold),
                **fold_metrics,
                **paired_change(base_fold, fold_frame, confidence),
            })
    summary = mark_pareto_frontier(pd.DataFrame(summary_rows), alpha)
    return (
        scores,
        summary,
        pd.DataFrame(fold_rows),
        pd.concat(decision_frames, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--fold-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    scores, summary, folds, decisions = evaluate_ensembles(
        pd.read_parquet(args.baseline_scores),
        pd.read_parquet(args.candidate_scores),
        minimum_history=args.minimum_history,
        alpha=args.alpha,
        mode=args.mode,
        confidence=args.confidence,
    )
    for path in (args.scores_output, args.decisions_output, args.summary_output, args.fold_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.scores_output, index=False)
    decisions.to_parquet(args.decisions_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    folds.to_csv(args.fold_output, index=False)
    manifest = {
        "status": "development-only",
        "evaluation_accessed": False,
        "minimum_history": args.minimum_history,
        "alpha": args.alpha,
        "mode": args.mode,
        "confidence": args.confidence,
        "combination_semantics": {
            "lower_score": "safer",
            "maximum": "pointwise pessimistic before recalibration",
            "minimum": "pointwise optimistic before recalibration",
        },
        "summary": summary.to_dict(orient="records"),
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
