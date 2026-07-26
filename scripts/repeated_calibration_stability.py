"""Repeated calibration/test stability on fixed development OOF scores.

These repeats vary only the event-level calibration split. They are correlated
stability diagnostics, not independent model replications.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import calibrate_positive_threshold, cp_upper, first_safe_decision_table, history_gated_event_table

METHODS = ("catboost_snapshot", "catboost_tail_aligned", "minimum")


def validate_oof_scores(scores: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    required = {
        "event_id", "time_to_tca", "y", "eligible_history_count", *methods
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")
    if scores.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("event_id/time_to_tca rows must be unique")
    if not scores["y"].isin([0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    if (scores.groupby("event_id")["y"].nunique() != 1).any():
        raise ValueError("Each event_id must have one event-level label")
    for method in methods:
        values = pd.to_numeric(scores[method], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{method} must contain finite scores")
    return scores.copy()


def repeated_calibration_stability(
    scores: pd.DataFrame,
    methods: tuple[str, ...] = METHODS,
    repeats: int = 250,
    test_fraction: float = 0.5,
    seed_base: int = 24072026,
    minimum_history: int = 3,
    alpha: float = 0.10,
    mode: str = "pac",
    confidence: float = 0.95,
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0, 1)")
    scores = validate_oof_scores(scores, methods)
    labels = scores.groupby("event_id", as_index=False).agg(y=("y", "first"))
    rows = []
    for repeat in range(repeats):
        seed = seed_base + repeat
        calibration_ids, test_ids = train_test_split(
            labels["event_id"].to_numpy(),
            test_size=test_fraction,
            random_state=seed,
            stratify=labels["y"].to_numpy(),
        )
        calibration = scores.loc[scores["event_id"].isin(calibration_ids)]
        test = scores.loc[scores["event_id"].isin(test_ids)]
        repeat_records = {}
        for method in methods:
            calibration_events = history_gated_event_table(
                calibration, method, minimum_history
            )
            rule = calibrate_positive_threshold(
                calibration_events.loc[calibration_events["y"] == 1, "min_score"],
                alpha=alpha,
                mode=mode,
                confidence=confidence,
            )
            decisions = first_safe_decision_table(
                test,
                method,
                threshold=rule["threshold"],
                minimum_history=minimum_history,
            )
            positive = decisions["y"] == 1
            negative = ~positive
            dangerous = decisions["safe_exclude"] & positive
            safe_negative = decisions["safe_exclude"] & negative
            first_safe = decisions.loc[safe_negative, "first_safe_tca"]
            record = {
                "method": method,
                "repeat": repeat,
                "seed": seed,
                "mode": mode,
                "alpha": alpha,
                "confidence": confidence,
                "minimum_history": minimum_history,
                "calibration_events": int(calibration_events.shape[0]),
                "calibration_positives": int((calibration_events["y"] == 1).sum()),
                "calibration_rank": rule["rank"],
                "calibration_pac_bound": rule["pac_bound"],
                "threshold": rule["threshold"],
                "danger_k": int(dangerous.sum()),
                "danger_n": int(positive.sum()),
                "danger_rate": float(dangerous.sum() / positive.sum()),
                "danger_ucb": cp_upper(int(dangerous.sum()), int(positive.sum()), confidence),
                "safe_negative": int(safe_negative.sum()),
                "negative_n": int(negative.sum()),
                "safe_negative_rate": float(safe_negative.sum() / negative.sum()),
                "median_first_safe_tca_days": (
                    None if first_safe.empty else float(first_safe.median())
                ),
            }
            repeat_records[method] = (record, decisions)
        baseline_decisions = repeat_records["catboost_snapshot"][1].set_index("event_id")
        for method in methods:
            record, decisions = repeat_records[method]
            aligned = decisions.set_index("event_id").loc[baseline_decisions.index]
            negative = aligned["y"] == 0
            positive = ~negative
            coverage_delta = (
                aligned.loc[negative, "safe_exclude"].astype(int)
                - baseline_decisions.loc[negative, "safe_exclude"].astype(int)
            )
            danger_delta = (
                aligned.loc[positive, "safe_exclude"].astype(int)
                - baseline_decisions.loc[positive, "safe_exclude"].astype(int)
            )
            record["coverage_delta_vs_snapshot"] = float(coverage_delta.mean())
            record["danger_delta_vs_snapshot"] = int(danger_delta.sum())
            rows.append(record)
    return pd.DataFrame(rows)


def summarize_stability(detail: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "repeat", "danger_rate", "danger_ucb", "safe_negative_rate", "coverage_delta_vs_snapshot", "danger_delta_vs_snapshot", "calibration_positives"}
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"Missing detail columns: {sorted(missing)}")
    rows = []
    for method, frame in detail.groupby("method", sort=False):
        rows.append({
            "method": method,
            "repeats": int(frame["repeat"].nunique()),
            "calibration_positives": float(frame["calibration_positives"].mean()),
            "mean_danger_rate": float(frame["danger_rate"].mean()),
            "median_danger_rate": float(frame["danger_rate"].median()),
            "danger_rate_q05": float(frame["danger_rate"].quantile(0.05)),
            "danger_rate_q95": float(frame["danger_rate"].quantile(0.95)),
            "median_danger_ucb": float(frame["danger_ucb"].median()),
            "ucb_le_alpha_fraction": float((frame["danger_ucb"] <= frame["alpha"]).mean()),
            "mean_safe_negative_rate": float(frame["safe_negative_rate"].mean()),
            "median_safe_negative_rate": float(frame["safe_negative_rate"].median()),
            "safe_negative_rate_q05": float(frame["safe_negative_rate"].quantile(0.05)),
            "safe_negative_rate_q95": float(frame["safe_negative_rate"].quantile(0.95)),
            "mean_coverage_delta_vs_snapshot": float(frame["coverage_delta_vs_snapshot"].mean()),
            "median_coverage_delta_vs_snapshot": float(frame["coverage_delta_vs_snapshot"].median()),
            "coverage_delta_q05": float(frame["coverage_delta_vs_snapshot"].quantile(0.05)),
            "coverage_delta_q95": float(frame["coverage_delta_vs_snapshot"].quantile(0.95)),
            "coverage_delta_positive_fraction": float((frame["coverage_delta_vs_snapshot"] > 0).mean()),
            "mean_danger_delta_vs_snapshot": float(frame["danger_delta_vs_snapshot"].mean()),
            "danger_delta_q05": float(frame["danger_delta_vs_snapshot"].quantile(0.05)),
            "danger_delta_q95": float(frame["danger_delta_vs_snapshot"].quantile(0.95)),
            "danger_not_worse_fraction": float((frame["danger_delta_vs_snapshot"] <= 0).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=250)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed-base", type=int, default=24072026)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    detail = repeated_calibration_stability(
        pd.read_parquet(args.scores),
        repeats=args.repeats,
        test_fraction=args.test_fraction,
        seed_base=args.seed_base,
        minimum_history=args.minimum_history,
        alpha=args.alpha,
        mode=args.mode,
        confidence=args.confidence,
    )
    summary = summarize_stability(detail)
    for path in (args.detail_output, args.summary_output, args.manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    manifest = {
        "status": "development-only calibration stability diagnostic",
        "evaluation_accessed": False,
        "independent_model_replications": False,
        "shared_fixed_oof_scores": True,
        "methods": list(METHODS),
        "repeats": args.repeats,
        "test_fraction": args.test_fraction,
        "seed_base": args.seed_base,
        "minimum_history": args.minimum_history,
        "alpha": args.alpha,
        "mode": args.mode,
        "confidence": args.confidence,
        "summary": summary.to_dict(orient="records"),
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
