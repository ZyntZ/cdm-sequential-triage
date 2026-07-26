"""Cross-fitted diagnostics for minimum history inside the decision window."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import (
    calibrate_positive_threshold,
    cp_upper,
    first_safe_decision_table,
    history_gated_event_table,
)


def crossfit_history_gate(
    scores: pd.DataFrame,
    score_col: str,
    minimum_histories: list[int],
    alpha: float,
    mode: str,
    confidence: float,
) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", "y", "fold", score_col}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ordered = scores.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()
    ordered["eligible_history_count"] = (
        ordered.groupby("event_id", sort=False).cumcount() + 1
    )

    rows = []
    for minimum_history in minimum_histories:
        held_out_decisions = []
        ranks = []
        thresholds = []
        for fold in sorted(ordered["fold"].unique()):
            calibration_prefixes = ordered.loc[ordered["fold"] != fold]
            held_out_prefixes = ordered.loc[ordered["fold"] == fold]
            calibration_events = history_gated_event_table(
                calibration_prefixes, score_col, minimum_history
            )
            rule = calibrate_positive_threshold(
                calibration_events.loc[calibration_events["y"] == 1, "min_score"],
                alpha=alpha,
                mode=mode,
                confidence=confidence,
            )
            decisions = first_safe_decision_table(
                held_out_prefixes,
                score_col,
                threshold=rule["threshold"],
                minimum_history=minimum_history,
            )
            decisions["fold"] = fold
            held_out_decisions.append(decisions)
            ranks.append(rule["rank"])
            thresholds.append(rule["threshold"])

        combined = pd.concat(held_out_decisions, ignore_index=True)
        positive = combined["y"] == 1
        negative = ~positive
        danger_k = int((combined["safe_exclude"] & positive).sum())
        safe_negative = combined["safe_exclude"] & negative
        first_safe_tca = combined.loc[safe_negative, "first_safe_tca"]
        rows.append({
            "minimum_history_in_window": minimum_history,
            "danger_k": danger_k,
            "danger_n": int(positive.sum()),
            "danger_rate": danger_k / int(positive.sum()),
            "danger_ucb": cp_upper(danger_k, int(positive.sum()), confidence),
            "safe_negative": int(safe_negative.sum()),
            "negative_n": int(negative.sum()),
            "safe_negative_rate": safe_negative.sum() / int(negative.sum()),
            "median_first_safe_tca_days": first_safe_tca.median(),
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "threshold_min": min(thresholds),
            "threshold_max": max(thresholds),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "artifacts" / "development_oof_model_comparison_v2.parquet",
    )
    parser.add_argument("--score-col", default="catboost_snapshot")
    parser.add_argument("--minimum-history", type=int, nargs="+", default=[1, 3, 4, 5])
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = crossfit_history_gate(
        pd.read_parquet(args.scores),
        score_col=args.score_col,
        minimum_histories=args.minimum_history,
        alpha=args.alpha,
        mode=args.mode,
        confidence=args.confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
