"""Run event-level calibration diagnostics on saved development OOF scores."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import calibrate_positive_threshold, cp_upper, event_policy_table


METHODS = {
    "current_risk": "risk",
    "logistic_snapshot": "logistic_snapshot",
    "catboost_snapshot": "catboost_snapshot",
    "catboost_dynamic": "catboost_dynamic",
}


def crossfit(scores: pd.DataFrame, alpha: float, mode: str, confidence: float) -> pd.DataFrame:
    rows = []
    for method, score_col in METHODS.items():
        decisions = []
        for fold in sorted(scores["fold"].unique()):
            calibration = event_policy_table(scores.loc[scores["fold"] != fold], score_col)
            held_out = event_policy_table(scores.loc[scores["fold"] == fold], score_col)
            rule = calibrate_positive_threshold(
                calibration.loc[calibration["y"] == 1, "min_score"],
                alpha=alpha,
                mode=mode,
                confidence=confidence,
            )
            held_out = held_out.assign(
                safe_exclude=held_out["min_score"] <= rule["threshold"],
                fold=fold,
            )
            decisions.append(held_out)

        combined = pd.concat(decisions, ignore_index=True)
        positive = combined["y"] == 1
        negative = ~positive
        danger_k = int((combined["safe_exclude"] & positive).sum())
        danger_n = int(positive.sum())
        safe_negative = int((combined["safe_exclude"] & negative).sum())
        negative_n = int(negative.sum())
        rows.append({
            "method": method,
            "mode": mode,
            "alpha": alpha,
            "danger_k": danger_k,
            "danger_n": danger_n,
            "danger_rate": danger_k / danger_n,
            "danger_ucb": cp_upper(danger_k, danger_n, confidence),
            "safe_negative": safe_negative,
            "negative_n": negative_n,
            "safe_negative_rate": safe_negative / negative_n,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "artifacts" / "development_oof_model_comparison_v2.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    scores = pd.read_parquet(args.scores)
    result = crossfit(scores, args.alpha, args.mode, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
