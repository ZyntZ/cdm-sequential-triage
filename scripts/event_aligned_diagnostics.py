"""OOF development experiment for positive-tail reweighting."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_aligned_model import (
    event_equal_weights,
    fit_dynamic_model,
    positive_tail_weights,
    prepare_dynamic_frame,
    score_dynamic_model,
)
from policy import calibrate_positive_threshold, cp_upper, first_safe_decision_table, history_gated_event_table


def attach_event_folds(
    prepared: pd.DataFrame, fold_table: pd.DataFrame
) -> pd.DataFrame:
    """Attach pre-existing event folds by an exact one-to-one prefix join."""
    keys = ["event_id", "time_to_tca"]
    required = {*keys, "fold"}
    missing = required.difference(fold_table.columns)
    if missing:
        raise ValueError(f"Missing fold columns: {sorted(missing)}")
    if prepared.duplicated(keys).any() or fold_table.duplicated(keys).any():
        raise ValueError("event_id/time_to_tca keys must be unique")
    attached = prepared.merge(
        fold_table.loc[:, keys + ["fold"]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not attached["_merge"].eq("both").all():
        raise ValueError("Fold table does not cover every prepared prefix")
    if (attached.groupby("event_id")["fold"].nunique() != 1).any():
        raise ValueError("Each event_id must occur in exactly one fold")
    return attached.drop(columns="_merge")


def inner_oof_scores(
    outer_training: pd.DataFrame,
    model_params: dict | None = None,
) -> np.ndarray:
    """Produce base scores without using the outer evaluation fold.

    Each outer-training event is scored by a model trained on the other inner
    folds. These scores may therefore define stage-two weights without leaking
    outer-evaluation outcomes into model fitting.
    """
    scores = pd.Series(np.nan, index=outer_training.index, dtype=float)
    inner_folds = sorted(outer_training["fold"].unique())
    if len(inner_folds) < 2:
        raise ValueError("At least two inner folds are required")
    for inner_fold in inner_folds:
        inner_training = outer_training.loc[outer_training["fold"] != inner_fold]
        inner_held_out = outer_training.loc[outer_training["fold"] == inner_fold]
        model = fit_dynamic_model(
            inner_training,
            event_equal_weights(inner_training),
            model_params=model_params,
        )
        scores.loc[inner_held_out.index] = score_dynamic_model(model, inner_held_out)
    if scores.isna().any():
        raise RuntimeError("Inner OOF scoring left unscored prefixes")
    return scores.loc[outer_training.index].to_numpy(dtype=float)


def crossfit_event_aligned(
    training: pd.DataFrame,
    fold_table: pd.DataFrame,
    hard_fraction: float = 0.25,
    hard_mass: float = 0.50,
    minimum_history: int = 3,
    alpha: float = 0.10,
    confidence: float = 0.95,
    mode: str = "pac",
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    prepared = attach_event_folds(prepare_dynamic_frame(training), fold_table)
    predictions = []
    for fold in sorted(prepared["fold"].unique()):
        train = prepared.loc[prepared["fold"] != fold].copy()
        held_out = prepared.loc[prepared["fold"] == fold].copy()
        base_scores = inner_oof_scores(train, model_params=model_params)
        weights = positive_tail_weights(
            train, base_scores, hard_fraction=hard_fraction, hard_mass=hard_mass
        )
        model = fit_dynamic_model(train, weights, model_params=model_params)
        held_out["catboost_tail_aligned"] = score_dynamic_model(model, held_out)
        predictions.append(held_out.loc[:, [
            "event_id", "time_to_tca", "y", "fold",
            "eligible_history_count", "catboost_tail_aligned",
        ]])
    scored = pd.concat(predictions, ignore_index=True)
    if scored.duplicated(["event_id", "time_to_tca"]).any():
        raise RuntimeError("OOF scoring produced duplicate prefixes")

    decisions = []
    thresholds = []
    ranks = []
    for fold in sorted(scored["fold"].unique()):
        calibration = scored.loc[scored["fold"] != fold]
        held_out = scored.loc[scored["fold"] == fold]
        calibration_events = history_gated_event_table(
            calibration, "catboost_tail_aligned", minimum_history
        )
        rule = calibrate_positive_threshold(
            calibration_events.loc[calibration_events["y"] == 1, "min_score"],
            alpha=alpha,
            mode=mode,
            confidence=confidence,
        )
        fold_decisions = first_safe_decision_table(
            held_out,
            "catboost_tail_aligned",
            threshold=rule["threshold"],
            minimum_history=minimum_history,
        )
        fold_decisions["fold"] = fold
        decisions.append(fold_decisions)
        thresholds.append(rule["threshold"])
        ranks.append(rule["rank"])
    events = pd.concat(decisions, ignore_index=True)
    positive = events["y"] == 1
    negative = ~positive
    dangerous = events["safe_exclude"] & positive
    safe_negative = events["safe_exclude"] & negative
    first_safe = events.loc[safe_negative, "first_safe_tca"]
    danger_n = int(positive.sum())
    negative_n = int(negative.sum())
    metrics = {
        "method": "catboost_tail_aligned",
        "base_score_for_weights": "nested_inner_oof_dynamic",
        "hard_fraction": hard_fraction,
        "hard_mass": hard_mass,
        "minimum_history": minimum_history,
        "alpha": alpha,
        "mode": mode,
        "confidence": confidence,
        "danger_k": int(dangerous.sum()),
        "danger_n": danger_n,
        "danger_rate": float(dangerous.sum() / danger_n),
        "danger_ucb": cp_upper(int(dangerous.sum()), danger_n, confidence),
        "safe_negative": int(safe_negative.sum()),
        "negative_n": negative_n,
        "safe_negative_rate": float(safe_negative.sum() / negative_n),
        "median_first_safe_tca_days": float(first_safe.median()),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "prefix_rows": int(len(scored)),
        "events": int(events.shape[0]),
    }
    return scored, metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--oof-scores", type=Path, required=True)
    parser.add_argument("--hard-fraction", type=float, default=0.25)
    parser.add_argument("--hard-mass", type=float, default=0.50)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    scored, metrics = crossfit_event_aligned(
        pd.read_parquet(args.training),
        pd.read_parquet(args.oof_scores),
        hard_fraction=args.hard_fraction,
        hard_mass=args.hard_mass,
        minimum_history=args.minimum_history,
        alpha=args.alpha,
        confidence=args.confidence,
        mode=args.mode,
        model_params={"iterations": args.iterations},
    )
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.scores_output, index=False)
    pd.DataFrame([metrics]).to_csv(args.report_output, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development-only",
        "evaluation_accessed": False,
        "method": "two-stage positive-tail reweighting with nested inner-OOF base scores",
        "input_training": str(args.training),
        "input_oof_scores": str(args.oof_scores),
        "model_iterations": args.iterations,
        "metrics": metrics,
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    write_json(args.manifest_output, manifest)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
