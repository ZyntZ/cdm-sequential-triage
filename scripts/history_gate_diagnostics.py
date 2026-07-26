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
from shift_gate import ConformalShiftGate


def _validate_event_folds(scores: pd.DataFrame) -> list[int]:
    event_fold_counts = scores.groupby("event_id")["fold"].nunique()
    if (event_fold_counts != 1).any():
        raise ValueError("Each event_id must occur in exactly one fold")
    folds = sorted(int(value) for value in scores["fold"].unique())
    if len(folds) < 5:
        raise ValueError("Gate-aware diagnostics require at least five event folds")
    return folds


def nested_gate_roles(folds: list[int], evaluation_fold: int) -> dict[str, list[int]]:
    """Assign disjoint folds to gate fit, gate calibration, policy calibration and evaluation."""
    position = folds.index(evaluation_fold)
    policy_calibration = folds[(position + 1) % len(folds)]
    gate_calibration = folds[(position + 2) % len(folds)]
    reserved = {evaluation_fold, policy_calibration, gate_calibration}
    gate_training = [fold for fold in folds if fold not in reserved]
    return {
        "gate_training": gate_training,
        "gate_calibration": [gate_calibration],
        "policy_calibration": [policy_calibration],
        "evaluation": [evaluation_fold],
    }


def attach_gate_features(
    scores: pd.DataFrame,
    feature_scores: pd.DataFrame,
    gate_features: list[str],
) -> pd.DataFrame:
    """Attach gate features by an exact one-to-one prefix-key join."""
    keys = ["event_id", "time_to_tca"]
    required = {*keys, *gate_features}
    missing = required.difference(feature_scores.columns)
    if missing:
        raise ValueError(f"Missing gate-feature columns: {sorted(missing)}")
    if scores.duplicated(keys).any() or feature_scores.duplicated(keys).any():
        raise ValueError("Gate-feature join keys must be unique in both inputs")
    overlap = set(gate_features).intersection(scores.columns)
    if overlap:
        scores = scores.drop(columns=sorted(overlap))
    selected = feature_scores.loc[:, keys + gate_features]
    merged = scores.merge(selected, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        missing_rows = int(merged["_merge"].ne("both").sum())
        raise ValueError(f"Gate features are missing for {missing_rows} score rows")
    return merged.drop(columns="_merge")


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


def crossfit_shift_gate(
    scores: pd.DataFrame,
    score_col: str,
    gate_features: list[str],
    minimum_histories: list[int],
    alpha: float,
    mode: str,
    confidence: float,
    gate_alpha: float,
) -> pd.DataFrame:
    """Evaluate a gate-aware policy with disjoint fold roles.

    For each held-out fold, two folds estimate gate location and scale, one
    calibrates the event-level gate threshold, and one calibrates the policy
    threshold. This keeps the complete policy fixed before policy calibration.
    """
    required = {"event_id", "time_to_tca", "y", "fold", score_col, *gate_features}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not gate_features or len(set(gate_features)) != len(gate_features):
        raise ValueError("gate_features must be non-empty and unique")
    if any(value < 1 for value in minimum_histories):
        raise ValueError("minimum histories must be at least one")

    ordered = scores.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()
    ordered["eligible_history_count"] = (
        ordered.groupby("event_id", sort=False).cumcount() + 1
    )
    folds = _validate_event_folds(ordered)
    rows = []
    for minimum_history in minimum_histories:
        held_out_decisions = []
        control_decisions = []
        policy_ranks = []
        policy_thresholds = []
        gate_thresholds = []
        role_records = []
        for evaluation_fold in folds:
            roles = nested_gate_roles(folds, evaluation_fold)
            gate_training = ordered.loc[ordered["fold"].isin(roles["gate_training"])]
            gate_calibration = ordered.loc[
                ordered["fold"].isin(roles["gate_calibration"])
            ]
            policy_calibration = ordered.loc[
                ordered["fold"].isin(roles["policy_calibration"])
            ]
            held_out = ordered.loc[ordered["fold"] == evaluation_fold]

            gate = ConformalShiftGate(gate_features).fit(gate_training)
            gate_rule = gate.calibrate_events(gate_calibration, alpha=gate_alpha)
            control_events = history_gated_event_table(
                policy_calibration, score_col, minimum_history
            )
            control_rule = calibrate_positive_threshold(
                control_events.loc[control_events["y"] == 1, "min_score"],
                alpha=alpha,
                mode=mode,
                confidence=confidence,
            )
            control = first_safe_decision_table(
                held_out,
                score_col,
                threshold=control_rule["threshold"],
                minimum_history=minimum_history,
            )
            control["fold"] = evaluation_fold
            control_decisions.append(control)

            gated_calibration = policy_calibration.copy()
            gate_allowed = gate.allows_safe_exclude(gated_calibration)
            gated_calibration.loc[~gate_allowed, score_col] = float("inf")
            calibration_events = history_gated_event_table(
                gated_calibration, score_col, minimum_history
            )
            policy_rule = calibrate_positive_threshold(
                calibration_events.loc[calibration_events["y"] == 1, "min_score"],
                alpha=alpha,
                mode=mode,
                confidence=confidence,
            )
            decisions = first_safe_decision_table(
                held_out,
                score_col,
                threshold=policy_rule["threshold"],
                minimum_history=minimum_history,
                shift_gate=gate,
            )
            decisions["fold"] = evaluation_fold
            held_out_decisions.append(decisions)
            policy_ranks.append(policy_rule["rank"])
            policy_thresholds.append(policy_rule["threshold"])
            gate_thresholds.append(gate_rule.threshold)
            role_records.append(
                f"{evaluation_fold}:train={','.join(map(str, roles['gate_training']))};"
                f"gate_cal={roles['gate_calibration'][0]};"
                f"policy_cal={roles['policy_calibration'][0]}"
            )

        combined = pd.concat(held_out_decisions, ignore_index=True)
        control_combined = pd.concat(control_decisions, ignore_index=True)
        if combined["event_id"].duplicated().any() or control_combined["event_id"].duplicated().any():
            raise RuntimeError("Cross-fitted evaluation produced duplicate events")
        control_combined = control_combined.set_index("event_id").loc[combined["event_id"]].reset_index()
        positive = combined["y"] == 1
        negative = ~positive
        danger = combined["safe_exclude"] & positive
        safe_negative = combined["safe_exclude"] & negative
        control_danger = control_combined["safe_exclude"] & positive
        control_safe_negative = control_combined["safe_exclude"] & negative
        blocked = combined["shift_gate_blocked"]
        first_safe_tca = combined.loc[safe_negative, "first_safe_tca"]
        danger_n = int(positive.sum())
        negative_n = int(negative.sum())
        rows.append({
            "minimum_history_in_window": minimum_history,
            "gate_features": ",".join(gate_features),
            "gate_alpha": gate_alpha,
            "danger_k": int(danger.sum()),
            "danger_n": danger_n,
            "danger_rate": float(danger.sum() / danger_n),
            "danger_ucb": cp_upper(int(danger.sum()), danger_n, confidence),
            "safe_negative": int(safe_negative.sum()),
            "negative_n": negative_n,
            "safe_negative_rate": float(safe_negative.sum() / negative_n),
            "control_danger_k": int(control_danger.sum()),
            "control_danger_ucb": cp_upper(int(control_danger.sum()), danger_n, confidence),
            "control_safe_negative": int(control_safe_negative.sum()),
            "control_safe_negative_rate": float(control_safe_negative.sum() / negative_n),
            "delta_danger_k_vs_control": int(danger.sum() - control_danger.sum()),
            "delta_safe_negative_vs_control": int(safe_negative.sum() - control_safe_negative.sum()),
            "median_first_safe_tca_days": float(first_safe_tca.median()),
            "gate_blocked_events": int(blocked.sum()),
            "gate_blocked_positive": int((blocked & positive).sum()),
            "gate_blocked_negative": int((blocked & negative).sum()),
            "policy_rank_min": min(policy_ranks),
            "policy_rank_max": max(policy_ranks),
            "policy_threshold_min": min(policy_thresholds),
            "policy_threshold_max": max(policy_thresholds),
            "gate_threshold_min": min(gate_thresholds),
            "gate_threshold_max": max(gate_thresholds),
            "fold_roles": " | ".join(role_records),
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
    parser.add_argument("--gate-features", nargs="*", default=[])
    parser.add_argument("--gate-feature-scores", type=Path)
    parser.add_argument("--gate-alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores = pd.read_parquet(args.scores)
    if args.gate_feature_scores is not None:
        if not args.gate_features:
            raise ValueError("--gate-feature-scores requires --gate-features")
        scores = attach_gate_features(
            scores, pd.read_parquet(args.gate_feature_scores), args.gate_features
        )
    if args.gate_features:
        result = crossfit_shift_gate(
            scores,
            score_col=args.score_col,
            gate_features=args.gate_features,
            minimum_histories=args.minimum_history,
            alpha=args.alpha,
            mode=args.mode,
            confidence=args.confidence,
            gate_alpha=args.gate_alpha,
        )
    else:
        result = crossfit_history_gate(
            scores,
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
