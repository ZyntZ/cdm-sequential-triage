"""Paired development robustness diagnostics for the event-aligned candidate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import calibrate_positive_threshold, first_safe_decision_table, history_gated_event_table
from robustness import subgroup_metrics


def attach_candidate_scores(
    baseline_scores: pd.DataFrame, candidate_scores: pd.DataFrame
) -> pd.DataFrame:
    keys = ["event_id", "time_to_tca"]
    baseline_required = {*keys, "y", "fold", "catboost_snapshot"}
    candidate_required = {*keys, "y", "fold", "eligible_history_count", "catboost_tail_aligned"}
    missing = baseline_required.difference(baseline_scores.columns)
    missing |= candidate_required.difference(candidate_scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if baseline_scores.duplicated(keys).any() or candidate_scores.duplicated(keys).any():
        raise ValueError("event_id/time_to_tca keys must be unique")
    merged = baseline_scores.loc[:, list(baseline_required)].merge(
        candidate_scores.loc[:, list(candidate_required)],
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Baseline and candidate prefix sets do not match")
    if not merged["y_baseline"].eq(merged["y_candidate"]).all():
        raise ValueError("Baseline and candidate labels do not match")
    if not merged["fold_baseline"].eq(merged["fold_candidate"]).all():
        raise ValueError("Baseline and candidate folds do not match")
    return merged.rename(columns={
        "y_baseline": "y",
        "fold_baseline": "fold",
    }).drop(columns=["y_candidate", "fold_candidate", "_merge"])


def crossfit_decisions(
    scores: pd.DataFrame,
    score_col: str,
    minimum_history: int = 3,
    alpha: float = 0.10,
    mode: str = "pac",
    confidence: float = 0.95,
) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", "y", "fold", "eligible_history_count", score_col}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    decisions = []
    for fold in sorted(scores["fold"].unique()):
        calibration = scores.loc[scores["fold"] != fold]
        held_out = scores.loc[scores["fold"] == fold]
        calibration_events = history_gated_event_table(
            calibration, score_col, minimum_history
        )
        rule = calibrate_positive_threshold(
            calibration_events.loc[calibration_events["y"] == 1, "min_score"],
            alpha=alpha,
            mode=mode,
            confidence=confidence,
        )
        result = first_safe_decision_table(
            held_out,
            score_col,
            threshold=rule["threshold"],
            minimum_history=minimum_history,
        )
        result["fold"] = fold
        decisions.append(result)
    combined = pd.concat(decisions, ignore_index=True)
    if combined["event_id"].duplicated().any():
        raise RuntimeError("Cross-fit produced duplicate event decisions")
    return combined


def event_groups(training: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", "mission_id"}
    missing = required.difference(training.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    if training.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Training prefix keys must be unique")
    ordered = training.sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False]
    ).copy()
    ordered["eligible_history_count"] = ordered.groupby("event_id", sort=False).cumcount() + 1
    feature_columns = [
        column for column in training.columns
        if column not in {"event_id", "time_to_tca", "y"}
    ]
    first = ordered.drop_duplicates("event_id", keep="first").copy()
    first["first_message_missing_fraction"] = first[feature_columns].isna().mean(axis=1)
    first["missingness_group"] = pd.cut(
        first["first_message_missing_fraction"],
        bins=[-np.inf, 0.0, 0.10, np.inf],
        labels=["none", "low", "high"],
        right=True,
    )
    histories = ordered.groupby("event_id", as_index=False).agg(
        messages_in_window=("eligible_history_count", "max")
    )
    histories["history_group"] = pd.cut(
        histories["messages_in_window"],
        bins=[0, 4, 10, np.inf],
        labels=["1-4", "5-10", "11+"],
        right=True,
    )
    groups = first.loc[:, [
        "event_id", "mission_id", "missingness_group", "first_message_missing_fraction"
    ]].merge(histories, on="event_id", validate="one_to_one")
    if groups.isna().any().any():
        raise ValueError("Subgroup assignment produced missing values")
    return groups


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


def paired_subgroup_table(
    paired: pd.DataFrame,
    group_col: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    baseline = subgroup_metrics(
        paired.rename(columns={"safe_exclude_baseline": "safe_exclude"}),
        group_col,
        confidence=confidence,
    ).add_prefix("baseline_").rename(columns={f"baseline_{group_col}": group_col})
    candidate = subgroup_metrics(
        paired.rename(columns={"safe_exclude_candidate": "safe_exclude"}),
        group_col,
        confidence=confidence,
    ).add_prefix("candidate_").rename(columns={f"candidate_{group_col}": group_col})
    summary = baseline.merge(candidate, on=group_col, validate="one_to_one")
    rows = []
    for group, frame in paired.groupby(group_col, observed=True, dropna=False):
        negative = frame["y"] == 0
        positive = ~negative
        gained = int((
            negative & ~frame["safe_exclude_baseline"] & frame["safe_exclude_candidate"]
        ).sum())
        lost = int((
            negative & frame["safe_exclude_baseline"] & ~frame["safe_exclude_candidate"]
        ).sum())
        danger_gained = int((
            positive & ~frame["safe_exclude_baseline"] & frame["safe_exclude_candidate"]
        ).sum())
        danger_lost = int((
            positive & frame["safe_exclude_baseline"] & ~frame["safe_exclude_candidate"]
        ).sum())
        neg_delta = (
            frame.loc[negative, "safe_exclude_candidate"].astype(int).to_numpy()
            - frame.loc[negative, "safe_exclude_baseline"].astype(int).to_numpy()
        )
        ci_low, ci_high = paired_bootstrap_ci(
            neg_delta, confidence=confidence, seed=24072026 + len(rows)
        )
        discordant = gained + lost
        danger_discordant = danger_gained + danger_lost
        both_safe = negative & frame["safe_exclude_baseline"] & frame["safe_exclude_candidate"]
        timing_delta = (
            frame.loc[both_safe, "first_safe_tca_candidate"]
            - frame.loc[both_safe, "first_safe_tca_baseline"]
        )
        rows.append({
            group_col: group,
            "coverage_gained_events": gained,
            "coverage_lost_events": lost,
            "coverage_net_events": gained - lost,
            "coverage_delta": float(neg_delta.mean()) if neg_delta.size else np.nan,
            "coverage_delta_ci_low": ci_low,
            "coverage_delta_ci_high": ci_high,
            "coverage_mcnemar_p": (
                float(binomtest(min(gained, lost), discordant, p=0.5).pvalue)
                if discordant else 1.0
            ),
            "danger_gained_events": danger_gained,
            "danger_lost_events": danger_lost,
            "danger_net_events": danger_gained - danger_lost,
            "danger_mcnemar_p": (
                float(binomtest(min(danger_gained, danger_lost), danger_discordant, p=0.5).pvalue)
                if danger_discordant else 1.0
            ),
            "both_safe_negative_events": int(both_safe.sum()),
            "median_timing_delta_days": (
                float(timing_delta.median()) if not timing_delta.empty else np.nan
            ),
        })
    paired_stats = pd.DataFrame(rows)
    result = summary.merge(paired_stats, on=group_col, validate="one_to_one")
    result.insert(0, "grouping", group_col)
    result = result.rename(columns={group_col: "group"})
    return result


def build_paired_decisions(
    training: pd.DataFrame,
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    minimum_history: int = 3,
    alpha: float = 0.10,
    mode: str = "pac",
    confidence: float = 0.95,
) -> pd.DataFrame:
    scores = attach_candidate_scores(baseline_scores, candidate_scores)
    baseline = crossfit_decisions(
        scores, "catboost_snapshot", minimum_history, alpha, mode, confidence
    ).rename(columns={
        "safe_exclude": "safe_exclude_baseline",
        "first_safe_tca": "first_safe_tca_baseline",
        "first_safe_score": "first_safe_score_baseline",
    })
    candidate = crossfit_decisions(
        scores, "catboost_tail_aligned", minimum_history, alpha, mode, confidence
    ).rename(columns={
        "safe_exclude": "safe_exclude_candidate",
        "first_safe_tca": "first_safe_tca_candidate",
        "first_safe_score": "first_safe_score_candidate",
    })
    keep = [
        "event_id", "y", "fold", "safe_exclude_candidate",
        "first_safe_tca_candidate", "first_safe_score_candidate",
    ]
    paired = baseline.loc[:, [
        "event_id", "y", "fold", "safe_exclude_baseline",
        "first_safe_tca_baseline", "first_safe_score_baseline",
    ]].merge(candidate.loc[:, keep], on=["event_id", "y", "fold"], validate="one_to_one")
    paired = paired.merge(event_groups(training), on="event_id", validate="one_to_one")
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mode", choices=("marginal", "pac"), default="pac")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()

    paired = build_paired_decisions(
        pd.read_parquet(args.training),
        pd.read_parquet(args.baseline_scores),
        pd.read_parquet(args.candidate_scores),
        minimum_history=args.minimum_history,
        alpha=args.alpha,
        mode=args.mode,
        confidence=args.confidence,
    )
    reports = []
    for group_col in ("mission_id", "history_group", "missingness_group", "fold"):
        reports.append(paired_subgroup_table(paired, group_col, args.confidence))
    report = pd.concat(reports, ignore_index=True)
    args.decisions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(args.decisions_output, index=False)
    report.to_csv(args.report_output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
