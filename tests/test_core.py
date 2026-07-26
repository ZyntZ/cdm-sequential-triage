import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from history_gate_diagnostics import (
    attach_gate_features, crossfit_shift_gate, nested_gate_roles,
)
from event_aligned_diagnostics import attach_event_folds
from event_aligned_model import (
    fit_dynamic_model, positive_tail_weights, prepare_dynamic_frame, score_dynamic_frame,
)
from event_aligned_robustness import (
    attach_candidate_scores, event_groups, paired_subgroup_table,
)
from score_ensemble import combine_scores
from score_ensemble_diagnostics import evaluate_ensembles
from repeated_calibration_stability import (
    repeated_calibration_stability, summarize_stability, validate_oof_scores,
)
from freeze_next_validation import freeze_plan
from train_final_tail_aligned import read_locked_candidate
from validation_plan import (
    evaluation_planning_table, maximum_passing_failures,
    minimum_positive_events, pass_probability,
)
from prefix_features import build_prefix_features, eligible_prefixes
from robustness import subgroup_metrics
from shift_gate import ConformalShiftGate
from triage import Decision, SequentialTriagePolicy
from confirmation import (
    POLICY, acquire_confirmation_lock, attach_event_labels, calibrate, evaluate,
    policy_from_model_manifest, prepare_prefix_scores,
)
from snapshot_model import (
    assert_disjoint_splits, event_equal_weights, fit_snapshot_model,
    prepare_snapshot_frame, score_snapshot_model,
)
from partitions import event_labels, split_event_ids
from study import (
    freeze_study, read_locked_study, validate_feature_cohort, validate_label_roster,
)
from policy import (
    calibration_rank, calibrate_positive_threshold, cp_upper,
    event_policy_table, evaluate_sequential_policy, evaluate_threshold,
    first_safe_decision_table, history_gated_event_table,
)


def sample():
    return pd.DataFrame({
        "event_id":[1,1,1,2,2], "time_to_tca":[5.,4.,3.,5.,3.],
        "risk":[-8.,-7.,-5.,-9.,-10.],
        "max_risk_estimate":[-7.,-6.,-4.,-8.,-9.],
        "miss_distance":[100.,90.,80.,200.,180.],
        "mahalanobis_distance":[5.,4.,3.,7.,6.],
        "y":[1,1,1,0,0]
    })


def test_prefix_features_do_not_use_future_rows():
    full=build_prefix_features(sample())
    truncated=build_prefix_features(sample().query("not (event_id == 1 and time_to_tca < 4)"))
    cols=[c for c in full.columns if c not in ["y"]]
    a=full.query("event_id == 1 and time_to_tca >= 4")[cols].reset_index(drop=True)
    b=truncated.query("event_id == 1")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)


def test_order_and_counts():
    x=build_prefix_features(sample())
    assert x.query("event_id == 1").time_to_tca.tolist()==[5.,4.,3.]
    assert x.query("event_id == 1").n_cdm_so_far.tolist()==[1,2,3]


def test_policy_is_event_level_any_time():
    x=sample().rename(columns={"risk":"score"})
    e=event_policy_table(x,"score")
    result=evaluate_threshold(e,-8.5)
    assert result["danger_k"]==0
    assert result["safe_negative_rate"]==1.0


def test_cp_known_zero_failure_bound():
    assert abs(cp_upper(0,73)-0.0402067943)<1e-8



def test_marginal_calibration_rank():
    assert calibration_rank(73, 0.05, mode="marginal") == 3
    assert calibration_rank(73, 0.10, mode="marginal") == 7


def test_pac_calibration_rank():
    assert calibration_rank(73, 0.05, mode="pac", confidence=0.95) == 1
    assert calibration_rank(73, 0.10, mode="pac", confidence=0.95) == 3


def test_calibrated_threshold_is_strict_with_ties():
    scores = np.array([0.1, 0.1, 0.2, 0.3, 0.4])
    result = calibrate_positive_threshold(scores, alpha=0.4, mode="marginal")
    assert result["rank"] == 2
    assert result["threshold"] < 0.1
    assert np.sum(scores <= result["threshold"]) == 0


def test_event_labels_must_be_constant():
    x = sample().rename(columns={"risk": "score"})
    x.loc[x.index[1], "y"] = 0
    with np.testing.assert_raises(ValueError):
        event_policy_table(x, "score")



def test_subgroup_metrics_keep_unsupported_bounds_missing():
    events = pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "y": [1, 0, 0, 0],
        "group": ["a", "a", "b", "b"],
        "safe_exclude": [False, True, True, False],
    })
    result = subgroup_metrics(events, "group").set_index("group")
    assert result.loc["a", "danger_rate"] == 0.0
    assert result.loc["a", "safe_negative_rate"] == 1.0
    assert np.isnan(result.loc["b", "danger_rate"])
    assert np.isnan(result.loc["b", "danger_ucb"])


def test_subgroup_metrics_require_event_level_rows():
    events = pd.DataFrame({
        "event_id": [1, 1],
        "y": [1, 1],
        "group": ["a", "a"],
        "safe_exclude": [False, True],
    })
    with np.testing.assert_raises(ValueError):
        subgroup_metrics(events, "group")



def test_calibration_retains_infinite_scores():
    scores = np.array([0.1, 0.2, np.inf])
    result = calibrate_positive_threshold(scores, alpha=0.5, mode="marginal")
    assert result["n_positive"] == 3
    assert result["rank"] == 2


def test_calibration_rejects_nan_scores():
    scores = np.array([0.1, np.nan, 0.2])
    with np.testing.assert_raises(ValueError):
        calibrate_positive_threshold(scores, alpha=0.5, mode="marginal")



def test_history_gate_keeps_ineligible_events():
    prefixes = pd.DataFrame({
        "event_id": [1, 1, 2],
        "y": [1, 1, 0],
        "time_to_tca": [5.0, 4.0, 5.0],
        "score": [0.4, 0.2, 0.1],
        "n_cdm_so_far": [1, 2, 1],
    })
    events = history_gated_event_table(
        prefixes, "score", minimum_history=2, history_col="n_cdm_so_far"
    )
    events = events.set_index("event_id")
    assert events.loc[1, "min_score"] == 0.2
    assert np.isinf(events.loc[2, "min_score"])
    assert len(events) == 2


def test_history_gate_rejects_invalid_minimum():
    prefixes = pd.DataFrame({
        "event_id": [1], "y": [1], "time_to_tca": [5.0],
        "score": [0.2], "n_cdm_so_far": [1],
    })
    with np.testing.assert_raises(ValueError):
        history_gated_event_table(
            prefixes, "score", minimum_history=0, history_col="n_cdm_so_far"
        )



def test_eligible_history_count_starts_when_decision_window_opens():
    raw = pd.DataFrame({
        "event_id": [1, 1, 1, 1],
        "time_to_tca": [9.0, 8.0, 7.0, 6.0],
        "risk": [-9.0, -9.0, -9.0, -9.0],
        "max_risk_estimate": [-8.0, -8.0, -8.0, -8.0],
        "miss_distance": [100.0, 100.0, 100.0, 100.0],
        "mahalanobis_distance": [5.0, 5.0, 5.0, 5.0],
        "y": [0, 0, 0, 0],
    })
    selected = eligible_prefixes(build_prefix_features(raw))
    assert selected["n_cdm_so_far"].tolist() == [3, 4]
    assert selected["eligible_history_count"].tolist() == [1, 2]


def test_first_safe_decision_uses_window_history_and_keeps_all_events():
    prefixes = pd.DataFrame({
        "event_id": [1, 1, 1, 2],
        "y": [0, 0, 0, 1],
        "time_to_tca": [7.0, 6.0, 5.0, 7.0],
        "score": [0.10, 0.30, 0.15, 0.90],
        "eligible_history_count": [1, 2, 3, 1],
    })
    decisions = first_safe_decision_table(
        prefixes, "score", threshold=0.20, minimum_history=2
    ).set_index("event_id")
    assert decisions.loc[1, "safe_exclude"]
    assert decisions.loc[1, "first_safe_tca"] == 5.0
    assert not decisions.loc[2, "safe_exclude"]
    assert np.isnan(decisions.loc[2, "first_safe_tca"])


def test_sequential_evaluation_reports_event_level_timing():
    prefixes = pd.DataFrame({
        "event_id": [1, 1, 2, 3],
        "y": [0, 0, 0, 1],
        "time_to_tca": [7.0, 6.0, 5.0, 4.0],
        "score": [0.30, 0.10, 0.15, 0.90],
        "eligible_history_count": [1, 2, 1, 1],
    })
    result = evaluate_sequential_policy(
        prefixes, "score", threshold=0.20, minimum_history=1
    )
    assert result["danger_k"] == 0
    assert result["safe_negative"] == 2
    assert result["safe_negative_rate"] == 1.0
    assert result["median_first_safe_tca"] == 5.5


def test_offline_shift_gate_matches_runtime_safe_decisions():
    proper = pd.DataFrame({"gate_feature": [0.0, 0.5, 1.0, 1.5, 2.0]})
    calibration = pd.DataFrame({
        "gate_feature": np.linspace(0.0, 2.0, 39),
    })
    gate = ConformalShiftGate(["gate_feature"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    prefixes = pd.DataFrame({
        "event_id": [1, 1, 2, 2, 3],
        "y": [0, 0, 0, 0, 1],
        "time_to_tca": [6.0, 5.0, 6.0, 5.0, 6.0],
        "score": [0.10, 0.10, 0.10, 0.10, 0.90],
        "eligible_history_count": [1, 2, 1, 2, 1],
        "gate_feature": [20.0, 1.0, 1.0, 1.0, 1.0],
    })

    decisions = first_safe_decision_table(
        prefixes, "score", threshold=0.20, minimum_history=1, shift_gate=gate
    ).set_index("event_id")

    assert decisions.loc[1, "safe_exclude"]
    assert decisions.loc[1, "first_safe_tca"] == 5.0
    assert decisions.loc[1, "shift_gate_blocked"]
    assert decisions.loc[1, "first_blocked_safe_tca"] == 6.0
    assert decisions.loc[2, "safe_exclude"]
    assert not decisions.loc[2, "shift_gate_blocked"]
    assert not decisions.loc[3, "safe_exclude"]


def test_offline_shift_gate_blocks_all_threshold_crossings_with_missing_features():
    proper = pd.DataFrame({"gate_feature": [0.0, 0.5, 1.0, 1.5, 2.0]})
    calibration = pd.DataFrame({
        "gate_feature": np.linspace(0.0, 2.0, 39),
    })
    gate = ConformalShiftGate(["gate_feature"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    prefixes = pd.DataFrame({
        "event_id": [1, 2],
        "y": [0, 1],
        "time_to_tca": [5.0, 5.0],
        "score": [0.10, 0.90],
        "eligible_history_count": [1, 1],
        "gate_feature": [np.nan, 1.0],
    })

    result = evaluate_sequential_policy(
        prefixes, "score", threshold=0.20, shift_gate=gate
    )

    assert result["safe_negative"] == 0
    assert result["shift_gate_blocked_events"] == 1
    assert result["shift_gate_blocked_negative"] == 1
    assert result["shift_gate_blocked_positive"] == 0




def test_gate_feature_join_requires_exact_prefix_keys():
    scores = pd.DataFrame({
        "event_id": [1, 1, 2],
        "time_to_tca": [6.0, 5.0, 6.0],
        "score": [0.1, 0.2, 0.3],
    })
    features = pd.DataFrame({
        "event_id": [1, 1, 2],
        "time_to_tca": [6.0, 5.0, 6.0],
        "gate_feature": [10.0, 11.0, 12.0],
    })
    merged = attach_gate_features(scores, features, ["gate_feature"])
    assert merged["gate_feature"].tolist() == [10.0, 11.0, 12.0]

    with np.testing.assert_raises(ValueError):
        attach_gate_features(scores, features.iloc[:-1], ["gate_feature"])


def test_positive_tail_weights_preserve_event_mass_and_focus_hard_positives():
    frame = pd.DataFrame({
        "event_id": [1, 1, 1, 1, 2, 2],
        "y": [1, 1, 1, 1, 0, 0],
    })
    scores = np.array([0.10, 0.20, 0.80, 0.90, 0.10, 0.90])
    weights = positive_tail_weights(
        frame, scores, hard_fraction=0.25, hard_mass=0.50
    )
    totals = pd.Series(weights).groupby(frame["event_id"]).sum()

    np.testing.assert_allclose(totals.to_numpy(), 1.0)
    assert weights[0] > weights[1]
    np.testing.assert_allclose(weights[4:], [0.5, 0.5])


def test_positive_tail_weights_reject_incomplete_or_in_sample_shape():
    frame = pd.DataFrame({"event_id": [1, 1], "y": [1, 1]})
    with np.testing.assert_raises(ValueError):
        positive_tail_weights(frame, [0.1], hard_fraction=0.25, hard_mass=0.50)
    with np.testing.assert_raises(ValueError):
        positive_tail_weights(frame, [0.1, np.nan], hard_fraction=0.25, hard_mass=0.50)


def test_event_aligned_fold_join_is_exact():
    prepared = pd.DataFrame({
        "event_id": [1, 1, 2],
        "time_to_tca": [6.0, 5.0, 6.0],
        "y": [1, 1, 0],
    })
    oof = pd.DataFrame({
        "event_id": [1, 1, 2],
        "time_to_tca": [6.0, 5.0, 6.0],
        "fold": [0, 0, 1],
        "base": [0.2, 0.3, 0.1],
    })
    attached = attach_event_folds(prepared, oof)
    assert attached["fold"].tolist() == [0, 0, 1]
    with np.testing.assert_raises(ValueError):
        attach_event_folds(prepared, oof.iloc[:-1])


def test_event_aligned_score_join_rejects_fold_mismatch():
    baseline = pd.DataFrame({
        "event_id": [1, 2], "time_to_tca": [6.0, 6.0],
        "y": [1, 0], "fold": [0, 1], "catboost_snapshot": [0.8, 0.1],
    })
    candidate = pd.DataFrame({
        "event_id": [1, 2], "time_to_tca": [6.0, 6.0],
        "y": [1, 0], "fold": [0, 2], "eligible_history_count": [1, 1],
        "catboost_tail_aligned": [0.9, 0.2],
    })
    with np.testing.assert_raises(ValueError):
        attach_candidate_scores(baseline, candidate)


def test_event_groups_use_window_history_and_first_message_missingness():
    frame = pd.DataFrame({
        "event_id": [1, 1, 2], "time_to_tca": [6.0, 5.0, 6.0],
        "mission_id": [3, 3, 4], "feature": [np.nan, 1.0, 2.0],
    })
    groups = event_groups(frame).set_index("event_id")
    assert groups.loc[1, "messages_in_window"] == 2
    assert groups.loc[1, "history_group"] == "1-4"
    assert groups.loc[1, "missingness_group"] == "high"
    assert groups.loc[2, "missingness_group"] == "none"


def test_paired_subgroup_table_reports_directional_changes():
    paired = pd.DataFrame({
        "event_id": [1, 2, 3, 4], "y": [0, 0, 1, 1], "group": ["a"] * 4,
        "safe_exclude_baseline": [False, True, False, True],
        "safe_exclude_candidate": [True, True, True, False],
        "first_safe_tca_baseline": [np.nan, 4.0, np.nan, 4.0],
        "first_safe_tca_candidate": [5.0, 5.0, 3.0, np.nan],
    })
    result = paired_subgroup_table(paired, "group", confidence=0.95).iloc[0]
    assert result["coverage_gained_events"] == 1
    assert result["coverage_lost_events"] == 0
    assert result["danger_gained_events"] == 1
    assert result["danger_lost_events"] == 1
    assert result["median_timing_delta_days"] == 1.0


def test_score_combinations_preserve_lower_is_safer_semantics():
    frame = pd.DataFrame({
        "catboost_snapshot": [0.1, 0.8],
        "catboost_tail_aligned": [0.2, 0.6],
    })
    combined = combine_scores(frame)
    assert combined.loc[0, "arithmetic_mean"] < combined.loc[1, "arithmetic_mean"]
    assert combined.loc[0, "geometric_mean"] < combined.loc[1, "geometric_mean"]
    assert np.all(combined["maximum"] >= combined["catboost_snapshot"])
    assert np.all(combined["maximum"] >= combined["catboost_tail_aligned"])
    assert np.all(combined["minimum"] <= combined["catboost_snapshot"])
    assert np.all(combined["minimum"] <= combined["catboost_tail_aligned"])


def test_score_combinations_reject_nonfinite_and_out_of_range_inputs():
    with np.testing.assert_raises(ValueError):
        combine_scores(pd.DataFrame({
            "catboost_snapshot": [0.1, np.nan],
            "catboost_tail_aligned": [0.2, 0.3],
        }))
    with np.testing.assert_raises(ValueError):
        combine_scores(pd.DataFrame({
            "catboost_snapshot": [0.1, 1.1],
            "catboost_tail_aligned": [0.2, 0.3],
        }))


def test_repeated_calibration_uses_event_level_stratified_halves():
    rows = []
    for event_id in range(40):
        label = int(event_id < 20)
        for step, time_to_tca in enumerate([6.0, 5.0, 4.0]):
            snapshot = 0.8 if label else 0.1 + 0.01 * step
            tail = 0.85 if label else 0.12 + 0.01 * step
            rows.append({
                "event_id": event_id,
                "time_to_tca": time_to_tca,
                "y": label,
                "eligible_history_count": step + 1,
                "catboost_snapshot": snapshot,
                "catboost_tail_aligned": tail,
                "minimum": min(snapshot, tail),
            })
    detail = repeated_calibration_stability(
        pd.DataFrame(rows),
        repeats=3,
        test_fraction=0.5,
        seed_base=10,
        minimum_history=1,
        alpha=0.30,
        mode="marginal",
    )
    assert len(detail) == 9
    assert detail["calibration_positives"].eq(10).all()
    assert detail["danger_n"].eq(10).all()
    assert detail.groupby("repeat")["seed"].nunique().eq(1).all()


def test_repeated_stability_summary_labels_correlated_oof_diagnostics():
    detail = pd.DataFrame({
        "method": ["a", "a"], "repeat": [0, 1], "alpha": [0.1, 0.1],
        "calibration_positives": [10, 10], "danger_rate": [0.0, 0.1],
        "danger_ucb": [0.05, 0.15], "safe_negative_rate": [0.6, 0.8],
        "coverage_delta_vs_snapshot": [0.01, -0.01],
        "danger_delta_vs_snapshot": [0, 1],
    })
    result = summarize_stability(detail).iloc[0]
    assert result["repeats"] == 2
    assert result["ucb_le_alpha_fraction"] == 0.5
    assert result["coverage_delta_positive_fraction"] == 0.5
    assert result["danger_not_worse_fraction"] == 0.5


def test_repeated_stability_rejects_nonfinite_scores():
    frame = pd.DataFrame({
        "event_id": [1], "time_to_tca": [5.0], "y": [0],
        "eligible_history_count": [1], "catboost_snapshot": [0.1],
        "catboost_tail_aligned": [np.nan], "minimum": [0.1],
    })
    with np.testing.assert_raises(ValueError):
        validate_oof_scores(
            frame, ("catboost_snapshot", "catboost_tail_aligned", "minimum")
        )


def test_validation_plan_known_clopper_pearson_requirements():
    assert minimum_positive_events(0) == 29
    assert minimum_positive_events(4) == 89
    assert maximum_passing_failures(200) == 12
    assert 0.79 < pass_probability(200, 0.05) < 0.80


def test_evaluation_planning_table_is_monotone_in_true_risk():
    table = evaluation_planning_table(positive_counts=(100, 200))
    assert table["positive_events"].tolist() == [100, 200]
    assert (
        table["pass_probability_if_true_rate_0.04"]
        >= table["pass_probability_if_true_rate_0.05"]
    ).all()
    assert (
        table["pass_probability_if_true_rate_0.05"]
        >= table["pass_probability_if_true_rate_0.06"]
    ).all()


def test_freeze_next_validation_creates_immutable_lock(tmp_path):
    terminal = tmp_path / "terminal.json"
    terminal.write_text('{"status":"development-complete"}\n')
    output = tmp_path / "preregistration.json"
    lock = tmp_path / "preregistration.lock"
    planning = tmp_path / "planning.csv"
    payload = freeze_plan([terminal], output, lock, planning)
    assert payload["candidate"]["score"] == "catboost_tail_aligned"
    assert payload["evaluation_accessed"] is False
    assert output.exists() and lock.exists() and planning.exists()
    with np.testing.assert_raises(FileExistsError):
        freeze_plan([terminal], output, lock, planning)


def test_final_trainer_requires_matching_preregistration_lock(tmp_path):
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(json.dumps({
        "status": "frozen-before-new-data",
        "evaluation_accessed": False,
        "candidate": {
            "score": "catboost_tail_aligned",
            "model": "two-stage CatBoost with nested inner-OOF positive-tail weights",
            "hard_fraction": 0.25,
            "hard_mass": 0.50,
            "iterations": 500,
        },
    }) + "\n")
    import hashlib
    digest = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    lock = tmp_path / "preregistration.lock"
    lock.write_text(json.dumps({"preregistration_sha256": digest}) + "\n")
    candidate, actual = read_locked_candidate(preregistration, lock)
    assert candidate["score"] == "catboost_tail_aligned"
    assert actual == digest

    lock.write_text(json.dumps({"preregistration_sha256": "0" * 64}) + "\n")
    with np.testing.assert_raises(ValueError):
        read_locked_candidate(preregistration, lock)


def test_final_trainer_rejects_changed_candidate(tmp_path):
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(json.dumps({
        "status": "frozen-before-new-data",
        "evaluation_accessed": False,
        "candidate": {"score": "different", "model": "different"},
    }) + "\n")
    import hashlib
    lock = tmp_path / "preregistration.lock"
    lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n")
    with np.testing.assert_raises(ValueError):
        read_locked_candidate(preregistration, lock)

def test_nested_gate_roles_are_disjoint():
    folds = [0, 1, 2, 3, 4]
    for evaluation_fold in folds:
        roles = nested_gate_roles(folds, evaluation_fold)
        assigned = [fold for values in roles.values() for fold in values]
        assert sorted(assigned) == folds
        assert len(assigned) == len(set(assigned))
        assert roles["evaluation"] == [evaluation_fold]
        assert len(roles["gate_training"]) == 2


def test_gate_aware_crossfit_keeps_complete_event_denominators():
    rows = []
    for fold in range(5):
        for local_event in range(10):
            event_id = fold * 100 + local_event
            label = int(local_event < 4)
            for step, time_to_tca in enumerate([6.0, 5.0]):
                rows.append({
                    "event_id": event_id,
                    "time_to_tca": time_to_tca,
                    "y": label,
                    "fold": fold,
                    "score": 0.8 if label else 0.1 + 0.01 * step,
                    "gate_feature": float(local_event) + 0.1 * step,
                })
    result = crossfit_shift_gate(
        pd.DataFrame(rows),
        score_col="score",
        gate_features=["gate_feature"],
        minimum_histories=[1],
        alpha=0.30,
        mode="marginal",
        confidence=0.95,
        gate_alpha=0.20,
    ).iloc[0]

    assert result["danger_n"] == 20
    assert result["negative_n"] == 30
    assert result["danger_k"] == 0
    assert result["safe_negative"] <= result["negative_n"]
    assert result["gate_flagged_events"] >= result["gate_blocked_events"]
    assert result["gate_flagged_positive"] + result["gate_flagged_negative"] == result["gate_flagged_events"]
    assert "gate_cal=" in result["fold_roles"]
    assert "policy_cal=" in result["fold_roles"]

def test_shift_gate_event_calibration_uses_complete_event_paths():
    proper = pd.DataFrame({"gate_feature": np.linspace(0.0, 1.0, 20)})
    calibration = pd.DataFrame({
        "event_id": np.repeat(np.arange(20), 2),
        "gate_feature": np.tile([0.25, 0.75], 20),
    })
    gate = ConformalShiftGate(["gate_feature"]).fit(proper)
    result = gate.calibrate_events(calibration, alpha=0.10)

    assert result.n_calibration == 20
    assert result.rank == 19
    assert result.marginal_flag_bound <= 0.10


def test_shift_gate_round_trip_preserves_decisions_and_fingerprint(tmp_path):
    proper = pd.DataFrame({"gate_feature": np.linspace(0.0, 1.0, 20)})
    calibration = pd.DataFrame({
        "event_id": np.repeat(np.arange(20), 2),
        "gate_feature": np.tile([0.25, 0.75], 20),
    })
    gate = ConformalShiftGate(["gate_feature"]).fit(proper)
    gate.calibrate_events(calibration, alpha=0.10)
    path = tmp_path / "shift_gate.json"
    gate.save(path)
    restored = ConformalShiftGate.load(path)
    probe = pd.DataFrame({"gate_feature": [0.5, 20.0, np.nan]})

    assert restored.fingerprint() == gate.fingerprint()
    assert restored.allows_safe_exclude(probe).tolist() == gate.allows_safe_exclude(probe).tolist()


def test_confirmation_requires_matching_shift_gate():
    proper = confirmation_scores().query("event_id <= 20")
    gate_calibration = confirmation_scores().query("event_id > 20")
    gate = ConformalShiftGate(["gate_feature"]).fit(proper)
    gate.calibrate_events(gate_calibration, alpha=0.10)
    calibration = calibrate(confirmation_scores(), shift_gate=gate)

    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)
    result = evaluate(
        confirmation_scores(event_offset=100), calibration, shift_gate=gate
    )
    assert result["evaluation"]["shift_gate_blocked_events"] >= 0

def test_shift_gate_calibration_and_blocking():
    proper = pd.DataFrame({
        "risk": [-9.0, -8.5, -8.0, -7.5, -7.0],
        "miss_distance": [100.0, 110.0, 90.0, 105.0, 95.0],
    })
    calibration = pd.DataFrame({
        "risk": np.linspace(-9.0, -7.0, 39),
        "miss_distance": np.linspace(90.0, 110.0, 39),
    })
    gate = ConformalShiftGate(["risk", "miss_distance"]).fit(proper)
    result = gate.calibrate(calibration, alpha=0.10)
    decisions = gate.allows_safe_exclude(pd.DataFrame({
        "risk": [-8.0, 20.0, np.nan],
        "miss_distance": [100.0, 100.0, 100.0],
    }))
    assert result.rank == 36
    assert result.marginal_flag_bound <= 0.10
    assert decisions.tolist() == [True, False, False]


def test_shift_gate_small_calibration_is_conservative():
    proper = pd.DataFrame({"risk": [-9.0, -8.0, -7.0]})
    calibration = pd.DataFrame({"risk": [-8.5, -8.0, -7.5]})
    gate = ConformalShiftGate(["risk"]).fit(proper)
    result = gate.calibrate(calibration, alpha=0.05)
    assert np.isinf(result.threshold)
    assert gate.allows_safe_exclude(pd.DataFrame({"risk": [1000.0]})).item()
    assert not gate.allows_safe_exclude(pd.DataFrame({"risk": [np.inf]})).item()


def test_shift_gate_requires_independent_finite_calibration_features():
    gate = ConformalShiftGate(["risk"]).fit(pd.DataFrame({"risk": [-9.0, -8.0, -7.0]}))
    with np.testing.assert_raises(ValueError):
        gate.calibrate(pd.DataFrame({"risk": [-8.0, np.nan]}), alpha=0.10)


def test_eligible_prefixes_include_window_boundaries():
    features = build_prefix_features(sample())
    selected = eligible_prefixes(features, min_days=3.0, max_days=5.0)
    assert selected["time_to_tca"].between(3.0, 5.0).all()
    assert {3.0, 5.0}.issubset(set(selected["time_to_tca"]))


def test_delta_previous_follows_causal_ingestion_order():
    features = build_prefix_features(sample()).query("event_id == 1")
    assert np.isnan(features.iloc[0]["risk_delta_prev"])
    assert features.iloc[1]["risk_delta_prev"] == 1.0
    assert features.iloc[2]["risk_delta_prev"] == 2.0
    assert features["dt_prev"].iloc[1:].tolist() == [1.0, 1.0]


def test_sequential_triage_history_gate_shift_gate_and_audit():
    proper = pd.DataFrame({"risk": [-9.0, -8.5, -8.0, -7.5, -7.0]})
    calibration = pd.DataFrame({"risk": np.linspace(-9.0, -7.0, 39)})
    gate = ConformalShiftGate(["risk"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    policy = SequentialTriagePolicy(
        safe_threshold=0.20,
        minimum_history=3,
        escalation_threshold=0.80,
        shift_gate=gate,
    )

    first = policy.update(1, 6.0, 0.10, {"risk": -8.0})
    second = policy.update(1, 5.0, 0.90, {"risk": -8.0})
    third = policy.update(1, 4.0, 0.10, {"risk": 20.0})
    fourth = policy.update(1, 3.0, 0.10, {"risk": -8.0})

    assert [first.decision, second.decision, third.decision, fourth.decision] == [
        Decision.MONITOR, Decision.ESCALATE, Decision.MONITOR, Decision.SAFE_EXCLUDE
    ]
    assert third.reason == "safe_exclude_blocked_by_shift_gate"
    audit = policy.audit_log()
    assert audit["sequence_number"].tolist() == [1, 2, 3, 4]
    assert audit.iloc[-1]["decision"] == "SAFE-EXCLUDE"


def test_sequential_triage_escalates_and_rejects_out_of_order_updates():
    policy = SequentialTriagePolicy(0.20, escalation_threshold=0.80)
    result = policy.update("event", 5.0, 0.90)
    assert result.decision == Decision.ESCALATE
    with np.testing.assert_raises(ValueError):
        policy.update("event", 5.5, 0.10)
    assert len(policy.audit_log()) == 1


def test_missing_shift_features_fail_safe():
    gate = ConformalShiftGate(["risk"]).fit(pd.DataFrame({"risk": [-9.0, -8.0, -7.0]}))
    gate.calibrate(pd.DataFrame({"risk": np.linspace(-9.0, -7.0, 39)}), alpha=0.10)
    policy = SequentialTriagePolicy(0.20, shift_gate=gate)
    result = policy.update(1, 5.0, 0.10)
    assert result.decision == Decision.MONITOR
    assert result.reason == "safe_exclude_blocked_by_shift_gate"


def test_runtime_safe_exclude_is_limited_to_calibrated_window():
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    early = policy.update("event", 8.0, 0.10)
    opening = policy.update("event", 7.0, 0.10)
    closing = policy.update("event", 2.0, 0.10)
    late = policy.update("event", 1.5, 0.10)

    assert [early.decision, opening.decision, closing.decision, late.decision] == [
        Decision.MONITOR,
        Decision.SAFE_EXCLUDE,
        Decision.SAFE_EXCLUDE,
        Decision.MONITOR,
    ]
    assert early.reason == "decision_window_not_open"
    assert late.reason == "decision_window_closed"


def test_minimum_history_counts_only_updates_inside_decision_window():
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=3)
    decisions = [
        policy.update("event", 9.0, 0.10),
        policy.update("event", 8.0, 0.10),
        policy.update("event", 7.0, 0.10),
        policy.update("event", 6.0, 0.10),
        policy.update("event", 5.0, 0.10),
    ]

    assert [item.eligible_history_count for item in decisions] == [0, 0, 1, 2, 3]
    assert decisions[-2].decision == Decision.MONITOR
    assert decisions[-1].decision == Decision.SAFE_EXCLUDE


def test_runtime_rejects_invalid_decision_window():
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy(0.20, min_days_to_tca=7.0, max_days_to_tca=2.0)


def test_reset_event_resets_eligible_history_count():
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=2)
    policy.update("event", 6.0, 0.10)
    policy.reset_event("event")
    restarted = policy.update("event", 6.0, 0.10)

    assert restarted.sequence_number == 1
    assert restarted.eligible_history_count == 1
    assert restarted.decision == Decision.MONITOR

def confirmation_scores(event_offset=0):
    rows = []
    for event_id in range(1, 41):
        label = 1
        base = 0.30 + event_id / 1000
        values = [base + 0.02, base + 0.01, base]
        for time_to_tca, score in zip([7.0, 6.0, 5.0], values):
            rows.append({
                "event_id": event_id + event_offset,
                "time_to_tca": time_to_tca,
                "y": label,
                "catboost_snapshot": score,
                "model_sha256": "test-model-sha256",
                "gate_feature": score,
            })
    for event_id, values in [(41, [0.20, 0.10, 0.05]), (42, [0.30, 0.20, 0.10])]:
        for time_to_tca, score in zip([7.0, 6.0, 5.0], values):
            rows.append({
                "event_id": event_id + event_offset,
                "time_to_tca": time_to_tca,
                "y": 0,
                "catboost_snapshot": score,
                "model_sha256": "test-model-sha256",
                "gate_feature": score,
            })
    return pd.DataFrame(rows)

def test_frozen_calibration_and_confirmation_are_disjoint():
    calibration = calibrate(confirmation_scores())
    assert calibration["policy"] == POLICY
    assert calibration["calibration"]["n_positive"] == 40

    result = evaluate(confirmation_scores(event_offset=100), calibration)
    assert result["evaluation_events"] == 42
    assert result["evaluation"]["danger_n"] == 40
    assert result["evaluation"]["negative_n"] == 2


def test_confirmation_rejects_event_overlap():
    calibration = calibrate(confirmation_scores())
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(), calibration)


def test_confirmation_rejects_policy_drift():
    calibration = calibrate(confirmation_scores())
    calibration["policy"]["minimum_history"] = 1
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_window_counter_is_recomputed():
    frame = confirmation_scores()
    frame["eligible_history_count"] = 99
    prepared = prepare_prefix_scores(frame)
    assert prepared.query("event_id == 1")["eligible_history_count"].tolist() == [1, 2, 3]

def test_confirmation_lock_cannot_be_reused(tmp_path):
    lock = tmp_path / "confirmation.lock"
    acquire_confirmation_lock(lock, {"run": 1})
    with np.testing.assert_raises(RuntimeError):
        acquire_confirmation_lock(lock, {"run": 2})

def test_frozen_calibration_rejects_too_few_positive_events():
    small = confirmation_scores().query("event_id <= 3 or event_id >= 41")
    with np.testing.assert_raises(ValueError):
        calibrate(small)

def snapshot_training_frame(event_offset=0):
    rows = []
    for event_id in range(1, 9):
        label = int(event_id > 4)
        for step, time_to_tca in enumerate([7.0, 6.0, 5.0]):
            rows.append({
                "event_id": event_id + event_offset,
                "time_to_tca": time_to_tca,
                "y": label,
                "risk": -9.0 + 3.0 * label + 0.1 * step,
                "max_risk_estimate": -8.0 + 2.0 * label,
                "max_risk_scaling": 1.0,
                "miss_distance": 200.0 - 80.0 * label + step,
                "relative_speed": 10.0 + label,
                "mahalanobis_distance": 6.0 - 2.0 * label,
                "t_position_covariance_det": 2.0 + label,
                "c_position_covariance_det": 3.0 + label,
                "t_obs_available": 20.0 + step,
                "t_obs_used": 18.0 + step,
                "t_weighted_rms": 0.5,
                "c_obs_available": 15.0 + step,
                "c_obs_used": 14.0 + step,
                "c_weighted_rms": 0.6,
                "mission_id": event_id % 3,
                "c_object_type": "PAYLOAD" if event_id % 2 else "DEBRIS",
            })
    return pd.DataFrame(rows)


def test_snapshot_model_scores_required_confirmation_columns():
    training = snapshot_training_frame()
    model = fit_snapshot_model(training, {"iterations": 10})
    scores = score_snapshot_model(model, snapshot_training_frame(100))
    assert scores.columns.tolist() == [
        "event_id", "time_to_tca", "y", "catboost_snapshot"
    ]
    assert scores["catboost_snapshot"].between(0.0, 1.0).all()
    assert scores["event_id"].nunique() == 8



def test_snapshot_model_can_keep_numeric_gate_features():
    training = snapshot_training_frame()
    model = fit_snapshot_model(training, {"iterations": 10})
    scores = score_snapshot_model(
        model, snapshot_training_frame(100), passthrough_columns=["risk", "miss_distance"]
    )
    assert "risk" in scores.columns
    assert "miss_distance" in scores.columns


def test_snapshot_model_rejects_unknown_gate_features():
    training = snapshot_training_frame()
    model = fit_snapshot_model(training, {"iterations": 10})
    with np.testing.assert_raises(ValueError):
        score_snapshot_model(model, training, passthrough_columns=["not_a_feature"])

def test_snapshot_event_weights_sum_to_one_per_event():
    prepared = prepare_snapshot_frame(snapshot_training_frame())
    weighted = prepared.assign(weight=event_equal_weights(prepared))
    totals = weighted.groupby("event_id")["weight"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)


def test_snapshot_splits_must_be_disjoint():
    frame = snapshot_training_frame()
    with np.testing.assert_raises(ValueError):
        assert_disjoint_splits({"training": frame, "calibration": frame.copy()})

def test_confirmation_rejects_scores_from_different_models():
    calibration = calibrate(confirmation_scores())
    evaluation_scores = confirmation_scores(event_offset=100)
    evaluation_scores["model_sha256"] = "other-model-sha256"
    with np.testing.assert_raises(ValueError):
        evaluate(evaluation_scores, calibration)

def test_evaluation_scores_are_label_blind_until_confirmation():
    training = snapshot_training_frame()
    model = fit_snapshot_model(training, {"iterations": 10})
    features = snapshot_training_frame(100).drop(columns="y")
    scores = score_snapshot_model(model, features, include_labels=False)
    assert "y" not in scores.columns
    scores["model_sha256"] = "test-model-sha256"
    labels = snapshot_training_frame(100).loc[:, ["event_id", "y"]].drop_duplicates()
    attached = attach_event_labels(scores, labels)
    assert attached["y"].notna().all()


def test_evaluation_label_join_requires_exact_event_set():
    scores = confirmation_scores(event_offset=100).drop(columns="y")
    labels = confirmation_scores(event_offset=100).loc[:, ["event_id", "y"]].drop_duplicates()
    with np.testing.assert_raises(ValueError):
        attach_event_labels(scores, labels.iloc[:-1])

def test_final_event_label_uses_minimum_time_to_tca():
    frame = pd.DataFrame({
        "event_id": [1, 1, 2, 2],
        "time_to_tca": [3.0, 0.5, 4.0, 0.2],
        "risk": [-4.0, -7.0, -8.0, -5.0],
    })
    labels = event_labels(frame).set_index("event_id")
    assert labels.loc[1, "y"] == 0
    assert labels.loc[2, "y"] == 1


def test_event_split_is_stratified_and_disjoint():
    labels = pd.DataFrame({
        "event_id": np.arange(100),
        "y": np.repeat([0, 1], 50),
    })
    splits = split_event_ids(labels)
    assert [len(splits[name]) for name in ["development", "calibration", "evaluation"]] == [60, 20, 20]
    assert not splits["development"].intersection(splits["calibration"])
    assert not splits["development"].intersection(splits["evaluation"])
    assert not splits["calibration"].intersection(splits["evaluation"])
    indexed = labels.set_index("event_id")["y"]
    assert [int(indexed.loc[list(splits[name])].sum()) for name in ["development", "calibration", "evaluation"]] == [30, 10, 10]


def test_complete_rosters_keep_events_without_eligible_prefixes():
    calibration_scores = confirmation_scores()
    calibration_labels = calibration_scores.loc[:, ["event_id", "y"]].drop_duplicates()
    calibration_labels = pd.concat([
        calibration_labels,
        pd.DataFrame({"event_id": [999], "y": [1]}),
    ], ignore_index=True)
    artifact = calibrate(calibration_scores, calibration_labels)
    assert artifact["calibration_events"] == 43
    assert artifact["calibration"]["n_positive"] == 41

    evaluation_scores = confirmation_scores(event_offset=100)
    evaluation_labels = evaluation_scores.loc[:, ["event_id", "y"]].drop_duplicates()
    evaluation_labels = pd.concat([
        evaluation_labels,
        pd.DataFrame({"event_id": [1999], "y": [0]}),
    ], ignore_index=True)
    result = evaluate(evaluation_scores, artifact, evaluation_labels)
    assert result["evaluation_events"] == 43
    assert result["evaluation"]["negative_n"] == 3



def tail_manifest():
    return {
        "score_column": "catboost_tail_aligned",
        "candidate": {
            "alpha": 0.10,
            "calibration_confidence": 0.95,
            "calibration_mode": "pac",
            "minimum_history": 3,
            "decision_window_days": [2.0, 7.0],
        },
    }


def test_dynamic_scoring_is_label_blind_and_returns_confirmation_columns():
    training = snapshot_training_frame()
    prepared = prepare_dynamic_frame(training)
    model = fit_dynamic_model(prepared, event_equal_weights(prepared), {"iterations": 10})
    features = snapshot_training_frame(100).drop(columns="y")
    scores = score_dynamic_frame(model, features)
    assert scores.columns.tolist() == [
        "event_id", "time_to_tca", "eligible_history_count",
        "catboost_tail_aligned",
    ]
    assert scores["catboost_tail_aligned"].between(0.0, 1.0).all()
    with np.testing.assert_raises(ValueError):
        score_dynamic_frame(model, snapshot_training_frame(100))


def test_tail_policy_is_loaded_from_model_manifest():
    policy = policy_from_model_manifest(tail_manifest())
    assert policy["score_column"] == "catboost_tail_aligned"
    assert policy["minimum_history"] == 3
    assert policy["min_days_to_tca"] == 2.0
    assert policy["max_days_to_tca"] == 7.0


def test_label_blind_tail_scores_can_be_calibrated_with_separate_labels():
    policy = policy_from_model_manifest(tail_manifest())
    scores = confirmation_scores().rename(
        columns={"catboost_snapshot": "catboost_tail_aligned"}
    ).drop(columns="y")
    labels = confirmation_scores().loc[:, ["event_id", "y"]].drop_duplicates()
    calibration = calibrate(scores, labels, policy=policy)
    assert calibration["policy"] == policy
    assert calibration["calibration"]["n_positive"] == 40


def test_tail_confirmation_requires_the_same_frozen_policy():
    policy = policy_from_model_manifest(tail_manifest())
    calibration_scores = confirmation_scores().rename(
        columns={"catboost_snapshot": "catboost_tail_aligned"}
    )
    calibration = calibrate(calibration_scores, policy=policy)
    evaluation_scores = confirmation_scores(event_offset=100).rename(
        columns={"catboost_snapshot": "catboost_tail_aligned"}
    )
    result = evaluate(evaluation_scores, calibration, policy=policy)
    assert result["policy"]["score_column"] == "catboost_tail_aligned"
    with np.testing.assert_raises(ValueError):
        evaluate(evaluation_scores, calibration)


def test_freeze_study_locks_disjoint_label_blind_cohorts(tmp_path):
    calibration_path = tmp_path / "calibration.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": [1, 1, 2], "time_to_tca": [6.0, 5.0, 6.0], "risk": [-8.0, -7.5, -9.0]
    }).to_parquet(calibration_path, index=False)
    pd.DataFrame({
        "event_id": [3, 4], "time_to_tca": [6.0, 6.0], "risk": [-8.0, -9.0]
    }).to_parquet(evaluation_path, index=False)
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(json.dumps({"status": "frozen-before-new-data"}) + "\n")
    import hashlib
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n")
    manifest_path = tmp_path / "study.json"
    lock_path = tmp_path / "study.lock"

    manifest = freeze_study(
        calibration_path, evaluation_path, preregistration, preregistration_lock,
        manifest_path, lock_path,
    )
    locked, digest = read_locked_study(manifest_path, lock_path)

    assert manifest["outcomes_accessed"] is False
    assert locked["cohorts"]["calibration"]["event_ids"] == ["1", "2"]
    assert locked["cohorts"]["evaluation"]["event_ids"] == ["3", "4"]
    assert validate_feature_cohort(
        calibration_path, manifest_path, lock_path, "calibration"
    ) == digest
    with np.testing.assert_raises(FileExistsError):
        freeze_study(
            calibration_path, evaluation_path, preregistration, preregistration_lock,
            manifest_path, lock_path,
        )


def test_freeze_study_rejects_overlap_and_outcome_columns(tmp_path):
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n")
    import hashlib
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n")
    calibration_path = tmp_path / "calibration.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pd.DataFrame({"event_id": [1], "time_to_tca": [6.0]}).to_parquet(calibration_path)
    pd.DataFrame({"event_id": [1], "time_to_tca": [5.0]}).to_parquet(evaluation_path)
    with np.testing.assert_raises(ValueError):
        freeze_study(
            calibration_path, evaluation_path, preregistration, preregistration_lock,
            tmp_path / "overlap.json", tmp_path / "overlap.lock",
        )
    pd.DataFrame({"event_id": [2], "time_to_tca": [5.0], "y": [0]}).to_parquet(evaluation_path)
    with np.testing.assert_raises(ValueError):
        freeze_study(
            calibration_path, evaluation_path, preregistration, preregistration_lock,
            tmp_path / "labelled.json", tmp_path / "labelled.lock",
        )


def test_frozen_study_detects_file_and_label_roster_drift(tmp_path):
    calibration_path = tmp_path / "calibration.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pd.DataFrame({"event_id": [1, 2], "time_to_tca": [6.0, 6.0]}).to_parquet(calibration_path)
    pd.DataFrame({"event_id": [3, 4], "time_to_tca": [6.0, 6.0]}).to_parquet(evaluation_path)
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n")
    import hashlib
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n")
    manifest_path, lock_path = tmp_path / "study.json", tmp_path / "study.lock"
    freeze_study(
        calibration_path, evaluation_path, preregistration, preregistration_lock,
        manifest_path, lock_path,
    )
    manifest, _ = read_locked_study(manifest_path, lock_path)
    validate_label_roster(pd.DataFrame({"event_id": [1, 2], "y": [0, 1]}), manifest, "calibration")
    with np.testing.assert_raises(ValueError):
        validate_label_roster(pd.DataFrame({"event_id": [1], "y": [0]}), manifest, "calibration")

    pd.DataFrame({"event_id": [1, 2], "time_to_tca": [6.0, 5.0]}).to_parquet(calibration_path)
    with np.testing.assert_raises(ValueError):
        validate_feature_cohort(calibration_path, manifest_path, lock_path, "calibration")
