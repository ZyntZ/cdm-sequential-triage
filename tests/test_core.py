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
from score_final_tail_aligned import score_file as score_tail_file
from replay_scores import (
    load_runtime, replay_scores, run_replay, validate_runtime_continuation,
    validate_score_stream,
)
from operator_dashboard import build_dashboard, current_events, load_audits
from evidence_dashboard import build_evidence_dashboard
from run_demo import run_demo
from confirm_policy import calibration_command
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
    policy_from_model_manifest, prepare_prefix_scores, write_json,
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


def test_shift_gate_save_preserves_target_after_fsync_failure(tmp_path, monkeypatch):
    import shift_gate as shift_gate_module
    frame = pd.DataFrame({"f": np.linspace(0.0, 1.0, 40)})
    gate = ConformalShiftGate(["f"]).fit(frame)
    gate.calibrate_events(
        pd.DataFrame({"event_id": np.arange(40), "f": np.linspace(0.0, 1.0, 40)}),
        alpha=0.10,
    )
    target = tmp_path / "gate.json"
    gate.save(target)
    original = target.read_bytes()

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(shift_gate_module.os, "fsync", fail_fsync)
    with np.testing.assert_raises(OSError):
        gate.save(target)

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".gate.json.*.tmp"))


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


def test_reset_event_rejects_reuse_after_audit():
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=2)
    policy.update("event", 6.0, 0.10)

    with np.testing.assert_raises(RuntimeError):
        policy.reset_event("event")

    continued = policy.update("event", 5.0, 0.10)
    assert continued.sequence_number == 2
    assert continued.eligible_history_count == 2


def test_reset_event_allows_untouched_unknown_event():
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=2)
    policy.reset_event("unknown")
    assert policy.continuation_limits() == {}

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

def test_atomic_json_write_cleans_up_after_fsync_failure(tmp_path, monkeypatch):
    import confirmation as confirmation_module
    target = tmp_path / "artifact.json"
    target.write_text("sentinel\n", encoding="utf-8")
    original = target.read_bytes()

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(confirmation_module.os, "fsync", fail_fsync)
    with np.testing.assert_raises(OSError):
        write_json(target, {"status": "new"})

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_confirmation_lock_is_fsynced(tmp_path, monkeypatch):
    import confirmation as confirmation_module
    calls = []
    real_fsync = confirmation_module.os.fsync

    def record_fsync(descriptor):
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(confirmation_module.os, "fsync", record_fsync)
    lock = tmp_path / "confirmation.lock"
    acquire_confirmation_lock(lock, {"run": 1})

    assert len(calls) == 1
    assert json.loads(lock.read_text(encoding="utf-8")) == {"run": 1}


def test_confirmation_lock_failure_removes_incomplete_file(tmp_path, monkeypatch):
    import confirmation as confirmation_module
    lock = tmp_path / "confirmation.lock"

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(confirmation_module.os, "fsync", fail_fsync)
    with np.testing.assert_raises(OSError):
        acquire_confirmation_lock(lock, {"run": 1})

    assert not lock.exists()


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


def test_freeze_study_cleans_temporary_manifest_after_fsync_failure(tmp_path, monkeypatch):
    import hashlib
    import study as study_module
    calibration_path = tmp_path / "calibration.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pd.DataFrame({"event_id": [1], "time_to_tca": [6.0]}).to_parquet(
        calibration_path, index=False
    )
    pd.DataFrame({"event_id": [2], "time_to_tca": [6.0]}).to_parquet(
        evaluation_path, index=False
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n", encoding="utf-8")
    output = tmp_path / "study.json"
    lock = tmp_path / "study.lock"

    def fail_fsync(_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(study_module.os, "fsync", fail_fsync)
    with np.testing.assert_raises(OSError):
        freeze_study(
            calibration_path, evaluation_path, preregistration,
            preregistration_lock, output, lock,
        )

    assert not output.exists()
    assert not lock.exists()
    assert not list(tmp_path.glob(".study.json.*.tmp"))


def test_freeze_study_removes_incomplete_lock_after_lock_fsync_failure(tmp_path, monkeypatch):
    import hashlib
    import study as study_module
    calibration_path = tmp_path / "calibration.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    pd.DataFrame({"event_id": [1], "time_to_tca": [6.0]}).to_parquet(
        calibration_path, index=False
    )
    pd.DataFrame({"event_id": [2], "time_to_tca": [6.0]}).to_parquet(
        evaluation_path, index=False
    )
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}\n", encoding="utf-8")
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
    }) + "\n", encoding="utf-8")
    output = tmp_path / "study.json"
    lock = tmp_path / "study.lock"
    real_fsync = study_module.os.fsync
    calls = {"count": 0}

    def fail_second_fsync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated lock fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(study_module.os, "fsync", fail_second_fsync)
    with np.testing.assert_raises(OSError):
        freeze_study(
            calibration_path, evaluation_path, preregistration,
            preregistration_lock, output, lock,
        )

    assert not output.exists()
    assert not lock.exists()
    assert not list(tmp_path.glob(".study.json.*.tmp"))


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


def test_runtime_checkpoint_round_trip_continues_event_state(tmp_path):
    policy = SequentialTriagePolicy(
        safe_threshold=0.20, minimum_history=3, escalation_threshold=0.80
    )
    policy.update("event", 8.0, 0.10)
    policy.update("event", 7.0, 0.10)
    policy.update("event", 6.0, 0.90)
    checkpoint = tmp_path / "runtime.json"
    digest = policy.checkpoint(checkpoint)

    restored = SequentialTriagePolicy.restore(checkpoint)
    decision = restored.update("event", 5.0, 0.10)

    assert len(digest) == 64
    assert decision.sequence_number == 4
    assert decision.eligible_history_count == 3
    assert decision.decision == Decision.SAFE_EXCLUDE
    assert restored.audit_log()["sequence_number"].tolist() == [1, 2, 3, 4]
    with np.testing.assert_raises(ValueError):
        restored.update("event", 5.5, 0.10)


def test_runtime_checkpoint_preserves_event_id_types(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update(7, 6.0, 0.10)
    policy.update("7", 6.0, 0.10)
    checkpoint = tmp_path / "typed-events.json"
    policy.checkpoint(checkpoint)

    restored = SequentialTriagePolicy.restore(checkpoint)

    assert set(restored._counts) == {7, "7"}
    assert restored.update(7, 5.0, 0.10).sequence_number == 2
    assert restored.update("7", 5.0, 0.10).sequence_number == 2


def test_runtime_checkpoint_verifies_shift_gate_fingerprint(tmp_path):
    proper = pd.DataFrame({"risk": [-9.0, -8.5, -8.0, -7.5, -7.0]})
    calibration = pd.DataFrame({"risk": np.linspace(-9.0, -7.0, 39)})
    gate = ConformalShiftGate(["risk"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    policy = SequentialTriagePolicy(0.20, shift_gate=gate)
    policy.update(1, 5.0, 0.10)
    checkpoint = tmp_path / "gated-runtime.json"
    policy.checkpoint(checkpoint)

    restored = SequentialTriagePolicy.restore(checkpoint, shift_gate=gate)
    assert np.isinf(restored.audit_log().iloc[0]["shift_score"])
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)

    other_gate = ConformalShiftGate(["risk"]).fit(proper)
    other_gate.calibrate(calibration, alpha=0.20)
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint, shift_gate=other_gate)


def test_runtime_checkpoint_detects_tampering(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 5.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["payload"]["state"][0]["sequence_number"] = 99
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_runtime_checkpoint_overwrites_atomically(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    checkpoint = tmp_path / "runtime.json"
    policy.update("event", 6.0, 0.10)
    first_digest = policy.checkpoint(checkpoint)
    policy.update("event", 5.0, 0.10)
    second_digest = policy.checkpoint(checkpoint)

    restored = SequentialTriagePolicy.restore(checkpoint)
    assert first_digest != second_digest
    assert restored._counts == {"event": 2}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_dynamic_scoring_passes_gate_features_and_custom_score_name():
    training = snapshot_training_frame()
    prepared = prepare_dynamic_frame(training)
    model = fit_dynamic_model(prepared, event_equal_weights(prepared), {"iterations": 10})
    features = snapshot_training_frame(100).drop(columns="y")
    scores = score_dynamic_frame(model, features, score_column="frozen_score", passthrough_columns=["risk", "miss_distance", "risk_range"])
    assert scores.columns.tolist() == ["event_id", "time_to_tca", "eligible_history_count", "risk", "miss_distance", "risk_range", "frozen_score"]
    assert scores["frozen_score"].between(0.0, 1.0).all()


def test_dynamic_scoring_rejects_invalid_passthrough_columns():
    training = snapshot_training_frame()
    prepared = prepare_dynamic_frame(training)
    model = fit_dynamic_model(prepared, event_equal_weights(prepared), {"iterations": 10})
    features = snapshot_training_frame(100).drop(columns="y")
    for columns in (["risk", "risk"], ["not_a_feature"], ["event_id"]):
        with np.testing.assert_raises(ValueError):
            score_dynamic_frame(model, features, passthrough_columns=columns)


def test_tail_score_file_retains_gate_features(tmp_path):
    root = Path(__file__).resolve().parents[1]
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "scores.parquet"
    snapshot_training_frame(500).drop(columns="y").to_parquet(features_path, index=False)
    scores = score_tail_file(
        features_path, root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json", output_path,
        gate_features=["risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance"],
    )
    assert output_path.exists()
    assert {"risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance"}.issubset(scores.columns)
    pd.testing.assert_frame_equal(scores, pd.read_parquet(output_path))


def runtime_calibration_artifact(model_hash="runtime-model", threshold=0.20):
    return {
        "policy": {
            "score_column": "catboost_tail_aligned",
            "alpha": 0.10,
            "confidence": 0.95,
            "calibration_mode": "pac",
            "minimum_history": 3,
            "min_days_to_tca": 2.0,
            "max_days_to_tca": 7.0,
        },
        "calibration": {"threshold": threshold},
        "model_sha256": model_hash,
        "shift_gate_sha256": None,
        "model_manifest_sha256": None,
        "calibration_event_ids": ["calibration-event"],
    }


def test_replay_scores_writes_complete_operator_audit(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["event", "event", "event", "other", "other", "other"],
        "time_to_tca": [5.0, 7.0, 6.0, 7.0, 6.0, 5.0],
        "catboost_tail_aligned": [0.10, 0.10, 0.10, 0.90, 0.90, 0.90],
        "model_sha256": ["runtime-model"] * 6,
    }).to_parquet(scores_path, index=False)
    output = tmp_path / "audit.parquet"
    checkpoint = tmp_path / "runtime.json"

    audit = run_replay(scores_path, calibration, output, checkpoint_path=checkpoint)

    assert output.exists() and checkpoint.exists()
    assert len(audit) == 6
    event = audit.loc[audit["event_id"] == "event"]
    assert event["time_to_tca"].tolist() == [7.0, 6.0, 5.0]
    assert event["decision"].tolist() == ["MONITOR", "MONITOR", "SAFE-EXCLUDE"]
    pd.testing.assert_frame_equal(audit, pd.read_parquet(output))
    assert audit["scores_sha256"].nunique() == 1
    assert audit["calibration_sha256"].nunique() == 1
    assert audit["runtime_checkpoint_sha256"].str.len().eq(64).all()
    assert audit["runtime_configuration_sha256"].str.len().eq(64).all()
    assert audit["safe_threshold"].eq(0.20).all()
    assert audit["escalation_threshold"].isna().all()
    assert audit["minimum_history"].eq(3).all()
    assert audit["min_days_to_tca"].eq(2.0).all()
    assert audit["max_days_to_tca"].eq(7.0).all()
    restored = SequentialTriagePolicy.restore(checkpoint)
    assert len(restored.audit_log()) == 6


def test_replay_audit_records_escalation_policy_and_rejects_drift(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    checkpoint = tmp_path / "runtime.json"
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["high"],
        "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.90],
        "model_sha256": ["runtime-model"],
    }).to_parquet(scores, index=False)

    audit = run_replay(
        scores, calibration, tmp_path / "audit.parquet",
        checkpoint_path=checkpoint, escalation_threshold=0.80,
    )

    assert audit.iloc[0]["decision"] == "ESCALATE"
    assert audit["escalation_threshold"].eq(0.80).all()
    assert audit["runtime_configuration_sha256"].nunique() == 1
    restored = SequentialTriagePolicy.restore(checkpoint)
    assert restored.configuration_fingerprint() == audit.iloc[0]["runtime_configuration_sha256"]

    next_scores = tmp_path / "next.parquet"
    pd.DataFrame({
        "event_id": ["other"],
        "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.90],
        "model_sha256": ["runtime-model"],
    }).to_parquet(next_scores, index=False)
    with np.testing.assert_raises(ValueError):
        run_replay(
            next_scores, calibration, tmp_path / "next-audit.parquet",
            checkpoint_path=checkpoint, escalation_threshold=0.85,
        )


def test_replay_scores_resumes_from_checkpoint(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    checkpoint = tmp_path / "runtime.json"
    first_scores = tmp_path / "first.parquet"
    pd.DataFrame({
        "event_id": [1, 1],
        "time_to_tca": [7.0, 6.0],
        "catboost_tail_aligned": [0.10, 0.10],
        "model_sha256": ["runtime-model", "runtime-model"],
    }).to_parquet(first_scores, index=False)
    run_replay(first_scores, calibration, tmp_path / "first-audit.parquet", checkpoint_path=checkpoint)

    next_scores = tmp_path / "next.parquet"
    pd.DataFrame({
        "event_id": [1],
        "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["runtime-model"],
    }).to_parquet(next_scores, index=False)
    audit = run_replay(next_scores, calibration, tmp_path / "next-audit.parquet", checkpoint_path=checkpoint)

    assert audit["sequence_number"].tolist() == [3]
    assert audit.iloc[-1]["decision"] == "SAFE-EXCLUDE"
    restored = SequentialTriagePolicy.restore(checkpoint)
    assert restored.audit_log()["sequence_number"].tolist() == [1, 2, 3]


def test_replay_scores_rejects_model_overlap_and_duplicate_updates(tmp_path):
    artifact = runtime_calibration_artifact()
    valid = pd.DataFrame({
        "event_id": ["new"], "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    })
    validate_score_stream(valid, artifact, "catboost_tail_aligned", None)
    with np.testing.assert_raises(ValueError):
        validate_score_stream(
            valid.assign(model_sha256="other-model"), artifact,
            "catboost_tail_aligned", None,
        )
    with np.testing.assert_raises(ValueError):
        validate_score_stream(
            valid.assign(event_id="calibration-event"), artifact,
            "catboost_tail_aligned", None,
        )
    with np.testing.assert_raises(ValueError):
        validate_score_stream(
            pd.concat([valid, valid], ignore_index=True), artifact,
            "catboost_tail_aligned", None,
        )


def test_replay_scores_does_not_overwrite_output_or_silently_replay(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": [1], "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    }).to_parquet(scores_path, index=False)
    checkpoint = tmp_path / "runtime.json"
    output = tmp_path / "audit.parquet"
    run_replay(scores_path, calibration, output, checkpoint_path=checkpoint)
    with np.testing.assert_raises(FileExistsError):
        run_replay(scores_path, calibration, output, checkpoint_path=checkpoint)
    with np.testing.assert_raises(ValueError):
        run_replay(
            scores_path, calibration, tmp_path / "second-audit.parquet",
            checkpoint_path=checkpoint,
        )


def test_replay_scores_requires_matching_gate_and_gate_columns(tmp_path):
    proper = pd.DataFrame({"risk": [-9.0, -8.5, -8.0, -7.5, -7.0]})
    gate_calibration = pd.DataFrame({
        "event_id": np.arange(39), "risk": np.linspace(-9.0, -7.0, 39)
    })
    gate = ConformalShiftGate(["risk"]).fit(proper)
    gate.calibrate_events(gate_calibration, alpha=0.10)
    gate_path = tmp_path / "gate.json"
    gate.save(gate_path)
    artifact = runtime_calibration_artifact()
    artifact["shift_gate_sha256"] = gate.fingerprint()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(artifact) + "\n")

    runtime, policy = load_runtime(calibration, gate_path=gate_path)
    missing_gate_column = pd.DataFrame({
        "event_id": [1], "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    })
    with np.testing.assert_raises(ValueError):
        validate_score_stream(
            missing_gate_column, artifact, policy["score_column"], runtime.shift_gate
        )
    with np.testing.assert_raises(ValueError):
        load_runtime(calibration)


def test_calibration_command_refuses_existing_output(tmp_path):
    import argparse
    output = tmp_path / "calibration.json"
    output.write_text("sentinel\n", encoding="utf-8")
    args = argparse.Namespace(
        scores=tmp_path / "missing-scores.parquet",
        labels=tmp_path / "missing-labels.parquet",
        output=output,
        gate=None,
        model_manifest=None,
        study_manifest=None,
        study_lock=None,
    )
    with np.testing.assert_raises(FileExistsError):
        calibration_command(args)
    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_replay_scores_public_loop_requires_prevalidated_columns():
    runtime = SequentialTriagePolicy(safe_threshold=0.20)
    frame = pd.DataFrame({"event_id": ["event"], "time_to_tca": [5.0]})
    with np.testing.assert_raises(ValueError):
        replay_scores(frame, runtime, "catboost_tail_aligned")
    assert runtime.audit_log().empty


def test_replay_resume_output_keeps_current_batch_lineage(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    checkpoint = tmp_path / "runtime.json"
    first_scores = tmp_path / "first.parquet"
    second_scores = tmp_path / "second.parquet"
    pd.DataFrame({
        "event_id": [1], "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    }).to_parquet(first_scores, index=False)
    first = run_replay(
        first_scores, calibration, tmp_path / "first-audit.parquet",
        checkpoint_path=checkpoint,
    )
    pd.DataFrame({
        "event_id": [1], "time_to_tca": [6.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    }).to_parquet(second_scores, index=False)
    second = run_replay(
        second_scores, calibration, tmp_path / "second-audit.parquet",
        checkpoint_path=checkpoint,
    )
    from confirmation import file_sha256
    assert len(first) == len(second) == 1
    assert first.iloc[0]["scores_sha256"] == file_sha256(first_scores)
    assert second.iloc[0]["scores_sha256"] == file_sha256(second_scores)
    assert second.iloc[0]["sequence_number"] == 2


def test_replay_audit_failure_does_not_advance_checkpoint(tmp_path, monkeypatch):
    import replay_scores as replay_module
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": [1], "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.10], "model_sha256": ["runtime-model"],
    }).to_parquet(scores, index=False)
    checkpoint = tmp_path / "runtime.json"
    before = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=3)
    before.checkpoint(checkpoint)
    original = checkpoint.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("simulated audit write failure")

    monkeypatch.setattr(replay_module, "_write_parquet_atomic", fail_write)
    with np.testing.assert_raises(OSError):
        replay_module.run_replay(
            scores, calibration, tmp_path / "audit.parquet",
            checkpoint_path=checkpoint,
        )
    assert checkpoint.read_bytes() == original
    assert not (tmp_path / "audit.parquet").exists()


def test_replay_marks_current_decision_within_each_output_batch(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["a", "a", "b"],
        "time_to_tca": [7.0, 6.0, 5.0],
        "catboost_tail_aligned": [0.10, 0.10, 0.90],
        "model_sha256": ["runtime-model"] * 3,
    }).to_parquet(scores, index=False)
    audit = run_replay(scores, calibration, tmp_path / "audit.parquet")
    assert audit["is_current_decision"].tolist() == [False, True, True]


def dashboard_audit_frame(calibration_path, event_id="event", decision="MONITOR", sequence=1):
    from confirmation import file_sha256
    return pd.DataFrame({
        "event_id": [event_id],
        "sequence_number": [sequence],
        "time_to_tca": [5.0],
        "score": [0.4],
        "decision": [decision],
        "reason": ["score_between_decision_thresholds"],
        "shift_score": [np.nan],
        "shift_gate_allowed": [True],
        "decision_window_eligible": [True],
        "eligible_history_count": [3],
        "scores_sha256": ["a" * 64],
        "calibration_sha256": [file_sha256(calibration_path)],
        "model_sha256": ["c" * 64],
        "shift_gate_sha256": [None],
        "model_manifest_sha256": [None],
        "runtime_checkpoint_sha256": [None],
        "runtime_configuration_sha256": ["d" * 64],
        "safe_threshold": [0.20],
        "escalation_threshold": [np.nan],
        "minimum_history": [3],
        "min_days_to_tca": [2.0],
        "max_days_to_tca": [7.0],
        "is_current_decision": [True],
    })


def test_operator_dashboard_builds_self_contained_html(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration, decision="SAFE-EXCLUDE").to_parquet(audit_path, index=False)
    output = tmp_path / "dashboard.html"

    summary = build_dashboard([audit_path], calibration, output)
    document = output.read_text(encoding="utf-8")

    assert summary["events"] == 1
    assert summary["current_decisions"]["SAFE-EXCLUDE"] == 1
    assert "Operator Console" in document
    assert "SAFE-EXCLUDE" in document
    assert "c" * 64 in document
    assert "https://" not in document


def test_operator_dashboard_combines_resumed_batches_by_latest_sequence(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    first = dashboard_audit_frame(calibration, decision="MONITOR", sequence=1)
    second = dashboard_audit_frame(calibration, decision="SAFE-EXCLUDE", sequence=2)
    first_path, second_path = tmp_path / "first.parquet", tmp_path / "second.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)

    combined = load_audits([first_path, second_path], runtime_calibration_artifact(model_hash="c" * 64), calibration)
    current = current_events(combined)

    assert len(current) == 1
    assert current.iloc[0]["decision"] == "SAFE-EXCLUDE"
    assert current.iloc[0]["sequence_number"] == 2


def test_operator_dashboard_escapes_event_ids(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration, event_id="<script>alert(1)</script>").to_parquet(audit_path, index=False)
    output = tmp_path / "dashboard.html"

    build_dashboard([audit_path], calibration, output)
    document = output.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document


def test_operator_dashboard_rejects_wrong_lineage_and_overlapping_batches(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    valid = dashboard_audit_frame(calibration)
    wrong = valid.copy()
    wrong["calibration_sha256"] = "b" * 64
    wrong_path = tmp_path / "wrong.parquet"
    wrong.to_parquet(wrong_path, index=False)
    with np.testing.assert_raises(ValueError):
        load_audits([wrong_path], runtime_calibration_artifact(model_hash="c" * 64), calibration)

    first_path, duplicate_path = tmp_path / "first.parquet", tmp_path / "duplicate.parquet"
    valid.to_parquet(first_path, index=False)
    valid.to_parquet(duplicate_path, index=False)
    with np.testing.assert_raises(ValueError):
        load_audits([first_path, duplicate_path], runtime_calibration_artifact(model_hash="c" * 64), calibration)


def test_operator_dashboard_shows_failed_confirmation_honestly(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit_path, index=False)
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({"evaluation": {
        "danger_k": 4, "danger_n": 73, "danger_rate": 4 / 73,
        "danger_ucb": 0.121, "safe_negative_rate": 0.7064,
        "median_first_safe_tca": 5.53,
    }}) + "\n")
    output = tmp_path / "dashboard.html"

    summary = build_dashboard([audit_path], calibration, output, confirmation)
    document = output.read_text(encoding="utf-8")

    assert summary["confirmation"]["passed"] is False
    assert "NOT MET" in document
    assert "12.10%" in document
    assert "does not validate" in document


def test_operator_dashboard_refuses_overwrite(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit_path, index=False)
    output = tmp_path / "dashboard.html"
    output.write_text("sentinel", encoding="utf-8")
    with np.testing.assert_raises(FileExistsError):
        build_dashboard([audit_path], calibration, output)
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_one_command_demo_builds_verified_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-demo"

    summary = run_demo(output, root=root)

    assert summary["status"] == "historical-demo-not-for-operations"
    assert summary["message_updates"] == 22656
    assert summary["events_in_runtime_window"] == 2387
    assert summary["confirmation"]["passed"] is False
    assert (output / "replay-audit.parquet").exists()
    assert (output / "operator-console.html").exists()
    assert (output / "runtime-state.json").exists()
    assert (output / "summary.json").exists()
    assert (output / "evidence-dashboard.html").exists()
    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert stored["caveat"] == summary["caveat"]
    assert stored["evidence_dashboard"] == "evidence-dashboard.html"
    assert stored["evidence"]["confirmation_passed"] is False
    assert "NOT MET" in (output / "operator-console.html").read_text(encoding="utf-8")


def test_one_command_demo_refuses_protected_and_nonempty_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with np.testing.assert_raises(ValueError):
        run_demo(root / "artifacts" / "demo", root=root)

    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("sentinel", encoding="utf-8")
    with np.testing.assert_raises(FileExistsError):
        run_demo(output, root=root)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "sentinel"


def test_one_command_demo_cleans_staging_on_failure(tmp_path, monkeypatch):
    import run_demo as demo_module
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "failed-demo"

    def fail_dashboard(*args, **kwargs):
        raise OSError("simulated dashboard failure")

    monkeypatch.setattr(demo_module, "build_dashboard", fail_dashboard)
    with np.testing.assert_raises(OSError):
        demo_module.run_demo(output, root=root)
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-demo.*"))


def test_evidence_dashboard_builds_three_verified_tiers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evidence.html"
    summary = build_evidence_dashboard(root, output)
    document = output.read_text(encoding="utf-8")

    assert summary["confirmation_passed"] is False
    assert abs(summary["danger_ucb"] - 0.12101499810942579) < 1e-12
    assert summary["criterion"] == 0.10
    assert summary["development_pareto_methods"] == ["catboost_tail_aligned", "minimum"]
    assert summary["preregistration_frozen"] is True
    assert summary["calibration_accessed"] is False
    assert summary["evaluation_accessed"] is False
    assert summary["v13_threshold"] is None
    assert "id='development'" in document
    assert "id='confirmation'" in document
    assert "id='preregistered'" in document
    assert "CRITERION NOT MET" in document
    assert "12.10%" in document
    assert "not confirmation evidence" in document
    assert "https://" not in document


def test_evidence_dashboard_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evidence.html"
    output.write_text("sentinel", encoding="utf-8")
    with np.testing.assert_raises(FileExistsError):
        build_evidence_dashboard(root, output)
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_evidence_dashboard_verifies_preregistration_lock(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "copy"
    shutil.copytree(root / "artifacts", copy_root / "artifacts")
    shutil.copytree(root / "reports", copy_root / "reports")
    prereg = copy_root / "artifacts" / "next_validation_preregistration_v12.json"
    prereg.write_text(prereg.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        build_evidence_dashboard(copy_root, tmp_path / "evidence.html")



def test_checkpoint_v2_persists_processed_batch_ledger(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="a" * 64,
        calibration_sha256="b" * 64,
        rows=1,
        events=1,
        min_time_to_tca=7.0,
        max_time_to_tca=7.0,
        first_audit_row=1,
        last_audit_row=1,
    )
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)

    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    restored = SequentialTriagePolicy.restore(checkpoint)

    assert envelope["payload"]["schema_version"] == 2
    assert restored.processed_batches() == policy.processed_batches()
    assert restored.has_processed_batch("a" * 64)


def test_checkpoint_v1_restores_and_upgrades_to_v2(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["schema_version"] = 1
    envelope["payload"].pop("processed_batches")
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    import hashlib
    envelope["checkpoint_sha256"] = hashlib.sha256(canonical).hexdigest()
    checkpoint.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    restored = SequentialTriagePolicy.restore(checkpoint)
    assert restored.processed_batches() == []

    restored.checkpoint(checkpoint)
    upgraded = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert upgraded["payload"]["schema_version"] == 2
    assert upgraded["payload"]["processed_batches"] == []


def test_replay_ledger_accumulates_across_checkpoint_resumes(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    checkpoint = tmp_path / "runtime.json"
    first_scores = tmp_path / "first.parquet"
    second_scores = tmp_path / "second.parquet"
    pd.DataFrame({
        "event_id": [1, 1],
        "time_to_tca": [7.0, 6.0],
        "catboost_tail_aligned": [0.10, 0.10],
        "model_sha256": ["runtime-model"] * 2,
    }).to_parquet(first_scores, index=False)
    pd.DataFrame({
        "event_id": [1, 2],
        "time_to_tca": [5.0, 7.0],
        "catboost_tail_aligned": [0.10, 0.90],
        "model_sha256": ["runtime-model"] * 2,
    }).to_parquet(second_scores, index=False)

    run_replay(
        first_scores, calibration, tmp_path / "first-audit.parquet",
        checkpoint_path=checkpoint,
    )
    run_replay(
        second_scores, calibration, tmp_path / "second-audit.parquet",
        checkpoint_path=checkpoint,
    )
    restored = SequentialTriagePolicy.restore(checkpoint)
    ledger = restored.processed_batches()

    assert len(ledger) == 2
    assert [entry["rows"] for entry in ledger] == [2, 2]
    assert [entry["events"] for entry in ledger] == [1, 2]
    assert [(entry["first_audit_row"], entry["last_audit_row"]) for entry in ledger] == [(1, 2), (3, 4)]
    assert ledger[0]["scores_sha256"] != ledger[1]["scores_sha256"]


def test_replay_rejects_processed_score_batch_before_state_change(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact()) + "\n")
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": [1],
        "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["runtime-model"],
    }).to_parquet(scores, index=False)
    checkpoint = tmp_path / "runtime.json"
    run_replay(
        scores, calibration, tmp_path / "first-audit.parquet",
        checkpoint_path=checkpoint,
    )
    before = checkpoint.read_bytes()

    with np.testing.assert_raises(ValueError):
        run_replay(
            scores, calibration, tmp_path / "duplicate-audit.parquet",
            checkpoint_path=checkpoint,
        )

    assert checkpoint.read_bytes() == before
    assert not (tmp_path / "duplicate-audit.parquet").exists()
    restored = SequentialTriagePolicy.restore(checkpoint)
    assert len(restored.audit_log()) == 1
    assert len(restored.processed_batches()) == 1


def test_checkpoint_integrity_covers_processed_batch_ledger(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="a" * 64,
        calibration_sha256="b" * 64,
        rows=1,
        events=1,
        min_time_to_tca=7.0,
        max_time_to_tca=7.0,
        first_audit_row=1,
        last_audit_row=1,
    )
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["processed_batches"][0]["rows"] = 2
    checkpoint.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)



def _rewrite_checkpoint_with_valid_digest(path, mutate):
    import hashlib
    envelope = json.loads(path.read_text(encoding="utf-8"))
    mutate(envelope["payload"])
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    envelope["checkpoint_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")


def test_restore_rejects_state_audit_tail_mismatch(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.update("event", 6.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["state"][0].update({"last_time_to_tca": 5.0}),
    )

    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_noncontiguous_audit_sequence(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.update("event", 6.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][1].update({"sequence_number": 3}),
    )

    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_nonmonotone_audit_tca(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.update("event", 6.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][1].update({"time_to_tca": 8.0}),
    )

    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_processed_batch_ledger_rejects_calibration_change():
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="a" * 64, calibration_sha256="c" * 64,
        rows=1, events=1, min_time_to_tca=7.0, max_time_to_tca=7.0,
        first_audit_row=1, last_audit_row=1,
    )
    policy.update("event", 6.0, 0.10)

    with np.testing.assert_raises(ValueError):
        policy.record_processed_batch(
            scores_sha256="b" * 64, calibration_sha256="d" * 64,
            rows=1, events=1, min_time_to_tca=6.0, max_time_to_tca=6.0,
            first_audit_row=2, last_audit_row=2,
        )


def test_processed_batch_range_cannot_exceed_audit():
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)

    with np.testing.assert_raises(ValueError):
        policy.record_processed_batch(
            scores_sha256="a" * 64, calibration_sha256="b" * 64,
            rows=2, events=1, min_time_to_tca=6.0, max_time_to_tca=7.0,
            first_audit_row=1, last_audit_row=2,
            _require_current_tail=False,
        )
