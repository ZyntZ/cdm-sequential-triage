import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from prefix_features import build_prefix_features, eligible_prefixes
from robustness import subgroup_metrics
from shift_gate import ConformalShiftGate
from triage import Decision, SequentialTriagePolicy
from policy import (
    calibration_rank, calibrate_positive_threshold, cp_upper,
    event_policy_table, evaluate_threshold, history_gated_event_table,
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
    events = history_gated_event_table(prefixes, "score", minimum_history=2)
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
        history_gated_event_table(prefixes, "score", minimum_history=0)


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
