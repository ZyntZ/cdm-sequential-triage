import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from prefix_features import build_prefix_features, eligible_prefixes
from policy import (
    calibration_rank, calibrate_positive_threshold, cp_upper,
    event_policy_table, evaluate_threshold,
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
