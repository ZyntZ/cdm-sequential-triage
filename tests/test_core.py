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
    DYNAMIC_FEATURES, feature_contract_sha256, fit_dynamic_model, positive_tail_weights,
    prepare_dynamic_frame, score_dynamic_frame, score_dynamic_model,
    validate_dynamic_feature_contract,
)
from external_cdm import (
    REQUIRED_EXTERNAL_FEATURES, adapt_external_cdms, derive_event_labels,
    event_grouping_review,
    outcome_blind_features, parse_cdm_json, parse_cdm_source, readiness_report,
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
from operator_dashboard import (
    build_dashboard, current_events, explain_event_sequence, load_audits,
    select_showcase_gate_blocked, select_showcase_monitor_to_safe,
)
from evidence_dashboard import build_evidence_dashboard, validate_planning_table
from run_demo import run_demo, verify_demo_bundle
from confirm_policy import (
    _confirmation_preflight, calibration_command, confirmation_command,
)
from audit_external_cdms import audit_external_export
from collect_external_cdms import check_v13_readiness
from external_collection import (
    allocate_prospective_cohort, append_export, close_collection,
    collection_status, materialize_collection, read_collection, seal_collection,
)
from validation_plan import (
    calibration_design, evaluation_planning_table, maximum_passing_failures,
    minimum_positive_events, pass_probability, validation_design_summary,
)
from verify_oof_evidence import verify_oof_evidence
from prefix_features import build_prefix_features, eligible_prefixes
from robustness import subgroup_metrics
from shift_gate import ConformalShiftGate
from triage import Decision, SequentialTriagePolicy
from confirmation import (
    POLICY, acquire_confirmation_lock, attach_event_labels, calibrate, evaluate,
    confirmation_status_path, policy_from_model_manifest, prepare_prefix_scores,
    read_confirmation_status, write_confirmation_status_sidecar, write_json,
)
from snapshot_model import (
    assert_disjoint_splits, event_equal_weights, fit_snapshot_model,
    prepare_snapshot_frame, score_snapshot_model,
)
from partitions import event_labels, split_event_ids
from study import (
    file_sha256, freeze_study, read_locked_study, validate_feature_cohort,
    validate_label_roster, validate_scored_cohort_roster,
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


def external_cdm_record(message_id, creation_date, tca, probability=1e-7):
    return {
        "CDM_ID": message_id,
        "MESSAGE_ID": message_id,
        "CREATION_DATE": creation_date,
        "TCA": tca,
        "MISS_DISTANCE": "1200",
        "MISS_DISTANCE_UNIT": "m",
        "RELATIVE_SPEED": "10123",
        "RELATIVE_SPEED_UNIT": "m/s",
        "COLLISION_PROBABILITY": str(probability),
        "COLLISION_MAX_PROBABILITY": str(probability * 2),
        "COLLISION_MAX_PC_SCALE_FACTOR": "1.4",
        "SAT1_OBJECT_DESIGNATOR": "25544",
        "SAT2_OBJECT_DESIGNATOR": "90001",
        "SAT2_OBJECT_TYPE": "DEBRIS",
        "SAT1_OBS_AVAILABLE": "120",
        "SAT1_OBS_USED": "118",
        "SAT1_WEIGHTED_RMS": "0.8",
        "SAT2_OBS_AVAILABLE": "80",
        "SAT2_OBS_USED": "75",
        "SAT2_WEIGHTED_RMS": "1.1",
        "RELATIVE_POSITION_R": "100",
        "RELATIVE_POSITION_T": "200",
        "RELATIVE_POSITION_N": "50",
        "SAT1_CR_R": "100", "SAT1_CT_R": "5", "SAT1_CT_T": "120",
        "SAT1_CN_R": "2", "SAT1_CN_T": "3", "SAT1_CN_N": "80",
        "SAT2_CR_R": "90", "SAT2_CT_R": "4", "SAT2_CT_T": "110",
        "SAT2_CN_R": "1", "SAT2_CN_T": "2", "SAT2_CN_N": "70",
    }


def test_external_cdm_adapter_builds_causal_v13_feature_rows():
    records = [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z", 2e-7),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z", 3e-6),
    ]
    frame = adapt_external_cdms(records, tca_tolerance_minutes=30)

    assert frame["event_id"].nunique() == 1
    assert frame["source_message_id"].tolist() == ["m1", "m2", "m3"]
    assert np.allclose(frame["time_to_tca"], [6.0, 5 + 10/1440, 4 + 20/1440])
    assert np.isfinite(frame["mahalanobis_distance"]).all()
    assert (frame["t_position_covariance_det"] > 0).all()
    assert (frame["c_position_covariance_det"] > 0).all()
    assert frame["mission_id"].eq("25544").all()
    prepared = prepare_dynamic_frame(frame, require_labels=False)
    assert len(prepared) == 3
    assert prepared["eligible_history_count"].tolist() == [1, 2, 3]


def test_external_cdm_event_grouping_splits_distant_tcas():
    records = [
        external_cdm_record("near", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("far", "2026-01-01T00:00:00Z", "2026-01-07T02:00:00Z"),
    ]
    frame = adapt_external_cdms(records, tca_tolerance_minutes=30)
    assert frame["event_id"].nunique() == 2


def test_event_grouping_review_flags_chained_tca_drift():
    records = [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-01T01:00:00Z", "2026-01-07T00:20:00Z"),
        external_cdm_record("m3", "2026-01-01T02:00:00Z", "2026-01-07T00:40:00Z"),
    ]
    frame = adapt_external_cdms(records, tca_tolerance_minutes=30)

    review = event_grouping_review(frame, tca_tolerance_minutes=30)

    assert frame["event_id"].nunique() == 1
    assert review["manual_review_required"] is True
    assert review["flagged_events"] == 1
    assert review["flags"][0]["tca_span_minutes"] == 40.0
    assert review["flags"][0]["reasons"] == [
        "within_event_tca_span_exceeds_grouping_tolerance"
    ]
    report = readiness_report(frame)
    assert report["scientific_status"] == "manual-event-grouping-review-required"


def test_event_grouping_review_clears_compact_cluster():
    records = [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z"),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z"),
    ]
    review = event_grouping_review(adapt_external_cdms(records))
    assert review["manual_review_required"] is False
    assert review["flagged_events"] == 0
    assert review["flags"] == []


def test_external_cdm_features_and_labels_are_separated():
    records = [
        external_cdm_record("early", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", 1e-8),
        external_cdm_record("late", "2026-01-06T12:00:00Z", "2026-01-07T00:10:00Z", 2e-5),
    ]
    frame = adapt_external_cdms(records, tca_tolerance_minutes=30)
    features = outcome_blind_features(frame, min_days=2.0)
    assert features["source_message_id"].tolist() == ["early"]
    with np.testing.assert_raises(ValueError):
        derive_event_labels(frame)
    labels = derive_event_labels(frame, collection_complete=True)
    assert labels["y"].tolist() == [1]


def test_external_cdm_readiness_reports_sequential_and_positive_shortfall():
    records = [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", 1e-7),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z", 2e-7),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z", 2e-5),
    ]
    report = readiness_report(
        adapt_external_cdms(records), collection_complete=True
    )
    assert report["events"] == 1
    assert report["events_eligible_minimum_history"] == 1
    assert report["collection_complete_attested"] is True
    assert report["positive_counts_suppressed"] is False
    assert report["provisional_positive_events"] == 1
    assert report["provisional_total_positive_target_met"] is False
    assert report["scientific_status"] == "candidate-collection-only"
    assert len(report["limitations"]) == 3


def test_readiness_report_suppresses_terminal_risk_counts_by_default():
    records = [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", 2e-5
        ),
        external_cdm_record(
            "m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z", 2e-5
        ),
        external_cdm_record(
            "m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z", 2e-5
        ),
    ]
    report = readiness_report(adapt_external_cdms(records))

    assert report["collection_complete_attested"] is False
    assert report["positive_counts_suppressed"] is True
    assert report["provisionally_labelled_events"] is None
    assert report["provisional_positive_events"] is None
    assert report["positive_rate"] is None
    assert report["provisional_calibration_positive_target_met"] is None
    assert report["provisional_evaluation_positive_target_met"] is None
    assert report["provisional_total_positive_target_met"] is None


def test_readiness_report_uses_literal_terminal_row_after_attestation():
    frame = pd.DataFrame({
        "event_id": ["event", "event"],
        "time_to_tca": [1.0, 5.0],
        "risk": [np.nan, -4.0],
        **{
            column: [1.0, 1.0]
            for column in REQUIRED_EXTERNAL_FEATURES
            if column not in {"risk"}
        },
    })
    frame["event_pair"] = "1|2"
    frame["tca"] = pd.Timestamp("2026-01-07T00:00:00Z")

    report = readiness_report(frame, collection_complete=True)

    assert report["provisionally_labelled_events"] == 0
    assert report["events_without_finite_final_pc"] == 1
    assert report["provisional_positive_events"] == 0


def test_external_cdm_parser_accepts_array_and_ndjson():
    first = external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    second = external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z")
    assert len(parse_cdm_json(json.dumps([first, second]))) == 2
    assert len(parse_cdm_json(json.dumps(first) + "\n" + json.dumps(second))) == 2


def test_external_cdm_inline_json_is_not_resolved_as_a_path(monkeypatch):
    inline = json.dumps(external_cdm_record(
        "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
    ))

    def unexpected_exists(_path):
        raise AssertionError("inline JSON must not be checked as a filesystem path")

    monkeypatch.setattr(Path, "exists", unexpected_exists)
    assert parse_cdm_source(inline)[0]["CDM_ID"] == "m1"


def cdm_kvn(message_id, creation_date, tca, probability=1e-7):
    return f"""CCSDS_CDM_VERS = 1.0
CREATION_DATE = {creation_date}
MESSAGE_ID = {message_id}
TCA = {tca}
MISS_DISTANCE = 1200 [m]
RELATIVE_SPEED = 10123 [m/s]
COLLISION_PROBABILITY = {probability}
COLLISION_MAX_PROBABILITY = {probability * 2}
COLLISION_MAX_PC_SCALE_FACTOR = 1.4
RELATIVE_POSITION_R = 100 [m]
RELATIVE_POSITION_T = 200 [m]
RELATIVE_POSITION_N = 50 [m]
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 25544
OBS_AVAILABLE = 120
OBS_USED = 118
WEIGHTED_RMS = 0.8
CR_R = 100
CT_R = 5
CT_T = 120
CN_R = 2
CN_T = 3
CN_N = 80
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 90001
OBJECT_TYPE = DEBRIS
OBS_AVAILABLE = 80
OBS_USED = 75
WEIGHTED_RMS = 1.1
CR_R = 90
CT_R = 4
CT_T = 110
CN_R = 1
CN_T = 2
CN_N = 70
"""


def test_external_cdm_parser_accepts_standard_kvn_and_object_sections():
    records = parse_cdm_source(
        cdm_kvn("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
        + cdm_kvn("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z")
    )
    assert len(records) == 2
    assert records[0]["SAT1_OBJECT_DESIGNATOR"] == "25544"
    assert records[0]["SAT2_OBJECT_TYPE"] == "DEBRIS"
    assert records[0]["MISS_DISTANCE_UNIT"] == "m"
    frame = adapt_external_cdms(records)
    assert frame["event_id"].nunique() == 1
    assert frame["source_message_id"].tolist() == ["m1", "m2"]
    assert np.isfinite(frame["mahalanobis_distance"]).all()


def test_external_cdm_parser_rejects_invalid_or_duplicate_kvn_fields():
    with np.testing.assert_raises(ValueError):
        parse_cdm_source("CCSDS_CDM_VERS = 1.0\nBROKEN LINE")
    duplicate = cdm_kvn(
        "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
    ).replace("TCA = 2026", "TCA = 2026\nTCA = 2026", 1)
    with np.testing.assert_raises(ValueError):
        parse_cdm_source(duplicate)


def test_external_cdm_adapter_rejects_duplicates_and_wrong_units():
    record = external_cdm_record("same", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    with np.testing.assert_raises(ValueError):
        adapt_external_cdms([record, record])
    wrong = dict(record, MESSAGE_ID="other", MISS_DISTANCE_UNIT="km")
    with np.testing.assert_raises(ValueError):
        adapt_external_cdms([wrong])


def test_external_cdm_audit_cli_writes_outcome_blind_features(tmp_path):
    source = tmp_path / "external.json"
    records = [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z"),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z", 2e-5),
    ]
    source.write_text(json.dumps(records), encoding="utf-8")
    features = tmp_path / "features.parquet"
    readiness = tmp_path / "readiness.json"

    report = audit_external_export(source, features, readiness)

    stored = pd.read_parquet(features)
    assert "y" not in stored.columns
    assert stored["time_to_tca"].ge(2.0).all()
    assert report["events_eligible_minimum_history"] == 1
    assert report["labels_written"] == 0
    assert json.loads(readiness.read_text(encoding="utf-8"))["source"]["messages"] == 3


def test_external_cdm_audit_requires_collection_complete_for_labels(tmp_path):
    source = tmp_path / "external.json"
    source.write_text(json.dumps([
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    ]), encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        audit_external_export(
            source,
            tmp_path / "features.parquet",
            tmp_path / "readiness.json",
            labels_output=tmp_path / "labels.parquet",
        )
    assert not list(tmp_path.glob("*.parquet"))
    assert not (tmp_path / "readiness.json").exists()


def test_external_cdm_audit_writes_labels_after_completion_attestation(tmp_path):
    source = tmp_path / "external.json"
    source.write_text(json.dumps([
        external_cdm_record("early", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", 1e-8),
        external_cdm_record("late", "2026-01-06T12:00:00Z", "2026-01-07T00:10:00Z", 2e-5),
    ]), encoding="utf-8")
    labels = tmp_path / "labels.parquet"
    report = audit_external_export(
        source,
        tmp_path / "features.parquet",
        tmp_path / "readiness.json",
        labels_output=labels,
        collection_complete=True,
    )
    assert report["labels_written"] == 1
    assert pd.read_parquet(labels)["y"].tolist() == [1]


def write_external_export(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_external_collection_appends_exports_and_preserves_event_id_under_tca_drift(tmp_path):
    first = write_external_export(tmp_path / "first.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z"),
    ])
    second = write_external_export(tmp_path / "second.json", [
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z"),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:25:00Z"),
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    append_export(first, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    result = append_export(second, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    stored, complete = read_collection(ledger)

    assert result["messages"] == 3
    assert result["events"] == 1
    assert result["batches"][1]["duplicate_messages"] == 1
    assert complete["event_id"].nunique() == 1
    assert complete["source_message_id"].tolist() == ["m1", "m2", "m3"]
    assert stored["batch_chain_head"] == stored["batches"][-1]["entry_sha256"]
    assert stored["batches"][1]["previous_entry_sha256"] == stored["batches"][0]["entry_sha256"]


def test_external_collection_rejects_duplicate_source_and_conflicting_message(tmp_path):
    original = external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    first = write_external_export(tmp_path / "first.json", [original])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    append_export(first, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    with np.testing.assert_raises(ValueError):
        append_export(first, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")

    conflict = dict(original, COLLISION_PROBABILITY="0.001")
    second = write_external_export(tmp_path / "second.json", [conflict])
    with np.testing.assert_raises(ValueError):
        append_export(second, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    _, complete = read_collection(ledger)
    assert len(complete) == 1


def test_append_export_recovers_verified_orphan_after_hard_crash(tmp_path, monkeypatch):
    import external_collection as collection_module

    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    common = dict(
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    real_atomic_json = collection_module._atomic_json

    def hard_crash(_payload, _path):
        raise SystemExit("simulated hard crash")

    monkeypatch.setattr(collection_module, "_atomic_json", hard_crash)
    with np.testing.assert_raises(SystemExit):
        append_export(source, ledger, batches, **common)
    orphan = next(batches.glob("batch-*.parquet"))
    orphan_sha256 = file_sha256(orphan)
    assert not ledger.exists()

    monkeypatch.setattr(collection_module, "_atomic_json", real_atomic_json)
    result = append_export(source, ledger, batches, **common)
    _, complete = read_collection(ledger)

    assert len(result["batches"]) == 1
    assert result["batches"][0]["recovered_orphan_batch"] is True
    assert result["batches"][0]["artifact_sha256"] == orphan_sha256
    assert file_sha256(orphan) == orphan_sha256
    assert complete["source_message_id"].tolist() == ["m1"]
    assert len(list(batches.glob("batch-*.parquet"))) == 1


def test_append_export_rejects_mismatched_orphan_without_modifying_it(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    batches.mkdir()
    source_sha256 = file_sha256(source)
    orphan = batches / f"batch-000001-{source_sha256[:12]}.parquet"
    pd.DataFrame({"source_message_id": ["other"], "risk": [-4.0]}).to_parquet(
        orphan, index=False
    )
    original_sha256 = file_sha256(orphan)

    with np.testing.assert_raises_regex(FileExistsError, "differs from expected"):
        append_export(
            source, ledger, batches,
            collection_start_utc="2026-01-01T00:00:00Z",
            collection_end_utc="2026-02-01T00:00:00Z",
        )

    assert not ledger.exists()
    assert orphan.exists()
    assert file_sha256(orphan) == original_sha256


def test_recovered_orphan_survives_retryable_ledger_failure(tmp_path, monkeypatch):
    import external_collection as collection_module

    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    common = dict(
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    real_atomic_json = collection_module._atomic_json
    monkeypatch.setattr(
        collection_module, "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with np.testing.assert_raises(SystemExit):
        append_export(source, ledger, batches, **common)
    orphan = next(batches.glob("batch-*.parquet"))
    orphan_sha256 = file_sha256(orphan)

    monkeypatch.setattr(
        collection_module, "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with np.testing.assert_raises(OSError):
        append_export(source, ledger, batches, **common)

    assert not ledger.exists()
    assert orphan.exists()
    assert file_sha256(orphan) == orphan_sha256

    monkeypatch.setattr(collection_module, "_atomic_json", real_atomic_json)
    result = append_export(source, ledger, batches, **common)
    assert result["batches"][0]["recovered_orphan_batch"] is True


def test_concurrent_append_exports_preserve_all_batches(tmp_path):
    import threading

    sources = []
    for index in range(3):
        record = external_cdm_record(
            f"m{index}",
            f"2026-01-0{index + 1}T00:00:00Z",
            f"2026-01-{7 + index:02d}T00:00:00Z",
        )
        record["SAT2_OBJECT_DESIGNATOR"] = str(90001 + index)
        sources.append(
            write_external_export(tmp_path / f"source-{index}.json", [record])
        )
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    common = dict(
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    barrier = threading.Barrier(len(sources))
    errors = []

    def worker(source):
        try:
            barrier.wait()
            append_export(source, ledger, batches, **common)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(source,)) for source in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    stored, complete = read_collection(ledger)
    assert [batch["batch"] for batch in stored["batches"]] == [1, 2, 3]
    assert stored["batch_chain_head"] == stored["batches"][-1]["entry_sha256"]
    assert complete["source_message_id"].nunique() == 3
    assert len(list(batches.glob("batch-*.parquet"))) == 3


def test_concurrent_duplicate_append_commits_once(tmp_path):
    import threading

    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    common = dict(
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    barrier = threading.Barrier(3)
    successes = []
    failures = []

    def worker():
        barrier.wait()
        try:
            append_export(source, ledger, batches, **common)
            successes.append(True)
        except ValueError as error:
            failures.append(str(error))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 2
    assert all("already been appended" in error for error in failures)
    stored, complete = read_collection(ledger)
    assert len(stored["batches"]) == 1
    assert len(complete) == 1


def test_ledger_lock_is_released_after_exception(tmp_path):
    import external_collection as collection_module

    ledger = tmp_path / "collection.json"
    with np.testing.assert_raises(RuntimeError):
        with collection_module.ledger_lock(ledger):
            raise RuntimeError("simulated failure")

    with collection_module.ledger_lock(ledger):
        assert (tmp_path / "collection.json.lock").exists()


def test_ledger_lock_uses_windows_byte_range_lock(monkeypatch, tmp_path):
    import external_collection as collection_module

    calls = []

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(file_descriptor, mode, length):
            calls.append((file_descriptor, mode, length))

    monkeypatch.setattr(collection_module, "fcntl", None)
    monkeypatch.setattr(collection_module, "msvcrt", FakeMsvcrt)

    ledger = tmp_path / "collection.json"
    with collection_module.ledger_lock(ledger):
        lock_path = tmp_path / "collection.json.lock"
        assert lock_path.read_bytes() == b"\0"
        assert calls[-1][1:] == (FakeMsvcrt.LK_LOCK, 1)

    assert [call[1:] for call in calls] == [
        (FakeMsvcrt.LK_LOCK, 1),
        (FakeMsvcrt.LK_UNLCK, 1),
    ]


def test_ledger_lock_rejects_platform_without_locking_api(monkeypatch, tmp_path):
    import external_collection as collection_module

    monkeypatch.setattr(collection_module, "fcntl", None)
    monkeypatch.setattr(collection_module, "msvcrt", None)
    with np.testing.assert_raises_regex(RuntimeError, "unavailable on this platform"):
        with collection_module.ledger_lock(tmp_path / "collection.json"):
            pass


def test_external_collection_verifies_source_and_batch_lineage(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    result = append_export(source, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    batch_path = ledger.parent / result["batches"][0]["artifact_path"]
    batch = pd.read_parquet(batch_path)
    batch.loc[0, "risk"] = -3.0
    batch.to_parquet(batch_path, index=False)
    with np.testing.assert_raises(ValueError):
        read_collection(ledger)


def test_external_collection_rejects_tampered_hash_chain(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    ])
    ledger = tmp_path / "collection.json"
    append_export(source, ledger, tmp_path / "batches", collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["batches"][0]["accepted_messages"] = 9
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        read_collection(ledger)


def test_external_collection_status_is_outcome_blind_and_cohort_specific(tmp_path):
    records = []
    for index in range(1, 25):
        for message_index in range(3):
            creation_day = 1 + message_index
            record = external_cdm_record(
                f"m{index}-{message_index}",
                f"2026-01-{creation_day:02d}T00:00:00Z",
                f"2026-01-{7 + index % 10:02d}T00:00:00Z",
                probability=2e-5 if message_index == 2 else 1e-8,
            )
            record["SAT2_OBJECT_DESIGNATOR"] = str(90000 + index)
            records.append(record)
    source = write_external_export(tmp_path / "source.json", records)
    ledger_path = tmp_path / "collection.json"
    append_export(
        source, ledger_path, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
        allocation_seed=42,
        calibration_fraction=0.5,
    )

    status = collection_status(ledger_path)
    ledger, complete = read_collection(ledger_path)

    assert status["integrity_verified"] is True
    assert status["status"] == "collecting"
    assert status["outcomes_accessed"] is False
    assert "positive_events" not in status
    assert status["messages"] == len(complete)
    assert status["events"] == complete["event_id"].nunique()
    window = complete.loc[complete["time_to_tca"].between(2.0, 7.0, inclusive="both")]
    expected_eligible = int((window.groupby("event_id").size() >= 3).sum())
    assert status["events_eligible_minimum_history"] == expected_eligible
    cohorts = status["allocation"]["cohorts"]
    assert cohorts["calibration"]["assigned_events"] > 0
    assert cohorts["evaluation"]["assigned_events"] > 0
    assert sum(item["assigned_events"] for item in cohorts.values()) == status["events"]
    assert status["allocation"]["assignments_sha256"] == ledger["allocation"]["assignments_sha256"]


def test_external_collection_status_rejects_tampered_batch(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    result = append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    batch_path = ledger.parent / result["batches"][0]["artifact_path"]
    batch = pd.read_parquet(batch_path)
    batch.loc[0, "risk"] = -3.0
    batch.to_parquet(batch_path, index=False)

    with np.testing.assert_raises(ValueError):
        collection_status(ledger)


def test_external_collection_status_validates_window_parameters(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    with np.testing.assert_raises(ValueError):
        collection_status(ledger, minimum_history=0)
    with np.testing.assert_raises(ValueError):
        collection_status(ledger, min_days=7.0, max_days=2.0)


def test_external_collection_snapshot_is_outcome_blind_and_source_bound(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z"),
        external_cdm_record("m3", "2026-01-03T00:00:00Z", "2026-01-07T00:20:00Z", 2e-5),
    ])
    ledger = tmp_path / "collection.json"
    append_export(source, ledger, tmp_path / "batches", collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z")
    features = tmp_path / "features.parquet"
    readiness = tmp_path / "readiness.json"
    report = materialize_collection(ledger, features, readiness)
    frame = pd.read_parquet(features)

    assert "y" not in frame.columns
    assert frame["time_to_tca"].ge(2.0).all()
    assert report["collection"]["batches"] == 1
    assert len(report["collection"]["source_sha256s"]) == 1
    assert report["collection_complete_attested"] is False
    assert report["positive_counts_suppressed"] is True
    assert report["provisional_positive_events"] is None
    assert report["provisional_total_positive_target_met"] is None
    assert len(report["collection"]["ledger_sha256"]) == 64
    assert report["features_sha256"] == __import__("hashlib").sha256(features.read_bytes()).hexdigest()


def test_external_collection_close_requires_frozen_study(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("early", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", 1e-8),
        external_cdm_record("late", "2026-01-06T12:00:00Z", "2026-01-07T00:10:00Z", 2e-5),
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    append_export(
        source, ledger, batches,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    with np.testing.assert_raises(ValueError):
        close_collection(
            ledger, tmp_path / "labels.parquet",
            study_manifest=tmp_path / "missing-study.json",
            study_lock=tmp_path / "missing-study.lock",
            calibration_labels_output=tmp_path / "calibration-labels.parquet",
            evaluation_labels_output=tmp_path / "evaluation-labels.parquet",
        )
    seal_collection(ledger)
    with np.testing.assert_raises(FileNotFoundError):
        close_collection(
            ledger, tmp_path / "labels.parquet",
            study_manifest=tmp_path / "missing-study.json",
            study_lock=tmp_path / "missing-study.lock",
            calibration_labels_output=tmp_path / "calibration-labels.parquet",
            evaluation_labels_output=tmp_path / "evaluation-labels.parquet",
        )
    assert not (tmp_path / "labels.parquet").exists()



def test_external_collection_locks_tca_tolerance(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    append_export(source, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z", tca_tolerance_minutes=30)
    second = write_external_export(tmp_path / "second.json", [
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z")
    ])
    with np.testing.assert_raises(ValueError):
        append_export(second, ledger, batches, collection_start_utc="2026-01-01T00:00:00Z", collection_end_utc="2026-02-01T00:00:00Z", tca_tolerance_minutes=60)


def test_external_collection_records_and_locks_collection_period(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2026-01-05T00:00:00Z", "2026-01-10T00:00:00Z")
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    result = append_export(
        source, ledger, batches,
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    assert result["collection_period"] == {
        "start_utc": "2026-01-01T00:00:00+00:00",
        "end_utc": "2026-02-01T00:00:00+00:00",
    }
    second = write_external_export(tmp_path / "second.json", [
        external_cdm_record("m2", "2026-01-06T00:00:00Z", "2026-01-10T00:10:00Z")
    ])
    with np.testing.assert_raises(ValueError):
        append_export(
            second, ledger, batches,
            collection_start_utc="2026-01-02T00:00:00Z",
            collection_end_utc="2026-02-01T00:00:00Z",
        )


def test_external_collection_rejects_messages_outside_frozen_period(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record("m1", "2025-12-31T23:00:00Z", "2026-01-07T00:00:00Z")
    ])
    ledger = tmp_path / "collection.json"
    with np.testing.assert_raises(ValueError):
        append_export(
            source, ledger, tmp_path / "batches",
            collection_start_utc="2026-01-01T00:00:00Z",
            collection_end_utc="2026-02-01T00:00:00Z",
        )
    assert not ledger.exists()
    assert not (tmp_path / "batches").exists()


def test_prospective_allocator_is_deterministic_and_label_blind():
    assignments = [
        allocate_prospective_cohort(f"event-{index}", 42, 1 / 3)
        for index in range(1000)
    ]
    repeated = [
        allocate_prospective_cohort(f"event-{index}", 42, 1 / 3)
        for index in range(1000)
    ]
    assert assignments == repeated
    calibration = assignments.count("calibration")
    assert 280 < calibration < 390
    assert assignments != [
        allocate_prospective_cohort(f"event-{index}", 43, 1 / 3)
        for index in range(1000)
    ]


def test_prospective_assignment_is_persisted_before_outcomes(tmp_path):
    first = write_external_export(tmp_path / "first.json", [
        external_cdm_record("a1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"),
        dict(external_cdm_record("b1", "2026-01-01T00:00:00Z", "2026-01-08T00:00:00Z"), SAT2_OBJECT_DESIGNATOR="90002"),
    ])
    ledger_path = tmp_path / "collection.json"
    ledger = append_export(
        first, ledger_path, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
        allocation_seed=99,
        calibration_fraction=0.5,
    )
    assert ledger["status"] == "collecting"
    assert ledger["allocation"]["assigned_before_outcome_access"] is True
    assert set(ledger["allocation"]["assignments"]) == set(
        read_collection(ledger_path)[1]["event_id"].astype(str)
    )
    assert len(ledger["allocation"]["assignments_sha256"]) == 64


def test_prospective_allocation_parameters_are_locked(tmp_path):
    first = write_external_export(tmp_path / "first.json", [
        external_cdm_record("m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z")
    ])
    second = write_external_export(tmp_path / "second.json", [
        external_cdm_record("m2", "2026-01-02T00:00:00Z", "2026-01-07T00:10:00Z")
    ])
    ledger = tmp_path / "collection.json"
    batches = tmp_path / "batches"
    common = dict(
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
        allocation_seed=7,
        calibration_fraction=0.4,
    )
    append_export(first, ledger, batches, **common)
    with np.testing.assert_raises(ValueError):
        append_export(second, ledger, batches, **{**common, "allocation_seed": 8})
    with np.testing.assert_raises(ValueError):
        append_export(second, ledger, batches, **{**common, "calibration_fraction": 0.5})


def _prospective_two_cohort_collection(tmp_path):
    records = []
    for index in range(1, 25):
        record = external_cdm_record(
            f"m{index}", "2026-01-01T00:00:00Z", f"2026-01-{7 + index % 10:02d}T00:00:00Z"
        )
        record["SAT2_OBJECT_DESIGNATOR"] = str(90000 + index)
        records.append(record)
    source = write_external_export(tmp_path / "source.json", records)
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
        allocation_seed=42,
        calibration_fraction=0.5,
    )
    seal_collection(ledger)
    return ledger


def test_materialize_collection_writes_disjoint_label_blind_cohorts(tmp_path):
    ledger = _prospective_two_cohort_collection(tmp_path)
    outputs = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }
    report = materialize_collection(ledger, **outputs)
    calibration = pd.read_parquet(outputs["calibration_features"])
    evaluation = pd.read_parquet(outputs["evaluation_features"])
    cal_roster = pd.read_parquet(outputs["calibration_roster"])
    eval_roster = pd.read_parquet(outputs["evaluation_roster"])
    allocation = json.loads(outputs["allocation_output"].read_text(encoding="utf-8"))

    assert "y" not in calibration and "y" not in evaluation
    assert set(calibration["event_id"]).isdisjoint(set(evaluation["event_id"]))
    assert set(cal_roster["event_id"]).isdisjoint(set(eval_roster["event_id"]))
    assert set(cal_roster["event_id"]) | set(eval_roster["event_id"]) == set(
        read_collection(ledger)[1]["event_id"]
    )
    assert allocation["status"] == "assigned-before-outcome-access"
    assert allocation["outcomes_accessed"] is False
    assert report["prospective_allocation"]["assignments_sha256"] == allocation["assignments_sha256"]


def test_materialize_collection_blocks_ambiguous_event_grouping(tmp_path):
    records = [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        ),
        external_cdm_record(
            "m2", "2026-01-01T01:00:00Z", "2026-01-07T00:20:00Z"
        ),
        external_cdm_record(
            "m3", "2026-01-01T02:00:00Z", "2026-01-07T00:40:00Z"
        ),
    ]
    for index, record in enumerate(records, start=1):
        record["SAT2_OBJECT_DESIGNATOR"] = "90001"
    source = write_external_export(tmp_path / "source.json", records)
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
        allocation_seed=42,
        calibration_fraction=0.5,
    )
    seal_collection(ledger)
    outputs = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }

    with np.testing.assert_raises_regex(
        ValueError, "blocked by ambiguous event grouping"
    ):
        materialize_collection(ledger, **outputs)

    assert not any(path.exists() for path in outputs.values())


def test_materialize_collection_allows_audit_only_ambiguous_snapshot(tmp_path):
    records = [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        ),
        external_cdm_record(
            "m2", "2026-01-01T01:00:00Z", "2026-01-07T00:20:00Z"
        ),
        external_cdm_record(
            "m3", "2026-01-01T02:00:00Z", "2026-01-07T00:40:00Z"
        ),
    ]
    source = write_external_export(tmp_path / "source.json", records)
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )
    features = tmp_path / "features.parquet"
    readiness = tmp_path / "readiness.json"

    report = materialize_collection(ledger, features, readiness)

    assert features.exists() and readiness.exists()
    assert report["event_grouping"]["manual_review_required"] is True
    assert report["scientific_status"] == "manual-event-grouping-review-required"


def test_freeze_study_uses_denominator_rosters_and_allocation_lineage(tmp_path):
    ledger = _prospective_two_cohort_collection(tmp_path)
    paths = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }
    materialize_collection(ledger, **paths)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    manifest_path, lock_path = tmp_path / "study.json", tmp_path / "study.lock"
    manifest = freeze_study(
        paths["calibration_features"], paths["evaluation_features"],
        preregistration, preregistration_lock, manifest_path, lock_path,
        calibration_roster=paths["calibration_roster"],
        evaluation_roster=paths["evaluation_roster"],
        allocation_manifest=paths["allocation_output"],
    )
    assert manifest["schema_version"] == 4
    assert len(manifest["allocation"]["sha256"]) == 64
    assert manifest["cohorts"]["calibration"]["events"] == len(
        pd.read_parquet(paths["calibration_roster"])
    )
    validate_feature_cohort(
        paths["calibration_features"], manifest_path, lock_path, "calibration"
    )


def test_prospective_collection_cannot_be_sealed_before_period_end(tmp_path):
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2100-01-01T00:00:00Z",
    )
    with np.testing.assert_raises(ValueError):
        seal_collection(ledger)
    assert read_collection(ledger)[0]["status"] == "collecting"


def test_freeze_study_rejects_tampered_allocation_rule(tmp_path):
    ledger = _prospective_two_cohort_collection(tmp_path)
    paths = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }
    materialize_collection(ledger, **paths)
    allocation = json.loads(paths["allocation_output"].read_text(encoding="utf-8"))
    allocation["seed"] += 1
    paths["allocation_output"].write_text(json.dumps(allocation), encoding="utf-8")
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    with np.testing.assert_raises(ValueError):
        freeze_study(
            paths["calibration_features"], paths["evaluation_features"],
            preregistration, preregistration_lock,
            tmp_path / "study.json", tmp_path / "study.lock",
            calibration_roster=paths["calibration_roster"],
            evaluation_roster=paths["evaluation_roster"],
            allocation_manifest=paths["allocation_output"],
        )
    assert not (tmp_path / "study.json").exists()
    assert not (tmp_path / "study.lock").exists()


def test_full_prospective_lifecycle_freezes_split_before_labels(tmp_path):
    ledger = _prospective_two_cohort_collection(tmp_path)
    paths = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }
    materialize_collection(ledger, **paths)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    study_path, study_lock = tmp_path / "study.json", tmp_path / "study.lock"
    manifest = freeze_study(
        paths["calibration_features"], paths["evaluation_features"],
        preregistration, preregistration_lock, study_path, study_lock,
        calibration_roster=paths["calibration_roster"],
        evaluation_roster=paths["evaluation_roster"],
        allocation_manifest=paths["allocation_output"],
    )
    labels = tmp_path / "labels.parquet"
    calibration_labels = tmp_path / "calibration-labels.parquet"
    evaluation_labels = tmp_path / "evaluation-labels.parquet"
    closed = close_collection(
        ledger, labels,
        study_manifest=study_path,
        study_lock=study_lock,
        calibration_labels_output=calibration_labels,
        evaluation_labels_output=evaluation_labels,
    )

    cal = pd.read_parquet(calibration_labels)
    evaluation = pd.read_parquet(evaluation_labels)
    assert closed["status"] == "closed"
    assert set(cal["event_id"].astype(str)) == set(
        manifest["cohorts"]["calibration"]["event_ids"]
    )
    assert set(evaluation["event_id"].astype(str)) == set(
        manifest["cohorts"]["evaluation"]["event_ids"]
    )
    assert set(cal["event_id"]).isdisjoint(set(evaluation["event_id"]))
    assert len(cal) + len(evaluation) == len(pd.read_parquet(labels))
    assert closed["study_manifest_sha256"] == file_sha256(study_path)
    with np.testing.assert_raises(ValueError):
        append_export(
            tmp_path / "source.json", ledger, tmp_path / "batches",
            collection_start_utc="2026-01-01T00:00:00Z",
            collection_end_utc="2026-02-01T00:00:00Z",
        )


def test_denominator_roster_can_include_event_without_feature_rows(tmp_path):
    calibration_features = tmp_path / "calibration-features.parquet"
    evaluation_features = tmp_path / "evaluation-features.parquet"
    pd.DataFrame({"event_id": ["cal-feature"], "time_to_tca": [6.0]}).to_parquet(calibration_features)
    pd.DataFrame({"event_id": ["eval-feature"], "time_to_tca": [6.0]}).to_parquet(evaluation_features)
    calibration_roster = tmp_path / "calibration-roster.parquet"
    evaluation_roster = tmp_path / "evaluation-roster.parquet"
    pd.DataFrame({"event_id": ["cal-feature", "cal-no-history"]}).to_parquet(calibration_roster)
    pd.DataFrame({"event_id": ["eval-feature"]}).to_parquet(evaluation_roster)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    manifest = freeze_study(
        calibration_features, evaluation_features,
        preregistration, preregistration_lock,
        tmp_path / "study.json", tmp_path / "study.lock",
        calibration_roster=calibration_roster,
        evaluation_roster=evaluation_roster,
    )
    assert manifest["cohorts"]["calibration"]["events"] == 2
    assert manifest["cohorts"]["calibration"]["feature_events"] == 1
    assert manifest["cohorts"]["calibration"]["no_feature_events"] == 1
    validate_label_roster(
        pd.DataFrame({"event_id": ["cal-feature", "cal-no-history"], "y": [0, 1]}),
        manifest,
        "calibration",
    )


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


def test_event_policy_table_rejects_invalid_numeric_inputs():
    frame = sample().rename(columns={"risk": "score"})
    for column, value in (("time_to_tca", -1.0), ("time_to_tca", np.inf), ("score", np.nan)):
        invalid = frame.copy()
        invalid.loc[0, column] = value
        with np.testing.assert_raises(ValueError):
            event_policy_table(invalid, "score")


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


def test_calibration_design_matches_frozen_v13_claim():
    design = calibration_design(100, alpha=0.10, confidence=0.95)

    assert design["rank"] == 5
    assert design["finite_threshold_available"] is True
    assert 0.0891 < design["pac_bound"] < 0.0893
    assert calibration_design(28)["finite_threshold_available"] is False
    assert calibration_design(29)["rank"] == 1


def test_validation_design_summary_matches_frozen_v13_claims():
    design = validation_design_summary(100, 200)

    assert design["calibration"]["rank"] == 5
    assert design["evaluation"]["maximum_passing_dangerous_exclusions"] == 12
    assert 0.79 < design["evaluation"]["pass_probability_at_assumed_rate"] < 0.80


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


def test_runtime_rejects_non_finite_safe_threshold():
    for threshold in (np.nan, np.inf, -np.inf):
        with np.testing.assert_raises(ValueError):
            SequentialTriagePolicy(threshold)


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
    metrics = result["evaluation"]
    assert result["evaluation_events"] == 42
    assert metrics["danger_n"] == 40
    assert metrics["negative_n"] == 2
    assert metrics["calibration_rank"] == calibration["calibration"]["rank"]
    assert metrics["calibration_n_positive"] == 40
    assert metrics["calibration_pac_bound"] == calibration["calibration"]["pac_bound"]
    assert metrics["calibration_marginal_bound"] == calibration["calibration"]["marginal_bound"]
    assert metrics["calibration_bound_satisfied"] is True


def test_calibration_event_roster_digest_matches_declared_ids():
    calibration = calibrate(confirmation_scores())
    from confirmation import event_id_digest
    assert calibration["calibration_event_ids_sha256"] == event_id_digest(
        pd.Series(calibration["calibration_event_ids"])
    )


def test_confirmation_rejects_tampered_calibration_event_roster():
    calibration = calibrate(confirmation_scores())
    calibration["calibration_event_ids"] = calibration["calibration_event_ids"][1:]
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_rejects_tampered_calibration_roster_digest():
    calibration = calibrate(confirmation_scores())
    calibration["calibration_event_ids_sha256"] = "0" * 64
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_rejects_duplicate_calibration_roster_ids():
    calibration = calibrate(confirmation_scores())
    calibration["calibration_event_ids"].append(
        calibration["calibration_event_ids"][0]
    )
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_rejects_event_overlap():
    calibration = calibrate(confirmation_scores())
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(), calibration)


def test_confirmation_rejects_policy_drift():
    calibration = calibrate(confirmation_scores())
    calibration["policy"]["minimum_history"] = 1
    with np.testing.assert_raises(ValueError):
        evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_rejects_tampered_calibration_rule_fields():
    mutations = {
        "rank": 2,
        "n_positive": 41,
        "alpha": 0.2,
        "mode": "marginal",
        "confidence": 0.9,
        "marginal_bound": 0.9,
        "pac_bound": 0.9,
        "threshold": float("nan"),
    }
    for field, value in mutations.items():
        calibration = calibrate(confirmation_scores())
        calibration["calibration"][field] = value
        with np.testing.assert_raises(ValueError):
            evaluate(confirmation_scores(event_offset=100), calibration)


def test_confirmation_rejects_incomplete_calibration_rule():
    calibration = calibrate(confirmation_scores())
    del calibration["calibration"]["pac_bound"]
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


def test_v13_feature_contract_matches_code_manifest_and_model():
    from catboost import CatBoostClassifier

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts" / "catboost_tail_aligned_final_v13.json").read_text(
            encoding="utf-8"
        )
    )
    model = CatBoostClassifier()
    model.load_model(
        str(root / "artifacts" / "catboost_tail_aligned_final_v13.cbm")
    )

    contract = validate_dynamic_feature_contract(model, manifest["features"])
    assert contract == DYNAMIC_FEATURES
    assert list(contract) == manifest["features"]
    assert list(contract) == model.feature_names_
    assert len(contract) == 45


def test_dynamic_scoring_rejects_model_feature_contract_drift():
    training = snapshot_training_frame()
    prepared = prepare_dynamic_frame(training)
    model = fit_dynamic_model(
        prepared, event_equal_weights(prepared), {"iterations": 5}
    )
    model.set_feature_names(list(reversed(model.feature_names_)))

    with np.testing.assert_raises_regex(ValueError, "Model feature contract"):
        score_dynamic_model(model, prepared)


def test_manifest_feature_contract_rejects_order_and_membership_drift():
    from catboost import CatBoostClassifier

    root = Path(__file__).resolve().parents[1]
    model = CatBoostClassifier()
    model.load_model(
        str(root / "artifacts" / "catboost_tail_aligned_final_v13.cbm")
    )
    reordered = list(DYNAMIC_FEATURES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with np.testing.assert_raises_regex(ValueError, "Manifest feature contract"):
        validate_dynamic_feature_contract(model, reordered)
    missing = list(DYNAMIC_FEATURES[:-1])
    with np.testing.assert_raises_regex(ValueError, "Manifest feature contract"):
        validate_dynamic_feature_contract(model, missing)


def test_tail_score_file_rejects_manifest_feature_contract_drift(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts" / "catboost_tail_aligned_final_v13.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["features"][0], manifest["features"][1] = (
        manifest["features"][1], manifest["features"][0]
    )
    manifest_path = tmp_path / "model.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    features_path = tmp_path / "features.parquet"
    snapshot_training_frame(500).drop(columns="y").to_parquet(
        features_path, index=False
    )
    output_path = tmp_path / "scores.parquet"

    with np.testing.assert_raises_regex(ValueError, "Manifest feature contract"):
        score_tail_file(
            features_path,
            root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
            manifest_path,
            output_path,
        )
    assert not output_path.exists()


def test_feature_contract_sha256_is_order_sensitive_and_stable():
    digest = feature_contract_sha256()
    assert len(digest) == 64
    int(digest, 16)
    assert digest == feature_contract_sha256(list(DYNAMIC_FEATURES))
    reordered = list(DYNAMIC_FEATURES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert feature_contract_sha256(reordered) != digest


def test_calibration_and_evaluation_bind_feature_contract():
    digest = feature_contract_sha256()
    calibration_scores = confirmation_scores().assign(
        feature_contract_sha256=digest
    )
    artifact = calibrate(calibration_scores)
    assert artifact["feature_contract_sha256"] == digest

    evaluation_scores = confirmation_scores(event_offset=100).assign(
        feature_contract_sha256=digest
    )
    result = evaluate(evaluation_scores, artifact)
    assert result["feature_contract_sha256"] == digest

    mismatched = evaluation_scores.assign(feature_contract_sha256="0" * 64)
    with np.testing.assert_raises_regex(ValueError, "different feature contracts"):
        evaluate(mismatched, artifact)


def test_confirmation_v1_style_scores_remain_backward_compatible():
    artifact = calibrate(confirmation_scores())
    assert artifact["feature_contract_sha256"] is None
    result = evaluate(confirmation_scores(event_offset=100), artifact)
    assert result["feature_contract_sha256"] is None


def test_prepare_prefix_scores_rejects_invalid_or_mixed_contract_hashes():
    frame = confirmation_scores().assign(feature_contract_sha256="not-a-digest")
    with np.testing.assert_raises_regex(ValueError, "64-character"):
        prepare_prefix_scores(frame)
    mixed = confirmation_scores().assign(feature_contract_sha256="a" * 64)
    mixed.loc[mixed.index[-1], "feature_contract_sha256"] = "b" * 64
    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        prepare_prefix_scores(mixed)


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


def _locked_preregistration(tmp_path, decision_window=(2.0, 7.0)):
    import hashlib
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text(json.dumps({
        "candidate": {"decision_window_days": list(decision_window)}
    }) + "\n", encoding="utf-8")
    lock = tmp_path / "preregistration.lock"
    lock.write_text(json.dumps({
        "preregistration_sha256": hashlib.sha256(
            preregistration.read_bytes()
        ).hexdigest()
    }) + "\n", encoding="utf-8")
    return preregistration, lock


def test_freeze_study_rejects_post_window_rows_before_lock(tmp_path):
    calibration = tmp_path / "calibration.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["cal", "cal"],
        "time_to_tca": [6.0, 1.0],
        "risk": [-8.0, -5.5],
    }).to_parquet(calibration, index=False)
    pd.DataFrame({
        "event_id": ["eval"], "time_to_tca": [6.0], "risk": [-9.0]
    }).to_parquet(evaluation, index=False)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    manifest, study_lock = tmp_path / "study.json", tmp_path / "study.lock"

    with np.testing.assert_raises(ValueError):
        freeze_study(
            calibration, evaluation, preregistration, preregistration_lock,
            manifest, study_lock,
        )

    assert not manifest.exists()
    assert not study_lock.exists()


def test_freeze_study_rejects_explicit_final_outcome_columns(tmp_path):
    calibration = tmp_path / "calibration.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["cal"], "time_to_tca": [6.0], "final_risk": [-5.5]
    }).to_parquet(calibration, index=False)
    pd.DataFrame({
        "event_id": ["eval"], "time_to_tca": [6.0], "risk": [-9.0]
    }).to_parquet(evaluation, index=False)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)

    with np.testing.assert_raises(ValueError):
        freeze_study(
            calibration, evaluation, preregistration, preregistration_lock,
            tmp_path / "study.json", tmp_path / "study.lock",
        )


def test_frozen_study_records_outcome_firewall_and_window_coverage(tmp_path):
    calibration = tmp_path / "calibration.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["cal", "cal"],
        "time_to_tca": [8.0, 6.0],
        "risk": [-9.0, -8.0],
    }).to_parquet(calibration, index=False)
    pd.DataFrame({
        "event_id": ["eval-a", "eval-b"],
        "time_to_tca": [7.0, 9.0],
        "risk": [-9.0, -10.0],
    }).to_parquet(evaluation, index=False)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    manifest_path, lock_path = tmp_path / "study.json", tmp_path / "study.lock"

    freeze_study(
        calibration, evaluation, preregistration, preregistration_lock,
        manifest_path, lock_path,
    )
    manifest, _ = read_locked_study(manifest_path, lock_path)

    assert manifest["schema_version"] == 4
    assert manifest["outcome_firewall"] == {
        "decision_window_days": [2.0, 7.0],
        "explicit_outcome_columns_forbidden": True,
        "post_window_rows_forbidden": True,
    }
    assert manifest["cohorts"]["calibration"]["time_to_tca_min"] == 6.0
    assert manifest["cohorts"]["calibration"]["decision_window_rows"] == 1
    assert manifest["cohorts"]["evaluation"]["decision_window_events"] == 1
    assert manifest["cohorts"]["evaluation"]["decision_window_event_ids"] == ["eval-a"]
    assert manifest["cohorts"]["calibration"]["decision_window_event_ids"] == ["cal"]
    assert manifest["cohorts"]["evaluation"]["columns"] == [
        "event_id", "risk", "time_to_tca"
    ]
    validate_feature_cohort(
        calibration, manifest_path, lock_path, "calibration"
    )


def test_scored_cohort_roster_requires_every_window_event(tmp_path):
    calibration = tmp_path / "calibration.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["cal"], "time_to_tca": [6.0], "risk": [-9.0]
    }).to_parquet(calibration, index=False)
    pd.DataFrame({
        "event_id": ["eval-a", "eval-b", "pre-window-only"],
        "time_to_tca": [7.0, 5.0, 9.0],
        "risk": [-9.0, -8.0, -10.0],
    }).to_parquet(evaluation, index=False)
    preregistration, preregistration_lock = _locked_preregistration(tmp_path)
    manifest_path, lock_path = tmp_path / "study.json", tmp_path / "study.lock"
    freeze_study(
        calibration, evaluation, preregistration, preregistration_lock,
        manifest_path, lock_path,
    )
    manifest, _ = read_locked_study(manifest_path, lock_path)

    validate_scored_cohort_roster(
        pd.DataFrame({"event_id": ["eval-a", "eval-b"]}),
        manifest,
        "evaluation",
    )
    with np.testing.assert_raises(ValueError):
        validate_scored_cohort_roster(
            pd.DataFrame({"event_id": ["eval-a"]}),
            manifest,
            "evaluation",
        )
    with np.testing.assert_raises(ValueError):
        validate_scored_cohort_roster(
            pd.DataFrame({"event_id": ["eval-a", "eval-b", "other"]}),
            manifest,
            "evaluation",
        )


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
    assert scores["feature_contract_sha256"].nunique() == 1
    assert scores["feature_contract_sha256"].iloc[0] == feature_contract_sha256()
    pd.testing.assert_frame_equal(scores, pd.read_parquet(output_path))


def test_tail_score_file_cleans_temporary_output_after_fsync_failure(tmp_path, monkeypatch):
    import score_final_tail_aligned as scoring_module
    root = Path(__file__).resolve().parents[1]
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "scores.parquet"
    snapshot_training_frame(700).drop(columns="y").to_parquet(
        features_path, index=False
    )

    def fail_fsync(_descriptor):
        raise OSError("simulated score fsync failure")

    monkeypatch.setattr(scoring_module.os, "fsync", fail_fsync)
    with np.testing.assert_raises(OSError):
        score_tail_file(
            features_path,
            root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
            root / "artifacts" / "catboost_tail_aligned_final_v13.json",
            output_path,
        )

    assert not output_path.exists()
    assert not list(tmp_path.glob(".scores.parquet.*.tmp"))


def test_tail_score_file_commits_with_os_replace(tmp_path, monkeypatch):
    import score_final_tail_aligned as scoring_module
    root = Path(__file__).resolve().parents[1]
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "scores.parquet"
    snapshot_training_frame(900).drop(columns="y").to_parquet(
        features_path, index=False
    )
    calls = []
    real_replace = scoring_module.os.replace

    def record_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(scoring_module.os, "replace", record_replace)
    scores = score_tail_file(
        features_path,
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        output_path,
    )

    assert len(calls) == 1
    assert calls[0][1] == output_path
    assert calls[0][0].parent == output_path.parent
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
        "calibration": {
            "threshold": threshold,
            "rank": 1,
            "n_positive": 40,
            "alpha": 0.10,
            "mode": "pac",
            "confidence": 0.95,
            "marginal_bound": 1 / 41,
            "pac_bound": 0.0721575245055145,
        },
        "model_sha256": model_hash,
        "shift_gate_sha256": None,
        "model_manifest_sha256": None,
        "calibration_event_ids": ["calibration-event"],
        "calibration_event_ids_sha256": __import__("hashlib").sha256(
            b"calibration-event"
        ).hexdigest(),
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


def test_confirmation_preflight_rejects_feature_contract_mismatch(tmp_path):
    import argparse

    digest = feature_contract_sha256()
    artifact = calibrate(
        confirmation_scores().assign(feature_contract_sha256=digest)
    )
    scores = tmp_path / "evaluation.parquet"
    confirmation_scores(event_offset=100).drop(columns="y").assign(
        feature_contract_sha256="0" * 64
    ).to_parquet(scores, index=False)
    args = argparse.Namespace(
        scores=scores, gate=None, model_manifest=None,
        study_manifest=None, study_lock=None,
    )
    with np.testing.assert_raises_regex(ValueError, "different feature contracts"):
        _confirmation_preflight(args, artifact)


def test_runtime_replay_binds_feature_contract(tmp_path):
    digest = feature_contract_sha256()
    artifact = runtime_calibration_artifact()
    artifact["feature_contract_sha256"] = digest
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["event"], "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["runtime-model"],
        "feature_contract_sha256": [digest],
    }).to_parquet(scores_path, index=False)
    audit = run_replay(scores_path, calibration, tmp_path / "audit.parquet")
    assert audit["feature_contract_sha256"].eq(digest).all()

    bad_scores = tmp_path / "bad-scores.parquet"
    pd.read_parquet(scores_path).assign(
        feature_contract_sha256="0" * 64
    ).to_parquet(bad_scores, index=False)
    with np.testing.assert_raises_regex(ValueError, "feature contract"):
        run_replay(bad_scores, calibration, tmp_path / "bad-audit.parquet")


def test_confirmation_status_sidecar_tracks_started_completed_and_legacy(tmp_path):
    lock = tmp_path / "confirmation.lock"
    output = tmp_path / "confirmation.json"
    assert read_confirmation_status(lock, output)["status"] == "not-started"
    acquire_confirmation_lock(lock, {"run": 1})
    original = lock.read_bytes()
    assert read_confirmation_status(lock, output)["status"] == "in-progress"
    output.write_text('{"evaluation": {}}\n', encoding="utf-8")
    assert read_confirmation_status(lock, output)["status"] == "legacy-completed"
    write_confirmation_status_sidecar(
        lock, status="completed", payload={
            "completed_at_utc": "2026-07-30T00:00:00+00:00",
            "output": str(output),
            "output_sha256": file_sha256(output),
        },
    )
    assert lock.read_bytes() == original
    status = read_confirmation_status(lock, output)
    assert status["status"] == "completed"
    assert status["output_sha256"] == file_sha256(output)
    assert confirmation_status_path(lock).exists()


def test_confirmation_status_rejects_tampered_lock_or_output(tmp_path):
    lock = tmp_path / "confirmation.lock"
    output = tmp_path / "confirmation.json"
    acquire_confirmation_lock(lock, {"run": 1})
    output.write_text('{}\n', encoding="utf-8")
    write_confirmation_status_sidecar(
        lock, status="completed", payload={
            "output": str(output), "output_sha256": file_sha256(output),
        },
    )
    output.write_text('{"tampered": true}\n', encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "output"):
        read_confirmation_status(lock, output)


def test_confirmation_status_rejects_invalid_terminal_state(tmp_path):
    lock = tmp_path / "confirmation.lock"
    acquire_confirmation_lock(lock, {"run": 1})
    with np.testing.assert_raises(ValueError):
        write_confirmation_status_sidecar(lock, status="unknown", payload={})


def test_confirmation_preflight_failure_does_not_burn_lock(tmp_path):
    import argparse
    calibration = tmp_path / "calibration.json"
    artifact = runtime_calibration_artifact()
    artifact["study_manifest_sha256"] = "expected-study"
    calibration.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    scores = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["evaluation-event"],
        "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["runtime-model"],
        "study_manifest_sha256": ["wrong-study"],
        "study_cohort": ["evaluation"],
    }).to_parquet(scores, index=False)
    args = argparse.Namespace(
        scores=scores,
        calibration=calibration,
        labels=tmp_path / "unopened-labels.parquet",
        output=tmp_path / "confirmation.json",
        lock=tmp_path / "confirmation.lock",
        gate=None,
        model_manifest=None,
        study_manifest=None,
        study_lock=None,
    )

    with np.testing.assert_raises(ValueError):
        confirmation_command(args)

    assert not args.lock.exists()
    assert not args.output.exists()


def test_confirmation_preflight_rejects_tampered_rule_before_lock(tmp_path):
    import argparse
    artifact = calibrate(confirmation_scores())
    artifact["calibration"]["pac_bound"] = 0.9
    scores = tmp_path / "evaluation.parquet"
    confirmation_scores(event_offset=100).drop(columns="y").to_parquet(scores, index=False)
    args = argparse.Namespace(
        scores=scores,
        gate=None,
        model_manifest=None,
        study_manifest=None,
        study_lock=None,
    )

    with np.testing.assert_raises(ValueError):
        _confirmation_preflight(args, artifact)


def test_confirmation_post_lock_failure_writes_failed_status(tmp_path, monkeypatch):
    import argparse
    import confirm_policy as confirm_module

    calibration = tmp_path / "calibration.json"
    artifact = runtime_calibration_artifact()
    artifact["policy"] = POLICY.copy()
    artifact["study_manifest_sha256"] = None
    calibration.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    scores = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["positive"] * 3,
        "time_to_tca": [7.0, 6.0, 5.0],
        "catboost_snapshot": [0.90] * 3,
        "model_sha256": ["runtime-model"] * 3,
    }).to_parquet(scores, index=False)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame({"event_id": ["positive"], "y": [1]}).to_parquet(labels, index=False)
    args = argparse.Namespace(
        scores=scores, calibration=calibration, labels=labels,
        output=tmp_path / "confirmation.json", lock=tmp_path / "confirmation.lock",
        gate=None, model_manifest=None, study_manifest=None, study_lock=None,
    )
    def fail_evaluate(*_args, **_kwargs):
        raise ValueError("simulated post-lock failure")
    monkeypatch.setattr(confirm_module, "evaluate", fail_evaluate)
    with np.testing.assert_raises_regex(ValueError, "simulated post-lock failure"):
        confirmation_command(args)
    assert args.lock.exists()
    assert not args.output.exists()
    status = read_confirmation_status(args.lock, args.output)
    assert status["status"] == "failed"
    assert status["failure_type"] == "ValueError"
    assert "simulated post-lock failure" in status["failure_message"]


def test_confirmation_command_writes_lock_and_result_after_valid_preflight(tmp_path):
    import argparse
    calibration = tmp_path / "calibration.json"
    artifact = runtime_calibration_artifact()
    artifact["policy"] = POLICY.copy()
    artifact["study_manifest_sha256"] = None
    calibration.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    scores = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["positive"] * 3 + ["negative"] * 3,
        "time_to_tca": [7.0, 6.0, 5.0] * 2,
        "catboost_snapshot": [0.90] * 3 + [0.10] * 3,
        "model_sha256": ["runtime-model"] * 6,
    }).to_parquet(scores, index=False)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame({
        "event_id": ["positive", "negative"], "y": [1, 0]
    }).to_parquet(labels, index=False)
    args = argparse.Namespace(
        scores=scores,
        calibration=calibration,
        labels=labels,
        output=tmp_path / "confirmation.json",
        lock=tmp_path / "confirmation.lock",
        gate=None,
        model_manifest=None,
        study_manifest=None,
        study_lock=None,
    )

    confirmation_command(args)

    assert args.lock.exists()
    assert read_confirmation_status(args.lock, args.output)["status"] == "completed"
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["evaluation"]["danger_k"] == 0
    assert result["evaluation"]["danger_n"] == 1
    assert result["evaluation"]["safe_negative"] == 1
    assert result["evaluation"]["negative_n"] == 1


def test_confirmation_preflight_rejects_model_mismatch_before_lock(tmp_path):
    import argparse
    calibration = tmp_path / "calibration.json"
    artifact = runtime_calibration_artifact()
    artifact["study_manifest_sha256"] = None
    calibration.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    scores = tmp_path / "evaluation.parquet"
    pd.DataFrame({
        "event_id": ["evaluation-event"],
        "time_to_tca": [5.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["different-model"],
    }).to_parquet(scores, index=False)
    args = argparse.Namespace(
        scores=scores,
        calibration=calibration,
        labels=tmp_path / "unopened-labels.parquet",
        output=tmp_path / "confirmation.json",
        lock=tmp_path / "confirmation.lock",
        gate=None,
        model_manifest=None,
        study_manifest=None,
        study_lock=None,
    )

    with np.testing.assert_raises(ValueError):
        confirmation_command(args)

    assert not args.lock.exists()
    assert not args.output.exists()


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


def test_event_explanation_tracks_first_safe_exclude():
    audit = pd.DataFrame({
        "event_id": ["event"] * 4,
        "sequence_number": [1, 2, 3, 4],
        "time_to_tca": [8.0, 7.0, 6.0, 5.0],
        "score": [0.10, 0.10, 0.10, 0.30],
        "decision": ["MONITOR", "MONITOR", "SAFE-EXCLUDE", "MONITOR"],
        "reason": [
            "decision_window_not_open", "minimum_history_not_reached",
            "score_at_or_below_calibrated_threshold",
            "score_between_decision_thresholds",
        ],
        "shift_score": [np.nan] * 4,
        "shift_gate_allowed": [True] * 4,
        "decision_window_eligible": [False, True, True, True],
        "eligible_history_count": [0, 1, 3, 4],
    })
    policy = {"minimum_history": 3, "min_days_to_tca": 2.0, "max_days_to_tca": 7.0}
    result = explain_event_sequence("event", audit, policy, threshold=0.20)
    assert result["total_updates"] == 4
    assert result["current_decision"] == "MONITOR"
    assert result["first_safe_exclude_sequence"] == 3
    assert result["first_safe_exclude_tca"] == 6.0
    assert result["steps"][0]["decision_window_eligible"] is False
    assert result["steps"][1]["history_sufficient"] is False
    assert result["steps"][2]["score_at_or_below_safe_threshold"] is True
    assert "calibrated safe threshold" in result["steps"][2]["explanation"]


def test_event_explanation_reports_shift_gate_block():
    audit = pd.DataFrame({
        "event_id": [1], "sequence_number": [3], "time_to_tca": [5.0],
        "score": [0.10], "decision": ["MONITOR"],
        "reason": ["safe_exclude_blocked_by_shift_gate"],
        "shift_score": [4.2], "shift_gate_allowed": [False],
        "decision_window_eligible": [True], "eligible_history_count": [3],
    })
    policy = {"minimum_history": 3, "min_days_to_tca": 2.0, "max_days_to_tca": 7.0}
    step = explain_event_sequence(1, audit, policy, threshold=0.20)["steps"][0]
    assert step["score_at_or_below_safe_threshold"] is True
    assert step["shift_gate_allowed"] is False
    assert step["shift_score"] == 4.2
    assert "applicability gate blocked" in step["explanation"]


def test_event_explanation_rejects_unknown_event_and_reason():
    audit = pd.DataFrame({
        "event_id": ["known"], "sequence_number": [1], "time_to_tca": [5.0],
        "score": [0.3], "decision": ["MONITOR"], "reason": ["unknown_reason"],
        "shift_score": [np.nan], "shift_gate_allowed": [True],
        "decision_window_eligible": [True], "eligible_history_count": [1],
    })
    policy = {"minimum_history": 1, "min_days_to_tca": 2.0, "max_days_to_tca": 7.0}
    with np.testing.assert_raises_regex(ValueError, "not present"):
        explain_event_sequence("missing", audit, policy, threshold=0.20)
    with np.testing.assert_raises_regex(ValueError, "Unsupported runtime decision reason"):
        explain_event_sequence("known", audit, policy, threshold=0.20)


def test_showcase_selector_prefers_maximum_first_safe_lead():
    rows = []
    for event_id, safe_tca, updates in (("A", 6.0, 3), ("B", 5.0, 4)):
        rows.append({
            "event_id": event_id, "sequence_number": 1, "time_to_tca": 6.5,
            "decision": "MONITOR", "reason": "minimum_history_not_reached",
            "shift_score": np.nan, "audit_batch": 1,
        })
        for sequence in range(2, updates + 1):
            rows.append({
                "event_id": event_id, "sequence_number": sequence,
                "time_to_tca": safe_tca - 0.1 * (sequence - 2),
                "decision": "SAFE-EXCLUDE",
                "reason": "score_at_or_below_calibrated_threshold",
                "shift_score": np.nan, "audit_batch": 1,
            })
    audit = pd.DataFrame(rows)
    audit["__event_key"] = audit["event_id"].map(lambda value: f"str:{value}")
    selected = select_showcase_monitor_to_safe(audit)
    assert selected["event_id"] == "A"
    assert selected["first_safe_tca"] == 6.0
    assert selected["first_safe_sequence"] == 2


def test_showcase_selector_requires_prior_monitor():
    audit = pd.DataFrame([{
        "event_id": "A", "sequence_number": 1, "time_to_tca": 6.0,
        "decision": "SAFE-EXCLUDE",
        "reason": "score_at_or_below_calibrated_threshold",
        "shift_score": np.nan, "audit_batch": 1, "__event_key": "str:A",
    }])
    with np.testing.assert_raises_regex(ValueError, "MONITOR-to-SAFE-EXCLUDE"):
        select_showcase_monitor_to_safe(audit)


def test_gate_showcase_returns_none_without_real_block():
    audit = pd.DataFrame([{
        "event_id": "A", "sequence_number": 1, "time_to_tca": 6.0,
        "decision": "SAFE-EXCLUDE",
        "reason": "score_at_or_below_calibrated_threshold",
        "shift_score": np.nan, "audit_batch": 1, "__event_key": "str:A",
    }])
    assert select_showcase_gate_blocked(audit) is None


def test_gate_showcase_prefers_current_monitor_case():
    audit = pd.DataFrame([
        {"event_id": "A", "sequence_number": 1, "time_to_tca": 6.0,
         "decision": "MONITOR", "reason": "safe_exclude_blocked_by_shift_gate",
         "shift_score": 3.0, "audit_batch": 1, "__event_key": "str:A"},
        {"event_id": "B", "sequence_number": 1, "time_to_tca": 6.0,
         "decision": "MONITOR", "reason": "safe_exclude_blocked_by_shift_gate",
         "shift_score": 4.0, "audit_batch": 1, "__event_key": "str:B"},
        {"event_id": "B", "sequence_number": 2, "time_to_tca": 5.5,
         "decision": "SAFE-EXCLUDE", "reason": "score_at_or_below_calibrated_threshold",
         "shift_score": np.nan, "audit_batch": 1, "__event_key": "str:B"},
    ])
    selected = select_showcase_gate_blocked(audit)
    assert selected["event_id"] == "A"
    assert selected["current_decision"] == "MONITOR"


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
    assert "Decision explanation" in document
    assert "The score did not cross a decision threshold" in document
    assert "c" * 64 in document
    assert "https://" not in document


def test_operator_dashboard_showcase_follows_current_event_priority(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    audit = pd.concat([
        dashboard_audit_frame(
            calibration, event_id="safe-event", decision="SAFE-EXCLUDE"
        ),
        dashboard_audit_frame(
            calibration, event_id="monitor-event", decision="MONITOR"
        ),
        dashboard_audit_frame(
            calibration, event_id="escalate-event", decision="ESCALATE"
        ),
    ], ignore_index=True)
    audit_path = tmp_path / "audit.parquet"
    audit.to_parquet(audit_path, index=False)
    output = tmp_path / "dashboard.html"

    build_dashboard([audit_path], calibration, output)
    document = output.read_text(encoding="utf-8")

    timeline = document.split("const events=", 1)[1].split(
        ";const select=", 1
    )[0]
    assert timeline.index('"label": "escalate-event"') < timeline.index(
        '"label": "monitor-event"'
    ) < timeline.index('"label": "safe-event"')


def test_operator_dashboard_contains_print_layout(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit_path, index=False)
    output = tmp_path / "dashboard.html"

    build_dashboard([audit_path], calibration, output)
    document = output.read_text(encoding="utf-8")

    assert "@media print" in document
    assert "select,footer,.active,.lineage{display:none}" in document
    assert ".timeline{grid-column:span 12}" in document
    assert "break-inside:avoid" in document


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


def test_operator_dashboard_verifies_processed_batch_chain(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["event", "event", "event"],
        "time_to_tca": [7.0, 6.0, 5.0],
        "catboost_tail_aligned": [0.10, 0.10, 0.10],
        "model_sha256": ["c" * 64] * 3,
    }).to_parquet(scores, index=False)
    audit_path = tmp_path / "audit.parquet"
    checkpoint = tmp_path / "runtime.json"
    run_replay(
        scores, calibration, audit_path, checkpoint_path=checkpoint,
    )
    output = tmp_path / "dashboard.html"

    summary = build_dashboard(
        [audit_path], calibration, output, checkpoint_path=checkpoint,
    )
    document = output.read_text(encoding="utf-8")

    assert summary["chain"]["status"] == "VERIFIED"
    assert summary["chain"]["length"] == 1
    assert len(summary["chain"]["head_sha256"]) == 64
    assert "Batch chain" in document
    assert "VERIFIED" in document
    assert summary["chain"]["head_sha256"][:16] in document


def test_operator_dashboard_rejects_checkpoint_audit_mismatch(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["event"],
        "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["c" * 64],
    }).to_parquet(scores, index=False)
    audit_path = tmp_path / "audit.parquet"
    checkpoint = tmp_path / "runtime.json"
    run_replay(scores, calibration, audit_path, checkpoint_path=checkpoint)
    audit = pd.read_parquet(audit_path)
    audit["runtime_checkpoint_sha256"] = "0" * 64
    audit.to_parquet(audit_path, index=False)

    with np.testing.assert_raises(ValueError):
        build_dashboard(
            [audit_path], calibration, tmp_path / "dashboard.html",
            checkpoint_path=checkpoint,
        )


def test_operator_dashboard_rejects_tampered_checkpoint(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    scores = tmp_path / "scores.parquet"
    pd.DataFrame({
        "event_id": ["event"],
        "time_to_tca": [7.0],
        "catboost_tail_aligned": [0.10],
        "model_sha256": ["c" * 64],
    }).to_parquet(scores, index=False)
    audit_path = tmp_path / "audit.parquet"
    checkpoint = tmp_path / "runtime.json"
    run_replay(scores, calibration, audit_path, checkpoint_path=checkpoint)
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["processed_batches"][0]["rows"] = 2
    checkpoint.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    with np.testing.assert_raises(ValueError):
        build_dashboard(
            [audit_path], calibration, tmp_path / "dashboard.html",
            checkpoint_path=checkpoint,
        )


def _two_batch_dashboard_inputs(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    checkpoint = tmp_path / "runtime.json"
    first_scores = tmp_path / "first-scores.parquet"
    second_scores = tmp_path / "second-scores.parquet"
    first_audit = tmp_path / "first-audit.parquet"
    second_audit = tmp_path / "second-audit.parquet"
    pd.DataFrame({
        "event_id": [1, 1],
        "time_to_tca": [7.0, 6.0],
        "catboost_tail_aligned": [0.10, 0.10],
        "model_sha256": ["c" * 64] * 2,
    }).to_parquet(first_scores, index=False)
    pd.DataFrame({
        "event_id": [1, 2],
        "time_to_tca": [5.0, 7.0],
        "catboost_tail_aligned": [0.10, 0.90],
        "model_sha256": ["c" * 64] * 2,
    }).to_parquet(second_scores, index=False)
    run_replay(
        first_scores, calibration, first_audit, checkpoint_path=checkpoint,
    )
    run_replay(
        second_scores, calibration, second_audit, checkpoint_path=checkpoint,
    )
    return calibration, checkpoint, first_scores, second_scores, first_audit, second_audit


def test_operator_dashboard_lists_two_processed_batches(tmp_path):
    from confirmation import file_sha256
    calibration, checkpoint, first_scores, second_scores, first_audit, second_audit = (
        _two_batch_dashboard_inputs(tmp_path)
    )
    output = tmp_path / "dashboard.html"

    summary = build_dashboard(
        [first_audit, second_audit], calibration, output,
        checkpoint_path=checkpoint,
    )
    document = output.read_text(encoding="utf-8")

    assert summary["chain"]["status"] == "VERIFIED"
    assert summary["chain"]["length"] == 2
    assert summary["chain"]["displayed_rows"] == 2
    assert summary["chain"]["clipped"] is False
    assert "Processed batches" in document
    assert file_sha256(first_scores)[:16] in document
    assert file_sha256(second_scores)[:16] in document
    assert "GENESIS" in document


def test_operator_dashboard_clips_processed_batch_table(tmp_path):
    from confirmation import file_sha256
    calibration, checkpoint, first_scores, second_scores, first_audit, second_audit = (
        _two_batch_dashboard_inputs(tmp_path)
    )
    output = tmp_path / "dashboard.html"

    summary = build_dashboard(
        [first_audit, second_audit], calibration, output,
        checkpoint_path=checkpoint, max_chain_rows=1,
    )
    document = output.read_text(encoding="utf-8")

    assert summary["chain"]["length"] == 2
    assert summary["chain"]["displayed_rows"] == 1
    assert summary["chain"]["clipped"] is True
    assert file_sha256(first_scores)[:16] not in document
    assert file_sha256(second_scores)[:16] in document
    assert "showing last 1 of 2 batches" in document


def test_operator_dashboard_rejects_invalid_chain_row_limit(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    audit = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit, index=False)

    with np.testing.assert_raises(ValueError):
        build_dashboard(
            [audit], calibration, tmp_path / "dashboard.html",
            max_chain_rows=0,
        )


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


def test_delivery_requirements_are_exactly_pinned():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    expected = {
        "pandas", "numpy", "scipy", "scikit-learn",
        "catboost", "pyarrow", "pytest",
    }
    parsed = {}
    for line in requirements:
        name, separator, version = line.partition("==")
        assert separator == "==" and name and version
        parsed[name] = version
    assert set(parsed) == expected


def test_run_demo_initializes_local_import_paths_before_imports():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_demo.py").read_text(encoding="utf-8")
    src_path = 'sys.path.insert(0, str(ROOT / "src"))'
    scripts_path = 'sys.path.insert(0, str(ROOT / "scripts"))'
    first_local_import = "from operator_dashboard import build_dashboard"
    assert source.index(src_path) < source.index(first_local_import)
    assert source.index(scripts_path) < source.index(first_local_import)


def test_one_command_demo_builds_verified_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-demo"

    summary = run_demo(output, root=root)

    assert summary["status"] == "historical-demo-not-for-operations"
    assert summary["message_updates"] == 22656
    assert summary["events_in_runtime_window"] == 2387
    assert summary["confirmation"]["passed"] is False
    assert summary["batch_chain"]["status"] == "VERIFIED"
    assert summary["batch_chain"]["length"] == 1
    assert len(summary["batch_chain"]["head_sha256"]) == 64
    assert (output / "replay-audit.parquet").exists()
    assert (output / "operator-console.html").exists()
    assert (output / "runtime-state.json").exists()
    assert (output / "summary.json").exists()
    assert (output / "evidence-dashboard.html").exists()
    stored = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert stored["caveat"] == summary["caveat"]
    assert stored["evidence_dashboard"] == "evidence-dashboard.html"
    assert stored["evidence"]["confirmation_passed"] is False
    assert stored["batch_chain"] == summary["batch_chain"]
    assert stored["bundle_verification"]["status"] == "VERIFIED"
    assert set(stored["input_artifacts"]) == {
        "confirmation_evaluation_scores", "confirmation_calibration",
        "confirmation_result", "confirmation_lock",
        "development_event_aligned_manifest", "development_oof_scores",
        "development_score_ensemble_manifest",
        "development_repeated_stability_manifest",
        "development_fold_diagnostics", "next_validation_planning",
        "next_validation_preregistration",
        "next_validation_preregistration_lock",
        "v13_model_manifest", "v13_model",
    }
    assert all(len(digest) == 64 for digest in stored["input_artifacts"].values())
    assert set(stored["bundle_verification"]["artifacts"]) == {
        "replay-audit.parquet", "operator-console.html",
        "runtime-state.json", "evidence-dashboard.html",
    }
    assert all(
        len(digest) == 64
        for digest in stored["bundle_verification"]["artifacts"].values()
    )
    verified = verify_demo_bundle(output, root=root)
    assert verified["status"] == "VERIFIED"
    assert verified["artifacts"] == stored["bundle_verification"]["artifacts"]
    assert verified["input_artifacts"] == stored["input_artifacts"]
    console = (output / "operator-console.html").read_text(encoding="utf-8")
    assert "NOT MET" in console
    assert "Batch chain" in console
    assert "VERIFIED" in console
    assert "Key figures" in console
    assert "Example trajectory" in console
    assert "22,656" in console
    assert "2,387" in console
    assert "687" in console
    assert "5.53 d" in console
    assert "Event <strong>5126</strong>" in console
    assert "TCA <strong>6.542 d</strong>" in console
    assert "No synthetic case is shown" in console
    assert summary["showcase_monitor_to_safe"]["event_id"] == 5126
    assert summary["showcase_gate_blocked"] is None


def test_demo_bundle_verification_rejects_tampered_artifact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-demo"
    run_demo(output, root=root)
    console = output / "operator-console.html"
    console.write_text(
        console.read_text(encoding="utf-8") + "\n<!-- tampered -->\n",
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(
        ValueError, "operator-console.html"
    ):
        verify_demo_bundle(output, root=root)


def test_demo_bundle_verification_rejects_input_lineage_drift(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-demo"
    run_demo(output, root=root)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_artifacts"]["v13_model"] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "v13_model"):
        verify_demo_bundle(output, root=root)


def test_demo_bundle_verification_rejects_incomplete_digest_roster(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-demo"
    run_demo(output, root=root)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["bundle_verification"]["artifacts"].pop("runtime-state.json")
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with np.testing.assert_raises_regex(
        ValueError, "invalid artifact digest roster"
    ):
        verify_demo_bundle(output, root=root)


def test_runtime_audit_event_index_survives_restore(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.2, minimum_history=1)
    policy.update("event-a", 6.0, 0.1)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)

    restored = SequentialTriagePolicy.restore(checkpoint)

    assert restored._audit_event_ids == {"event-a"}
    with np.testing.assert_raises_regex(RuntimeError, "Cannot reset an event"):
        restored.reset_event("event-a")
    restored.reset_event("event-without-audit")


def test_one_command_demo_refuses_protected_and_nonempty_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for directory in ("artifacts", "data", "reports", "src", "scripts", "tests"):
        with np.testing.assert_raises(ValueError):
            run_demo(root / directory / "demo", root=root)

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


def test_evidence_dashboard_blocks_nonterminal_confirmation_states(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    for state in ("not-started", "in-progress", "failed"):
        copy_root = tmp_path / state
        shutil.copytree(root / "artifacts", copy_root / "artifacts")
        shutil.copytree(root / "reports", copy_root / "reports")
        confirmation_dir = copy_root / "artifacts" / "confirmation_v1"
        lock = confirmation_dir / "confirmation.lock"
        result = confirmation_dir / "confirmation.json"
        confirmation_status_path(lock).unlink(missing_ok=True)
        if state == "not-started":
            lock.unlink()
        elif state == "in-progress":
            result.unlink()
        else:
            write_confirmation_status_sidecar(
                lock, status="failed", payload={
                    "failure_type": "ValueError",
                    "failure_message": "simulated",
                },
            )
        output = tmp_path / f"{state}.html"
        with np.testing.assert_raises_regex(ValueError, "terminal completed state"):
            build_evidence_dashboard(copy_root, output)
        assert not output.exists()


def test_evidence_dashboard_accepts_legacy_completed_confirmation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert read_confirmation_status(
        root / "artifacts" / "confirmation_v1" / "confirmation.lock",
        root / "artifacts" / "confirmation_v1" / "confirmation.json",
    )["status"] == "legacy-completed"
    output = tmp_path / "evidence.html"
    summary = build_evidence_dashboard(root, output)
    assert summary["confirmation_terminal_status"] == "legacy-completed"
    assert output.exists()


def test_evidence_dashboard_builds_three_verified_tiers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evidence.html"
    summary = build_evidence_dashboard(root, output)
    document = output.read_text(encoding="utf-8")

    assert summary["confirmation_passed"] is False
    assert summary["confirmation_terminal_status"] == "legacy-completed"
    assert abs(summary["danger_ucb"] - 0.12101499810942579) < 1e-12
    assert summary["criterion"] == 0.10
    assert summary["development_pareto_methods"] == ["catboost_tail_aligned", "minimum"]
    assert summary["preregistration_frozen"] is True
    assert summary["calibration_accessed"] is False
    assert summary["evaluation_accessed"] is False
    assert summary["v13_threshold"] is None
    assert abs(summary["historical_correct_safe_excludes_per_1000"] - 686.8110984416572) < 1e-12
    assert abs(summary["historical_remaining_per_1000"] - 313.1889015583428) < 1e-12
    assert summary["candidate_fold_safety_passes"] == 1
    assert summary["candidate_fold_safety_total"] == 5
    assert summary["oof_verification_passed"] is True
    assert summary["oof_verification_checks"] == 40
    assert summary["oof_verification_schema"] == 2
    assert summary["oof_fold_crosscheck_passed"] is True
    assert len(summary["folds_csv_sha256"]) == 64
    assert len(summary["oof_fold_results"]) == 5
    assert "id='development'" in document
    assert "id='confirmation'" in document
    assert "id='preregistered'" in document
    assert "CRITERION NOT MET" in document
    assert "12.10%" in document
    assert "not confirmation evidence" in document
    assert "https://" not in document


def test_evidence_dashboard_embeds_frontier_and_fold_stability(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "evidence.html"
    build_evidence_dashboard(root, output)
    document = output.read_text(encoding="utf-8")

    assert document.count("<svg class='evidence-plot'") == 2
    assert "Development safety-automation frontier" in document
    assert "Outer-fold coverage stability" in document
    assert "UCB criterion 10%" in document
    assert "catboost_tail_aligned" in document
    assert "fold 0" in document and "-2.16 pp" in document
    assert "fold 1" in document and "+6.97 pp" in document
    assert "12/192 dangerous exclusions" in document
    assert "2.43 percentage points" in document
    assert "1/5 folds meet the pooled 10% UCB criterion" in document
    assert "Live OOF evidence verification" in document
    assert "40/40 integrity and metric checks passed" in document
    assert "verifier schema 2" in document
    assert "38–39 positive events" in document
    assert "687 correct SAFE-EXCLUDE decisions per 1,000" in document
    assert "313" in document and "Remaining per 1,000" in document
    assert "not confirmation evidence" in document
    assert "https://" not in document


def test_oof_fold_results_match_independent_fold_diagnostics():
    root = Path(__file__).resolve().parents[1]
    result = verify_oof_evidence(
        root / "artifacts" / "development_event_aligned_oof_v8.parquet",
        root / "artifacts" / "development_event_aligned_v8.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
    )
    folds = pd.read_csv(root / "reports" / "development_score_ensemble_folds_v10.csv")
    folds = folds.loc[folds["method"].eq("catboost_tail_aligned")].set_index("fold")
    for live in result["fold_results"]:
        stored = folds.loc[live["fold"]]
        for field in ("danger_k", "danger_n", "safe_negative", "negative_n"):
            assert live[field] == int(stored[field])
        for field in (
            "danger_rate", "danger_ucb", "safe_negative_rate",
            "median_first_safe_tca_days",
        ):
            assert abs(live[field] - float(stored[field])) <= 1e-12


def test_evidence_dashboard_rejects_fold_diagnostic_divergence(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "copy"
    shutil.copytree(root / "artifacts", copy_root / "artifacts")
    shutil.copytree(root / "reports", copy_root / "reports")
    path = copy_root / "reports" / "development_score_ensemble_folds_v10.csv"
    folds = pd.read_csv(path)
    mask = folds["method"].eq("catboost_tail_aligned") & folds["fold"].eq(0)
    folds.loc[mask, "danger_k"] = 9
    folds.to_csv(path, index=False)

    with np.testing.assert_raises_regex(ValueError, "fold 0 danger_k"):
        build_evidence_dashboard(copy_root, tmp_path / "evidence.html")


def test_evidence_planning_validation_rejects_recomputed_claim_mismatch():
    plan = pd.DataFrame({
        "positive_events": [200],
        "maximum_passing_failures": [11],
        "upper_bound_at_maximum": [0.09540141772987734],
        "pass_probability_if_true_rate_0.05": [0.7964843459884561],
    })
    with np.testing.assert_raises_regex(ValueError, "maximum-passing-failures"):
        validate_planning_table(plan, alpha=0.10, confidence=0.95)


def test_frontier_svg_marks_feasibility_and_pareto_status():
    from evidence_dashboard import frontier_svg
    summary = [
        {"method": "safe", "danger_ucb": 0.09, "safe_negative_rate": 0.70,
         "safety_feasible": True, "pareto_frontier": True},
        {"method": "unsafe", "danger_ucb": 0.11, "safe_negative_rate": 0.80,
         "safety_feasible": False, "pareto_frontier": False},
    ]
    document = frontier_svg(summary, 0.10)
    assert "class='point pareto'" in document
    assert "class='point infeasible'" in document
    assert "UCB criterion 10%" in document


def test_fold_stability_svg_rejects_incomplete_input():
    from evidence_dashboard import fold_stability_svg
    with np.testing.assert_raises(ValueError):
        fold_stability_svg(pd.DataFrame({"method": ["catboost_tail_aligned"]}))


def test_russian_demo_localizes_operator_and_evidence_dashboards(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "russian-demo"
    summary = run_demo(output, root=root, locale="ru")
    operator = (output / "operator-console.html").read_text(encoding="utf-8")
    evidence = (output / "evidence-dashboard.html").read_text(encoding="utf-8")

    assert summary["locale"] == "ru"
    assert summary["evidence"]["locale"] == "ru"
    assert "<html lang='ru'>" in operator
    assert "Операторская консоль" in operator
    assert "Очередь событий" in operator
    assert "НЕ ВЫПОЛНЕН" in operator
    assert "<html lang='ru'>" in evidence
    assert "Последовательный триаж CDM · результаты" in evidence
    assert "Граница безопасность–автоматизация" in evidence
    assert "КРИТЕРИЙ НЕ ВЫПОЛНЕН" in evidence
    assert "Диагностика безопасности по folds" in evidence
    assert "корректных решений SAFE-EXCLUDE на 1 000" in evidence
    assert evidence.count("<svg class='evidence-plot'") == 2
    assert "https://" not in operator
    assert "https://" not in evidence


def test_run_demo_rejects_unsupported_locale_before_replay(tmp_path, monkeypatch):
    import run_demo as demo_module
    output = tmp_path / "invalid-locale-demo"
    replay_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal replay_called
        replay_called = True
        raise AssertionError("replay must not start for an unsupported locale")

    monkeypatch.setattr(demo_module, "run_replay", fail_if_called)
    with np.testing.assert_raises_regex(ValueError, "Unsupported locale"):
        demo_module.run_demo(output, root=Path(__file__).resolve().parents[1], locale="xx")
    assert replay_called is False
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid-locale-demo.*"))


def test_operator_dashboard_rejects_incomplete_audit_at_load_time(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n"
    )
    audit = dashboard_audit_frame(calibration).drop(columns="decision_window_eligible")
    audit_path = tmp_path / "audit.parquet"
    audit.to_parquet(audit_path, index=False)
    output = tmp_path / "operator.html"

    with np.testing.assert_raises_regex(ValueError, "decision_window_eligible"):
        build_dashboard([audit_path], calibration, output)
    assert not output.exists()


def test_operator_dashboard_rejects_tampered_calibration_rule(tmp_path):
    calibration = tmp_path / "calibration.json"
    artifact = runtime_calibration_artifact(model_hash="c" * 64)
    artifact["calibration"]["pac_bound"] = 0.9
    calibration.write_text(json.dumps(artifact) + "\n")
    audit_path = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit_path, index=False)
    output = tmp_path / "operator.html"

    with np.testing.assert_raises_regex(ValueError, "PAC bound"):
        build_dashboard([audit_path], calibration, output)
    assert not output.exists()


def test_evidence_dashboard_rejects_tampered_oof_scores(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "copy"
    shutil.copytree(root / "artifacts", copy_root / "artifacts")
    shutil.copytree(root / "reports", copy_root / "reports")
    oof_path = copy_root / "artifacts" / "development_event_aligned_oof_v8.parquet"
    oof = pd.read_parquet(oof_path)
    oof.loc[0, "catboost_tail_aligned"] = 2.0
    oof.to_parquet(oof_path, index=False)

    output = tmp_path / "evidence.html"
    with np.testing.assert_raises_regex(ValueError, "OOF evidence verification failed"):
        build_evidence_dashboard(copy_root, output)
    assert not output.exists()


def test_evidence_dashboard_rejects_hash_consistent_calibration_tampering(tmp_path):
    import shutil
    from confirmation import file_sha256

    root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "copy"
    shutil.copytree(root / "artifacts", copy_root / "artifacts")
    shutil.copytree(root / "reports", copy_root / "reports")
    calibration = copy_root / "artifacts" / "confirmation_v1" / "calibration.json"
    artifact = json.loads(calibration.read_text(encoding="utf-8"))
    artifact["calibration"]["pac_bound"] = 0.9
    calibration.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    tampered_hash = file_sha256(calibration)
    lock_path = copy_root / "artifacts" / "confirmation_v1" / "confirmation.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["calibration_artifact_sha256"] = tampered_hash
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    result_path = copy_root / "artifacts" / "confirmation_v1" / "confirmation.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["calibration_artifact_sha256"] = tampered_hash
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output = tmp_path / "evidence.html"

    with np.testing.assert_raises_regex(ValueError, "PAC bound"):
        build_evidence_dashboard(copy_root, output)
    assert not output.exists()


def test_dashboards_reject_unsupported_locale(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with np.testing.assert_raises(ValueError):
        build_evidence_dashboard(root, tmp_path / "evidence.html", locale="xx")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(runtime_calibration_artifact(model_hash="c" * 64)) + "\n")
    audit = tmp_path / "audit.parquet"
    dashboard_audit_frame(calibration).to_parquet(audit, index=False)
    with np.testing.assert_raises(ValueError):
        build_dashboard([audit], calibration, tmp_path / "operator.html", locale="xx")


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



def test_checkpoint_v3_persists_processed_batch_hash_chain(tmp_path):
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

    assert envelope["payload"]["schema_version"] == 3
    assert restored.processed_batches() == policy.processed_batches()
    assert restored.has_processed_batch("a" * 64)
    entry = restored.processed_batches()[0]
    assert entry["previous_entry_sha256"] is None
    assert len(entry["entry_sha256"]) == 64
    assert restored.processed_batch_chain_head() == entry["entry_sha256"]


def test_checkpoint_v1_restores_and_upgrades_to_v3(tmp_path):
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
    assert upgraded["payload"]["schema_version"] == 3
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
    assert ledger[0]["previous_entry_sha256"] is None
    assert ledger[1]["previous_entry_sha256"] == ledger[0]["entry_sha256"]
    assert restored.processed_batch_chain_head() == ledger[1]["entry_sha256"]


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


def test_restore_rejects_forged_eligible_history_with_valid_digest(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=3)
    policy.update("event", 7.0, 0.10)
    policy.update("event", 6.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)

    def forge_history(payload):
        payload["audit"][0]["eligible_history_count"] = 0

    _rewrite_checkpoint_with_valid_digest(checkpoint, forge_history)
    with np.testing.assert_raises_regex(ValueError, "eligible history"):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_forged_window_flag_with_valid_digest(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 8.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][0].update(
            {"decision_window_eligible": True}
        ),
    )

    with np.testing.assert_raises_regex(ValueError, "decision-window flag"):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_forged_decision_reason_with_valid_digest(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20, minimum_history=3)
    policy.update("event", 7.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][0].update({
            "decision": "SAFE-EXCLUDE",
            "reason": "score_at_or_below_calibrated_threshold",
        }),
    )

    with np.testing.assert_raises_regex(ValueError, "decision or reason"):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_forged_shift_gate_decision_with_valid_digest(tmp_path):
    proper = pd.DataFrame({"risk": [-9.0, -8.5, -8.0, -7.5, -7.0]})
    calibration = pd.DataFrame({"risk": np.linspace(-9.0, -7.0, 39)})
    gate = ConformalShiftGate(["risk"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    policy = SequentialTriagePolicy(0.20, shift_gate=gate)
    policy.update("event", 5.0, 0.10)
    checkpoint = tmp_path / "gated-runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][0].update({
            "shift_gate_allowed": True,
            "decision": "SAFE-EXCLUDE",
            "reason": "score_at_or_below_calibrated_threshold",
        }),
    )

    with np.testing.assert_raises_regex(ValueError, "shift-gate decision"):
        SequentialTriagePolicy.restore(checkpoint, shift_gate=gate)


def test_restore_rejects_shift_score_missing_with_active_gate(tmp_path):
    proper = pd.DataFrame({"risk": [-9.0, -8.5, -8.0, -7.5, -7.0]})
    calibration = pd.DataFrame({"risk": np.linspace(-9.0, -7.0, 39)})
    gate = ConformalShiftGate(["risk"]).fit(proper)
    gate.calibrate(calibration, alpha=0.10)
    policy = SequentialTriagePolicy(0.20, shift_gate=gate)
    policy.update("event", 5.0, 0.10, {"risk": -8.0})
    checkpoint = tmp_path / "gated-runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][0].update({"shift_score": None}),
    )

    with np.testing.assert_raises_regex(ValueError, "shift score is missing"):
        SequentialTriagePolicy.restore(checkpoint, shift_gate=gate)


def test_restore_rejects_shift_gate_fields_without_gate(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["audit"][0].update({
            "shift_score": 3.0,
            "shift_gate_allowed": False,
        }),
    )

    with np.testing.assert_raises_regex(ValueError, "shift-gate fields"):
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



def test_checkpoint_v2_ledger_upgrades_to_v3_hash_chain(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="a" * 64, calibration_sha256="b" * 64,
        rows=1, events=1, min_time_to_tca=7.0, max_time_to_tca=7.0,
        first_audit_row=1, last_audit_row=1,
    )
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)

    def downgrade(payload):
        payload["schema_version"] = 2
        for entry in payload["processed_batches"]:
            entry.pop("entry_sha256")
            entry.pop("previous_entry_sha256")

    _rewrite_checkpoint_with_valid_digest(checkpoint, downgrade)
    restored = SequentialTriagePolicy.restore(checkpoint)
    ledger = restored.processed_batches()
    assert len(ledger) == 1
    assert ledger[0]["previous_entry_sha256"] is None
    assert len(ledger[0]["entry_sha256"]) == 64

    restored.checkpoint(checkpoint)
    upgraded = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert upgraded["payload"]["schema_version"] == 3
    assert len(upgraded["payload"]["processed_batches"][0]["entry_sha256"]) == 64


def _two_batch_chain_checkpoint(tmp_path):
    policy = SequentialTriagePolicy(safe_threshold=0.20)
    policy.update("event", 7.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="a" * 64, calibration_sha256="b" * 64,
        rows=1, events=1, min_time_to_tca=7.0, max_time_to_tca=7.0,
        first_audit_row=1, last_audit_row=1,
    )
    policy.update("event", 6.0, 0.10)
    policy.record_processed_batch(
        scores_sha256="c" * 64, calibration_sha256="b" * 64,
        rows=1, events=1, min_time_to_tca=6.0, max_time_to_tca=6.0,
        first_audit_row=2, last_audit_row=2,
    )
    checkpoint = tmp_path / "runtime.json"
    policy.checkpoint(checkpoint)
    return checkpoint


def test_restore_rejects_tampered_batch_entry_with_valid_checkpoint_digest(tmp_path):
    checkpoint = _two_batch_chain_checkpoint(tmp_path)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["processed_batches"][0].update({"events": 2}),
    )
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_broken_batch_chain_link(tmp_path):
    checkpoint = _two_batch_chain_checkpoint(tmp_path)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["processed_batches"][1].update(
            {"previous_entry_sha256": "0" * 64}
        ),
    )
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_reordered_batch_chain(tmp_path):
    checkpoint = _two_batch_chain_checkpoint(tmp_path)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["processed_batches"].reverse(),
    )
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)


def test_restore_rejects_nonnull_genesis_link(tmp_path):
    checkpoint = _two_batch_chain_checkpoint(tmp_path)
    _rewrite_checkpoint_with_valid_digest(
        checkpoint,
        lambda payload: payload["processed_batches"][0].update(
            {"previous_entry_sha256": "0" * 64}
        ),
    )
    with np.testing.assert_raises(ValueError):
        SequentialTriagePolicy.restore(checkpoint)

def test_oof_evidence_verification_matches_frozen_development_artifacts():
    root = Path(__file__).resolve().parents[1]
    result = verify_oof_evidence(
        root / "artifacts" / "development_event_aligned_oof_v8.parquet",
        root / "artifacts" / "development_event_aligned_v8.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
    )

    assert result["schema_version"] == 2
    assert result["status"] == "development-only-oof-verification"
    assert result["passed"] is True
    assert len(result["checks"]) == 40
    assert [row["fold"] for row in result["fold_results"]] == [0, 1, 2, 3, 4]
    assert sum(row["danger_k"] for row in result["fold_results"]) == 12
    assert sum(row["danger_n"] for row in result["fold_results"]) == 192
    assert min(row["rank"] for row in result["fold_results"]) == 9
    assert max(row["rank"] for row in result["fold_results"]) == 10
    assert result["recomputed_metrics"]["danger_k"] == 12
    assert result["recomputed_metrics"]["danger_n"] == 192
    assert result["recomputed_metrics"]["safe_negative"] == 5286
    assert abs(result["recomputed_metrics"]["danger_ucb"] - 0.09929771751247034) < 1e-15
    assert "not cryptographically bound" in result["limitations"][0]


def test_oof_evidence_verification_detects_metric_tampering(tmp_path):
    root = Path(__file__).resolve().parents[1]
    development = json.loads(
        (root / "artifacts" / "development_event_aligned_v8.json").read_text(
            encoding="utf-8"
        )
    )
    development["metrics"]["danger_k"] = 11
    development_path = tmp_path / "development_event_aligned_v8.json"
    development_path.write_text(
        json.dumps(development) + "\n", encoding="utf-8"
    )
    preregistration = json.loads(
        (root / "artifacts" / "next_validation_preregistration_v12.json").read_text(
            encoding="utf-8"
        )
    )
    terminal = preregistration["candidate_selection"]["terminal_development_artifacts"]
    key = next(key for key in terminal if Path(key).name == development_path.name)
    terminal[key] = file_sha256(development_path)
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration) + "\n", encoding="utf-8"
    )
    preregistration_lock = tmp_path / "preregistration.lock"
    preregistration_lock.write_text(json.dumps({
        "preregistration_sha256": file_sha256(preregistration_path)
    }) + "\n", encoding="utf-8")
    v13 = json.loads(
        (root / "artifacts" / "catboost_tail_aligned_final_v13.json").read_text(
            encoding="utf-8"
        )
    )
    v13["preregistration"]["sha256"] = file_sha256(preregistration_path)
    v13["preregistration"]["lock_sha256"] = file_sha256(preregistration_lock)
    v13_path = tmp_path / "v13.json"
    v13_path.write_text(json.dumps(v13) + "\n", encoding="utf-8")

    result = verify_oof_evidence(
        root / "artifacts" / "development_event_aligned_oof_v8.parquet",
        development_path,
        v13_path,
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        preregistration_path,
        preregistration_lock,
    )

    assert result["passed"] is False
    failed = {check["name"] for check in result["checks"] if not check["passed"]}
    assert "metric_danger_k" in failed


def test_v13_readiness_verifies_frozen_artifacts_without_outcomes():
    root = Path(__file__).resolve().parents[1]
    result = check_v13_readiness(
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
    )

    assert result["candidate"] == "catboost_tail_aligned"
    assert result["ready_for_scientific_confirmation"] is False
    assert set(result["missing_prerequisites"]) == {
        "prospective_collection", "frozen_new_study"
    }
    assert result["failed_checks"] == []
    assert result["collection"] is None
    assert result["targets"]["recommended_calibration_positive_events"] == 100
    assert result["targets"]["recommended_evaluation_positive_events"] == 200
    assert result["targets"]["minimum_evaluation_positive_events_if_four_failures"] == 89
    assert result["statistical_design"]["calibration"]["rank"] == 5
    assert result["statistical_design"]["evaluation"][
        "maximum_passing_dangerous_exclusions"
    ] == 12
    design_check = next(
        check for check in result["checks"] if check["name"] == "statistical_design"
    )
    assert design_check["passed"] is True


def test_v13_readiness_rejects_inconsistent_sample_size_claims(tmp_path):
    root = Path(__file__).resolve().parents[1]
    preregistration = json.loads(
        (root / "artifacts" / "next_validation_preregistration_v12.json").read_text(
            encoding="utf-8"
        )
    )
    preregistration["new_study"]["recommended_total_positive_events"] = 299
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration) + "\n", encoding="utf-8"
    )
    lock_path = tmp_path / "preregistration.lock"
    lock_path.write_text(json.dumps({
        "preregistration_sha256": file_sha256(preregistration_path)
    }) + "\n", encoding="utf-8")
    manifest = json.loads(
        (root / "artifacts" / "catboost_tail_aligned_final_v13.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["preregistration"]["sha256"] = file_sha256(preregistration_path)
    manifest["preregistration"]["lock_sha256"] = file_sha256(lock_path)
    manifest_path = tmp_path / "model.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = check_v13_readiness(
        preregistration_path, lock_path, manifest_path,
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
    )

    assert "statistical_design" in result["failed_checks"]
    assert result["ready_for_scientific_confirmation"] is False


def test_v13_readiness_rejects_tampered_model_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "artifacts" / "catboost_tail_aligned_final_v13.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["outputs"]["model"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "v13.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = check_v13_readiness(
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        manifest_path,
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
    )

    assert result["ready_for_scientific_confirmation"] is False
    assert "model_binary" in result["failed_checks"]


def test_v13_readiness_reports_unsealed_collection_as_blocking(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = write_external_export(tmp_path / "source.json", [
        external_cdm_record(
            "m1", "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"
        )
    ])
    ledger = tmp_path / "collection.json"
    append_export(
        source, ledger, tmp_path / "batches",
        collection_start_utc="2026-01-01T00:00:00Z",
        collection_end_utc="2026-02-01T00:00:00Z",
    )

    result = check_v13_readiness(
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        ledger=ledger,
    )

    assert "prospective_collection" in result["failed_checks"]
    assert result["collection"]["status"] == "collecting"
    assert result["collection"]["outcomes_accessed"] is False




def _frozen_prospective_study(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = _prospective_two_cohort_collection(tmp_path)
    outputs = {
        "features_output": tmp_path / "all-features.parquet",
        "readiness_output": tmp_path / "readiness.json",
        "calibration_features": tmp_path / "calibration-features.parquet",
        "evaluation_features": tmp_path / "evaluation-features.parquet",
        "calibration_roster": tmp_path / "calibration-roster.parquet",
        "evaluation_roster": tmp_path / "evaluation-roster.parquet",
        "allocation_output": tmp_path / "allocation.json",
    }
    materialize_collection(ledger, **outputs)
    study_manifest = tmp_path / "study.json"
    study_lock = tmp_path / "study.lock"
    freeze_study(
        outputs["calibration_features"],
        outputs["evaluation_features"],
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        study_manifest,
        study_lock,
        calibration_roster=outputs["calibration_roster"],
        evaluation_roster=outputs["evaluation_roster"],
        allocation_manifest=outputs["allocation_output"],
    )
    return ledger, study_manifest, study_lock


def test_close_collection_recovers_matching_outputs_after_interrupted_ledger_write(
    tmp_path, monkeypatch
):
    import external_collection as collection_module

    ledger, study_manifest, study_lock = _frozen_prospective_study(tmp_path)
    labels = tmp_path / "labels.parquet"
    calibration_labels = tmp_path / "calibration-labels.parquet"
    evaluation_labels = tmp_path / "evaluation-labels.parquet"
    real_atomic_json = collection_module._atomic_json

    def interrupt_ledger_write(_payload, _path):
        raise SystemExit("simulated interruption")

    monkeypatch.setattr(collection_module, "_atomic_json", interrupt_ledger_write)
    with np.testing.assert_raises(SystemExit):
        close_collection(
            ledger,
            labels,
            study_manifest=study_manifest,
            study_lock=study_lock,
            calibration_labels_output=calibration_labels,
            evaluation_labels_output=evaluation_labels,
        )
    assert not labels.exists()
    assert not calibration_labels.exists()
    assert not evaluation_labels.exists()
    assert read_collection(ledger)[0]["status"] == "sealed"

    monkeypatch.setattr(collection_module, "_atomic_json", real_atomic_json)
    expected_labels = collection_module.derive_event_labels(
        read_collection(ledger)[1], collection_complete=True
    )
    expected_labels.to_parquet(labels, index=False)
    expected_calibration = expected_labels.loc[
        expected_labels["event_id"].astype(str).isin(
            read_locked_study(study_manifest, study_lock)[0]["cohorts"]["calibration"]["event_ids"]
        )
    ]
    expected_calibration.to_parquet(calibration_labels, index=False)

    closed = close_collection(
        ledger,
        labels,
        study_manifest=study_manifest,
        study_lock=study_lock,
        calibration_labels_output=calibration_labels,
        evaluation_labels_output=evaluation_labels,
    )

    assert closed["status"] == "closed"
    assert closed["labels_sha256"] == file_sha256(labels)
    assert closed["calibration_labels_sha256"] == file_sha256(calibration_labels)
    assert closed["evaluation_labels_sha256"] == file_sha256(evaluation_labels)


def test_close_collection_rejects_mismatched_interrupted_label_output(tmp_path):
    ledger, study_manifest, study_lock = _frozen_prospective_study(tmp_path)
    labels = tmp_path / "labels.parquet"
    calibration_labels = tmp_path / "calibration-labels.parquet"
    evaluation_labels = tmp_path / "evaluation-labels.parquet"
    pd.DataFrame({"event_id": ["forged"], "y": [0]}).to_parquet(labels, index=False)
    original = labels.read_bytes()

    with np.testing.assert_raises_regex(FileExistsError, "does not match"):
        close_collection(
            ledger,
            labels,
            study_manifest=study_manifest,
            study_lock=study_lock,
            calibration_labels_output=calibration_labels,
            evaluation_labels_output=evaluation_labels,
        )

    assert labels.read_bytes() == original
    assert not calibration_labels.exists()
    assert not evaluation_labels.exists()
    assert read_collection(ledger)[0]["status"] == "sealed"


def test_v13_readiness_verifies_study_ledger_cross_binding(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ledger, study_manifest, study_lock = _frozen_prospective_study(tmp_path)

    result = check_v13_readiness(
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        ledger=ledger,
        study_manifest=study_manifest,
        study_lock=study_lock,
    )

    assert result["ready_for_scientific_confirmation"] is True
    assert result["failed_checks"] == []
    frozen = next(
        check for check in result["checks"] if check["name"] == "frozen_new_study"
    )
    assert frozen["passed"] is True
    assert "bound to the supplied sealed ledger" in frozen["detail"]


def test_v13_readiness_rejects_study_from_another_sealed_ledger(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ledger_a, study_manifest, study_lock = _frozen_prospective_study(tmp_path / "a")
    ledger_b, _, _ = _frozen_prospective_study(tmp_path / "b")
    assert file_sha256(ledger_a) != file_sha256(ledger_b)

    result = check_v13_readiness(
        root / "artifacts" / "next_validation_preregistration_v12.json",
        root / "artifacts" / "next_validation_preregistration_v12.lock",
        root / "artifacts" / "catboost_tail_aligned_final_v13.json",
        root / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
        ledger=ledger_b,
        study_manifest=study_manifest,
        study_lock=study_lock,
    )

    assert result["ready_for_scientific_confirmation"] is False
    assert "frozen_new_study" in result["failed_checks"]
    frozen = next(
        check for check in result["checks"] if check["name"] == "frozen_new_study"
    )
    assert frozen["passed"] is False
    assert frozen["detail"] == (
        "new-study allocation is not bound to the supplied sealed ledger"
    )
