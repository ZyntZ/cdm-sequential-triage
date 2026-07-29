"""Manage an append-only prospective collection of offline CDM exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study import read_locked_study
from validation_plan import minimum_positive_events, validation_design_summary

from external_collection import (
    append_export,
    close_collection,
    collection_status,
    materialize_collection,
    seal_collection,
)



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_v13_readiness(
    preregistration: Path,
    preregistration_lock: Path,
    model_manifest: Path,
    model: Path,
    ledger: Path | None = None,
    study_manifest: Path | None = None,
    study_lock: Path | None = None,
) -> dict:
    """Check v13 validation prerequisites without reading scores or outcomes."""
    checks = []

    def record(name: str, passed: bool | None, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    preregistration_sha256 = _sha256(preregistration)
    lock = json.loads(preregistration_lock.read_text(encoding="utf-8"))
    lock_ok = lock.get("preregistration_sha256") == preregistration_sha256
    record(
        "preregistration_lock",
        lock_ok,
        "preregistration matches its frozen SHA-256" if lock_ok
        else "preregistration SHA-256 does not match its lock",
    )

    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    manifest_binding_ok = (
        manifest.get("preregistration", {}).get("sha256") == preregistration_sha256
        and manifest.get("preregistration", {}).get("lock_sha256")
        == _sha256(preregistration_lock)
    )
    record(
        "model_preregistration_binding",
        manifest_binding_ok,
        "v13 manifest is bound to the frozen preregistration" if manifest_binding_ok
        else "v13 manifest does not match the preregistration or its lock",
    )

    expected_model_sha256 = manifest.get("outputs", {}).get("model", {}).get("sha256")
    actual_model_sha256 = _sha256(model)
    model_ok = bool(expected_model_sha256) and actual_model_sha256 == expected_model_sha256
    record(
        "model_binary",
        model_ok,
        f"v13 model SHA-256 verified: {actual_model_sha256}" if model_ok
        else "v13 model binary SHA-256 does not match its manifest",
    )

    firewall_ok = (
        manifest.get("calibration_accessed") is False
        and manifest.get("evaluation_accessed") is False
        and manifest.get("threshold") is None
    )
    record(
        "outcome_firewall",
        firewall_ok,
        "calibration and evaluation remain unopened; threshold is unset" if firewall_ok
        else "v13 outcome firewall flags are not in the frozen pre-calibration state",
    )

    new_study = prereg.get("new_study", {})
    requirements_ok = (
        new_study.get("requires_genuinely_new_events") is True
        and new_study.get("calibration_and_evaluation_event_sets_disjoint") is True
        and new_study.get("no_threshold_or_model_tuning_after_outcome_access") is True
        and manifest.get("calibration_required_on_genuinely_new_events") is True
    )
    record(
        "new_study_requirements",
        requirements_ok,
        "genuinely new, disjoint cohorts are required before v13 calibration" if requirements_ok
        else "frozen new-study requirements are incomplete or inconsistent",
    )

    collection = None
    if ledger is None:
        record(
            "prospective_collection",
            None,
            "no ledger supplied; real prospective collection has not been demonstrated",
        )
    else:
        collection = collection_status(ledger)
        collection_ok = (
            collection.get("integrity_verified") is True
            and collection.get("outcomes_accessed") is False
            and collection.get("status") == "sealed"
        )
        record(
            "prospective_collection",
            collection_ok,
            (
                f"sealed ledger integrity verified; "
                f"eligible events={collection['events_eligible_minimum_history']}"
            ) if collection_ok else (
                "collection must be integrity-verified, outcome-blind, and sealed; "
                f"current status={collection.get('status')}"
            ),
        )

    study = None
    study_options = (study_manifest, study_lock)
    if not any(value is not None for value in study_options):
        record(
            "frozen_new_study",
            None,
            "no new-study manifest and lock supplied",
        )
    elif not all(value is not None for value in study_options):
        record(
            "frozen_new_study",
            False,
            "study manifest and study lock must be supplied together",
        )
    else:
        try:
            study, _ = read_locked_study(study_manifest, study_lock)
            preregistration_record = study.get("preregistration", {})
            study_ok = (
                study.get("outcomes_accessed") is False
                and preregistration_record.get("sha256") == preregistration_sha256
                and preregistration_record.get("lock_sha256")
                == _sha256(preregistration_lock)
            )
            detail = (
                "new disjoint study is frozen before outcome access"
                if study_ok
                else "new-study manifest is not bound to the v13 preregistration and lock"
            )
            if study_ok and collection is not None:
                allocation = study.get("allocation")
                collection_allocation = collection.get("allocation")
                binding_ok = (
                    isinstance(allocation, dict)
                    and isinstance(collection_allocation, dict)
                    and allocation.get("collection_status") == "sealed"
                    and allocation.get("outcomes_accessed") is False
                    and allocation.get("ledger_sha256") == collection.get("ledger_sha256")
                    and allocation.get("assignments_sha256")
                    == collection_allocation.get("assignments_sha256")
                    and allocation.get("rule") == collection_allocation.get("rule")
                    and int(allocation.get("seed", -1))
                    == int(collection_allocation.get("seed", -2))
                    and float(allocation.get("calibration_fraction", -1.0))
                    == float(collection_allocation.get("calibration_fraction", -2.0))
                )
                study_ok = binding_ok
                detail = (
                    "new disjoint study is frozen and bound to the supplied sealed ledger"
                    if binding_ok
                    else "new-study allocation is not bound to the supplied sealed ledger"
                )
            record("frozen_new_study", study_ok, detail)
        except Exception as error:
            record("frozen_new_study", False, f"new-study lock check failed: {error}")

    targets = {
        "recommended_calibration_positive_events": int(
            new_study.get("recommended_calibration_positive_events", 0)
        ),
        "recommended_evaluation_positive_events": int(
            new_study.get("recommended_evaluation_positive_events", 0)
        ),
        "recommended_total_positive_events": int(
            new_study.get("recommended_total_positive_events", 0)
        ),
        "minimum_evaluation_positive_events_if_four_failures": int(
            new_study.get("minimum_positive_events_if_four_failures", 0)
        ),
        "evaluation_success_criterion": new_study.get("primary_success"),
    }
    alpha = float(prereg.get("candidate", {}).get("alpha", 0.10))
    calibration_confidence = float(
        prereg.get("candidate", {}).get("calibration_confidence", 0.95)
    )
    evaluation_confidence = float(
        prereg.get("candidate", {}).get("evaluation_confidence", 0.95)
    )
    statistical_design = None
    try:
        statistical_design = validation_design_summary(
            targets["recommended_calibration_positive_events"],
            targets["recommended_evaluation_positive_events"],
            alpha=alpha,
            confidence=calibration_confidence,
        )
        if evaluation_confidence != calibration_confidence:
            evaluation = validation_design_summary(
                targets["recommended_calibration_positive_events"],
                targets["recommended_evaluation_positive_events"],
                alpha=alpha,
                confidence=evaluation_confidence,
            )["evaluation"]
            statistical_design["evaluation"] = evaluation
        expected_total = (
            targets["recommended_calibration_positive_events"]
            + targets["recommended_evaluation_positive_events"]
        )
        expected_four_failure_minimum = minimum_positive_events(
            4, alpha=alpha, confidence=evaluation_confidence
        )
        design_ok = (
            statistical_design["calibration"]["finite_threshold_available"]
            and statistical_design["calibration"]["pac_bound"] <= alpha
            and statistical_design["evaluation"][
                "maximum_passing_dangerous_exclusions"
            ] >= 0
            and targets["recommended_total_positive_events"] == expected_total
            and targets["minimum_evaluation_positive_events_if_four_failures"]
            == expected_four_failure_minimum
        )
        record(
            "statistical_design",
            design_ok,
            (
                "finite-sample calibration and evaluation claims recomputed successfully"
                if design_ok else
                "preregistered sample-size claims are internally inconsistent"
            ),
        )
    except (TypeError, ValueError):
        record(
            "statistical_design", False,
            "preregistered sample-size or error-control parameters are invalid",
        )

    failed = [check["name"] for check in checks if check["passed"] is False]
    missing = [check["name"] for check in checks if check["passed"] is None]
    ready_for_scientific_confirmation = (
        not failed and not missing and collection is not None and study is not None
    )
    return {
        "schema_version": 1,
        "candidate": manifest.get("score_column"),
        "ready_for_scientific_confirmation": ready_for_scientific_confirmation,
        "failed_checks": failed,
        "missing_prerequisites": missing,
        "checks": checks,
        "targets": targets,
        "statistical_design": statistical_design,
        "collection": collection,
        "interpretation": (
            "v13 is frozen and technically ready, but scientific confirmation requires "
            "a real prospective ledger, a locked new study, PAC calibration, and one disjoint evaluation."
        ),
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append, audit, and close a prospective external CDM collection"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    append = commands.add_parser("append")
    append.add_argument("--input", type=Path, required=True)
    append.add_argument("--ledger", type=Path, required=True)
    append.add_argument("--batches-dir", type=Path, required=True)
    append.add_argument("--collection-start-utc", required=True)
    append.add_argument("--collection-end-utc", required=True)
    append.add_argument("--tca-tolerance-minutes", type=int, default=30)
    append.add_argument("--allocation-seed", type=int, default=24072026)
    append.add_argument("--calibration-fraction", type=float, default=1 / 3)

    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--ledger", type=Path, required=True)
    snapshot.add_argument("--features-output", type=Path, required=True)
    snapshot.add_argument("--readiness-output", type=Path, required=True)
    snapshot.add_argument("--calibration-features", type=Path)
    snapshot.add_argument("--evaluation-features", type=Path)
    snapshot.add_argument("--calibration-roster", type=Path)
    snapshot.add_argument("--evaluation-roster", type=Path)
    snapshot.add_argument("--allocation-output", type=Path)

    seal = commands.add_parser("seal")
    seal.add_argument("--ledger", type=Path, required=True)

    close = commands.add_parser("close")
    close.add_argument("--ledger", type=Path, required=True)
    close.add_argument("--labels-output", type=Path, required=True)
    close.add_argument("--calibration-labels-output", type=Path, required=True)
    close.add_argument("--evaluation-labels-output", type=Path, required=True)
    close.add_argument("--study-manifest", type=Path, required=True)
    close.add_argument("--study-lock", type=Path, required=True)

    status = commands.add_parser("status")
    status.add_argument("--ledger", type=Path, required=True)
    status.add_argument("--minimum-history", type=int, default=3)
    status.add_argument("--min-days", type=float, default=2.0)
    status.add_argument("--max-days", type=float, default=7.0)

    readiness = commands.add_parser("check-v13")
    readiness.add_argument(
        "--preregistration", type=Path,
        default=ROOT / "artifacts" / "next_validation_preregistration_v12.json",
    )
    readiness.add_argument(
        "--preregistration-lock", type=Path,
        default=ROOT / "artifacts" / "next_validation_preregistration_v12.lock",
    )
    readiness.add_argument(
        "--model-manifest", type=Path,
        default=ROOT / "artifacts" / "catboost_tail_aligned_final_v13.json",
    )
    readiness.add_argument(
        "--model", type=Path,
        default=ROOT / "artifacts" / "catboost_tail_aligned_final_v13.cbm",
    )
    readiness.add_argument("--ledger", type=Path)
    readiness.add_argument("--study-manifest", type=Path)
    readiness.add_argument("--study-lock", type=Path)

    args = parser.parse_args()
    if args.command == "append":
        result = append_export(
            args.input, args.ledger, args.batches_dir,
            collection_start_utc=args.collection_start_utc,
            collection_end_utc=args.collection_end_utc,
            tca_tolerance_minutes=args.tca_tolerance_minutes,
            allocation_seed=args.allocation_seed,
            calibration_fraction=args.calibration_fraction,
        )
    elif args.command == "snapshot":
        result = materialize_collection(
            args.ledger, args.features_output, args.readiness_output,
            calibration_features=args.calibration_features,
            evaluation_features=args.evaluation_features,
            calibration_roster=args.calibration_roster,
            evaluation_roster=args.evaluation_roster,
            allocation_output=args.allocation_output,
        )
    elif args.command == "seal":
        result = seal_collection(args.ledger)
    elif args.command == "close":
        result = close_collection(
            args.ledger, args.labels_output,
            study_manifest=args.study_manifest,
            study_lock=args.study_lock,
            calibration_labels_output=args.calibration_labels_output,
            evaluation_labels_output=args.evaluation_labels_output,
        )
    elif args.command == "check-v13":
        result = check_v13_readiness(
            args.preregistration,
            args.preregistration_lock,
            args.model_manifest,
            args.model,
            ledger=args.ledger,
            study_manifest=args.study_manifest,
            study_lock=args.study_lock,
        )
    else:
        result = collection_status(
            args.ledger,
            minimum_history=args.minimum_history,
            min_days=args.min_days,
            max_days=args.max_days,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
