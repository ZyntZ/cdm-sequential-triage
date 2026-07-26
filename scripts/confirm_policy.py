"""Calibrate the frozen policy and run a single confirmation pass."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confirmation import (
    acquire_confirmation_lock,
    attach_event_labels,
    calibrate,
    evaluate,
    file_sha256,
    policy_from_model_manifest,
    read_json,
    write_json,
)
from shift_gate import ConformalShiftGate
from study import read_locked_study, validate_label_roster


def _study_context(args: argparse.Namespace, cohort: str, labels: pd.DataFrame) -> tuple[dict | None, str | None]:
    options = (args.study_manifest, args.study_lock)
    if any(value is not None for value in options):
        if not all(value is not None for value in options):
            raise ValueError("--study-manifest and --study-lock must be used together")
        manifest, digest = read_locked_study(args.study_manifest, args.study_lock)
        validate_label_roster(labels, manifest, cohort)
        return manifest, digest
    return None, None


def _validate_scored_study(prefixes: pd.DataFrame, digest: str | None, cohort: str) -> None:
    if digest is None:
        return
    required = {"study_manifest_sha256", "study_cohort"}
    missing = required.difference(prefixes.columns)
    if missing:
        raise ValueError(f"Scores are not bound to the frozen study: {sorted(missing)}")
    if set(prefixes["study_manifest_sha256"].astype(str).unique()) != {digest}:
        raise ValueError("Scores reference a different frozen study")
    if set(prefixes["study_cohort"].astype(str).unique()) != {cohort}:
        raise ValueError(f"Scores are not from the frozen {cohort} cohort")


def calibration_command(args: argparse.Namespace) -> None:
    prefixes = pd.read_parquet(args.scores)
    labels = pd.read_parquet(args.labels)
    _, study_hash = _study_context(args, "calibration", labels)
    _validate_scored_study(prefixes, study_hash, "calibration")
    shift_gate = None if args.gate is None else ConformalShiftGate.load(args.gate)
    manifest = None if args.model_manifest is None else read_json(args.model_manifest)
    if manifest is not None and manifest.get("calibration_required_on_genuinely_new_events") is True:
        if study_hash is None:
            raise ValueError("The frozen model requires a locked genuinely new study")
    policy = None if manifest is None else policy_from_model_manifest(manifest)
    artifact = calibrate(prefixes, labels, shift_gate=shift_gate, policy=policy)
    artifact["model_manifest_sha256"] = (
        None if args.model_manifest is None else file_sha256(args.model_manifest)
    )
    artifact["calibration_scores_sha256"] = file_sha256(args.scores)
    artifact["calibration_labels_sha256"] = file_sha256(args.labels)
    artifact["shift_gate_file_sha256"] = (
        None if args.gate is None else file_sha256(args.gate)
    )
    artifact["study_manifest_sha256"] = study_hash
    artifact["study_manifest_file_sha256"] = (
        None if args.study_manifest is None else file_sha256(args.study_manifest)
    )
    artifact["study_lock_file_sha256"] = (
        None if args.study_lock is None else file_sha256(args.study_lock)
    )
    artifact["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, artifact)
    rule = artifact["calibration"]
    print(
        f"threshold={rule['threshold']:.17g} "
        f"rank={rule['rank']} n_positive={rule['n_positive']} "
        f"pac_bound={rule['pac_bound']:.6f}"
    )


def confirmation_command(args: argparse.Namespace) -> None:
    artifact = read_json(args.calibration)
    if args.output.exists():
        raise FileExistsError(f"Confirmation output already exists: {args.output}")
    study_options = (args.study_manifest, args.study_lock)
    if any(value is not None for value in study_options):
        if not all(value is not None for value in study_options):
            raise ValueError("--study-manifest and --study-lock must be used together")
        _, study_hash = read_locked_study(args.study_manifest, args.study_lock)
    else:
        study_hash = None
    lock_payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_artifact": str(args.calibration),
        "calibration_artifact_sha256": file_sha256(args.calibration),
        "evaluation_scores": str(args.scores),
        "evaluation_scores_sha256": file_sha256(args.scores),
        "evaluation_labels": str(args.labels),
        "evaluation_labels_sha256": file_sha256(args.labels),
        "output": str(args.output),
        "study_manifest": None if args.study_manifest is None else str(args.study_manifest),
        "study_manifest_sha256": study_hash,
        "study_lock": None if args.study_lock is None else str(args.study_lock),
        "study_lock_file_sha256": None if args.study_lock is None else file_sha256(args.study_lock),
        "shift_gate": None if args.gate is None else str(args.gate),
        "shift_gate_file_sha256": (
            None if args.gate is None else file_sha256(args.gate)
        ),
        "model_manifest": (
            None if args.model_manifest is None else str(args.model_manifest)
        ),
        "model_manifest_sha256": (
            None if args.model_manifest is None else file_sha256(args.model_manifest)
        ),
    }
    acquire_confirmation_lock(args.lock, lock_payload)
    prefix_scores = pd.read_parquet(args.scores)
    event_labels = pd.read_parquet(args.labels)
    _, study_hash = _study_context(args, "evaluation", event_labels)
    _validate_scored_study(prefix_scores, study_hash, "evaluation")
    if artifact.get("study_manifest_sha256") != study_hash:
        raise ValueError("Calibration and evaluation do not use the same frozen study")
    scored_ids = set(prefix_scores["event_id"].astype(str).unique())
    labels_for_scored_events = event_labels.loc[
        event_labels["event_id"].astype(str).isin(scored_ids)
    ]
    prefixes = attach_event_labels(prefix_scores, labels_for_scored_events)
    shift_gate = None if args.gate is None else ConformalShiftGate.load(args.gate)
    manifest = None if args.model_manifest is None else read_json(args.model_manifest)
    if manifest is not None and manifest.get("calibration_required_on_genuinely_new_events") is True:
        if study_hash is None:
            raise ValueError("The frozen model requires a locked genuinely new study")
    policy = None if manifest is None else policy_from_model_manifest(manifest)
    expected_manifest_hash = artifact.get("model_manifest_sha256")
    supplied_manifest_hash = (
        None if args.model_manifest is None else file_sha256(args.model_manifest)
    )
    if expected_manifest_hash != supplied_manifest_hash:
        raise ValueError("Calibration artifact and supplied model manifest do not match")
    result = evaluate(
        prefixes, artifact, event_labels, shift_gate=shift_gate, policy=policy
    )
    result.update(lock_payload)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    metrics = result["evaluation"]
    print(
        f"danger={metrics['danger_k']}/{metrics['danger_n']} "
        f"ucb95={metrics['danger_ucb']:.6f} "
        f"safe_negative_rate={metrics['safe_negative_rate']:.6f} "
        f"median_first_safe_tca={metrics['median_first_safe_tca']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--scores", type=Path, required=True)
    calibration.add_argument("--labels", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--gate", type=Path)
    calibration.add_argument("--model-manifest", type=Path)
    calibration.add_argument("--study-manifest", type=Path)
    calibration.add_argument("--study-lock", type=Path)
    calibration.set_defaults(handler=calibration_command)

    confirmation = commands.add_parser("confirm")
    confirmation.add_argument("--scores", type=Path, required=True)
    confirmation.add_argument("--calibration", type=Path, required=True)
    confirmation.add_argument("--labels", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--lock", type=Path, required=True)
    confirmation.add_argument("--gate", type=Path)
    confirmation.add_argument("--model-manifest", type=Path)
    confirmation.add_argument("--study-manifest", type=Path)
    confirmation.add_argument("--study-lock", type=Path)
    confirmation.set_defaults(handler=confirmation_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
