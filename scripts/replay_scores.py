"""Replay a pre-scored CDM stream through the calibrated runtime policy."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confirmation import (
    file_sha256, policy_from_model_manifest, read_json, validate_policy,
)
from shift_gate import ConformalShiftGate
from triage import SequentialTriagePolicy


def load_runtime(
    calibration_path: Path,
    gate_path: Path | None = None,
    model_manifest_path: Path | None = None,
    checkpoint_path: Path | None = None,
    escalation_threshold: float | None = None,
) -> tuple[SequentialTriagePolicy, dict]:
    artifact = read_json(calibration_path)
    policy_config = validate_policy(artifact.get("policy"))
    calibration = artifact.get("calibration")
    if not isinstance(calibration, dict) or "threshold" not in calibration:
        raise ValueError("Calibration artifact has no decision threshold")

    expected_manifest_hash = artifact.get("model_manifest_sha256")
    supplied_manifest_hash = (
        None if model_manifest_path is None else file_sha256(model_manifest_path)
    )
    if expected_manifest_hash != supplied_manifest_hash:
        raise ValueError("Calibration artifact and model manifest do not match")
    if model_manifest_path is not None:
        manifest = read_json(model_manifest_path)
        if policy_from_model_manifest(manifest) != policy_config:
            raise ValueError("Calibration policy and model manifest do not match")

    gate = None if gate_path is None else ConformalShiftGate.load(gate_path)
    gate_fingerprint = None if gate is None else gate.fingerprint()
    if artifact.get("shift_gate_sha256") != gate_fingerprint:
        raise ValueError("Calibration artifact and shift gate do not match")

    if checkpoint_path is not None and checkpoint_path.exists():
        runtime = SequentialTriagePolicy.restore(checkpoint_path, shift_gate=gate)
        expected_configuration = {
            "safe_threshold": runtime._encode_float(float(calibration["threshold"])),
            "minimum_history": int(policy_config["minimum_history"]),
            "escalation_threshold": runtime._encode_float(escalation_threshold),
            "min_days_to_tca": float(policy_config["min_days_to_tca"]),
            "max_days_to_tca": float(policy_config["max_days_to_tca"]),
            "shift_gate_fingerprint": gate_fingerprint,
        }
        if runtime._configuration_payload() != expected_configuration:
            raise ValueError("Checkpoint does not match the calibrated runtime configuration")
    else:
        runtime = SequentialTriagePolicy(
            safe_threshold=float(calibration["threshold"]),
            minimum_history=int(policy_config["minimum_history"]),
            escalation_threshold=escalation_threshold,
            shift_gate=gate,
            min_days_to_tca=float(policy_config["min_days_to_tca"]),
            max_days_to_tca=float(policy_config["max_days_to_tca"]),
        )
    return runtime, policy_config


def validate_score_stream(
    scores: pd.DataFrame,
    artifact: dict,
    score_column: str,
    gate: ConformalShiftGate | None,
) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", score_column, "model_sha256"}
    if gate is not None:
        required.update(gate.feature_columns)
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing runtime score columns: {sorted(missing)}")
    if scores.empty:
        raise ValueError("Score stream must contain at least one row")
    if scores["event_id"].isna().any():
        raise ValueError("event_id must not contain missing values")
    if scores.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca updates are not allowed")
    tca = pd.to_numeric(scores["time_to_tca"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(scores[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tca).all() or (tca < 0).any():
        raise ValueError("time_to_tca must be finite and non-negative")
    if not np.isfinite(values).all():
        raise ValueError("Runtime scores must be finite")
    model_hashes = scores["model_sha256"].dropna().astype(str).unique()
    if len(model_hashes) != 1 or scores["model_sha256"].isna().any():
        raise ValueError("Scores must contain exactly one model_sha256")
    if str(model_hashes[0]) != str(artifact.get("model_sha256")):
        raise ValueError("Score stream and calibration artifact use different models")
    calibration_ids = set(artifact.get("calibration_event_ids", []))
    incoming_ids = set(scores["event_id"].astype(str).unique())
    overlap = calibration_ids.intersection(incoming_ids)
    if overlap:
        raise ValueError(f"Runtime stream overlaps calibration by {len(overlap)} event_id values")
    ordered = scores.copy()
    ordered["__event_order"] = pd.factorize(ordered["event_id"], sort=False)[0]
    ordered = ordered.sort_values(
        ["__event_order", "time_to_tca"], ascending=[True, False], kind="mergesort"
    )
    return ordered.drop(columns="__event_order")


def replay_scores(
    scores: pd.DataFrame,
    runtime: SequentialTriagePolicy,
    score_column: str,
) -> pd.DataFrame:
    ordered = validate_score_stream(
        scores,
        {"model_sha256": scores["model_sha256"].iloc[0], "calibration_event_ids": []},
        score_column,
        runtime.shift_gate,
    )
    gate_columns = [] if runtime.shift_gate is None else runtime.shift_gate.feature_columns
    for row in ordered.to_dict(orient="records"):
        gate_features = (
            None if not gate_columns else {column: row[column] for column in gate_columns}
        )
        runtime.update(
            event_id=row["event_id"],
            time_to_tca=row["time_to_tca"],
            score=row[score_column],
            gate_features=gate_features,
        )
    return runtime.audit_log()


def _write_parquet_atomic(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
        frame.to_parquet(temporary_name, index=False)
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def run_replay(
    scores_path: Path,
    calibration_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    gate_path: Path | None = None,
    model_manifest_path: Path | None = None,
    escalation_threshold: float | None = None,
) -> pd.DataFrame:
    if output_path.exists():
        raise FileExistsError(f"Replay output already exists: {output_path}")
    artifact = read_json(calibration_path)
    runtime, policy_config = load_runtime(
        calibration_path,
        gate_path=gate_path,
        model_manifest_path=model_manifest_path,
        checkpoint_path=checkpoint_path,
        escalation_threshold=escalation_threshold,
    )
    scores = pd.read_parquet(scores_path)
    ordered = validate_score_stream(
        scores, artifact, policy_config["score_column"], runtime.shift_gate
    )
    audit = replay_scores(ordered, runtime, policy_config["score_column"])
    audit["scores_sha256"] = file_sha256(scores_path)
    audit["calibration_sha256"] = file_sha256(calibration_path)
    audit["model_sha256"] = str(artifact["model_sha256"])
    audit["shift_gate_sha256"] = artifact.get("shift_gate_sha256")
    audit["model_manifest_sha256"] = artifact.get("model_manifest_sha256")
    checkpoint_digest = None
    if checkpoint_path is not None:
        checkpoint_digest = runtime.checkpoint(checkpoint_path)
    audit["runtime_checkpoint_sha256"] = checkpoint_digest
    _write_parquet_atomic(audit, output_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a pre-scored CDM stream through a calibrated policy"
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--escalation-threshold", type=float)
    args = parser.parse_args()
    audit = run_replay(
        args.scores, args.calibration, args.output,
        checkpoint_path=args.checkpoint, gate_path=args.gate,
        model_manifest_path=args.model_manifest,
        escalation_threshold=args.escalation_threshold,
    )
    counts = audit["decision"].value_counts().to_dict()
    print(f"rows={len(audit)} events={audit['event_id'].nunique()}")
    print(" ".join(f"{decision}={counts.get(decision, 0)}" for decision in ("SAFE-EXCLUDE", "MONITOR", "ESCALATE")))


if __name__ == "__main__":
    main()
