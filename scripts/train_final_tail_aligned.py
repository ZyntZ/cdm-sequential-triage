"""Train the preregistered tail-aligned model on the full development set."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_aligned_diagnostics import attach_event_folds, inner_oof_scores
from event_aligned_model import (
    DYNAMIC_FEATURES,
    fit_dynamic_model,
    positive_tail_weights,
    prepare_dynamic_frame,
)
from snapshot_model import MODEL_PARAMS, file_sha256

EXPECTED_SCORE = "catboost_tail_aligned"
EXPECTED_MODEL = "two-stage CatBoost with nested inner-OOF positive-tail weights"
EXPECTED_TRAINING_SHA256 = "0a27a6afd44a0b8094b5059c93f648597bcbd4fff0f87e8a90ffd6fa8883b968"


def read_locked_candidate(preregistration: Path, lock: Path) -> tuple[dict, str]:
    if not preregistration.is_file() or not lock.is_file():
        raise FileNotFoundError("Preregistration and lock files are required")
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    actual_hash = file_sha256(preregistration)
    if lock_payload.get("preregistration_sha256") != actual_hash:
        raise ValueError("Preregistration hash does not match its lock")
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-before-new-data":
        raise ValueError("Preregistration is not frozen-before-new-data")
    if payload.get("evaluation_accessed") is not False:
        raise ValueError("Preregistration does not declare evaluation_accessed=false")
    candidate = payload.get("candidate", {})
    if candidate.get("score") != EXPECTED_SCORE or candidate.get("model") != EXPECTED_MODEL:
        raise ValueError("Preregistered candidate does not match the tail-aligned trainer")
    return candidate, actual_hash


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_final_model(
    training_path: Path,
    fold_table_path: Path,
    preregistration_path: Path,
    lock_path: Path,
    model_path: Path,
    manifest_path: Path,
) -> dict:
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("Final model and manifest outputs must not already exist")
    candidate, preregistration_sha256 = read_locked_candidate(
        preregistration_path, lock_path
    )
    training_sha256 = file_sha256(training_path)
    if training_sha256 != EXPECTED_TRAINING_SHA256:
        raise ValueError("Development training SHA-256 does not match the frozen partition")

    raw_training = pd.read_parquet(training_path)
    prepared = prepare_dynamic_frame(raw_training)
    fold_table = pd.read_parquet(fold_table_path)
    prepared = attach_event_folds(prepared, fold_table)
    if prepared["fold"].nunique() != 5:
        raise ValueError("The final trainer requires the frozen five-fold assignment")

    model_params = MODEL_PARAMS.copy()
    model_params["iterations"] = int(candidate["iterations"])
    base_scores = inner_oof_scores(prepared, model_params=model_params)
    weights = positive_tail_weights(
        prepared,
        base_scores,
        hard_fraction=float(candidate["hard_fraction"]),
        hard_mass=float(candidate["hard_mass"]),
    )
    weight_totals = pd.Series(weights).groupby(prepared["event_id"].reset_index(drop=True)).sum()
    if not np.allclose(weight_totals.to_numpy(dtype=float), 1.0):
        raise RuntimeError("Final training weights do not sum to one per event")

    started = datetime.now(timezone.utc)
    model = fit_dynamic_model(prepared, weights, model_params=model_params)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_sha256 = file_sha256(model_path)

    event_labels = prepared.groupby("event_id")["y"].first()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preregistration-locked-development-fit",
        "score_column": EXPECTED_SCORE,
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "lock_path": str(lock_path),
            "lock_sha256": file_sha256(lock_path),
        },
        "candidate": candidate,
        "model_params": model_params,
        "features": list(DYNAMIC_FEATURES),
        "training_method": {
            "base_scores": "fresh five-fold inner-OOF dynamic CatBoost scores",
            "tail_weights": "positive events only; total weight one per event",
            "final_fit": "all development events",
        },
        "inputs": {
            "training": {
                "path": str(training_path),
                "sha256": training_sha256,
                "prefix_rows": int(len(prepared)),
                "events": int(prepared["event_id"].nunique()),
                "positive_events": int(event_labels.sum()),
            },
            "fold_table": {
                "path": str(fold_table_path),
                "sha256": file_sha256(fold_table_path),
                "folds": int(prepared["fold"].nunique()),
            },
        },
        "outputs": {
            "model": {"path": str(model_path), "sha256": model_sha256}
        },
        "calibration_accessed": False,
        "evaluation_accessed": False,
        "threshold": None,
        "calibration_required_on_genuinely_new_events": True,
        "elapsed_seconds_final_fit": (
            datetime.now(timezone.utc) - started
        ).total_seconds(),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--fold-table", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = train_final_model(
        args.training,
        args.fold_table,
        args.preregistration,
        args.lock,
        args.model,
        args.manifest,
    )
    print(f"model_sha256={manifest['outputs']['model']['sha256']}")
    print(f"events={manifest['inputs']['training']['events']}")
    print("threshold=UNSET; new-event calibration required")


if __name__ == "__main__":
    main()
