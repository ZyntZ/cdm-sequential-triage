"""Train the frozen snapshot model and score held-out event splits."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snapshot_model import (
    CATEGORICAL_FEATURES,
    MAX_DAYS_TO_TCA,
    MIN_DAYS_TO_TCA,
    MODEL_PARAMS,
    NUMERIC_FEATURES,
    assert_disjoint_splits,
    file_sha256,
    fit_snapshot_model,
    score_snapshot_model,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--evaluation-scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    frames = {
        "training": pd.read_parquet(args.training),
        "calibration": pd.read_parquet(args.calibration),
        "evaluation": pd.read_parquet(args.evaluation_features),
    }
    assert_disjoint_splits(frames)
    model = fit_snapshot_model(frames["training"])

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model)
    model_sha256 = file_sha256(args.model)
    calibration_scores = score_snapshot_model(model, frames["calibration"]).assign(
        model_sha256=model_sha256
    )
    evaluation_scores = score_snapshot_model(
        model, frames["evaluation"], include_labels=False
    ).assign(model_sha256=model_sha256)
    args.calibration_scores.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_scores.parent.mkdir(parents=True, exist_ok=True)
    calibration_scores.to_parquet(args.calibration_scores, index=False)
    evaluation_scores.to_parquet(args.evaluation_scores, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "CatBoostClassifier snapshot",
        "model_params": MODEL_PARAMS,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "window_days": [MIN_DAYS_TO_TCA, MAX_DAYS_TO_TCA],
        "prefix_weighting": "equal total weight per event",
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in {
                "training": args.training,
                "calibration": args.calibration,
                "evaluation": args.evaluation_features,
            }.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in {
                "model": args.model,
                "calibration_scores": args.calibration_scores,
                "evaluation_scores": args.evaluation_scores,
            }.items()
        },
        "events": {
            name: int(frame["event_id"].nunique())
            for name, frame in frames.items()
        },
    }
    write_json(args.manifest, manifest)
    print(f"model_sha256={manifest['outputs']['model']['sha256']}")
    print(
        f"calibration_events={calibration_scores['event_id'].nunique()} "
        f"evaluation_events={evaluation_scores['event_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
