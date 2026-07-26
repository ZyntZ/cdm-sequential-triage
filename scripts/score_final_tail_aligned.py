"""Score label-blind CDM histories with the frozen tail-aligned model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_aligned_model import score_dynamic_frame
from snapshot_model import file_sha256


def score_file(features_path: Path, model_path: Path, manifest_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        raise FileExistsError(f"Score output already exists: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("outputs", {}).get("model", {}).get("sha256")
    actual_hash = file_sha256(model_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Model SHA-256 does not match its manifest")
    if manifest.get("calibration_accessed") is not False or manifest.get("evaluation_accessed") is not False:
        raise ValueError("Model manifest does not preserve label-blind scoring status")

    features = pd.read_parquet(features_path)
    if "y" in features.columns:
        raise ValueError("Scoring features must not contain y")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    score_column = manifest.get("score_column")
    scores = score_dynamic_frame(model, features, score_column=score_column)
    scores["model_sha256"] = actual_hash
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output_path, index=False)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = score_file(args.features, args.model, args.manifest, args.output)
    print(f"rows={len(scores)} events={scores['event_id'].nunique()}")
    print(f"score_column={json.loads(args.manifest.read_text())['score_column']}")


if __name__ == "__main__":
    main()
