from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy import (
    calibrate_positive_threshold,
    cp_upper,
    first_safe_decision_table,
    history_gated_event_table,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _terminal_artifact_digest(preregistration: dict[str, Any], name: str) -> str | None:
    records = preregistration.get("candidate_selection", {}).get(
        "terminal_development_artifacts", {}
    )
    matches = [value for key, value in records.items() if Path(key).name == name]
    if len(matches) != 1:
        return None
    return matches[0]


def verify_oof_evidence(
    oof_path: str | Path,
    development_manifest_path: str | Path,
    v13_manifest_path: str | Path,
    model_path: str | Path,
    preregistration_path: str | Path,
    preregistration_lock_path: str | Path,
) -> dict[str, Any]:
    """Verify committed development-only OOF evidence without fitting a model."""
    oof_path = Path(oof_path)
    development_manifest_path = Path(development_manifest_path)
    v13_manifest_path = Path(v13_manifest_path)
    model_path = Path(model_path)
    preregistration_path = Path(preregistration_path)
    preregistration_lock_path = Path(preregistration_lock_path)

    development = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    v13 = json.loads(v13_manifest_path.read_text(encoding="utf-8"))
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    preregistration_lock = json.loads(
        preregistration_lock_path.read_text(encoding="utf-8")
    )
    oof = pd.read_parquet(oof_path)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        })

    preregistration_sha256 = file_sha256(preregistration_path)
    preregistration_lock_sha256 = file_sha256(preregistration_lock_path)
    development_manifest_sha256 = file_sha256(development_manifest_path)
    model_sha256 = file_sha256(model_path)
    oof_sha256 = file_sha256(oof_path)

    record(
        "preregistration_lock",
        preregistration_lock.get("preregistration_sha256") == preregistration_sha256,
        preregistration_lock.get("preregistration_sha256"),
        preregistration_sha256,
    )
    record(
        "v13_preregistration_binding",
        v13.get("preregistration", {}).get("sha256") == preregistration_sha256,
        v13.get("preregistration", {}).get("sha256"),
        preregistration_sha256,
    )
    record(
        "v13_preregistration_lock_binding",
        v13.get("preregistration", {}).get("lock_sha256")
        == preregistration_lock_sha256,
        v13.get("preregistration", {}).get("lock_sha256"),
        preregistration_lock_sha256,
    )
    expected_development_sha256 = _terminal_artifact_digest(
        preregistration, development_manifest_path.name
    )
    record(
        "development_manifest_binding",
        expected_development_sha256 == development_manifest_sha256,
        development_manifest_sha256,
        expected_development_sha256,
    )
    expected_model_sha256 = v13.get("outputs", {}).get("model", {}).get("sha256")
    record(
        "model_binary_binding",
        expected_model_sha256 == model_sha256,
        model_sha256,
        expected_model_sha256,
    )

    required = {
        "event_id", "time_to_tca", "y", "fold",
        "eligible_history_count", "catboost_tail_aligned",
    }
    missing = sorted(required.difference(oof.columns))
    record("oof_required_columns", not missing, missing, [])
    if missing:
        return {
            "schema_version": 2,
            "status": "development-only-oof-verification",
            "passed": False,
            "note": (
                "This verifies internal consistency of development artifacts only. "
                "It is not confirmation evidence and does not access calibration or "
                "evaluation outcomes."
            ),
            "artifacts": {"oof_sha256": oof_sha256},
            "checks": checks,
            "fold_results": None,
            "recomputed_metrics": None,
        }

    metrics = development.get("metrics", {})
    candidate = v13.get("candidate", {})
    rows = int(len(oof))
    events = int(oof["event_id"].nunique())
    event_labels = oof.groupby("event_id", sort=False)["y"].first()
    positives = int(event_labels.sum())
    folds = sorted(int(value) for value in oof["fold"].unique())
    finite_scores = bool(np.isfinite(oof["catboost_tail_aligned"]).all())
    bounded_scores = bool(oof["catboost_tail_aligned"].between(0.0, 1.0).all())
    labels_valid = bool(oof["y"].isin([0, 1]).all())
    labels_consistent = bool((oof.groupby("event_id")["y"].nunique() == 1).all())
    event_folds_consistent = bool(
        (oof.groupby("event_id")["fold"].nunique() == 1).all()
    )

    record("prefix_rows", rows == int(metrics.get("prefix_rows", -1)), rows, metrics.get("prefix_rows"))
    record("events", events == int(metrics.get("events", -1)), events, metrics.get("events"))
    record("positive_events", positives == int(v13.get("inputs", {}).get("training", {}).get("positive_events", -1)), positives, v13.get("inputs", {}).get("training", {}).get("positive_events"))
    record("fold_roster", folds == [0, 1, 2, 3, 4], folds, [0, 1, 2, 3, 4])
    record("unique_prefixes", not bool(oof.duplicated(["event_id", "time_to_tca"]).any()), int(oof.duplicated(["event_id", "time_to_tca"]).sum()), 0)
    record("finite_scores", finite_scores, finite_scores, True)
    record("bounded_scores", bounded_scores, bounded_scores, True)
    record("binary_labels", labels_valid, labels_valid, True)
    record("event_label_consistency", labels_consistent, labels_consistent, True)
    record("event_fold_consistency", event_folds_consistent, event_folds_consistent, True)

    preregistered_candidate = preregistration.get("candidate")
    record(
        "v13_candidate_preregistration_binding",
        isinstance(preregistered_candidate, dict) and candidate == preregistered_candidate,
        candidate,
        preregistered_candidate,
    )
    parameter_pairs = {
        "hard_fraction": (candidate.get("hard_fraction"), metrics.get("hard_fraction")),
        "hard_mass": (candidate.get("hard_mass"), metrics.get("hard_mass")),
        "iterations": (candidate.get("iterations"), development.get("model_iterations")),
        "minimum_history": (candidate.get("minimum_history"), metrics.get("minimum_history")),
        "alpha": (candidate.get("alpha"), metrics.get("alpha")),
        "calibration_mode": (candidate.get("calibration_mode"), metrics.get("mode")),
        "calibration_confidence": (candidate.get("calibration_confidence"), metrics.get("confidence")),
    }
    for name, (observed, expected) in parameter_pairs.items():
        record(f"candidate_{name}", observed == expected, observed, expected)
    record("v13_calibration_unopened", v13.get("calibration_accessed") is False, v13.get("calibration_accessed"), False)
    record("v13_evaluation_unopened", v13.get("evaluation_accessed") is False, v13.get("evaluation_accessed"), False)
    record("v13_threshold_unset", v13.get("threshold") is None, v13.get("threshold"), None)
    record("development_status", development.get("status") == "development-only", development.get("status"), "development-only")

    minimum_history = int(candidate["minimum_history"])
    alpha = float(candidate["alpha"])
    confidence = float(candidate["calibration_confidence"])
    mode = str(candidate["calibration_mode"])
    decisions = []
    thresholds: list[float] = []
    ranks: list[int] = []
    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        calibration = oof.loc[oof["fold"] != fold]
        held_out = oof.loc[oof["fold"] == fold]
        calibration_events = history_gated_event_table(
            calibration, "catboost_tail_aligned", minimum_history
        )
        rule = calibrate_positive_threshold(
            calibration_events.loc[calibration_events["y"] == 1, "min_score"],
            alpha=alpha,
            mode=mode,
            confidence=confidence,
        )
        held_out_decisions = first_safe_decision_table(
            held_out,
            "catboost_tail_aligned",
            threshold=rule["threshold"],
            minimum_history=minimum_history,
        )
        held_out_decisions["fold"] = fold
        decisions.append(held_out_decisions)
        threshold = float(rule["threshold"])
        rank = int(rule["rank"])
        thresholds.append(threshold)
        ranks.append(rank)
        fold_positive = held_out_decisions["y"] == 1
        fold_negative = ~fold_positive
        fold_dangerous = held_out_decisions["safe_exclude"] & fold_positive
        fold_safe_negative = held_out_decisions["safe_exclude"] & fold_negative
        fold_danger_k = int(fold_dangerous.sum())
        fold_danger_n = int(fold_positive.sum())
        fold_safe_negative_count = int(fold_safe_negative.sum())
        fold_negative_n = int(fold_negative.sum())
        fold_first_safe = held_out_decisions.loc[
            fold_safe_negative, "first_safe_tca"
        ]
        fold_results.append({
            "fold": int(fold),
            "calibration_positive_events": int(
                (calibration_events["y"] == 1).sum()
            ),
            "rank": rank,
            "threshold": threshold,
            "danger_k": fold_danger_k,
            "danger_n": fold_danger_n,
            "danger_rate": float(fold_danger_k / fold_danger_n),
            "danger_ucb": cp_upper(fold_danger_k, fold_danger_n, confidence),
            "safe_negative": fold_safe_negative_count,
            "negative_n": fold_negative_n,
            "safe_negative_rate": float(
                fold_safe_negative_count / fold_negative_n
            ),
            "median_first_safe_tca_days": (
                None if fold_first_safe.empty else float(fold_first_safe.median())
            ),
        })

    event_decisions = pd.concat(decisions, ignore_index=True)
    positive = event_decisions["y"] == 1
    negative = ~positive
    dangerous = event_decisions["safe_exclude"] & positive
    safe_negative = event_decisions["safe_exclude"] & negative
    first_safe = event_decisions.loc[safe_negative, "first_safe_tca"]
    danger_k = int(dangerous.sum())
    danger_n = int(positive.sum())
    negative_n = int(negative.sum())
    safe_negative_count = int(safe_negative.sum())
    recomputed = {
        "danger_k": danger_k,
        "danger_n": danger_n,
        "danger_rate": float(danger_k / danger_n),
        "danger_ucb": cp_upper(danger_k, danger_n, confidence),
        "safe_negative": safe_negative_count,
        "negative_n": negative_n,
        "safe_negative_rate": float(safe_negative_count / negative_n),
        "median_first_safe_tca_days": float(first_safe.median()),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
    }
    integer_metrics = (
        "danger_k", "danger_n", "safe_negative", "negative_n", "rank_min", "rank_max"
    )
    float_metrics = (
        "danger_rate", "danger_ucb", "safe_negative_rate",
        "median_first_safe_tca_days", "threshold_min", "threshold_max",
    )
    for name in integer_metrics:
        record(
            f"metric_{name}",
            int(recomputed[name]) == int(metrics.get(name, -1)),
            recomputed[name],
            metrics.get(name),
        )
    for name in float_metrics:
        expected = metrics.get(name)
        passed = expected is not None and math.isclose(
            float(recomputed[name]), float(expected), rel_tol=1e-12, abs_tol=1e-15
        )
        record(f"metric_{name}", passed, recomputed[name], expected)

    return {
        "schema_version": 2,
        "status": "development-only-oof-verification",
        "passed": all(check["passed"] for check in checks),
        "note": (
            "This verifies internal consistency of development artifacts only. "
            "It is not confirmation evidence and does not access calibration or "
            "evaluation outcomes."
        ),
        "limitations": [
            "The OOF parquet SHA-256 is reported but is not cryptographically bound by an earlier frozen manifest.",
            "Metric recomputation verifies stored scores; it does not reproduce model training.",
        ],
        "artifacts": {
            "oof_sha256": oof_sha256,
            "development_manifest_sha256": development_manifest_sha256,
            "v13_manifest_sha256": file_sha256(v13_manifest_path),
            "model_sha256": model_sha256,
            "preregistration_sha256": preregistration_sha256,
            "preregistration_lock_sha256": preregistration_lock_sha256,
        },
        "checks": checks,
        "fold_results": fold_results,
        "recomputed_metrics": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify committed development-only OOF evidence for the v13 candidate"
    )
    parser.add_argument("--oof", type=Path, default=ROOT / "artifacts" / "development_event_aligned_oof_v8.parquet")
    parser.add_argument("--development-manifest", type=Path, default=ROOT / "artifacts" / "development_event_aligned_v8.json")
    parser.add_argument("--v13-manifest", type=Path, default=ROOT / "artifacts" / "catboost_tail_aligned_final_v13.json")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "catboost_tail_aligned_final_v13.cbm")
    parser.add_argument("--preregistration", type=Path, default=ROOT / "artifacts" / "next_validation_preregistration_v12.json")
    parser.add_argument("--preregistration-lock", type=Path, default=ROOT / "artifacts" / "next_validation_preregistration_v12.lock")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_oof_evidence(
        args.oof,
        args.development_manifest,
        args.v13_manifest,
        args.model,
        args.preregistration,
        args.preregistration_lock,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Verification output already exists: {args.output}")
        _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
