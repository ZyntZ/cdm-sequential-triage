"""Freeze the next-candidate specification before acquiring new outcomes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validation_plan import evaluation_planning_table, file_sha256, minimum_positive_events

CANDIDATE = {
    "score": "catboost_tail_aligned",
    "model": "two-stage CatBoost with nested inner-OOF positive-tail weights",
    "hard_fraction": 0.25,
    "hard_mass": 0.50,
    "iterations": 500,
    "decision_window_days": [2.0, 7.0],
    "minimum_history": 3,
    "calibration_mode": "pac",
    "alpha": 0.10,
    "calibration_confidence": 0.95,
    "evaluation_confidence": 0.95,
    "label": "final log10(Pc) >= -6",
    "primary_error": "any SAFE-EXCLUDE during the eligible event trajectory given Y=1",
}


def freeze_plan(
    terminal_artifacts: list[Path],
    output: Path,
    lock: Path,
    planning_output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Preregistration output already exists: {output}")
    if lock.exists():
        raise FileExistsError(f"Preregistration lock already exists: {lock}")
    missing = [str(path) for path in terminal_artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing terminal artifacts: {missing}")
    fingerprints = {
        str(path): file_sha256(path) for path in terminal_artifacts
    }
    planning = evaluation_planning_table()
    payload = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-new-data",
        "candidate": CANDIDATE,
        "candidate_selection": {
            "terminal_development_artifacts": fingerprints,
            "reason": (
                "Selected as the high-automation Pareto candidate after v11. "
                "The minimum ensemble was not selected because its pooled safety "
                "advantage did not persist across calibration splits."
            ),
            "confirmation_v1": (
                "Previously unblinded historical evidence that predates this "
                "revised candidate. Subsequent development is post-confirmation; "
                "confirmation_v1 is not independent evidence for this candidate "
                "and cannot be reused as confirmation."
            ),
        },
        "new_study": {
            "requires_genuinely_new_events": True,
            "calibration_and_evaluation_event_sets_disjoint": True,
            "no_threshold_or_model_tuning_after_outcome_access": True,
            "primary_success": "one-sided 95% Clopper-Pearson UCB <= 0.10",
            "secondary_metrics": [
                "safe-negative rate",
                "median first SAFE-EXCLUDE time before TCA",
                "decision proportions",
            ],
            "recommended_calibration_positive_events": 100,
            "recommended_evaluation_positive_events": 200,
            "recommended_total_positive_events": 300,
            "rationale": (
                "At 100 calibration positives, PAC rank 5 has bound 0.0892. "
                "At 200 evaluation positives, up to 12 dangerous exclusions pass "
                "the 10% UCB criterion; pass probability is about 79.6% if true "
                "danger is 5%. Calibration and evaluation events must be disjoint."
            ),
            "minimum_positive_events_if_four_failures": minimum_positive_events(4),
        },
        "evaluation_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    planning_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    planning.to_csv(planning_output, index=False)
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({
            "preregistration": str(output),
            "preregistration_sha256": file_sha256(output),
            "planning": str(planning_output),
            "planning_sha256": file_sha256(planning_output),
            "frozen_at_utc": payload["frozen_at_utc"],
        }, stream, indent=2)
        stream.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--planning-output", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze_plan(
        args.terminal_artifact, args.output, args.lock, args.planning_output
    )
    print(json.dumps(payload["candidate"], indent=2))


if __name__ == "__main__":
    main()
