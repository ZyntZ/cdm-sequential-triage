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
    read_json,
    write_json,
)


def calibration_command(args: argparse.Namespace) -> None:
    prefixes = pd.read_parquet(args.scores)
    artifact = calibrate(prefixes)
    artifact["calibration_scores_sha256"] = file_sha256(args.scores)
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
    lock_payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_artifact": str(args.calibration),
        "calibration_artifact_sha256": file_sha256(args.calibration),
        "evaluation_scores": str(args.scores),
        "evaluation_scores_sha256": file_sha256(args.scores),
        "evaluation_labels": str(args.labels),
        "evaluation_labels_sha256": file_sha256(args.labels),
        "output": str(args.output),
    }
    acquire_confirmation_lock(args.lock, lock_payload)
    prefix_scores = pd.read_parquet(args.scores)
    event_labels = pd.read_parquet(args.labels)
    prefixes = attach_event_labels(prefix_scores, event_labels)
    result = evaluate(prefixes, artifact)
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
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(handler=calibration_command)

    confirmation = commands.add_parser("confirm")
    confirmation.add_argument("--scores", type=Path, required=True)
    confirmation.add_argument("--calibration", type=Path, required=True)
    confirmation.add_argument("--labels", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--lock", type=Path, required=True)
    confirmation.set_defaults(handler=confirmation_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
