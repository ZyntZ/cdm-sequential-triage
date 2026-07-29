"""Manage an append-only prospective collection of offline CDM exports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_collection import (
    append_export,
    close_collection,
    collection_status,
    materialize_collection,
    seal_collection,
)


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
