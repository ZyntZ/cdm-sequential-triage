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
    materialize_collection,
    read_collection,
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

    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--ledger", type=Path, required=True)
    snapshot.add_argument("--features-output", type=Path, required=True)
    snapshot.add_argument("--readiness-output", type=Path, required=True)

    close = commands.add_parser("close")
    close.add_argument("--ledger", type=Path, required=True)
    close.add_argument("--labels-output", type=Path, required=True)

    status = commands.add_parser("status")
    status.add_argument("--ledger", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "append":
        result = append_export(
            args.input, args.ledger, args.batches_dir,
            collection_start_utc=args.collection_start_utc,
            collection_end_utc=args.collection_end_utc,
            tca_tolerance_minutes=args.tca_tolerance_minutes,
        )
    elif args.command == "snapshot":
        result = materialize_collection(
            args.ledger, args.features_output, args.readiness_output
        )
    elif args.command == "close":
        result = close_collection(args.ledger, args.labels_output)
    else:
        result, _ = read_collection(args.ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
