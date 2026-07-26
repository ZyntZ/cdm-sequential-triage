"""Build the frozen event-level partitions from ESA train_data.zip."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from partitions import (
    EXPECTED_EVENTS,
    EXPECTED_POSITIVES,
    EXPECTED_ROWS,
    FINAL_RISK_THRESHOLD,
    SEED_STAGE1,
    SEED_STAGE2,
    build_partitions,
    event_labels,
    file_sha256,
    read_training_archive,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    archive_sha256 = file_sha256(args.archive)
    if args.expected_sha256 and archive_sha256 != args.expected_sha256:
        raise ValueError("Training archive SHA-256 does not match the expected value")

    frame = read_training_archive(args.archive)
    labels = event_labels(frame)
    partitions = build_partitions(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, partition in partitions.items():
        path = args.output_dir / f"{name}.parquet"
        partition.to_parquet(path, index=False)
        paths[name] = path

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(args.archive),
            "sha256": archive_sha256,
            "rows": int(len(frame)),
            "events": int(labels.shape[0]),
            "positive_events": int(labels["y"].sum()),
        },
        "expected": {
            "rows": EXPECTED_ROWS,
            "events": EXPECTED_EVENTS,
            "positive_events": EXPECTED_POSITIVES,
        },
        "label": {
            "definition": "final risk >= -6",
            "final_row": "minimum time_to_tca per event_id",
            "risk_threshold": FINAL_RISK_THRESHOLD,
        },
        "split": {
            "unit": "event_id",
            "fractions": {"development": 0.6, "calibration": 0.2, "evaluation": 0.2},
            "seed_stage1": SEED_STAGE1,
            "seed_stage2": SEED_STAGE2,
            "stratified_by": "y",
        },
        "decision_window_days": [2.0, 7.0],
        "outputs": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": int(len(partitions[name])),
                "events_with_rows": int(partitions[name]["event_id"].nunique()),
            }
            for name, path in paths.items()
        },
    }
    write_json(args.manifest, manifest)
    print(
        f"events={manifest['source']['events']} "
        f"positive_events={manifest['source']['positive_events']}"
    )
    for name, values in manifest["outputs"].items():
        print(f"{name}: rows={values['rows']} events={values['events_with_rows']}")


if __name__ == "__main__":
    main()
