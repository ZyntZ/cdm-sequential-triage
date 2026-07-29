"""Normalize an offline JSON or CCSDS KVN CDM export and audit study readiness."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_cdm import (
    adapt_external_cdms,
    derive_event_labels,
    outcome_blind_features,
    parse_cdm_source,
    readiness_report,
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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


def audit_external_export(
    input_path: Path,
    features_output: Path,
    readiness_output: Path,
    *,
    labels_output: Path | None = None,
    collection_complete: bool = False,
    tca_tolerance_minutes: int = 30,
) -> dict:
    outputs = [features_output, readiness_output]
    if labels_output is not None:
        outputs.append(labels_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    if labels_output is not None and not collection_complete:
        raise ValueError("--labels-output requires --collection-complete")

    records = parse_cdm_source(input_path)
    normalized = adapt_external_cdms(
        records, tca_tolerance_minutes=tca_tolerance_minutes
    )
    report = readiness_report(normalized)
    report["source"] = {
        "path": str(input_path.resolve()),
        "messages": len(records),
        "tca_tolerance_minutes": tca_tolerance_minutes,
    }
    features = outcome_blind_features(normalized)

    staged: list[Path] = []
    try:
        _atomic_parquet(features, features_output)
        staged.append(features_output)
        if labels_output is not None:
            labels = derive_event_labels(
                normalized, collection_complete=collection_complete
            )
            _atomic_parquet(labels, labels_output)
            staged.append(labels_output)
            report["labels_written"] = int(len(labels))
        else:
            report["labels_written"] = 0
        _atomic_json(report, readiness_output)
        staged.append(readiness_output)
    except Exception:
        for path in staged:
            path.unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize an offline JSON or CCSDS KVN CDM export"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--collection-complete", action="store_true")
    parser.add_argument("--tca-tolerance-minutes", type=int, default=30)
    args = parser.parse_args()
    report = audit_external_export(
        args.input,
        args.features_output,
        args.readiness_output,
        labels_output=args.labels_output,
        collection_complete=args.collection_complete,
        tca_tolerance_minutes=args.tca_tolerance_minutes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
