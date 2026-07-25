"""Evaluate subgroup robustness of saved development decisions.

The input must contain one row per event with event-level labels, decisions,
and subgroup columns. This script does not select a new threshold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustness import subgroup_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        type=Path,
        default=ROOT / "artifacts" / "development_crossfit_decisions_v4.parquet",
    )
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    decisions = pd.read_parquet(args.decisions)
    result = subgroup_metrics(
        decisions,
        group_col=args.group,
        confidence=args.confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
