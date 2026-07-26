from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study import freeze_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_study(
        args.calibration_features, args.evaluation_features,
        args.preregistration, args.preregistration_lock,
        args.output, args.lock,
    )
    calibration = manifest["cohorts"]["calibration"]
    evaluation = manifest["cohorts"]["evaluation"]
    print(f"calibration={calibration['events']} events; evaluation={evaluation['events']} events")


if __name__ == "__main__":
    main()
