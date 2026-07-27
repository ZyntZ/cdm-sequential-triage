"""Run the locked historical replay and build the operator console."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from operator_dashboard import build_dashboard
from replay_scores import run_replay


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run_demo(output_dir: Path, root: Path = ROOT) -> dict:
    """Create a historical replay audit and console in a fresh external directory."""
    output_dir = output_dir.resolve()
    protected = [root / "artifacts", root / "data", root / "src", root / "scripts"]
    if any(_inside(output_dir, path) for path in protected):
        raise ValueError("Demo output directory must be outside source and artifact directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Demo output directory is not empty: {output_dir}")

    scores = root / "artifacts" / "confirmation_v1" / "evaluation_scores.parquet"
    calibration = root / "artifacts" / "confirmation_v1" / "calibration.json"
    confirmation = root / "artifacts" / "confirmation_v1" / "confirmation.json"
    for path in (scores, calibration, confirmation):
        if not path.exists():
            raise FileNotFoundError(f"Required historical artifact is missing: {path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        audit_path = staging / "replay-audit.parquet"
        console_path = staging / "operator-console.html"
        checkpoint_path = staging / "runtime-state.json"
        audit = run_replay(
            scores, calibration, audit_path, checkpoint_path=checkpoint_path
        )
        dashboard = build_dashboard(
            [audit_path], calibration, console_path,
            confirmation_path=confirmation,
        )
        summary = {
            "status": "historical-demo-not-for-operations",
            "output_directory": str(output_dir),
            "audit": "replay-audit.parquet",
            "console": "operator-console.html",
            "checkpoint": "runtime-state.json",
            "message_updates": int(len(audit)),
            "events_in_runtime_window": int(audit["event_id"].nunique()),
            "message_decisions": {
                key: int(value) for key, value in audit["decision"].value_counts().items()
            },
            "current_event_decisions": dashboard["current_decisions"],
            "confirmation": dashboard["confirmation"],
            "caveat": (
                "confirmation_v1 did not meet the pre-specified UCB criterion; "
                "this historical replay does not validate the v13 candidate"
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging.replace(output_dir)
        staging = None
        return summary
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked confirmation_v1 historical operator demo"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(tempfile.gettempdir()) / "cdm-sequential-triage-demo",
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
