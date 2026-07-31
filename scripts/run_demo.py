"""Run the locked historical replay and build the operator console."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from operator_dashboard import build_dashboard
from operator_dashboard import SUPPORTED_LOCALES
from evidence_dashboard import build_evidence_dashboard
from replay_scores import run_replay

from confirmation import file_sha256


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False



DEMO_ARTIFACTS = (
    "replay-audit.parquet",
    "operator-console.html",
    "runtime-state.json",
    "evidence-dashboard.html",
)


DEMO_INPUTS = {
    "confirmation_evaluation_scores": "artifacts/confirmation_v1/evaluation_scores.parquet",
    "confirmation_calibration": "artifacts/confirmation_v1/calibration.json",
    "confirmation_result": "artifacts/confirmation_v1/confirmation.json",
    "confirmation_lock": "artifacts/confirmation_v1/confirmation.lock",
    "development_event_aligned_manifest": "artifacts/development_event_aligned_v8.json",
    "development_oof_scores": "artifacts/development_event_aligned_oof_v8.parquet",
    "development_score_ensemble_manifest": "artifacts/development_score_ensemble_v10.json",
    "development_repeated_stability_manifest": "artifacts/development_score_ensemble_repeated_v11.json",
    "development_fold_diagnostics": "reports/development_score_ensemble_folds_v10.csv",
    "next_validation_planning": "reports/next_validation_sample_size_v12.csv",
    "next_validation_preregistration": "artifacts/next_validation_preregistration_v12.json",
    "next_validation_preregistration_lock": "artifacts/next_validation_preregistration_v12.lock",
    "v13_model_manifest": "artifacts/catboost_tail_aligned_final_v13.json",
    "v13_model": "artifacts/catboost_tail_aligned_final_v13.cbm",
}


def verify_demo_bundle(output_dir: Path, root: Path = ROOT) -> dict:
    """Verify generated outputs and the exact source artifacts used by the demo."""
    output_dir = output_dir.resolve()
    root = root.resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Demo summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary.get("bundle_verification", {}).get("artifacts")
    if not isinstance(expected, dict) or set(expected) != set(DEMO_ARTIFACTS):
        raise ValueError("Demo summary has an invalid artifact digest roster")
    expected_inputs = summary.get("input_artifacts")
    if not isinstance(expected_inputs, dict) or set(expected_inputs) != set(DEMO_INPUTS):
        raise ValueError("Demo summary has an invalid input artifact digest roster")
    verified_inputs = {}
    for name, relative_path in DEMO_INPUTS.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Demo input artifact is missing: {path}")
        actual = file_sha256(path)
        if actual != expected_inputs[name]:
            raise ValueError(f"Demo input artifact SHA-256 mismatch: {name}")
        verified_inputs[name] = actual
    verified = {}
    for name in DEMO_ARTIFACTS:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Demo artifact is missing: {path}")
        actual = file_sha256(path)
        if actual != expected[name]:
            raise ValueError(f"Demo artifact SHA-256 mismatch: {name}")
        verified[name] = actual
    return {
        "status": "VERIFIED",
        "artifacts": verified,
        "input_artifacts": verified_inputs,
        "summary": "summary.json",
    }

def run_demo(output_dir: Path, root: Path = ROOT, locale: str = "en") -> dict:
    """Create a historical replay audit and console in a fresh external directory."""
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")
    output_dir = output_dir.resolve()
    protected = [
        root / name for name in ("artifacts", "data", "reports", "src", "scripts", "tests")
    ]
    if any(_inside(output_dir, path) for path in protected):
        raise ValueError("Demo output directory must be outside repository content directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Demo output directory is not empty: {output_dir}")

    scores = root / "artifacts" / "confirmation_v1" / "evaluation_scores.parquet"
    calibration = root / "artifacts" / "confirmation_v1" / "calibration.json"
    confirmation = root / "artifacts" / "confirmation_v1" / "confirmation.json"
    preregistration = root / "artifacts" / "next_validation_preregistration_v12.json"
    input_artifacts = {}
    for name, relative_path in DEMO_INPUTS.items():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required historical artifact is missing: {path}")
        input_artifacts[name] = file_sha256(path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        audit_path = staging / "replay-audit.parquet"
        console_path = staging / "operator-console.html"
        checkpoint_path = staging / "runtime-state.json"
        evidence_path = staging / "evidence-dashboard.html"
        audit = run_replay(
            scores, calibration, audit_path, checkpoint_path=checkpoint_path
        )
        dashboard = build_dashboard(
            [audit_path], calibration, console_path,
            confirmation_path=confirmation,
            checkpoint_path=checkpoint_path,
            locale=locale,
        )
        evidence = build_evidence_dashboard(root, evidence_path, locale=locale)
        bundle_artifacts = {
            path.name: file_sha256(path)
            for path in (audit_path, console_path, checkpoint_path, evidence_path)
        }
        summary = {
            "status": "historical-demo-not-for-operations",
            "bundle_verification": {
                "status": "VERIFIED",
                "artifacts": bundle_artifacts,
            },
            "input_artifacts": input_artifacts,
            "locale": locale,
            "output_directory": str(output_dir),
            "audit": "replay-audit.parquet",
            "console": "operator-console.html",
            "checkpoint": "runtime-state.json",
            "evidence_dashboard": "evidence-dashboard.html",
            "evidence": evidence,
            "message_updates": int(len(audit)),
            "events_in_runtime_window": int(audit["event_id"].nunique()),
            "message_decisions": {
                key: int(value) for key, value in audit["decision"].value_counts().items()
            },
            "current_event_decisions": dashboard["current_decisions"],
            "batch_chain": dashboard["chain"],
            "confirmation": dashboard["confirmation"],
            "showcase_monitor_to_safe": dashboard["showcase_monitor_to_safe"],
            "showcase_gate_blocked": dashboard["showcase_gate_blocked"],
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
        verified = verify_demo_bundle(output_dir, root=root)
        if verified["artifacts"] != summary["bundle_verification"]["artifacts"]:
            raise ValueError("Published demo bundle verification changed after commit")
        if verified["input_artifacts"] != summary["input_artifacts"]:
            raise ValueError("Published demo input lineage changed after commit")
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
    parser.add_argument("--locale", choices=("en", "ru"), default="en")
    args = parser.parse_args()
    print(json.dumps(
        run_demo(args.output_dir, locale=args.locale), ensure_ascii=False, indent=2
    ))


if __name__ == "__main__":
    main()
