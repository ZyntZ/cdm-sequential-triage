# Sequential CDM Triage

Event-level risk-controlled triage for conjunction data message streams.

## Research question
Can a sequential policy reduce manual review of low-final-risk conjunction events while controlling the probability that a high-final-calculated-risk event is ever assigned `SAFE-EXCLUDE` during the 2–7 day decision window?

## Status
Frozen-policy research prototype with one completed confirmation run. Not for operational collision avoidance or maneuver decisions.

## Decisions
- `SAFE-EXCLUDE`: no current manual review; routine automated ingestion continues.
- `MONITOR`: priority automated watchlist; reassess at every new CDM.
- `ESCALATE`: manual analysis required.

## Current evidence
Current risk, logistic snapshot, CatBoost snapshot, and CatBoost dynamic scores were compared on identical event-level out-of-fold splits. CatBoost snapshot led at the strict end of the development frontier.

The single locked evaluation run observed 4/73 dangerous exclusions (5.48%; one-sided 95% upper bound 12.10%), 70.64% SAFE-EXCLUDE coverage among low-final-risk events, and a median first decision time of 5.53 days before TCA. The pre-specified 10% evaluation upper-bound criterion was not met. Immutable run artifacts are in `artifacts/confirmation_v1/`.

The policy module supports two event-level calibration statements:
- marginal rank calibration for average risk over the random calibration sample;
- PAC rank calibration with an explicit confidence level over the calibration sample.

Development results are in `reports/DEVELOPMENT_REPORT_002_RU.md`, `reports/DEVELOPMENT_NOTES_CALIBRATION.md`, and `reports/DEVELOPMENT_NOTES_ROBUSTNESS.md`.

Subgroup diagnostics cover mission, history length, and entry-message completeness. The three-CDM minimum-history gate is frozen. PAC calibration on 73 positive events selected rank 3 and threshold 0.002248533976580822; the single evaluation run is complete.

A split-conformal applicability gate is implemented for numeric event features. It blocks `SAFE-EXCLUDE` on non-finite inputs and on events outside the calibrated robust-deviation region. Its false-flag statement is marginal and requires event-level exchangeability; it is not a guarantee under arbitrary distribution shift. Offline sequential evaluation accepts the same fitted and calibrated gate as the runtime policy, reports blocked threshold crossings, and only counts a decision when both the calibrated score rule and gate permit `SAFE-EXCLUDE`.

The runtime policy combines the calibrated lower threshold, minimum-history rule, applicability gate, and the calibrated 2–7 day decision window. Updates outside the window remain auditable but cannot receive `SAFE-EXCLUDE`; the minimum-history counter includes only updates inside the window. Every accepted update records its sequence number, eligible-history count, score, gate result, decision, and reason. Out-of-order event updates are rejected. The `ESCALATE` threshold is an operational prioritisation setting and is not covered by the dangerous-exclusion guarantee.

Offline diagnostics now use `eligible_history_count`, which starts when the 2–7 day decision window opens. This matches the runtime counter; total pre-window history remains available separately as `n_cdm_so_far`.

## Reproducibility
- Dataset source/checksums: `data/manifest.json`
- Statistical protocol: `PROTOCOL.md`
- Causal feature builder: `src/prefix_features.py`
- Event-level policy and calibration utilities: `src/policy.py`
- Conformal applicability gate: `src/shift_gate.py`
- Stateful three-decision policy and audit log: `src/triage.py`
- Calibration diagnostics: `python scripts/calibration_diagnostics.py --output reports/calibration.csv`
- Minimum-history/timing diagnostics: `python scripts/history_gate_diagnostics.py --output reports/history_gate.csv`
- Build frozen event partitions: `python scripts/make_partitions.py --archive data/raw/train_data.zip --output-dir data/processed/partitions --manifest data/processed/partitions.json --expected-sha256 68362fe5629cc80f17291f2d73f733bf4e922675e37b91a8ee79afadb46f3edc`
- Train and score frozen snapshot model: `python scripts/train_snapshot.py --training data/processed/partitions/training.parquet --calibration data/processed/partitions/calibration.parquet --evaluation-features data/processed/partitions/evaluation_features.parquet --model artifacts/catboost_snapshot.cbm --calibration-scores artifacts/calibration_scores.parquet --evaluation-scores artifacts/evaluation_scores.parquet --manifest artifacts/snapshot_model.json`
- Frozen calibration: `python scripts/confirm_policy.py calibrate --scores artifacts/calibration_scores.parquet --labels data/processed/partitions/calibration_labels.parquet --output calibration.json`
- One-shot confirmation: `python scripts/confirm_policy.py confirm --scores artifacts/evaluation_scores.parquet --labels data/processed/partitions/evaluation_labels.parquet --calibration calibration.json --output confirmation.json --lock confirmation.lock`
- Subgroup diagnostics: `python scripts/robustness_diagnostics.py --group mission_id --output reports/mission.csv`
- Tests: `pytest -q`

The partition command reproduces the frozen 60/20/20 event split and keeps complete calibration/evaluation label rosters, including events with no CDM in the decision window. The training command scores evaluation features without loading their outcomes. The confirmation command creates an exclusive lock before loading event-level labels, rejects calibration/evaluation event overlap, verifies the frozen policy, and records input checksums. Keep the lock and result under version control after the first confirmatory run.

## Data
ESA Collision Avoidance Challenge, DOI: 10.5281/zenodo.4463683, CC BY 4.0. Raw data are not redistributed in this repository.

## Limitations
The target is high final calculated collision risk, not collision occurrence. Statistical control relies on event-level exchangeability with calibration data and does not constitute an operational safety guarantee under arbitrary distribution shift.
