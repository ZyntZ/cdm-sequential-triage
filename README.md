# Sequential CDM Triage

Event-level risk-controlled triage for conjunction data message streams.

## Research question
Can a sequential policy reduce manual review of low-final-risk conjunction events while controlling the probability that a high-final-calculated-risk event is ever assigned `SAFE-EXCLUDE` during the 2–7 day decision window?

## Status
Development-only research prototype. Calibration and evaluation partitions remain untouched. Not for operational collision avoidance or maneuver decisions.

## Decisions
- `SAFE-EXCLUDE`: no current manual review; routine automated ingestion continues.
- `MONITOR`: priority automated watchlist; reassess at every new CDM.
- `ESCALATE`: manual analysis required.

## Current evidence
Current risk, logistic snapshot, CatBoost snapshot, and CatBoost dynamic scores were compared on identical event-level out-of-fold splits. CatBoost snapshot currently leads at the strict end of the development frontier.

The policy module supports two event-level calibration statements:
- marginal rank calibration for average risk over the random calibration sample;
- PAC rank calibration with an explicit confidence level over the calibration sample.

Development results are in `reports/DEVELOPMENT_REPORT_002_RU.md`, `reports/DEVELOPMENT_NOTES_CALIBRATION.md`, and `reports/DEVELOPMENT_NOTES_ROBUSTNESS.md`.

Subgroup diagnostics cover mission, history length, and entry-message completeness. A three-CDM minimum-history gate remains a development candidate; it has not been frozen or evaluated on the calibration partition.

A split-conformal applicability gate is implemented for numeric event features. It blocks `SAFE-EXCLUDE` on non-finite inputs and on events outside the calibrated robust-deviation region. Its false-flag statement is marginal and requires event-level exchangeability; it is not a guarantee under arbitrary distribution shift.

The runtime policy combines the calibrated lower threshold, minimum-history rule, applicability gate, and the calibrated 2–7 day decision window. Updates outside the window remain auditable but cannot receive `SAFE-EXCLUDE`; the minimum-history counter includes only updates inside the window. Every accepted update records its sequence number, eligible-history count, score, gate result, decision, and reason. Out-of-order event updates are rejected. The `ESCALATE` threshold is an operational prioritisation setting and is not covered by the dangerous-exclusion guarantee.

## Reproducibility
- Dataset source/checksums: `data/manifest.json`
- Statistical protocol: `PROTOCOL.md`
- Causal feature builder: `src/prefix_features.py`
- Event-level policy and calibration utilities: `src/policy.py`
- Conformal applicability gate: `src/shift_gate.py`
- Stateful three-decision policy and audit log: `src/triage.py`
- Calibration diagnostics: `python scripts/calibration_diagnostics.py --output reports/calibration.csv`
- Subgroup diagnostics: `python scripts/robustness_diagnostics.py --group mission_id --output reports/mission.csv`
- Tests: `pytest -q`

## Data
ESA Collision Avoidance Challenge, DOI: 10.5281/zenodo.4463683, CC BY 4.0. Raw data are not redistributed in this repository.

## Limitations
The target is high final calculated collision risk, not collision occurrence. Statistical control relies on event-level exchangeability with calibration data and does not constitute an operational safety guarantee under arbitrary distribution shift.
