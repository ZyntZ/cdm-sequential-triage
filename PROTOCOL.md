# Development protocol v0.2 — 2026-07-26

## Confirmatory target
Y = 1{final log10(Pc) >= -6}, where final is the row with minimum time_to_tca for an event.

## Decision window
Primary sequential evaluation uses only CDMs with time_to_tca >= 2 days. This matches the early-decision setting of the ESA challenge and prevents use of near-TCA/final information.

## Unit and leakage control
The unit is event_id. All prefixes of an event remain in one split. Features at step t use only that row and earlier rows (larger time_to_tca). The final label is never included as a predictor.

## Decisions
SAFE-EXCLUDE: do not place the event into the current manual-review queue; automatic ingestion continues.
MONITOR: retain in automated watch state and await new CDM.
ESCALATE: route to manual analysis.

## Primary dangerous error
P(any SAFE-EXCLUDE during the eligible trajectory | Y=1).

## Development phases
1. Pilot: data audit, feature/model/policy comparison using train/development only.
2. Freeze: choose model, features, alpha, minimum useful effect, and subgroup tests.
3. Calibration: fit event-level threshold on untouched positive calibration events.
4. Confirmation: evaluate once on untouched internal evaluation events.
5. External stress test: ESA official test only if released labels are obtained; it is not called an independent random test.

## Historical confirmation candidate
The completed `confirmation_v1` candidate used the CatBoost snapshot score, PAC rank calibration, alpha = 0.10, 95% calibration confidence, a 2–7 day decision window, and minimum_history = 3 counted only after the window opens. This operating point was selected on development data before the locked evaluation. It remains the specification of the completed historical run and is not the candidate for the next study.

Primary confirmatory metrics are event-level dangerous-exclusion count/rate with a one-sided 95% Clopper-Pearson upper bound, safe-negative rate, and median first SAFE-EXCLUDE time before TCA. The confirmation pass is run once after calibration.

## Current status
The frozen policy was calibrated on 73 positive calibration events. PAC rank = 3, threshold = 0.002248533976580822, and the 95% calibration bound is 0.083741 at alpha = 0.10.

The single locked evaluation run produced 4 dangerous exclusions among 73 positive events: observed rate 5.48%, one-sided 95% Clopper-Pearson upper bound 12.10%. The policy assigned SAFE-EXCLUDE to 1,807 of 2,558 negative events (70.64%); median first decision time was 5.53 days before TCA. The pre-specified criterion requiring the evaluation upper bound to be no greater than 10% was not met. No second confirmatory run is permitted.

## Revised candidate for a genuinely new study
Development tuning closed after the v11 calibration-stability diagnostic. The revised candidate is frozen in `artifacts/next_validation_preregistration_v12.json`; its lock must not be replaced or deleted. This candidate is post-confirmation development and has no valid claim on `confirmation_v1`, which is previously unblinded historical evidence.

The next study requires new event sequences split into disjoint calibration and evaluation sets before outcome access. The planning target is at least 100 positive calibration events and 200 positive evaluation events. Model parameters, feature construction, hard-prefix weighting, decision window, minimum history, calibration mode, alpha, and primary success criterion are fixed by the preregistration artifact. No threshold or model tuning is permitted after new outcomes are accessed. The calibration artifact binds its event roster to a canonical SHA-256 digest; confirmation recomputes the digest and rejects missing, duplicated, or modified roster entries before using the artifact.
Before labels are loaded, `freeze_new_study.py` must lock the exact label-blind calibration and evaluation feature files and their disjoint event rosters. For the frozen 2–7 day policy, these feature files may retain earlier pre-window history but must not contain rows below 2 days to TCA or columns explicitly exposing final outcomes. The study manifest records the column roster, TCA range, decision-window coverage, and the exact event IDs represented inside the decision window. Calibration and confirmation require the scored event roster to match those IDs exactly; events with only pre-window history are not expected to produce a score. Subsequent scoring, calibration, and confirmation must reference that locked study. Immediately before the one-shot confirmation lock is created, label-blind preflight checks the score file, study binding, calibration/model/gate identities, required columns, model hash, duplicate updates, event overlap, and complete decision-window coverage. Label-roster checks occur only after the lock is acquired. This prevents correctable input errors from consuming the confirmation run while preserving the rule that outcome-dependent failures cannot be retried. It does not prove that externally prepared features with neutral names were generated without outcome access; source-data handling still requires procedural separation and audit.

