# Development protocol v0.1 — 2026-07-24

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

## Current status
Pilot only. No confirmatory result has been produced.
