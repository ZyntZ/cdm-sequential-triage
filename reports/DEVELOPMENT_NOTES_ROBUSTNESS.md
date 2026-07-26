# Development notes: subgroup robustness

## Scope
The current CatBoost snapshot score was evaluated by mission, available history length, and entry-message completeness. Decisions are cross-fitted PAC decisions at alpha 0.10: each development fold uses a threshold calibrated on the other four folds. The calibration and evaluation partitions were not accessed.

## Mission-level results
Mission-level estimates are descriptive. Several missions contain fewer than six positive events, so their dangerous-exclusion bounds are too wide for mission-specific claims.

The largest positive-event groups were mission 1 (51 positives), mission 2 (31), mission 5 (17), mission 3 (15), missions 6 and 15 (13 each), and mission 18 (11). Observed dangerous-exclusion counts were 2/51, 1/31, 1/17, 0/15, 2/13, 3/13, and 1/11, respectively.

No mission-level guarantee is claimed. Mission 15 had the largest dangerous count among the better represented missions (3/13), while mission 6 had 2/13. These groups should be included in later stress tests. Mission 24 had no positive event and therefore cannot support a dangerous-error estimate.

## Available history

> The minimum-history gate results below used total `n_cdm_so_far` rather than the runtime counter that starts when the decision window opens. They are retained as an audit trail but are superseded by `DEVELOPMENT_NOTE_003_RU.md` and `development_history_gate_window_v5.csv`. Subgroup counts by total CDMs available in the window remain descriptive.

Events were grouped by the number of CDMs available in the 2–7 day decision window.

| CDMs in window | Events | Positive events | Dangerous exclusions | Safe-negative rate |
|---|---:|---:|---:|---:|
| 1–4 | 2,023 | 93 | 5/93 | 55.80% |
| 5–10 | 1,642 | 62 | 6/62 | 71.96% |
| 11+ | 3,481 | 37 | 1/37 | 83.22% |

Safe-negative coverage increased strongly with longer histories (chi-square 473.41, 2 df, p=1.59e-103). The dangerous-exclusion comparison was not significant (chi-square 2.16, 2 df, p=0.340), but the smallest expected count was 2.31, so this test has limited power and should be treated as exploratory.

A minimum-history gate was also evaluated with the same cross-fitted PAC rule:

| Minimum available CDMs | Events still eligible | Dangerous exclusions | UCB95 | Safe-negative rate |
|---:|---:|---:|---:|---:|
| 1 | 7,146 | 12/192 | 9.93% | 73.05% |
| 2 | 6,830 | 12/192 | 9.93% | 72.25% |
| 3 | 6,160 | 9/192 | 8.04% | 67.95% |
| 4 | 5,585 | 6/192 | 6.07% | 57.91% |
| 5 | 5,123 | 5/192 | 5.40% | 55.09% |

Requiring at least three CDMs reduced the development upper bound from 9.93% to 8.04% at a coverage cost of 5.10 percentage points. This gate was selected after inspecting development results and is not a confirmatory result.

The comparison was repeated across 250 stratified 50/50 development splits. With PAC calibration at alpha 0.10, the ungated policy had a mean dangerous-event rate of 5.23% and median safe-negative coverage of 69.60%. The three-CDM gate reduced the mean dangerous-event rate to 3.63% and median coverage to 63.70%. Gates of four and five CDMs reduced mean dangerous-event rates further, to 2.57% and 1.97%, but median coverage fell to 51.57% and 46.07%.

## Entry-message completeness
Completeness groups were defined from the fraction of missing fields in the first CDM inside the decision window:

- `none`: no missing fields;
- `low`: more than 0% and at most 10% missing;
- `high`: more than 10% missing.

| Missingness | Events | Positive events | Dangerous exclusions | Safe-negative rate |
|---|---:|---:|---:|---:|
| none | 3,814 | 88 | 7/88 | 75.90% |
| low | 2,443 | 77 | 4/77 | 69.70% |
| high | 889 | 27 | 1/27 | 69.95% |

Safe-negative coverage differed across completeness groups (chi-square 33.08, 2 df, p=6.54e-8). The dangerous-exclusion comparison was not significant (chi-square 0.88, 2 df, p=0.644), but expected dangerous counts were too small for a strong conclusion.

Missingness is not monotonic with observed danger in this development sample. A blanket rejection rule based only on missing-field fraction is therefore not supported. Completeness remains a monitoring variable for the shift gate.

## Current decision
Superseded on 26.07.2026 after aligning the offline history counter with runtime semantics. The frozen development candidate is CatBoost snapshot with PAC calibration at alpha 0.10 and three eligible CDMs inside the decision window. See `DEVELOPMENT_NOTE_003_RU.md`.
