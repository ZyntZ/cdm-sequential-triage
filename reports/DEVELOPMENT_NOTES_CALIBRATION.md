# Development notes: calibration rules

## Scope
This iteration adds event-level threshold calibration to the policy module. Model scores and data partitions are unchanged. Calibration and evaluation partitions were not accessed.

## Implemented rules

### Marginal rank calibration
For `n` positive calibration events, the rank is

```
floor(alpha * (n + 1))
```

and `SAFE-EXCLUDE` is allowed below the selected positive order statistic. The threshold is moved one floating-point step down so ties do not make the rule anti-conservative. This rule controls risk averaged over the random calibration sample under event-level exchangeability.

### PAC calibration
The largest rank `r` is used for which

```
Beta(r, n + 1 - r).ppf(confidence) <= alpha
```

This gives a tolerance-bound interpretation: with the requested confidence over the calibration sample, the conditional dangerous-exclusion probability is no greater than `alpha`, assuming exchangeable event-level scores.

With 73 positive calibration events, the available ranks are:

| alpha | marginal rank | PAC rank, 95% confidence |
|---:|---:|---:|
| 0.05 | 3 | 1 |
| 0.10 | 7 | 3 |
| 0.15 | 11 | 6 |
| 0.20 | 14 | 9 |

The PAC rule is substantially more conservative at the available sample size.

## Development diagnostics
The rules were compared on the existing development out-of-fold scores using cross-fitting. With about 153–154 positive calibration events per fold, PAC calibration at alpha=0.10 gave:

| score | dangerous events | pooled UCB95 | safe-negative rate |
|---|---:|---:|---:|
| current risk | 14/192 | 11.16% | 70.07% |
| logistic snapshot | 14/192 | 11.16% | 71.97% |
| CatBoost snapshot | 12/192 | 9.93% | 73.05% |
| CatBoost dynamic | 13/192 | 10.55% | 73.99% |

Only the CatBoost snapshot score met the pooled 10% upper bound in this development diagnostic. This is not a confirmatory result.

A repeated 50/50 development split with 96 positive calibration events showed the expected trade-off. At alpha=0.05, median safe-negative rates were 51.39% for CatBoost snapshot PAC calibration and 64.47% for marginal calibration. Mean dangerous-event rates were 2.07% and 4.13%, respectively. At alpha=0.10, the corresponding median safe-negative rates were 69.60% and 79.42%.

The repeated-split summaries are stability diagnostics. Empirical error in a finite test half may exceed alpha even when a marginal or PAC calibration statement is valid.

## Current choice
CatBoost snapshot remains the leading score. Both calibration modes remain available in code:

- marginal mode for the distribution-free average-risk statement;
- PAC mode for a stronger conditional-on-calibration statement with an explicit confidence level.

No final mode or alpha is frozen yet. That decision will be made before accessing the calibration partition.
