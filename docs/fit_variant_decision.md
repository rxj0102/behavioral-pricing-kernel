# Decision rule for the default SVI fit variant (pre-registered 2026-10-15)

Written before any result of the noise study v2 was read. The study was still running when this rule was committed. The current defaults in config.py (robust loss, monotone enforcement on) are provisional and are replaced by the outcome of this rule, whichever way it goes.

## Candidates

Three pipeline variants, run on identical noisy quotes (see src/bpk/rnd/noise_study.py):

1. legacy: linear weighted least squares SVI, no maturity monotonicity enforcement.
2. robust: spread-weighted SVI with a soft-L1 loss, no enforcement.
3. robust_monotone: robust fit plus isotonic enforcement of monotone total variance in maturity.

The robust loss scale stays at one bid-ask spread (svi_f_scale = 1). It is not tuned here.

## Objective

On the two configurations stale 5 percent (stale_0.05) and combined level 2 (combined_2), for each variant:

- For each date take the L1 error of the 30 day density against the exact Q density. A date for which no 30 day density could be produced counts as the worst value 2 (the maximum of an L1 distance between two densities).
- Bootstrap the dates (2000 resamples, seed fixed by the key decision_boot) and compute the 90th percentile of the per-date values in each resample. The bootstrap-mean p90 is the mean of these over resamples.
- The objective of a variant is the mean of its bootstrap-mean p90 across the two configurations (equal weights).

## Coverage condition

Coverage is measured in two ways, on each of the two configurations: the share of dates with a 30 day density (horizon coverage) and the share of maturity slices that were fitted (slice coverage). A variant is eligible only if, on both configurations, both coverages are at least the legacy variant coverage minus 0.02 (two points). Legacy is always eligible.

## Rule

The default is the eligible variant with the lowest objective. An exact tie goes to the simpler variant in the order legacy, robust, robust_monotone. No other consideration (neither negative density points nor calendar counts, which are reported) can overrule the rule.

## Confirmation

After the primary set decides, the winner is confirmed on a disjoint set of dates:

- Same number of dates as the primary set (the 152-date grading set), drawn with a new seed key (noise_confirm) from the dates not in the primary set.
- Same two configurations, all three variants, the same objective and coverage condition.
- The winner is confirmed if the same variant is the eligible lowest-objective variant on the confirmation set. Both sets are reported in full.
- If a different variant wins on the confirmation set, the result is reported as not confirmed and the default reverts to legacy, so that no new method is adopted without confirmation. This is stated in the paper.

## Scale sensitivity

The robust loss scale is kept at one spread. A sensitivity study over other scales (0.5, 2 and 4 spreads) is a separate, logged study: each scale adds rows to results/results_log.csv (spec ids svi_fscale_*), uses the same dates and noise as the primary set, and is reported as sensitivity only. It cannot change the default scale, and it is run after the decision, not before.

## Outputs

src/bpk/rnd/decision.py reads the raw tables of the primary and confirmation runs and writes the objective table, the eligibility table and the decision (primary, confirmation, final default) with the config hash and date ranges, and appends one results-log row per variant and set.

---

## Outcome (appended after the rule was applied; the rule text above is unchanged)

Source tables: fit_variant_decision_primary / _confirm, noise_study_v2 and noise_study_v2_confirm (with _by_regime and _raw), fallback_diagnostic_primary / _confirm, all in results/tables with config hash and date ranges.

| Set | Dates | legacy | robust | robust_monotone | Winner |
|---|---|---|---|---|---|
| Primary | 152 (seed key noise_sample) | 0.142 | 0.599 | 0.600 | legacy |
| Confirmation | 152 disjoint (seed key noise_confirm) | 0.109 | 0.618 | 0.628 | legacy |

Objective is the mean over the two configurations of the bootstrap-mean p90 of the 30 day L1, failures at 2. All variants were eligible on coverage in both sets. Legacy is confirmed, so the defaults are svi_loss = linear and enforce_calendar = false (config.py).

Why the robust variants lost (reproducible from the fallback_diagnostic tables, not part of the rule):

- When SVI succeeds the robust fit is better (median native L1 0.0075 against 0.0186 on the primary set, 0.0075 against 0.0160 on the confirmation set).
- The robust fits fail the butterfly check far more often and the convex fallback then takes over (about 17 to 19 percent of slices at combined 2 against under 0.5 percent for legacy). The fallback densities are close to useless: median native L1 about 1.3 to 1.4 (the maximum possible is 2), and 30 day densities with a convex bracket slice have median L1 about 0.85 to 0.91.
- On the confirmation set the robust variant stale 5 percent advantage seen on the primary set did not hold at the 90th percentile (bootstrap-mean p90 0.118 against 0.031 on the primary set): a handful of fallback failures set the tail.
- Maturity monotonicity enforcement removed all calendar violations (counted before and after in the tables) but did not change L1 materially, and the change in negative density points was small and not consistently in one direction.

What this does and does not show:

- It shows legacy is the better default under the pre-registered rule on this synthetic panel. It does not show that robust fitting is a bad idea; the diagnostic points at the convex fallback as the failing component. The fallback was almost never used before robust fitting exposed it, which is itself a finding about untested paths.
- Any follow-up (a repaired fallback, a robust variant with it) is a new candidate. It must be registered as a new round in this file before it is run, with a fresh date set drawn with a new seed key, because the primary and confirmation sets have now been used.
