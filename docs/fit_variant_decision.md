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
