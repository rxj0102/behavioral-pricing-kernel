# Real-data sensitivity plan (to run once real option data exist)

Every choice in the RND and EPK pipeline that is not pinned by theory is fixed here before real data are looked at. Defaults are not changed after seeing real-data results. The synthetic tail-rule study (results/results_log.csv, spec ids tail_*) does not inform the default for real data, because synthetic tails are light by construction.

## Table to produce

One row per specification, same dates and same quotes in every row, columns for the quantities that matter downstream:

- 30 day RND moments (mean of the gross return, variance, skewness, excess kurtosis) and the mass in the extended tails.
- EPK validity region width, share of dates with insufficient region, nonmonotonicity measures inside the region, and their exceedance of the Phase 4 null thresholds.
- Phase 5 distortion estimate (weighting curvature) and its standard error, once Phase 5 exists.

## Specifications (default in bold)

| Dimension | Values |
|---|---|
| Tail family and shape | **GPD xi = 0**, GPD xi = -0.1, GPD xi = 0.2, GPD slope matched, GEV xi = 0, GEV xi = -0.1 |
| Tail splice probability | 0.01, **0.02**, 0.05 |
| Smile fit | **SVI with convex fallback**, convex masses only |
| Maturity handling | **total variance interpolation to 30 days**, nearest maturity without interpolation |
| Cleaning | **default minimum open interest and spread cap**, stricter (double minimum open interest), looser (half) |
| Forward | **near-ATM parity**, spot with carry from the risk-free rate and a dividend yield series |

## Step 0: calibrate the noise levels from the first real snapshot

Before the table above is read, the synthetic noise levels are matched to what real quotes look like, so that the Phase 2 noise study and the Phase 4 null study are run at realistic levels. From the first real snapshot (the collector output, after the layout was inspected by a person):

1. **Observed half-spreads.** Compute the relative half-spread (ask minus bid, over two, over mid) by moneyness bucket and maturity, and compare with the generator spread (max of 0.025 and 0.4 percent of mid, times 1 to 1.3). Report the ratio by bucket. If real spreads are systematically wider, scale the synthetic spread before the noise study is rerun.
2. **Jitter level.** Estimate the dispersion of the forward implied by put-call parity across the near-ATM strikes of each maturity, in units of the observed half-spread. Run the synthetic panel through the same estimator at jitter x0, x0.5, x1, x2, x4 and pick the synthetic level whose dispersion matches; report the matched level with the range of levels that fit within sampling error.
3. **Parity-violation rate.** Share of call-put pairs whose parity forward deviates from the maturity forward by more than the cleaning tolerance (1 percent by default), and the share of crossed or locked quotes, zero-bid quotes and missing sides, each by moneyness bucket. These match the dropped-strike level and the baseline defects of the generator.
4. **Stale rate.** A single snapshot cannot show staleness. It is calibrated from the first week of consecutive daily snapshots as the share of quotes whose bid and ask are unchanged from the previous day while the underlying moved by more than a stated amount, by moneyness bucket.
5. **Output.** A small table (code output with config hash and snapshot date) listing observed statistics, the synthetic noise level matched, and its confidence range. The noise study (all three pipeline variants) and the Phase 4 null are rerun at the matched levels, and the matched levels become the realistic joint case for all later power studies.

The first snapshot is one day, so the matched levels are provisional and are updated when more days exist. The calibration only sets the noise level of the validation studies; it never tunes the estimator on real data.

## Rules

- All rows are reported, including those that fail or are degenerate.
- A conclusion about the pricing kernel puzzle or the distortion is called robust only if it holds, in sign and rough size, across the tail and cleaning rows. If it flips, the paper says so.
- Each specification adds a row to the results log, so the Deflated Sharpe trial count (H4 power study) stays honest.
- The table is code output (a CSV with config hash and date range), not hand written.
