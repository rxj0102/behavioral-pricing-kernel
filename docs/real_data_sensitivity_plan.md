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

## Rules

- All rows are reported, including those that fail or are degenerate.
- A conclusion about the pricing kernel puzzle or the distortion is called robust only if it holds, in sign and rough size, across the tail and cleaning rows. If it flips, the paper says so.
- Each specification adds a row to the results log, so the Deflated Sharpe trial count (H4 power study) stays honest.
- The table is code output (a CSV with config hash and date range), not hand written.
