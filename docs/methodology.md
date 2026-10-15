# Methodology

Status: outline only, filled in as each phase lands. Each module states its math in its docstrings; this file collects the definitions that must be fixed before estimation.

1. Risk-neutral density: see Phase 2 section below.
2. Physical density: filtered historical simulation with GJR-GARCH, HAR-RV variant, PIT, Berkowitz and Knuppel calibration tests, Fackler-King recalibration.
3. EPK: ratio of densities times the gross risk-free rate on a common grid, bootstrap bands, nonmonotonicity measures.
4. Parametric SDFs: CRRA, habit-style, rank-dependent or CPT weighting with sentiment shift. Estimation by minimizing density distance with bounds and regularization.
5. Prediction: monthly excess returns, Newey-West and Hodrick standard errors, Clark-West, Campbell-Thompson out-of-sample R squared.
6. Strategy: costs, Deflated Sharpe Ratio with the true number of trials.

TODO (before Phase 5): fix the exact definition of the distortion measure and of the nonmonotonicity measures.

## Phase 2: risk-neutral density (implemented, graded on the synthetic panel)

Code in src/bpk/rnd. No numbers are quoted here; the tables under results/tables (rnd_*) are produced by code (make rnd) and carry a config hash.

Steps per date and maturity:
1. Cleaned OTM mids are converted to forward-normalized undiscounted prices and then to implied total variance w by bisection (bs.py). Weights are the inverse bid-ask width in total variance.
2. Raw SVI is fitted with 18 starts (svi.py). Residuals are (model - observed total variance) divided by the bid-ask width in total variance, so a residual of 1 is one spread, and the default loss is soft-L1 with scale one spread (robust to quotes far outside their spread; scale fixed a priori). The legacy linear loss is kept as an option for comparison. Butterfly arbitrage is checked with Gatheral's g(k) on a grid over the quote range plus 0.1. If violated, a penalized refit is tried; if still violated, a convex fit is used (convex.py).
3. Deviation from the spec: the fallback is a fit of nonnegative probability masses (convex by construction) rather than a smoothing spline followed by convexity repair. The masses are fitted by bounded least squares with a second-difference roughness penalty.
4. Density by Breeden-Litzenberger: second derivative of the OTM price in strike on a fine uniform grid (density.py), density of the log return k = ln(S_T/F).
5. Tails: generalized Pareto beyond splice points where the CDF reaches tail_alpha (default 2 percent) from each side, or at the edge of the quote range if that comes first. The tail matches the density and mass at the splice. Tail shape is fixed (xi = 0, exponential) by default. The slope-matching alternative scored worse against the exact synthetic density and is kept as an option; all variants are in results/results_log.csv (spec ids tail_*). That comparison used synthetic tails, which are light by construction, so it says nothing about which rule suits real option data. Deviation from the spec: GPD in log-return space, not GEV, and no GEV variant was tried.
6. Constant 30 day horizon: linear interpolation of total variance in maturity at fixed k between the two bracketing maturities, ln F interpolated linearly, on the intersection of the observed k ranges.

Arbitrage diagnostics: calendar violations are counted as grid points where total variance decreases in maturity (rnd_arbitrage table); butterfly violations of the raw SVI fit are counted per slice (rnd_fit_diagnostics); negative density values before clipping are counted (n_negative). Violations are counted and reported, not repaired, except for the SVI to convex fallback.

Grading (grade.py): the estimate is compared with the exact Q density from the generator at the same maturity, including the interpolated 30 day horizon where the exact density is evaluated at 30 days. L1 error is reported in total, by region of k, and for the generalized Pareto tail parts. The interpolated horizon therefore includes the error of total-variance interpolation itself.

Known limitations:
- Quotes survive cleaning only where the bid is positive, so at short maturities the quoted range is narrow and a few percent of probability mass sits in the extended tails. Tail extension error is graded separately for that reason.
- Everything above is validated on synthetic data only. Real option data will have stale quotes, wider spreads and different tail behavior, none of which this validation covers.

## Phase 5 design

The identification study design (oracle and estimated modes, grid of distortion strengths including zero, power curve, gate before real data) is fixed in docs/phase5_plan.md.

## Plans fixed before the next phases

- Phase 3 requirements (clock mapping, oracle and estimated drift, scoring, PIT and Berkowitz): docs/phase3_plan.md.
- Phase 4 requirements (EPK validity region, null distribution at alpha0 = 1.00): docs/phase4_plan.md.
- Real-data sensitivity table (tail rules, splice, smile, maturity, cleaning, forward): docs/real_data_sensitivity_plan.md.
- GEV tails were added to the tail-rule study as an option (tail_rule gev). Default stays the generalized Pareto with xi = 0. GEV and GPD variants scored within noise of each other on the synthetic truth, and the default was fixed before those scores were seen.
- Noise-injection study (jitter, stale quotes, dropped strikes, combined): src/bpk/rnd/noise_study.py, tables noise_study_*.

## Phase 2 follow-ups (robust fit, calendar enforcement)

- Spread-weighted SVI with a robust loss (config rnd.svi_loss, default soft_l1, rnd.svi_f_scale = 1). Analytic Jacobian for the unpenalized stage.
- Isotonic enforcement of monotone total variance in maturity before densities and interpolation (src/bpk/rnd/monotone.py, config rnd.enforce_calendar, default true). The fit is the least-squares nondecreasing sequence at each k and is a no-op when the input is already monotone. It does not guarantee the butterfly condition of the adjusted slices; negative density points are counted before and after.
- The default fit variant is chosen by the pre-registered rule in docs/fit_variant_decision.md (committed before the v2 results were read). Noise study v2 compares three pipeline variants (legacy, robust, robust_monotone) on identical noisy quotes over the 152-date grading set, with fit coverage next to every L1, failures by variance regime, L1 with failures counted at the worst value 2, and bootstrap intervals over dates.
