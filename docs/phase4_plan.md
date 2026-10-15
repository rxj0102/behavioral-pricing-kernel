# Phase 4 plan: empirical pricing kernel and nonmonotonicity (requirements fixed before the code)

## 1. EPK validity region

The risk-neutral density is data-driven only inside the observed quote range, and the generalized Pareto or GEV tails beyond the splice points are extrapolations set by the tail rule, not by data. The EPK in a region where either density is an extrapolation says nothing about preferences. So, for each date:

- Region from the risk-neutral side: V_Q = [k_left, k_right] intersect [k_lo, k_hi], where k_left and k_right are the splice points of the estimated density and [k_lo, k_hi] is the intersection of the observed log-moneyness ranges of the maturities used (for the 30 day horizon, the two bracketing maturities).
- Region from the physical side: V_P = grid points where the FHS density rests on enough simulated draws (minimum count per bin, fixed in the config before the run).
- Validity region V = V_Q intersect V_P, on the common return grid. If V is narrower than a minimum width set in the config (default plus and minus 0.05 in log return) the date is reported as insufficient and excluded from the measures, with the count of excluded dates by year and variance regime.
- Nonmonotonicity measures (share of grid points where the EPK increases, and size of the U-shape bump) are evaluated only inside V. Outside V the EPK may be plotted, shaded as extrapolated, but never enters a measure.
- Tests: a synthetic case where the true EPK is known to be non-monotone only outside the quote range must give no detection inside V; a kernel with a bump inside V must be detected.

## 2. Null distribution of the measure at alpha0 = 1.00

A nonmonotonicity measure computed from noisy estimated densities is positive even when the true kernel is monotone. So before any detection claim:

- Simulate the synthetic panel with alpha0 = 1.00 (CRRA kernel, monotone by construction) over many dates and several noise levels from the noise study, run the full pipeline (cleaning, RND, FHS physical density in oracle-density and estimated modes), compute both measures inside V.
- Report the null distribution of each measure: mean, median, 90th, 95th and 99th percentiles, by variance regime and by noise level, separately for oracle P and estimated P.
- A detection threshold is the 95th percentile of the null (99th as a second threshold), taken from the matching regime and noise level. The share of dates above threshold at alpha0 = 1.00 is the false positive rate and must be close to the nominal level, otherwise the measure is recalibrated or dropped.
- Power at alpha0 below 1 uses the same thresholds (this feeds the Phase 5 power curve).
- On real data, only the thresholds from the null study are used, and a real-data nonmonotonicity share below the threshold-implied false positive rate is reported as not distinguishable from zero.

## 3. Bootstrap bands

Bootstrap confidence bands for the EPK follow the project spec. The bootstrap resamples quotes and simulated draws; bands are drawn only inside V.
