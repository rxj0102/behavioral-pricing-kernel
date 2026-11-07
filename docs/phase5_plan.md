# Phase 5 plan: identification study and estimation (committed before Phase 5 starts)

Real-data SDF estimation does not start until this study has run and its gate (below) is reported. The design is fixed here so it cannot be tuned to the result.

## Two modes, both required

1. **Oracle mode.** The estimator receives the true physical density of the generator (SyntheticHestonAdapter.densities, physical part) and the estimated RND from quotes. This isolates the SDF estimator and the RND error.
2. **Estimated mode.** The physical density is the filtered historical simulation forecast (Phase 3) computed from the synthetic return path only, with information available at each date. This adds physical-density error. The generator has a known simplification (state path follows plain Heston P-dynamics while the pricing density is shifted by xi, stored per chain), so estimated mode also contains that gap, and it is reported alongside the results rather than hidden.

A result is reported only if both modes were run. If oracle mode passes and estimated mode fails, the conclusion is that the physical density is the bottleneck, and that is stated.

## Grid of distortion strengths, including zero

- Main grid: alpha0 in {1.00, 0.95, 0.90, 0.85, 0.80, 0.70} with delta = 0 (constant distortion). alpha0 = 1 means no probability weighting and is the zero-distortion cell, used to measure size (false positive rate). Distortion strength is 1 - alpha0.
- Sentiment link grid: delta in {0, 0.025, 0.05, 0.10} at alpha0 = 0.90. This tests recovery of delta only (see Amendment 2), never H2.
- Everything else at the default generator values (gamma = 2, beta = 1, Heston parameters as in configs).

## Replications and noise

R = 200 replications per cell in the default config (R = 10 in the fast config), each with its own seed derived from the global seed and the cell label. Each replication simulates a cross-section of 60 dates, builds quotes with the generator spread, tick and defects, and runs the full pipeline (cleaning, RND, estimation). No step uses generator truth except oracle mode where stated.

## Test and outputs

- Null: alpha = 1 (no probability weighting). Test: bootstrap likelihood-ratio style statistic, comparing the behavioral SDF with its nested CRRA special case, at the 5 percent level. The bootstrap resamples dates.
- **Power curve:** rejection rate versus distortion strength 1 - alpha0, one curve per mode, with Monte Carlo confidence intervals, written as a table and a figure by code.
- Also reported per cell: bias and RMSE of alpha, gamma, beta and delta; share of starts that converge; share of replications hitting a parameter bound.
- Out-of-sample pricing error comparison against CRRA and habit (H1 procedure), same grid, with the Diebold-Mariano style statistic.

## Gate before real data

- Size at alpha0 = 1 in both modes lies in [0.02, 0.08].
- In oracle mode, absolute bias of alpha is below 0.05 for alpha0 <= 0.90.
- Convergence rate across starts is reported; below 90 percent is a failing result that must be fixed or stated.
- The distortion strength at which power first exceeds 0.80 is reported for each mode. Real-data estimates smaller than that detectable strength are described as not distinguishable from zero.

Failures of the gate are reported in the paper as findings, not removed.
