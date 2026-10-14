# Methodology

Status: outline only, filled in as each phase lands. Each module states its math in its docstrings; this file collects the definitions that must be fixed before estimation.

1. Risk-neutral density: cleaning, SVI smile with no-arbitrage checks, Breeden-Litzenberger second derivative, generalized Pareto tails, total variance interpolation to a constant 30 day horizon.
2. Physical density: filtered historical simulation with GJR-GARCH, HAR-RV variant, PIT, Berkowitz and Knuppel calibration tests, Fackler-King recalibration.
3. EPK: ratio of densities times the gross risk-free rate on a common grid, bootstrap bands, nonmonotonicity measures.
4. Parametric SDFs: CRRA, habit-style, rank-dependent or CPT weighting with sentiment shift. Estimation by minimizing density distance with bounds and regularization.
5. Prediction: monthly excess returns, Newey-West and Hodrick standard errors, Clark-West, Campbell-Thompson out-of-sample R squared.
6. Strategy: costs, Deflated Sharpe Ratio with the true number of trials.

TODO (before Phase 5): fix the exact definition of the distortion measure and of the nonmonotonicity measures.
