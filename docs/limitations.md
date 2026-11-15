# Limitations

## Real-data study (Amendment 3)

These apply to the results in the README. The sections below them concern the option-chain extraction code.

- Joint hypothesis. The physical density is a GJR-GARCH filtered historical simulation with a constant expected excess return (the sample mean since 1926). Every pricing kernel result is conditional on it.
- Published statistics, not option prices. The risk-neutral side is nine statistics per date from the Minneapolis Fed, produced with its own smoothing and tail method. The 20 percent thresholds are read in log-return units; a test shows that reading fits the other statistics better than simple-return units.
- Location. The forward shift is taken from a two-normal reconstruction of each density and averages about 1.3 percent per six months, close to the bill rate over the period. An error here moves every published point by the same small amount.
- Horizon. Six and twelve months. Results need not carry over to the 30 day horizon of most of the literature.
- Stand-in. Daily US market returns from the Ken French library replace the S&P 500 on the physical side.
- Sentiment. One proxy (Baker-Wurgler, orthogonalised), built by its authors with full-sample information and available to 2025-12. The AAII file could not be downloaded.
- Dependence. Weekly densities for a six month horizon overlap heavily. Newey-West errors with 26 lags are used for the fit tests and may still understate uncertainty.
- Habit-style benchmark. Risk aversion linear in the drawdown is a reduced form. Its estimated sign is opposite to habit formation.
- Probability weighting. One-parameter Prelec with rank from the physical distribution. Asymmetric weighting was not pre-specified and is not tested.
- Sample end. Density dates after 2026-09-30, the last day of the market return file, are dropped.

Status: Phase 0. Known from the design, to be extended with evidence as phases complete.

- The physical density is a model forecast, so any EPK is a joint test of the risk-neutral and physical density estimates (a known issue in this literature).
- Out-of-sample predictive tests on a monthly horizon have few independent observations.
- If WRDS is unavailable, the sample will be short or synthetic, which limits what can be claimed.
