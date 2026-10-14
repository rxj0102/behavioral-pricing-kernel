# Pre-registered hypotheses

Committed in Phase 0, before any estimation has been run. Any change after this commit must be made in a new commit that states what changed and why, and the original text stays in git history.

## Research question

Can a behavioral stochastic discount factor with probability weighting (rank-dependent or cumulative prospect theory) and a sentiment shift explain the shape of the empirical pricing kernel (EPK) better than CRRA, habit and other rational benchmarks, and does the estimated behavioral distortion predict future S&P 500 excess returns out of sample?

## Definitions

- EPK at horizon 30 calendar days: risk-neutral density divided by physical density, scaled by the gross risk-free rate, on a common grid of gross returns.
- Distortion measure: the estimated weighting curvature parameter of the behavioral SDF (primary), with a nonmonotonicity measure of the EPK as a secondary measure. The exact definitions are fixed in docs/methodology.md before Phase 5.
- High-sentiment period: top tercile of the Baker-Wurgler sentiment index (primary) or of the AAII bull-bear spread (secondary), terciles computed on an expanding window so no future information is used.

## Hypotheses

**H1 (fit).** A probability-weighting SDF fits the cross-section of 30-day EPKs better than CRRA and a habit-style SDF. Judged by out-of-sample pricing error (distance between model-implied and observed risk-neutral density) and a Diebold-Mariano style comparison of the loss differential. Rejected if the behavioral model does not beat both benchmarks at the 5 percent level, or beats them only in sample.

**H2 (sentiment).** The estimated weighting curvature, and the nonmonotonicity measure of the EPK, are higher in high-sentiment periods than in other periods. Tested with a difference in means and a regression on the sentiment index with Newey-West standard errors. Rejected if the coefficient is not positive and significant at the 5 percent level for both sentiment proxies, or if it flips sign across subsamples.

**H3 (predictability).** The distortion measure predicts monthly SPX excess returns with positive out-of-sample R squared (Campbell-Thompson, expanding window, no lookahead) after controlling for the variance risk premium. Supported only if out-of-sample R squared is positive and the Clark-West test rejects at the 5 percent level, and the result survives the placebo test with shuffled dates.

**H4 (trading, negative result allowed).** Any trading signal built from the distortion measure does not survive transaction costs and the Deflated Sharpe Ratio. The count of trials entering the Deflated Sharpe Ratio is the true number of rows in the results log. Reported whichever way it comes out.

## Commitments

- Every specification tried is appended to the results log by code, including failures and rejected ones.
- The paper reports all four hypotheses, including those rejected or only partly supported.
- No parameter, window or sample choice is tuned on the out-of-sample period.
