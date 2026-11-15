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

---

## Amendment 1 (2026-10-14)

The original text above is unchanged. This amendment is made before any estimation has been run, after the data situation became known: WRDS is unavailable, only free sources are allowed, and the real option data will be a forward-collected snapshot series (CBOE delayed chain) with a short window that excludes 2008 and 2020.

- **H1 and H2 on real data are descriptive.** Over the collected window, results for H1 (fit comparison, Diebold-Mariano style test) and H2 (sentiment relation) are reported as descriptive statistics with their uncertainty. They are not confirmatory tests, and the paper must not describe them as supporting or rejecting the hypotheses beyond the collected window.
- **H3 and H4 are evaluated only as a power study on the synthetic panel.** The collected window will have too few independent monthly observations for out-of-sample predictive tests or for the Deflated Sharpe Ratio. H3 and H4 are therefore examined by simulation: generate synthetic panels with a known behavioral SDF and a known predictive link of chosen strength, and report how often the H3 procedure (Campbell-Thompson out-of-sample R squared with Clark-West) and the H4 procedure (net Sharpe and Deflated Sharpe Ratio) detect it. No real-data claim is made for H3 or H4. If the collected series later becomes long enough for a real test, that is a new amendment stating the threshold before the test is run.
- **Synthetic results are method validation only.** They show what the procedures can and cannot detect. They are never reported as evidence about markets.

---

## Amendment 2 (2026-10-15)

The original text and Amendment 1 are unchanged.

- **H2 cannot be evaluated on synthetic data.** In the synthetic generator the sentiment series is itself synthetic and drives the weighting curvature by construction: alpha_t = clip(alpha0 - delta s_t, alpha_min, 1). Any estimated relation between curvature (or the nonmonotonicity of the EPK) and that sentiment series therefore reflects the generating equation, not market behavior, so it carries no evidence for or against H2.
- What the synthetic panel can test about sentiment is identification only: whether the estimator recovers delta when delta is nonzero and returns approximately zero when delta is zero. That belongs to the Phase 5 identification study and is reported as such.
- H2 is evaluated only on real data, where it is descriptive over the collected window (Amendment 1), against Baker-Wurgler and AAII series that are not generated by this project. If no real sentiment series can be obtained, H2 is reported as not evaluated.

---

## Amendment 3 (2026-11-14)

The original text and Amendments 1 and 2 are unchanged. This amendment is committed before any estimation on real data. At the time of writing only the availability, date range and field definitions of the series below have been inspected.

### Why

Amendment 1 assumed the only free real data would be a forward-collected CBOE snapshot series, too short for any test. A longer free source exists. The Federal Reserve Bank of Minneapolis publishes statistics of option-implied (risk-neutral) densities for the S&P 500 at 6 and 12 month horizons from 2007-01-12, which covers 2008 and 2020. H1 to H4 are therefore evaluated on real data as stated below. The synthetic power study of Amendment 1 is not part of this pass and is reported as not done.

### Data, all free

- Risk-neutral side: Minneapolis Fed market-based probability statistics, series sp6m (primary) and sp12m (robustness), 2007-01-12 to 2026-10-28, 819 dates, every two weeks until 2013 and weekly after. Per date: mean, standard deviation, skewness, excess kurtosis, 10th, 50th and 90th percentile of the log return, and the probability of a fall and of a rise of 20 percent or more.
- Physical side: daily US market excess return and bill rate from the Ken French Data Library, used as a stand-in for the S&P 500. That file ends 2026-09-30, so later density dates are dropped.
- Sentiment: Baker-Wurgler index, column SENT_ORTH, monthly to 2025-12, from Jeffrey Wurgler's page. A date in month m uses the value of month m-1. The AAII survey file could not be downloaded (HTTP 403), so the secondary proxy is not evaluated and H2 rests on one proxy. That is weaker than the two-proxy rule of the original text.
- Variance risk premium: CBOE VIX from FRED (VIXCLS) and realised variance from the daily market return.

### Changes to the definitions

- Horizon: 6 months (126 trading days), not 30 days. The 12 month series repeats H1 only.
- Return variable: x, the log return over the horizon in excess of the forward. The forward shift on each date is the log of the risk-neutral mean gross return, taken from a mixture of two normals fitted to the nine published statistics.
- Physical density on date t: GJR-GARCH(1,1) with filtered historical simulation, 126 days ahead, 5000 paths, a fixed seed per date. Parameters are re-estimated each January on daily data from 1963-07 to the end of the previous year. The mean daily excess return is the sample mean from 1926-07 to the day before t.
- Models for the SDF m(x), each normalised so that the implied density integrates to one:
  - M0: m = 1 (no risk adjustment).
  - M1 CRRA: m = exp(-gamma x).
  - M2 habit-style: M1 with gamma_t = gamma0 + gamma1 d_t, where d_t is the log drawdown of the market index from its trailing five year maximum on the day before t.
  - M3 probability weighting: m = exp(-gamma x) w'(S(x)), Prelec weighting with curvature alpha and beta fixed at 1, S the physical survival function.
  - M4: M3 with alpha_t = clip(alpha0 - delta s_t, 0.2, 1), s_t the sentiment value.
- Loss on date t: mean squared difference between the model-implied and the published risk-neutral distribution function at five points: the 20 percent fall, the three published percentiles, and the 20 percent rise.
- Estimation: pooled parameters minimise the mean loss over the training dates. Expanding window, re-estimated each January. The first out-of-sample year is 2012.
- Distortion measure: on each date, alpha_t minimises that date's loss with gamma held at the latest pooled M3 estimate (so no later data enter). The reported measure is curvature c_t = 1 - alpha_t. Secondary measure: the sum of the upward moves of the log of the reconstructed pricing kernel between the 5th and 95th physical percentile.

### Tests

- H1: out-of-sample 2012 to 2026. Diebold-Mariano test of the loss differential of M3 against M1 and against M2, Newey-West with 26 lags, one-sided at 5 percent. Supported only if M3 beats both. In-sample losses are reported next to the out-of-sample ones.
- H2: regression of the month-end curvature c_t on lagged sentiment, Newey-West with 12 lags; and the difference in mean c_t between top-tercile sentiment months (expanding terciles) and the rest. Supported only if the slope is positive and significant at 5 percent and keeps its sign in both halves of the sample.
- H3: Amendment 1 required a stated threshold before a real test. The threshold is 120 out-of-sample months. Target: next month's market excess return. Predictors: c_t and the variance risk premium (VIX squared over 12 minus realised variance of the last 21 days). Expanding window from 60 months. Reported: Campbell-Thompson out-of-sample R squared of the model with c_t and the variance risk premium against the model with the variance risk premium alone, and against the historical mean; Clark-West; and a placebo with 1000 shuffles of c_t.
- H4: market weight clip(forecast / (3 x variance forecast), 0, 1.5) from the H3 model, 5 bp cost per unit of turnover, against buy and hold. Deflated Sharpe Ratio with the number of trials equal to the rows in the results log.

### What this cannot show

The physical density is a model forecast, so every result is a joint test of that model. The published statistics embed the Minneapolis Fed's own smoothing and tail method. Five points and four moments cannot resolve fine local structure of the kernel near the current price. The window has two crises, not many.
