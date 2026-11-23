# Robustness of the surviving finding to the physical density (post hoc)

Not pre-registered. Written by `python -m bpk.study.physical_robustness`. Same dates, published points, walk-forward and Diebold-Mariano settings as the main study, 6 month tenor. RMSE is the out-of-sample error in percentage points of probability. The t statistics are Diebold-Mariano, positive when probability weighting has the smaller loss.

| Physical density | No adjustment | CRRA | Habit | Weighting | Weighting vs CRRA t | Weighting vs habit t | Median alpha | Dates with alpha below 1 |
|---|---|---|---|---|---|---|---|---|
| GJR-GARCH filtered historical simulation, sample mean premium (pre-specified) | 8.4 | 7.8 | 8.3 | 7.2 | 4.5 | 1.2 | 0.80 | 96% |
| as fhs with a zero expected excess return | 3.1 | 3.0 | 3.0 | 2.9 | 0.4 | 0.4 | 0.92 | 79% |
| as fhs with twice the sample mean excess return | 15.5 | 14.6 | 14.0 | 12.5 | 3.7 | 2.4 | 0.62 | 99% |
| GJR-GARCH with Gaussian shocks instead of the empirical residuals | 8.0 | 6.1 | 4.7 | 6.2 | -1.3 | -3.5 | 0.84 | 90% |
| unfiltered historical simulation: daily excess returns drawn from the last 5 years | 14.6 | 10.0 | 9.3 | 9.8 | 1.4 | -1.7 | 0.93 | 55% |

Probability weighting beats CRRA at the 5% level under 2 of 5 physical densities and overweights tails on most dates under 5 of 5. It beats the habit-style benchmark at that level under 1 of 5.
