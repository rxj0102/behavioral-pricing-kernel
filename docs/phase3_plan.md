# Phase 3 plan: physical density by filtered historical simulation (requirements fixed before the code)

## 1. Calendar-to-trading-day mapping, in code and tested against the generator

The generator mixes two clocks and Phase 3 must not hide it:

- The simulated state path advances one business day per step with model time dt = 1/252 per step (pandas business days, no holidays, so about 261 steps per calendar year).
- Option prices and the true densities use model time T = dte / 365 for a maturity of dte calendar days.

The horizon of the study is 30 calendar days. The number of business days in the 30 calendar days after date t is 20, 21 or 22 depending on t. Required deliverables:

1. A function in src/bpk/physical (for example trading_days_in_horizon(date, calendar_days)) that returns the number of business days in (date, date + calendar_days], and a function that returns the model time n/252 of an n-step path horizon. Both documented with the convention above.
2. A test against the generator that (a) checks the function equals the count of simulated steps the generator actually takes between the two dates, (b) checks that the variance of n-step simulated log returns scales as n/252 within Monte Carlo error, and (c) asserts the size of the gap between pricing time dte / 365 and path time n/252 for each date, so that any silent change of the generator convention fails the test.
3. A decision, stated in the Phase 3 docs, on which clock the forecast uses. The FHS forecast is a path forecast over n trading days. The true P density used for scoring it is the generator's Heston P density evaluated at model time n/252 (a new generator method taking model time, not dte). The gap to the pricing-horizon P density at T = 30/365 is reported as its own table, because it is a fixed bias that the EPK inherits when Q is at T = 30/365 and P at n/252.

## 2. FHS with oracle drift and with estimated drift, both reported

- GJR-GARCH(1,1) fitted by quasi maximum likelihood on returns available at each date only (expanding or rolling window), standardized residuals resampled (filtered historical simulation), n-day paths simulated.
- Oracle drift: the conditional mean of the n-step log return implied by the generator (from the true state variance), used as the drift in the simulation. It isolates the volatility and shape model.
- Estimated drift: drift estimated from past returns only (rolling mean, shrunk to zero as a second variant). The estimated-drift run is the one that can be used on real data.
- Both runs are reported in the same tables. Oracle drift is never used alone as a headline.
- A HAR-RV variant is a robustness check as in the project spec.

## 3. Scoring against the true P density, plus PIT and Berkowitz

- Score the FHS density against the true P density of the generator at the same n-day horizon: L1 error in total and by region of the log return, and mean and standard deviation errors.
- Which true P: two exist and both are reported. (a) the law of the simulated path (plain Heston P at model time n/252): this is what FHS targets, and the only one for which PIT is valid. (b) the pricing P density, which is (a)-type density shifted by the stored location xi so that the same SDF prices bond and index. The shift xi per date and maturity is in the generator truth table. The difference between (a) and (b) is a known misspecification of the generator, and it enters the oracle mode of the Phase 5 identification study. Reported, not hidden.
- Calibration: probability integral transforms of realized n-day returns under the forecast density (realized returns from the simulated path), the Berkowitz likelihood ratio test on the PITs (with the overlap issue handled by non-overlapping horizons or a HAC-adjusted version, stated in the code), the Knuppel test, and the recalibration step (Fackler-King) with and without it.
- No lookahead: a test that fails if any future observation enters the forecast at date t.

## 4. Outputs

Tables with config hash and date range (as in Phase 1 and 2), results-log rows for every variant (oracle and estimated drift, GARCH and HAR, with and without recalibration), and figures. All on the synthetic panel; a real-data run waits for real return data.
