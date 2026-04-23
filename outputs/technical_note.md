# Technical Note

## Research interpretation

The empirical object is BOJ-related policy repricing, not a fully identified structural monetary policy shock. `boj_shock` uses the daily 2Y JGB yield change on BOJ decision dates. This captures market repricing around policy decisions but can include other same-day news.

## Core results from the current run

| Item | Result |
|---|---|
| Sample | 2016-05-19 to 2026-04-21 |
| Observations | 2,371 |
| BOJ event days | 79 |
| Selected lag | 1 |
| Baseline interaction LR p-value | 0.949 |
| STVAR transition slope | 1.889 |
| STVAR threshold | 0.740 |
| TVAR threshold | 0.298 |

For a +10bp BOJ repricing shock, the linear VAR estimates an average contemporaneous response of about +1.21% yen appreciation, +9.81bp in the 2Y JGB yield, +12.65bp in the 10Y JGB yield, and +1.71% in the EWJ equity proxy.

The model variables are daily returns and yield changes, so a flow IRF should normally decay toward zero. The economically relevant object is the cumulative response, which measures the implied level or log-level displacement after the repricing shock.

In the baseline STVAR, the high-fragility 20-day cumulative response for the same +10bp shock is about +3.20% yen appreciation, +10.79bp in the 2Y JGB yield, +12.67bp in the 10Y JGB yield, and +0.97% in the EWJ equity proxy. The high-minus-low 20-day cumulative difference is about +3.49 percentage points for yen appreciation, +1.02bp for the 2Y JGB yield, -0.38bp for the 10Y JGB yield, and -1.08 percentage points for the equity proxy.

## Caveats

1. The formal interaction test does not support strong state dependence in the baseline specification. The nonlinear response figures should therefore be interpreted as exploratory evidence, not a definitive rejection of the linear model.
2. EWJ is a U.S.-listed ETF and embeds USD exposure, U.S. trading hours, and ETF-market effects. It is a practical proxy, not a perfect TOPIX/Nikkei replacement.
3. USD/JPY comes from ECB reference rates through Frankfurter. It is suitable for daily repricing analysis but not for intraday announcement identification.
4. The STVAR is estimated by least squares conditional on a logistic transition function, not by a full structural likelihood with bootstrap confidence bands. Bootstrap inference is a natural next extension.
5. The Markov-switching output is a stress-probability proxy estimated on a reduced stress score, not a full MS-VAR with switching covariance matrices.

## Bottom line

The current public-data implementation supports a cautious reading: BOJ repricing shocks have persistent cumulative effects on Japanese yields and the yen, while the evidence for fragility amplification is suggestive in nonlinear specifications but not strongly confirmed by the baseline interaction test.
