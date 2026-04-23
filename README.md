# BOJ Tightening and Fragility-Dependent Transmission

A daily-data empirical framework for testing whether BOJ tightening shocks transmit differently when market fragility is high.

## What the pipeline produces

- `data/processed/model_daily.csv`: aligned daily modeling dataset.
- `data/processed/boj_event_mapping.csv`: BOJ announcement dates mapped to the next available model trading day.
- `outputs/tables/`: descriptive statistics, correlations, VAR coefficients, interaction tests, STVAR/TVAR summaries, cumulative IRF paths, flow-IRF diagnostics, and robustness summaries.
- `outputs/figures/`: market time series, fragility index, event-window response, STVAR/TVAR response plots, and Markov-switching stress proxy probabilities.
- `outputs/run_summary.json`: compact run metadata and key diagnostics.
- `data_sources.md`: exact data-source and transformation documentation.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python .\run_research.py
```

The script downloads public data, rebuilds the processed dataset, re-estimates the models, and overwrites the output tables and figures.

## Main implementation choices

The baseline endogenous vector is:

```text
fx_ret, jpy_2y_change, jpy_10y_change, eq_ret
```

`boj_shock` is the daily 2Y JGB yield change on BOJ decision dates and zero otherwise. This is a practical policy-repricing proxy. The baseline fragility index is a PCA index built from U.S.-Japan 2Y spread compression, yen volatility, JGB pressure, and VIX. The code also constructs equal-weight and Japan-only alternatives.

The main nonlinear model is a two-regime logistic STVAR estimated by nonlinear least squares. The transition threshold is constrained to the 20th-80th percentile of the fragility state and the transition slope is constrained to avoid an economically uninformative tiny high-state sample.

The endogenous variables are daily returns and yield changes. For that reason, the main response figures and legacy `stvar_irf_*` tables now report cumulative responses, which are the economically relevant level/log-level effects. The one-day flow responses are still saved separately as `stvar_flow_irf_*` and `tvar_flow_irf_*` diagnostics.

## Current run headline

Current sample: 2016-05-19 to 2026-04-21, 2,371 observations, 79 mapped BOJ event days. BIC selects one daily lag.

The interaction test does not reject constant transmission in the baseline PCA specification (`p = 0.949`). The constrained STVAR still shows a larger cumulative yen-appreciation response in high-fragility states, but this nonlinear result should be treated as suggestive rather than decisive because the formal interaction test is weak and the equity proxy is EWJ rather than TOPIX/Nikkei.
