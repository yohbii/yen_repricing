from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, MAJOR_BOJ_EVENTS, TABLES
from .models import TRANSMISSION_COLS, Y_COLS, cumulative_irf, regime_irf


def save_tables(df, lag_table, linear, interaction, stvar, tvar, event_summary):
    pd.DataFrame([
        {"variable": "fx_ret", "definition": "-100 * diff(log(USD/JPY)); positive means yen appreciation", "source": "Frankfurter ECB-based USD/JPY"},
        {"variable": "jpy_2y_change", "definition": "Daily change in 2Y JGB yield, basis points", "source": "Japan Ministry of Finance"},
        {"variable": "jpy_10y_change", "definition": "Daily change in 10Y JGB yield, basis points", "source": "Japan Ministry of Finance"},
        {"variable": "eq_ret", "definition": "100 * diff(log(EWJ close)); Japan equity proxy in USD", "source": "Nasdaq historical API"},
        {"variable": "eq_ret_local", "definition": "EWJ USD return minus yen-appreciation return; approximate local-currency Japan equity return", "source": "Constructed from EWJ and USD/JPY"},
        {"variable": "boj_shock", "definition": "2Y JGB yield change on BOJ decision dates, zero otherwise", "source": "BOJ release dates + MOF JGB yields"},
        {"variable": "boj_tightening_shock", "definition": "Positive part of BOJ event-day 2Y JGB yield change; zero for easing/non-tightening events", "source": "Constructed"},
        {"variable": "fragility_pca", "definition": "PC1 of standardized differential compression, yen vol, JGB pressure, VIX", "source": "Constructed"},
        {"variable": "fragility_equal", "definition": "Equal-weight average of standardized fragility components", "source": "Constructed"},
    ]).to_csv(TABLES / "variable_definitions.csv", index=False)
    desc_cols = Y_COLS + ["eq_ret_local", "boj_shock", "boj_tightening_shock", "fragility_pca", "fragility_equal"]
    df[desc_cols].describe().T.to_csv(TABLES / "descriptive_statistics.csv")
    df[Y_COLS + ["eq_ret_local", "boj_shock", "boj_tightening_shock", "fragility_pca", "vix", "us_2y", "jpy_2y", "jpy_10y"]].corr().to_csv(TABLES / "correlation_matrix.csv")
    lag_table.to_csv(TABLES / "lag_selection.csv", index=False)
    pd.DataFrame(linear["coef"], index=linear["names"], columns=Y_COLS).to_csv(TABLES / "linear_var_coefficients.csv")
    pd.DataFrame(interaction["coef"], index=interaction["names"], columns=Y_COLS).to_csv(TABLES / "interaction_coefficients.csv")
    pd.DataFrame({
        "metric": ["lr_stat", "lr_pvalue", "nobs"],
        "value": [interaction["lr_stat"], interaction["lr_pvalue"], interaction["n"]],
    }).to_csv(TABLES / "interaction_test.csv", index=False)
    pd.DataFrame({
        "metric": ["gamma", "threshold_c", "transition_mean", "ssr", "converged"],
        "value": [stvar["gamma"], stvar["threshold_c"], stvar["transition_mean"], stvar["ssr"], stvar["success"]],
    }).to_csv(TABLES / "stvar_summary.csv", index=False)
    pd.DataFrame({
        "metric": ["threshold", "low_n", "high_n", "ssr"],
        "value": [tvar["threshold"], tvar["low_n"], tvar["high_n"], tvar["ssr"]],
    }).to_csv(TABLES / "tvar_summary.csv", index=False)
    event_summary.to_csv(TABLES / "event_window_summary.csv", index=False)
    major_tightening_validation(df).to_csv(TABLES / "major_tightening_event_validation.csv", index=False)


def major_tightening_validation(df: pd.DataFrame, horizons=(0, 1, 2, 5, 10, 20)) -> pd.DataFrame:
    work = df.copy()
    if "eq_ret_local" not in work:
        work["eq_ret_local"] = work["eq_ret"] - work["fx_ret"]
    state = work["fragility_pca"].shift(1)
    event_locs = [
        i for i in range(len(work) - max(horizons))
        if work["boj_shock"].iloc[i] >= 2.0 and pd.notna(state.iloc[i])
    ]
    if not event_locs:
        return pd.DataFrame()
    threshold = float(np.median([state.iloc[i] for i in event_locs]))
    rows = []
    for regime, locs in [
        ("low_fragility", [i for i in event_locs if state.iloc[i] <= threshold]),
        ("high_fragility", [i for i in event_locs if state.iloc[i] > threshold]),
    ]:
        for h in horizons:
            values = []
            for i in locs:
                response = work[TRANSMISSION_COLS].iloc[i:i + h + 1].sum()
                values.append(response)
            mean = pd.DataFrame(values).mean()
            rows.append({
                "regime": regime,
                "horizon": h,
                "n_events": len(locs),
                "mean_shock_bp": float(work["boj_shock"].iloc[locs].mean()) if locs else np.nan,
                "fragility_split_median": threshold,
                **mean.to_dict(),
            })
    return pd.DataFrame(rows)


def event_window_summary(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    rows = []
    event_locs = np.flatnonzero(df["boj_event"].to_numpy() == 1)
    for h in range(-window, window + 1):
        vals = []
        for loc in event_locs:
            idx = loc + h
            if 0 <= idx < len(df):
                vals.append(df.loc[idx, Y_COLS])
        if vals:
            mean = pd.DataFrame(vals).mean()
            rows.append({"event_day": h, **mean.to_dict()})
    return pd.DataFrame(rows)


def plot_time_series(df: pd.DataFrame):
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    series = [("usdjpy", "USD/JPY"), ("jpy_2y", "JGB 2Y (%)"), ("jpy_10y", "JGB 10Y (%)"), ("eq_price", "EWJ equity proxy"), ("vix", "VIX")]
    for ax, (col, title) in zip(axes, series):
        ax.plot(df["date"], df[col], lw=1.1)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_market_time_series.png", dpi=160)
    plt.close(fig)


def plot_fragility(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["date"], df["fragility_pca"], label="PCA fragility", lw=1.2)
    ax.plot(df["date"], df["fragility_equal"], label="Equal-weight fragility", lw=0.9, alpha=0.7)
    for date, label in MAJOR_BOJ_EVENTS.items():
        date = pd.Timestamp(date)
        if df["date"].min() <= date <= df["date"].max():
            ax.axvline(date, color="black", alpha=0.18, lw=0.8)
    ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-09"), color="tab:red", alpha=0.12, label="Aug 2024 turbulence")
    ax.set_title("Fragility index and major BOJ dates")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_fragility_index.png", dpi=160)
    plt.close(fig)


def plot_event_window(event_summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, col in zip(axes.ravel(), Y_COLS):
        ax.plot(event_summary["event_day"], event_summary[col], marker="o", lw=1.2)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_event_window_mean.png", dpi=160)
    plt.close(fig)


def plot_irfs(stvar, tvar, linear, p: int, event_lp: dict | None = None):
    flow_low = regime_irf(stvar["low"], p)
    flow_high = regime_irf(stvar["high"], p)
    flow_diff = flow_high.copy()
    flow_diff[Y_COLS] = flow_high[Y_COLS] - flow_low[Y_COLS]
    flow_low.to_csv(TABLES / "stvar_flow_irf_low_fragility.csv", index=False)
    flow_high.to_csv(TABLES / "stvar_flow_irf_high_fragility.csv", index=False)
    flow_diff.to_csv(TABLES / "stvar_flow_irf_high_minus_low.csv", index=False)

    irf_low = cumulative_irf(flow_low)
    irf_high = cumulative_irf(flow_high)
    diff = irf_high.copy()
    diff[Y_COLS] = irf_high[Y_COLS] - irf_low[Y_COLS]
    irf_low.to_csv(TABLES / "stvar_cumulative_irf_low_fragility.csv", index=False)
    irf_high.to_csv(TABLES / "stvar_cumulative_irf_high_fragility.csv", index=False)
    diff.to_csv(TABLES / "stvar_cumulative_irf_high_minus_low.csv", index=False)
    # Backward-compatible filenames now contain the economically relevant cumulative responses.
    irf_low.to_csv(TABLES / "stvar_irf_low_fragility.csv", index=False)
    irf_high.to_csv(TABLES / "stvar_irf_high_fragility.csv", index=False)
    diff.to_csv(TABLES / "stvar_irf_high_minus_low.csv", index=False)

    if event_lp is None:
        for data, name, title in [
            (irf_low, "04_stvar_low_fragility_irf.png", "STVAR low-fragility cumulative response to +10bp BOJ shock"),
            (irf_high, "05_stvar_high_fragility_irf.png", "STVAR high-fragility cumulative response to +10bp BOJ shock"),
            (diff, "06_stvar_high_minus_low_irf.png", "High minus low cumulative response"),
        ]:
            _plot_irf_frame(data, name, title, Y_COLS)
    else:
        _save_and_plot_event_lp(event_lp)

    tvar_flow_low = regime_irf(tvar["low"], p)
    tvar_flow_high = regime_irf(tvar["high"], p)
    tvar_flow_diff = tvar_flow_high.copy()
    tvar_flow_diff[Y_COLS] = tvar_flow_high[Y_COLS] - tvar_flow_low[Y_COLS]
    tvar_flow_diff.to_csv(TABLES / "tvar_flow_irf_high_minus_low.csv", index=False)

    tvar_low = cumulative_irf(tvar_flow_low)
    tvar_high = cumulative_irf(tvar_flow_high)
    tvar_diff = tvar_high.copy()
    tvar_diff[Y_COLS] = tvar_high[Y_COLS] - tvar_low[Y_COLS]
    tvar_diff.to_csv(TABLES / "tvar_cumulative_irf_high_minus_low.csv", index=False)
    tvar_diff.to_csv(TABLES / "tvar_irf_high_minus_low.csv", index=False)
    _plot_irf_frame(tvar_diff, "07_tvar_high_minus_low_irf.png", "TVAR high minus low cumulative response", Y_COLS)

    pd.DataFrame([
        {
            "model": "Linear VAR",
            "result": "Average contemporaneous response to +10bp BOJ shock",
            **dict(zip(Y_COLS, linear["coef"][-1] * 10.0)),
        },
        {
            "model": "STVAR",
            "result": "High minus low 20-day cumulative response to +10bp BOJ shock",
            **irf_high.merge(irf_low, on="horizon", suffixes=("_high", "_low")).query("horizon == 20")
            .assign(**{col: lambda x, col=col: x[f"{col}_high"] - x[f"{col}_low"] for col in Y_COLS})[Y_COLS]
            .iloc[0].to_dict(),
        },
        {
            "model": "TVAR",
            "result": "High minus low 20-day cumulative response to +10bp BOJ shock",
            **tvar_diff.query("horizon == 20")[Y_COLS].iloc[0].to_dict(),
        },
    ]).to_csv(TABLES / "robustness_summary.csv", index=False)


def _save_and_plot_event_lp(event_lp: dict):
    event_lp["low"].to_csv(TABLES / "event_lp_low_fragility.csv", index=False)
    event_lp["high"].to_csv(TABLES / "event_lp_high_fragility.csv", index=False)
    event_lp["diff"].to_csv(TABLES / "event_lp_high_minus_low.csv", index=False)
    event_lp["coef"].to_csv(TABLES / "event_lp_coefficients.csv", index=False)
    pd.DataFrame([
        {"metric": "low_state_p25_positive_events", "value": event_lp["low_state"]},
        {"metric": "high_state_p75_positive_events", "value": event_lp["high_state"]},
        {"metric": "n_boj_events", "value": event_lp["n_events"]},
        {"metric": "n_tightening_events", "value": event_lp["n_tightening_events"]},
        {"metric": "mean_positive_tightening_shock_bp", "value": event_lp["mean_tightening_shock"]},
    ]).to_csv(TABLES / "event_lp_summary.csv", index=False)
    for data, name, title in [
        (event_lp["low"], "04_stvar_low_fragility_irf.png", "Event LP low-fragility cumulative response to +10bp tightening shock"),
        (event_lp["high"], "05_stvar_high_fragility_irf.png", "Event LP high-fragility cumulative response to +10bp tightening shock"),
        (event_lp["diff"], "06_stvar_high_minus_low_irf.png", "Event LP high minus low cumulative response"),
    ]:
        _plot_irf_frame(data, name, title, TRANSMISSION_COLS)


def _plot_irf_frame(data: pd.DataFrame, file_name: str, title: str, columns=None):
    columns = columns or Y_COLS
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, col in zip(axes.ravel(), columns):
        ax.plot(data["horizon"], data[col], lw=1.4)
        ax.axhline(0, color="black", lw=0.8, alpha=0.35)
        ax.set_title(col)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES / file_name, dpi=160)
    plt.close(fig)


def plot_ms_proxy(df: pd.DataFrame, probs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["fragility_pca"], color="tab:blue", lw=1.0, label="PCA fragility")
    ax2 = ax.twinx()
    ax2.plot(probs["date"], probs["ms_high_prob"], color="tab:red", lw=1.0, alpha=0.75, label="MS high-stress probability")
    ax.set_title("Fragility index and Markov-switching stress proxy")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_ms_proxy_probabilities.png", dpi=160)
    plt.close(fig)


def write_json_summary(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
