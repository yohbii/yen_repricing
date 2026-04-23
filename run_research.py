from __future__ import annotations

import warnings

import pandas as pd

from src.config import DIRS, LOGS, OUTPUTS, TABLES
from src.data_loader import load_all_sources
from src.models import Y_COLS, fit_event_lp_irf, fit_interaction, fit_linear_var, fit_ms_proxy, fit_stvar, fit_tvar, regime_irf, select_lag
from src.preprocess import build_model_dataset
from src.reporting import (
    event_window_summary,
    plot_event_window,
    plot_fragility,
    plot_irfs,
    plot_ms_proxy,
    plot_time_series,
    save_tables,
    write_json_summary,
)


def main():
    warnings.filterwarnings("ignore")
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    sources = load_all_sources()
    df = build_model_dataset(sources)

    lag_table = select_lag(df, max_lag=5)
    bic_lag = int(lag_table.loc[lag_table["bic"].idxmin(), "lag"])
    p = max(1, min(bic_lag, 2))

    linear = fit_linear_var(df, p)
    interaction = fit_interaction(df, p)
    stvar = fit_stvar(df, p)
    tvar = fit_tvar(df, p)
    event_lp = fit_event_lp_irf(df, p)
    ms_probs = fit_ms_proxy(df)
    ms_probs.to_csv(TABLES / "ms_proxy_probabilities.csv", index=False)

    equal_interaction = fit_interaction(df, p, frag_col="fragility_equal")
    equal_stvar = fit_stvar(df, p, frag_col="fragility_equal")
    dummy_df = df.copy()
    dummy_df["boj_shock"] = dummy_df["boj_event"].astype(float)
    dummy_linear = fit_linear_var(dummy_df, p)
    equal_low = regime_irf(equal_stvar["low"], p)
    equal_high = regime_irf(equal_stvar["high"], p)
    equal_diff0 = (equal_high.loc[0, Y_COLS] - equal_low.loc[0, Y_COLS]).to_dict()
    robustness_extra = pd.DataFrame([
        {
            "check": "equal_weight_fragility_interaction",
            "metric": "LR p-value",
            "value": equal_interaction["lr_pvalue"],
        },
        {
            "check": "equal_weight_fragility_stvar",
            "metric": "gamma",
            "value": equal_stvar["gamma"],
        },
        {
            "check": "equal_weight_fragility_stvar",
            "metric": "threshold_c",
            "value": equal_stvar["threshold_c"],
        },
        *[
            {"check": "equal_weight_fragility_stvar", "metric": f"h0_high_minus_low_{col}", "value": value}
            for col, value in equal_diff0.items()
        ],
        *[
            {"check": "event_dummy_linear_var", "metric": f"h0_response_{col}", "value": value}
            for col, value in zip(Y_COLS, dummy_linear["coef"][-1])
        ],
    ])
    robustness_extra.to_csv(TABLES / "additional_robustness.csv", index=False)

    event_summary = event_window_summary(df)
    save_tables(df, lag_table, linear, interaction, stvar, tvar, event_summary)

    plot_time_series(df)
    plot_fragility(df)
    plot_event_window(event_summary)
    plot_irfs(stvar, tvar, linear, p, event_lp=event_lp)
    plot_ms_proxy(df, ms_probs)

    summary = {
        "sample_start": str(df["date"].min().date()),
        "sample_end": str(df["date"].max().date()),
        "observations": int(len(df)),
        "selected_lag_bic": bic_lag,
        "estimated_lag_used": p,
        "boj_event_days_mapped": int(df["boj_event"].sum()),
        "interaction_lr_pvalue": float(interaction["lr_pvalue"]),
        "stvar_gamma": float(stvar["gamma"]),
        "stvar_threshold_c": float(stvar["threshold_c"]),
        "tvar_threshold": float(tvar["threshold"]),
        "event_lp_tightening_events": int(event_lp["n_tightening_events"]),
        "event_lp_mean_positive_tightening_shock_bp": float(event_lp["mean_tightening_shock"]),
    }
    write_json_summary(OUTPUTS / "run_summary.json", summary)
    (LOGS / "pipeline_complete.txt").write_text("Pipeline completed successfully.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
