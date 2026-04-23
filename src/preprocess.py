from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, END_DATE, START_DATE
from .fragility import add_fragility_indexes


def _next_available(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    pos = index.searchsorted(date)
    if pos >= len(index):
        return None
    return index[pos]


def build_model_dataset(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    price = sources["usdjpy"].merge(sources["equity"], on="date", how="outer").sort_values("date")
    price["fx_ret"] = -100.0 * np.log(price["usdjpy"]).diff()
    price["eq_ret"] = 100.0 * np.log(price["eq_price"]).diff()
    price["eq_ret_local"] = price["eq_ret"] - price["fx_ret"]

    rates = sources["jgb"].merge(sources["ust"], on="date", how="outer").sort_values("date")
    rates[["jpy_2y", "jpy_10y", "us_2y"]] = rates[["jpy_2y", "jpy_10y", "us_2y"]].ffill()
    rates["jpy_2y_change"] = rates["jpy_2y"].diff() * 100.0
    rates["jpy_10y_change"] = rates["jpy_10y"].diff() * 100.0
    rates["us_2y_change"] = rates["us_2y"].diff() * 100.0

    vix = sources["vix"].copy()
    vix["vix"] = vix["vix"].ffill()
    df = price.merge(rates, on="date", how="outer").merge(vix, on="date", how="outer")
    df = df.sort_values("date")
    df["vix"] = df["vix"].ffill()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]

    model_cols = [
        "fx_ret", "jpy_2y_change", "jpy_10y_change", "eq_ret", "eq_ret_local",
        "jpy_2y", "jpy_10y", "us_2y", "usdjpy", "eq_price", "vix",
    ]
    df = df.dropna(subset=["fx_ret", "jpy_2y_change", "jpy_10y_change", "eq_ret", "us_2y", "vix"])
    df = add_fragility_indexes(df)

    df["boj_event"] = 0
    df["boj_shock"] = 0.0
    idx = pd.DatetimeIndex(df["date"])
    mapped = []
    for event_date in sources["events"]["event_date"]:
        mapped_date = _next_available(idx, pd.Timestamp(event_date))
        if mapped_date is None:
            continue
        row = df["date"].eq(mapped_date)
        df.loc[row, "boj_event"] = 1
        df.loc[row, "boj_shock"] = df.loc[row, "jpy_2y_change"]
        mapped.append({"event_date": event_date, "mapped_trading_date": mapped_date})
    pd.DataFrame(mapped).to_csv(DATA_PROCESSED / "boj_event_mapping.csv", index=False)
    df["boj_tightening_shock"] = df["boj_shock"].clip(lower=0.0)
    df["boj_easing_shock"] = (-df["boj_shock"]).clip(lower=0.0)
    df["major_tightening_event"] = (df["boj_tightening_shock"] >= 2.0).astype(int)

    df = df.dropna(subset=["fragility_pca", "fragility_equal"]).reset_index(drop=True)
    df.to_csv(DATA_PROCESSED / "model_daily.csv", index=False)
    return df
