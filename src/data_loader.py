from __future__ import annotations

import json
import re
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .config import BOJ_EVENT_FALLBACK, DATA_RAW, END_DATE, START_DATE


HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout: int = 40, retries: int = 3) -> requests.Response:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
        errors="coerce",
    )


def load_mof_jgb() -> pd.DataFrame:
    urls = [
        "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
        "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
    ]
    frames = []
    for idx, url in enumerate(urls):
        text = _get(url).text
        (DATA_RAW / f"mof_jgb_{idx}.csv").write_text(text, encoding="utf-8")
        df = pd.read_csv(StringIO(text), skiprows=1, na_values=["-", ""])
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["date"])
        frames.append(df[["date", "2Y", "10Y"]])
    out = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
    out = out.sort_values("date")
    out["jpy_2y"] = _numeric(out["2Y"])
    out["jpy_10y"] = _numeric(out["10Y"])
    return out[["date", "jpy_2y", "jpy_10y"]]


def load_usdjpy() -> pd.DataFrame:
    url = f"https://api.frankfurter.app/{START_DATE}..{END_DATE}?from=USD&to=JPY"
    payload = _get(url).json()
    (DATA_RAW / "frankfurter_usdjpy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [{"date": date, "usdjpy": values["JPY"]} for date, values in payload["rates"].items()]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date")


def load_ewj() -> pd.DataFrame:
    url = (
        "https://api.nasdaq.com/api/quote/EWJ/historical"
        f"?assetclass=etf&fromdate={START_DATE}&todate={END_DATE}&limit=9999"
    )
    payload = _get(url).json()
    (DATA_RAW / "nasdaq_ewj.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["data"]["tradesTable"]["rows"]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out["eq_price"] = _numeric(out["close"])
    return out[["date", "eq_price"]].sort_values("date")


def load_vix() -> pd.DataFrame:
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    text = _get(url).text
    (DATA_RAW / "cboe_vix.csv").write_text(text, encoding="utf-8")
    out = pd.read_csv(StringIO(text))
    out["date"] = pd.to_datetime(out["DATE"])
    out["vix"] = _numeric(out["CLOSE"])
    return out[["date", "vix"]].sort_values("date")


def load_us_treasury() -> pd.DataFrame:
    frames = []
    for year in range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 1):
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        )
        text = _get(url).text
        (DATA_RAW / f"treasury_{year}.csv").write_text(text, encoding="utf-8")
        df = pd.read_csv(StringIO(text))
        df["date"] = pd.to_datetime(df["Date"])
        df["us_2y"] = _numeric(df["2 Yr"])
        frames.append(df[["date", "us_2y"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")


def load_boj_events() -> pd.DataFrame:
    dates = set(pd.to_datetime(BOJ_EVENT_FALLBACK))
    for year in range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 1):
        url = f"https://www.boj.or.jp/en/mopo/mpmdeci/mpr_{year}/index.htm"
        try:
            html = _get(url, timeout=20, retries=1).text
        except Exception:
            continue
        (DATA_RAW / f"boj_releases_{year}.html").write_text(html, encoding="utf-8")
        for match in re.finditer(r"([A-Z][a-z]{2}\.\s+\d{1,2},\s+%d).*?Statement on Monetary Policy" % year, html, re.S):
            dates.add(pd.to_datetime(match.group(1).replace(".", "")))
    events = pd.DataFrame({"event_date": sorted(d for d in dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE))})
    events["source"] = "BOJ release page or curated fallback"
    events.to_csv(DATA_RAW / "boj_events.csv", index=False)
    return events


def load_all_sources() -> dict[str, pd.DataFrame]:
    return {
        "jgb": load_mof_jgb(),
        "usdjpy": load_usdjpy(),
        "equity": load_ewj(),
        "vix": load_vix(),
        "ust": load_us_treasury(),
        "events": load_boj_events(),
    }
