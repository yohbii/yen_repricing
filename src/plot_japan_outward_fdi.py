from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import requests


OUTPUT_DIR = Path("outputs") / "figures"
DATA_DIR = Path("outputs") / "data"
WB_API_TEMPLATE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
INDICATOR = "BM.KLT.DINV.WD.GD.ZS"  # FDI, net outflows (% of GDP)
COUNTRIES = {
    "JPN": "Japan",
    "USA": "United States",
    "DEU": "Germany",
    "CHN": "China",
}
START_YEAR = 1982


def fetch_world_bank_series(country_code: str, indicator: str) -> pd.DataFrame:
    response = requests.get(
        WB_API_TEMPLATE.format(country=country_code, indicator=indicator),
        params={"format": "json", "per_page": 200},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected World Bank payload for {country_code}: {payload}")

    records = payload[1]
    rows = []
    for item in records:
        value = item.get("value")
        year = item.get("date")
        if value is None or year is None:
            continue
        rows.append({"year": int(year), "value": float(value), "country": country_code})

    if not rows:
        raise ValueError(f"No usable data points returned for {country_code}.")
    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    data_frames = [fetch_world_bank_series(code, INDICATOR) for code in COUNTRIES]
    df = pd.concat(data_frames, ignore_index=True)
    df = df[df["year"] >= START_YEAR].copy()
    df["country_name"] = df["country"].map(COUNTRIES)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df["ma3"] = df.groupby("country")["value"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return df


def plot_outward_fdi_trend(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data_out = DATA_DIR / "worldbank_outward_fdi_pct_gdp.csv"
    df.to_csv(data_out, index=False, encoding="utf-8-sig")

    # ── Global font: Times New Roman ──────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12.6, 7.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Lines ─────────────────────────────────────────────────────────────────
    colors = {"JPN": "#c62828", "USA": "#4e79a7", "DEU": "#59a14f", "CHN": "#9c6fb6"}
    for code, name in COUNTRIES.items():
        one = df[df["country"] == code]
        lw = 3.2 if code == "JPN" else 2.0
        alpha = 1.0 if code == "JPN" else 0.78
        zorder = 4 if code == "JPN" else 3
        ax.plot(
            one["year"],
            one["ma3"],
            color=colors[code],
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            label=f"{name} (3Y MA)",
        )
        ax.scatter(one["year"].iloc[-1], one["ma3"].iloc[-1], color=colors[code], s=32, zorder=5)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(0, color="#7d7d7d", linewidth=1, linestyle="--", alpha=0.8, zorder=2)
    ymax = max(df["ma3"].max(), 0)
    for xval, label, yoffset in [(1985, "Plaza Accord (1985)", 0.96), (2013, "Abenomics (2013)", 0.86)]:
        ax.axvline(xval, color="#888888", linewidth=1, linestyle=(0, (2, 2)), alpha=0.85, zorder=2)
        ax.text(xval + 0.3, ymax * yoffset, label, fontsize=9.5, color="#555555",
                fontstyle="italic")

    # ── Titles & labels ───────────────────────────────────────────────────────
    fig.suptitle(
        "Outward FDI Trend: Japan vs Major Economies, 1982–present",
        fontsize=17, fontweight="bold", y=0.98,
    )
    ax.set_title(
        "Three-year moving average of FDI net outflows (% of GDP); Japan highlighted for emphasis",
        fontsize=10.5, color="#4f4f4f", pad=8,
    )
    ax.set_ylabel("FDI net outflows (% of GDP)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)

    # ── Axis style ────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8b8b8")
    ax.spines["bottom"].set_color("#b8b8b8")
    ax.tick_params(axis="both", colors="#3f3f3f")
    ax.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.35, color="#bdbdbd")
    ax.grid(axis="x", visible=False)

    # ── Legend: upper-right, no overlap ──────────────────────────────────────
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=10.5)

    # ── Save PNG + PDF ────────────────────────────────────────────────────────
    png_path = OUTPUT_DIR / "japan_outward_fdi_trend.png"
    pdf_path = OUTPUT_DIR / "japan_outward_fdi_trend.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    df = build_dataset()
    output_path = plot_outward_fdi_trend(df)
    print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
