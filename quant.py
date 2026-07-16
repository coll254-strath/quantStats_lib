"""
=====================================================================================
MAG 7 QUANT ANALYSIS PIPELINE
=====================================================================================
A 5-stage quant equity research workflow:
    1. ACQUIRE    -> price history (yfinance) + fundamentals (finvizfinance)
    2. STRUCTURE  -> clean returns matrix (pandas / numpy)
    3. MEASURE    -> risk & return metrics (numpy / QuantStats)
    4. BENCHMARK  -> each name vs S&P 500 (QuantStats)
    5. VISUALIZE  -> executive dashboard + individual tearsheets

Output artifacts (all written to ./output/):
    - executive_summary.xlsx          (one-page metrics table, exec-ready)
    - dashboard.png                   (4-panel visual summary)
    - tearsheets/<TICKER>_tearsheet.html   (full QuantStats report per stock)
=====================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs
import matplotlib.pyplot as plt
import seaborn as sns
from finvizfinance.quote import finvizfinance

warnings.filterwarnings("ignore")
qs.extend_pandas()  # attaches .sharpe(), .sortino(), etc. directly onto pandas Series

# -------------------------------------------------------------------------------------
# STAGE 0: CONFIGURATION
# -------------------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
BENCHMARK = "SPY"
LOOKBACK_YEARS = 3
OUTPUT_DIR = "output"
TEARSHEET_DIR = os.path.join(OUTPUT_DIR, "tearsheets")

os.makedirs(TEARSHEET_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
COLOR_PALETTE = sns.color_palette("Set2", len(TICKERS))


# -------------------------------------------------------------------------------------
# STAGE 1: ACQUIRE
# -------------------------------------------------------------------------------------
def fetch_price_data(tickers, benchmark, years):
    """Pull adjusted close prices for all names + benchmark in one call."""
    all_symbols = tickers + [benchmark]
    print(f"[ACQUIRE] Downloading {len(all_symbols)} symbols, {years}y history...")
    data = yf.download(
        all_symbols,
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
    )["Close"]
    return data.dropna(how="all")


def fetch_fundamentals(tickers):
    """Pull a fundamental snapshot per ticker via Finviz (P/E, Market Cap, etc.)."""
    print("[ACQUIRE] Pulling Finviz fundamental snapshots...")
    rows = []
    fields = ["P/E", "Market Cap", "EPS (ttm)", "ROE", "Debt/Eq", "Beta", "Dividend"]
    for t in tickers:
        try:
            stock = finvizfinance(t)
            fund = stock.ticker_fundament()
            row = {"Ticker": t}
            row.update({f: fund.get(f, "N/A") for f in fields})
            rows.append(row)
        except Exception as e:
            rows.append({"Ticker": t, "Error": str(e)})
    return pd.DataFrame(rows).set_index("Ticker")


# -------------------------------------------------------------------------------------
# STAGE 2: STRUCTURE
# -------------------------------------------------------------------------------------
def build_returns_matrix(price_df):
    """Convert prices to daily simple returns, dropping the first NaN row."""
    print("[STRUCTURE] Building daily returns matrix...")
    returns = price_df.pct_change().dropna()
    return returns


# -------------------------------------------------------------------------------------
# STAGE 3 & 4: MEASURE + BENCHMARK
# -------------------------------------------------------------------------------------
def compute_metrics_table(returns, tickers, benchmark):
    """
    Build the executive metrics table: CAGR, Volatility, Sharpe, Sortino,
    Max Drawdown, Beta vs benchmark, and correlation to benchmark.
    """
    print("[MEASURE] Computing risk/return metrics per name...")
    bench_returns = returns[benchmark]
    records = []

    for t in tickers:
        r = returns[t]
        records.append({
            "Ticker": t,
            "CAGR %": round(qs.stats.cagr(r) * 100, 2),
            "Volatility % (ann.)": round(qs.stats.volatility(r) * 100, 2),
            "Sharpe": round(qs.stats.sharpe(r), 2),
            "Sortino": round(qs.stats.sortino(r), 2),
            "Max Drawdown %": round(qs.stats.max_drawdown(r) * 100, 2),
            "Beta vs SPY": round(qs.stats.greeks(r, bench_returns)["beta"], 2),
            "Corr. vs SPY": round(r.corr(bench_returns), 2),
        })

    df = pd.DataFrame(records).set_index("Ticker")
    return df.sort_values("Sharpe", ascending=False)


def generate_tearsheets(returns, tickers, benchmark):
    """Generate a full QuantStats HTML tearsheet per ticker vs the benchmark."""
    print("[BENCHMARK] Generating individual QuantStats tearsheets...")
    bench_returns = returns[benchmark]
    for t in tickers:
        path = os.path.join(TEARSHEET_DIR, f"{t}_tearsheet.html")
        qs.reports.html(
            returns[t],
            benchmark=bench_returns,
            output=path,
            title=f"{t} vs {benchmark}",
        )
        print(f"    -> {path}")


# -------------------------------------------------------------------------------------
# STAGE 5: VISUALIZE
# -------------------------------------------------------------------------------------
def build_dashboard(returns, metrics_df, tickers, benchmark):
    """4-panel executive dashboard: cumulative returns, drawdowns, correlation, risk-return."""
    print("[VISUALIZE] Rendering executive dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Magnificent 7 — Quant Performance Dashboard", fontsize=20, fontweight="bold")

    # Panel 1: Cumulative returns
    ax = axes[0, 0]
    cum_returns = (1 + returns[tickers]).cumprod() - 1
    for i, t in enumerate(tickers):
        ax.plot(cum_returns.index, cum_returns[t] * 100, label=t, color=COLOR_PALETTE[i], linewidth=1.8)
    bench_cum = (1 + returns[benchmark]).cumprod() - 1
    ax.plot(bench_cum.index, bench_cum * 100, label=benchmark, color="black", linewidth=2.5, linestyle="--")
    ax.set_title("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_ylabel("% Return")

    # Panel 2: Drawdown
    ax = axes[0, 1]
    for i, t in enumerate(tickers):
        dd = qs.stats.to_drawdown_series(returns[t]) * 100
        ax.plot(dd.index, dd, label=t, color=COLOR_PALETTE[i], linewidth=1.5)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("% Drawdown")

    # Panel 3: Correlation heatmap
    ax = axes[1, 0]
    corr = returns[tickers + [benchmark]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=False)
    ax.set_title("Correlation Matrix")

    # Panel 4: Risk-return scatter
    ax = axes[1, 1]
    ax.scatter(metrics_df["Volatility % (ann.)"], metrics_df["CAGR %"],
               s=200, c=COLOR_PALETTE[:len(metrics_df)], edgecolors="black")
    for t in metrics_df.index:
        ax.annotate(t, (metrics_df.loc[t, "Volatility % (ann.)"], metrics_df.loc[t, "CAGR %"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Risk vs Return")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, "dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    -> {out_path}")


def export_executive_summary(metrics_df, fundamentals_df):
    """Combine performance + fundamentals into one Excel workbook for leadership review."""
    print("[VISUALIZE] Exporting executive_summary.xlsx...")
    out_path = os.path.join(OUTPUT_DIR, "executive_summary.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Performance Metrics")
        fundamentals_df.to_excel(writer, sheet_name="Fundamentals")
    print(f"    -> {out_path}")


# -------------------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------------------
def main():
    prices = fetch_price_data(TICKERS, BENCHMARK, LOOKBACK_YEARS)
    returns = build_returns_matrix(prices)

    metrics_df = compute_metrics_table(returns, TICKERS, BENCHMARK)
    fundamentals_df = fetch_fundamentals(TICKERS)

    generate_tearsheets(returns, TICKERS, BENCHMARK)
    build_dashboard(returns, metrics_df, TICKERS, BENCHMARK)
    export_executive_summary(metrics_df, fundamentals_df)

    print("\n=== EXECUTIVE METRICS TABLE ===")
    print(metrics_df.to_string())
    print("\nPipeline complete. See ./output/ for all deliverables.")


if __name__ == "__main__":
    main()
