# run_momentum.py  (sits at repo root, never committed with results)

from pathlib import Path
import pandas as pd
from strategies.factors.momentum import MomentumStrategy
from backtesting.engine import run_backtest

DATA_DIR = Path("data/raw")

def ffill_after_first(series):
    first = series.first_valid_index()
    if first is None:
        return series
    out = series.copy()
    out.loc[first:] = series.loc[first:].ffill()
    return out

def load(filepath, train_end):
    df = pd.read_csv(filepath).set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().iloc[:-1]
    idx = pd.date_range(df.index.min(), pd.Timestamp(train_end), freq="D")
    return df.reindex(idx).apply(ffill_after_first)

TRAIN_END = "2025-01-01"

df_universe = pd.read_csv(DATA_DIR / "filtered_top_mcaps_enriched_23_01_2026_with_cex_lists_new.csv")
universe_set = set(df_universe.rename(columns={"id": "Coingecko-id"})["Coingecko-id"])

prices = load(DATA_DIR / "top1000_prices_2015_to_today_12_03_2026.csv", TRAIN_END)
mcap   = load(DATA_DIR / "top1000_mcap_2015_to_today_12_03_2026.csv",   TRAIN_END)

eligible = list(prices.columns.intersection(universe_set))
prices   = prices[eligible]
mcap     = mcap[mcap.columns.intersection(universe_set)]

# ── Run ───────────────────────────────────────────────────────────────────────
strategy = MomentumStrategy()   # uses all defaults
output   = run_backtest(strategy, prices, mcap)

print(output["metrics"])

# Save
output["results"].to_csv("results/momentum_results.csv")
output["weights"].to_csv("results/momentum_weights.csv")