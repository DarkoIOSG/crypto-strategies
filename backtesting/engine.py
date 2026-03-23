# backtesting/engine.py

import numpy as np
import pandas as pd
from pathlib import Path
from backtesting.metrics import full_report


def run_backtest(
    strategy,
    prices: pd.DataFrame,
    mcap: pd.DataFrame,
    fee_rate: float = None,
) -> dict:
    """
    Takes any BaseStrategy, runs it, returns results dict.
    fee_rate overrides strategy param if provided.
    """
    fee = fee_rate if fee_rate is not None else strategy.params.get("fee_rate", 0.001)
    p   = strategy.params

    weights = strategy.generate_signals(prices=prices, mcap=mcap)

    rebal_dates = weights.index.tolist()
    records     = []

    def snap(date, idx):
        candidates = idx[idx <= date]
        return candidates[-1] if not candidates.empty else None

    rebal_schedule = pd.date_range(
        start=rebal_dates[0],
        end=p["train_end"],
        freq=p["rebal_freq"],
    )

    for i, rebal_date in enumerate(rebal_dates):
        # Find next rebalance
        future = [d for d in rebal_dates if d > rebal_date]
        if not future:
            continue
        next_rebal = snap(future[0], prices.index)
        if next_rebal is None or next_rebal == rebal_date:
            continue

        w_row = weights.loc[rebal_date]
        w_row = w_row[w_row != 0]

        if w_row.empty:
            continue

        period_ret_gross = 0.0
        for token, weight in w_row.items():
            if token not in prices.columns:
                continue
            p0 = prices.loc[rebal_date, token] if rebal_date in prices.index else np.nan
            p1 = prices.loc[next_rebal,  token] if next_rebal  in prices.index else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                token_ret       = p1 / p0 - 1
                period_ret_gross += weight * token_ret

        period_ret = period_ret_gross - 2 * fee

        records.append({
            "rebal_date":       rebal_date,
            "next_rebal":       next_rebal,
            "period_ret_gross": period_ret_gross,
            "period_ret":       period_ret,
        })

    results              = pd.DataFrame(records).set_index("rebal_date")
    results["cum_ret"]   = (1 + results["period_ret"]).cumprod()
    results["drawdown"]  = results["cum_ret"] / results["cum_ret"].cummax() - 1

    metrics = full_report(results["period_ret"], label=strategy.NAME)

    return {
        "results":  results,
        "metrics":  metrics,
        "weights":  weights,
        "metadata": strategy.get_metadata(),
    }