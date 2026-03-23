# strategies/factors/momentum.py

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):

    NAME        = "volatility_adjusted_momentum"
    VERSION     = "1.0.0"
    AUTHOR      = "your-name"
    DESCRIPTION = "3-week rolling Sharpe-like score, long top 25% short bottom 25%"

    DEFAULTS = {
        "lookback_weeks":   3,
        "universe_size":    200,
        "long_pct":         0.25,
        "short_pct":        0.25,
        "fee_rate":         0.0005,
        "min_history_days": 21,
        "signal_type":      "sharpe",        # sharpe | mean | sortino
        "position_sizing":  "score_weighted", # equal | score_weighted
        "rebal_freq":       "W-MON",
        "backtest_start":   "2020-01-01",
        "train_end":        "2025-01-01",
    }

    def __init__(self, params: dict = {}):
        super().__init__({**self.DEFAULTS, **params})

    def _validate_params(self) -> None:
        p = self.params
        if not 0 < p["long_pct"] <= 0.5:
            raise ValueError("long_pct must be between 0 and 0.5")
        if not 0 < p["short_pct"] <= 0.5:
            raise ValueError("short_pct must be between 0 and 0.5")
        if p["signal_type"] not in ("sharpe", "mean", "sortino"):
            raise ValueError("signal_type must be sharpe | mean | sortino")
        if p["position_sizing"] not in ("equal", "score_weighted"):
            raise ValueError("position_sizing must be equal | score_weighted")
        if p["universe_size"] < 4:
            raise ValueError("universe_size must be at least 4")

    def _compute_score(
        self, ret_window: pd.DataFrame
    ) -> pd.Series:
        """Compute momentum score for each token over the lookback window."""
        mean_ret = ret_window.mean()
        std_ret  = ret_window.std()
        signal   = self.params["signal_type"]

        if signal == "sortino":
            downside = ret_window.clip(upper=0)
            std_down = downside.std().replace(0, np.nan)
            return (mean_ret / std_down).dropna()
        elif signal == "mean":
            return mean_ret.dropna()
        else:  # sharpe
            return (mean_ret / std_ret.replace(0, np.nan)).dropna()

    def _weight_positions(
        self, score: pd.Series, selected: list
    ) -> pd.Series:
        """Return normalized weights for selected tokens."""
        sizing = self.params["position_sizing"]
        if sizing == "score_weighted":
            w = score[selected].abs()
            w = w / w.sum() if w.sum() > 0 else pd.Series(
                1 / len(w), index=w.index
            )
        else:  # equal
            w = pd.Series(1 / len(selected), index=selected)
        return w

    def generate_signals(
        self,
        prices: pd.DataFrame,
        mcap: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of position weights.
        Index: rebal_dates | Columns: tokens | Values: weight (-0.5 to +0.5)
        Positive = long, Negative = short
        """
        p             = self.params
        lookback_days = p["lookback_weeks"] * 7
        daily_returns = prices.pct_change(fill_method=None)

        rebal_dates = pd.date_range(
            start=max(
                prices.index.min() + pd.Timedelta(days=lookback_days + 7),
                pd.Timestamp(p["backtest_start"]),
            ),
            end=p["train_end"],
            freq=p["rebal_freq"],
        )

        def snap(date, idx):
            candidates = idx[idx <= date]
            return candidates[-1] if not candidates.empty else None

        weight_records = {}

        for sched_date in rebal_dates:
            rebal_date = snap(sched_date, prices.index)
            if rebal_date is None:
                continue

            prev_dates = prices.index[prices.index < rebal_date]
            if len(prev_dates) < lookback_days:
                continue
            prev_day = prev_dates[-1]

            # Universe: top N by mcap
            mcap_snap = mcap.loc[prev_day].dropna() if prev_day in mcap.index else pd.Series(dtype=float)
            if mcap_snap.empty:
                continue

            top_tokens = mcap_snap.nlargest(p["universe_size"]).index.tolist()
            tradeable  = [
                t for t in top_tokens
                if t in prices.columns and pd.notna(prices.loc[rebal_date, t])
            ]
            if len(tradeable) < 4:
                continue

            # Signal window
            prev_loc      = prices.index.get_loc(prev_day)
            win_start_loc = max(0, prev_loc - lookback_days + 1)
            window_slice  = prices.index[win_start_loc : prev_loc + 1]

            ret_window   = daily_returns.loc[window_slice, tradeable]
            valid_tokens = ret_window.columns[
                ret_window.count() >= p["min_history_days"]
            ].tolist()
            if len(valid_tokens) < 4:
                continue

            ret_window = ret_window[valid_tokens].dropna(how="all")
            score      = self._compute_score(ret_window)
            if len(score) < 4:
                continue

            # Select longs and shorts
            n_long  = max(1, int(np.floor(len(score) * p["long_pct"])))
            n_short = max(1, int(np.floor(len(score) * p["short_pct"])))
            longs   = score.nlargest(n_long).index.tolist()
            shorts  = score.nsmallest(n_short).index.tolist()

            lw = self._weight_positions(score, longs)
            sw = self._weight_positions(score, shorts)

            w_dict = {}
            for t in lw.index:
                w_dict[t] =  0.5 * lw[t]   # long leg
            for t in sw.index:
                w_dict[t] = -0.5 * sw[t]   # short leg

            weight_records[rebal_date] = w_dict

        weights_df = pd.DataFrame(weight_records).T
        weights_df.index.name = "rebal_date"
        return weights_df.sort_index().fillna(0)