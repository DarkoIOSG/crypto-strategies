# tests/test_momentum.py

import pytest
import pandas as pd
import numpy as np
from strategies.factors.momentum import MomentumStrategy


def make_fake_data(n_tokens=20, n_days=60):
    dates  = pd.date_range("2021-01-01", periods=n_days, freq="D")
    np.random.seed(42)
    data   = np.random.randn(n_days, n_tokens) * 0.03 + 0.001
    cols   = [f"token_{i}" for i in range(n_tokens)]
    prices = pd.DataFrame(np.cumprod(1 + data, axis=0) * 100, index=dates, columns=cols)
    mcap   = pd.DataFrame(np.random.rand(n_days, n_tokens) * 1e9, index=dates, columns=cols)
    return prices, mcap


def test_default_instantiation():
    s = MomentumStrategy()
    assert s.params["lookback_weeks"] == 3


def test_param_override():
    s = MomentumStrategy({"lookback_weeks": 5})
    assert s.params["lookback_weeks"] == 5


def test_invalid_params():
    with pytest.raises(ValueError):
        MomentumStrategy({"long_pct": 1.5})
    with pytest.raises(ValueError):
        MomentumStrategy({"signal_type": "bad_signal"})


def test_generate_signals_shape():
    prices, mcap = make_fake_data()
    s = MomentumStrategy({
        "universe_size":  20,
        "backtest_start": "2021-01-01",
        "train_end":      "2021-03-01",
    })
    weights = s.generate_signals(prices=prices, mcap=mcap)
    assert isinstance(weights, pd.DataFrame)
    assert weights.index.name == "rebal_date"


def test_weights_sum_to_zero():
    """Long-short book: sum of weights should be near zero each period."""
    prices, mcap = make_fake_data()
    s = MomentumStrategy({
        "universe_size":  20,
        "backtest_start": "2021-01-01",
        "train_end":      "2021-03-01",
    })
    weights = s.generate_signals(prices=prices, mcap=mcap)
    for _, row in weights.iterrows():
        assert abs(row.sum()) < 1e-6, "Weights should sum to ~0 (long-short)"


def test_metadata():
    s = MomentumStrategy()
    meta = s.get_metadata()
    assert meta["name"] == "volatility_adjusted_momentum"
    assert "params" in meta