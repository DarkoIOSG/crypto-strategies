import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52


def sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return np.nan
    return (returns.mean() * PERIODS_PER_YEAR) / (returns.std() * np.sqrt(PERIODS_PER_YEAR))


def sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0].std() * np.sqrt(PERIODS_PER_YEAR)
    if downside == 0:
        return np.nan
    return (returns.mean() * PERIODS_PER_YEAR) / downside


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def cagr(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    n_years = (returns.index[-1] - returns.index[0]).days / 365.25
    return float(cum.iloc[-1] ** (1 / n_years) - 1)


def full_report(returns: pd.Series, label: str = "") -> dict:
    return {
        "label":      label,
        "total":      float((1 + returns).cumprod().iloc[-1] - 1),
        "cagr":       cagr(returns),
        "volatility": float(returns.std() * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe":     sharpe(returns),
        "sortino":    sortino(returns),
        "max_dd":     max_drawdown(returns),
        "win_rate":   float((returns > 0).mean()),
    }
