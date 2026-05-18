"""Pure functions computing technical indicators.

All functions are vector-ready (pandas Series) and handle short / missing
input gracefully – they return ``NaN`` instead of raising.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.astype(float).ewm(span=span, adjust=False, min_periods=1).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.astype(float).rolling(window=window, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if series is None or len(series) < 2:
        return pd.Series(dtype=float)
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    # Degenerate cases: no down moves -> RSI = 100, no up moves -> RSI = 0.
    rsi_val = rsi_val.where(~((avg_down == 0) & (avg_up > 0)), 100.0)
    rsi_val = rsi_val.where(~((avg_up == 0) & (avg_down > 0)), 0.0)
    return rsi_val.fillna(50.0)


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    if close is None or len(close) < 2:
        return pd.Series(dtype=float)
    log_ret = np.log(close.astype(float)).diff()
    return log_ret.rolling(window=window, min_periods=2).std()


def atr_like(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    if high is None or low is None or close is None or len(close) < 2:
        return pd.Series(dtype=float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = series.astype(float)
    mean = s.rolling(window=window, min_periods=5).mean()
    std = s.rolling(window=window, min_periods=5).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z


def bollinger_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    return rolling_zscore(close, window=window)


def safe_last(series: pd.Series) -> Optional[float]:
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def pct_distance(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100.0
