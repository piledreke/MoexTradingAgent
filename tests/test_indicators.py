"""Tests for indicator helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.indicators import (
    atr_like,
    ema,
    pct_distance,
    realized_volatility,
    rolling_zscore,
    rsi,
    safe_last,
    sma,
)


def test_ema_matches_known_values() -> None:
    s = pd.Series([10, 11, 12, 13, 14, 15], dtype=float)
    e = ema(s, span=3)
    assert len(e) == len(s)
    assert abs(safe_last(e) - 14.03125) < 1e-3


def test_sma_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert abs(safe_last(sma(s, window=2)) - 3.5) < 1e-9


def test_rsi_bounds() -> None:
    s = pd.Series(np.linspace(100, 200, 50))  # monotonically increasing
    out = rsi(s, period=14)
    assert safe_last(out) > 70


def test_rsi_downtrend() -> None:
    s = pd.Series(np.linspace(200, 100, 50))
    out = rsi(s, period=14)
    assert safe_last(out) < 30


def test_atr_like_positive() -> None:
    n = 30
    high = pd.Series(np.linspace(100, 130, n))
    low = high - 1.0
    close = high - 0.5
    out = atr_like(high, low, close, window=14)
    last = safe_last(out)
    assert last is not None and last > 0


def test_rolling_zscore_zero_when_constant() -> None:
    s = pd.Series([5.0] * 30)
    out = rolling_zscore(s, window=10)
    # constant series => std=0 -> NaN; last value should be NaN -> safe_last None
    assert safe_last(out) is None


def test_rolling_zscore_positive_when_spike() -> None:
    s = pd.Series([1.0] * 28 + [10.0, 20.0])
    out = rolling_zscore(s, window=20)
    last = safe_last(out)
    assert last is not None and last > 1.5


def test_realized_vol_non_negative() -> None:
    rng = np.random.default_rng(42)
    s = pd.Series(np.cumsum(rng.normal(0, 0.01, 50)) + 100)
    out = realized_volatility(s, window=20)
    last = safe_last(out)
    assert last is not None and last >= 0


def test_pct_distance() -> None:
    assert pct_distance(110, 100) == 10.0
    assert pct_distance(95, 100) == -5.0
    assert pct_distance(None, 100) is None
    assert pct_distance(100, 0) is None
