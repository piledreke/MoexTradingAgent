import numpy as np
import pandas as pd
import pytest

from analytics.technical import atr, detect_trend, ema, macd, rsi, sma


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5])
    out = sma(s, 3)
    assert out.iloc[-1] == 4.0


def test_ema_decay():
    s = pd.Series([1.0] * 50 + [10.0] * 50)
    out = ema(s, 10)
    assert out.iloc[49] < out.iloc[60] < out.iloc[99]


def test_rsi_bounds():
    np.random.seed(42)
    s = pd.Series(np.random.randn(200).cumsum() + 100)
    r = rsi(s, 14)
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_keys():
    s = pd.Series(np.linspace(100, 110, 100))
    d = macd(s)
    assert {"macd", "signal", "hist"} <= d.keys()


def test_detect_trend_uptrend():
    df = pd.DataFrame({
        "c": np.linspace(100, 130, 100),
        "h": np.linspace(101, 131, 100),
        "l": np.linspace(99, 129, 100),
        "v": np.ones(100) * 1000,
    })
    res = detect_trend(df)
    assert res["trend"] == "uptrend"


def test_atr_positive():
    df = pd.DataFrame({
        "h": np.linspace(101, 110, 50),
        "l": np.linspace(99, 108, 50),
        "c": np.linspace(100, 109, 50),
    })
    a = atr(df, 14)
    assert (a.dropna() > 0).all()