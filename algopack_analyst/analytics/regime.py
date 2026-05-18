"""Market regime detection — trending / ranging / volatile.

КРИТИЧНО для скоринга: разные стратегии работают в разных режимах.
Trend-following indicators (EMA cross) дают ложные сигналы в боковике.
Mean-reversion indicators (RSI) дают ложные в сильных трендах.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import pytz

from analytics.technical import adx, atr, bollinger, realized_volatility
from config import settings

MSK = pytz.timezone("Europe/Moscow")

VolRegime = Literal["low", "normal", "high", "extreme"]
TrendRegime = Literal["uptrend", "downtrend", "sideways"]
Session = Literal["premarket", "morning", "midday", "afternoon", "close", "evening", "closed"]


@dataclass
class MarketRegime:
    """Composite market regime descriptor for one ticker."""

    ticker: str
    timestamp: datetime
    trend_regime: TrendRegime
    volatility_regime: VolRegime
    volume_regime: Literal["low", "normal", "high"]
    session: Session
    adx: float
    rv_annualized: float
    bb_width: float
    is_breakout: bool

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "trend": self.trend_regime,
            "volatility": self.volatility_regime,
            "volume": self.volume_regime,
            "session": self.session,
            "adx": round(self.adx, 1),
            "rv": round(self.rv_annualized, 3),
            "bb_width": round(self.bb_width, 4),
            "is_breakout": self.is_breakout,
        }


def detect_session(now: datetime | None = None) -> Session:
    """Determine trading session phase."""
    now = (now or datetime.now(MSK)).astimezone(MSK)
    h, m = now.hour, now.minute
    minutes = h * 60 + m
    if now.weekday() >= 5:
        return "closed"
    if minutes < 9 * 60 + 50:
        return "premarket"
    if minutes < 11 * 60:
        return "morning"   # high vol
    if minutes < 14 * 60:
        return "midday"    # low vol
    if minutes < 17 * 60:
        return "afternoon"
    if minutes < 18 * 60 + 45:
        return "close"     # high vol
    if minutes < 23 * 60 + 50:
        return "evening"   # extended
    return "closed"


def detect_volatility_regime(df: pd.DataFrame) -> tuple[VolRegime, float]:
    """Classify volatility based on annualized realized vol."""
    if df.empty or len(df) < 30:
        return ("normal", 0.0)
    c = df.sort_values("ts")["c"].astype(float) if "ts" in df.columns else df["c"].astype(float)
    rv = float(realized_volatility(c, 30).iloc[-1]) if len(c) >= 30 else 0
    rv_pct = rv * 100

    if rv_pct < settings.LOW_VOLATILITY_THRESHOLD if hasattr(settings, 'LOW_VOLATILITY_THRESHOLD') else 12:
        return ("low", rv)
    if rv_pct < (settings.NORMAL_VOLATILITY_THRESHOLD if hasattr(settings, 'NORMAL_VOLATILITY_THRESHOLD') else 20):
        return ("normal", rv)
    if rv_pct < (settings.HIGH_VOLATILITY_THRESHOLD if hasattr(settings, 'HIGH_VOLATILITY_THRESHOLD') else 35):
        return ("high", rv)
    return ("extreme", rv)


def detect_trend_regime(df: pd.DataFrame) -> tuple[TrendRegime, float]:
    """Classify trend via ADX + EMA slope."""
    if df.empty or len(df) < 50:
        return ("sideways", 0.0)
    sorted_df = df.sort_values("ts") if "ts" in df.columns else df
    adx_d = adx(sorted_df, 14)
    a = float(adx_d["adx"].iloc[-1])
    plus = float(adx_d["plus_di"].iloc[-1])
    minus = float(adx_d["minus_di"].iloc[-1])

    if a < 20:
        return ("sideways", a)
    if plus > minus:
        return ("uptrend", a)
    return ("downtrend", a)


def detect_volume_regime(df: pd.DataFrame) -> Literal["low", "normal", "high"]:
    """Classify volume relative to recent average."""
    if df.empty or len(df) < 30:
        return "normal"
    sorted_df = df.sort_values("ts") if "ts" in df.columns else df
    last_v = sorted_df["v"].tail(5).mean()
    avg_v = sorted_df["v"].tail(50).mean()
    if avg_v == 0:
        return "normal"
    ratio = last_v / avg_v
    if ratio < 0.5:
        return "low"
    if ratio > 1.8:
        return "high"
    return "normal"


def detect_regime(ticker: str, ohlcv_5m: pd.DataFrame, ohlcv_1h: pd.DataFrame | None = None) -> MarketRegime:
    """Detect composite market regime for a ticker.

    Use 5m for vol/volume (intraday), 1h for trend (more stable).
    """
    trend_df = ohlcv_1h if ohlcv_1h is not None and not ohlcv_1h.empty else ohlcv_5m
    trend, adx_v = detect_trend_regime(trend_df)
    vol_regime, rv = detect_volatility_regime(ohlcv_5m)
    volume_regime = detect_volume_regime(ohlcv_5m)
    session = detect_session()

    # Breakout flag: bb_width compressed then expanding
    bb_width = 0.0
    is_breakout = False
    if not ohlcv_5m.empty and len(ohlcv_5m) >= 30:
        sorted_5m = ohlcv_5m.sort_values("ts") if "ts" in ohlcv_5m.columns else ohlcv_5m
        bb = bollinger(sorted_5m["c"], 20, 2.0)
        bb_width = float(bb["width"].iloc[-1])
        bb_width_avg = float(bb["width"].tail(20).mean())
        is_breakout = bb_width > bb_width_avg * 1.5

    return MarketRegime(
        ticker=ticker,
        timestamp=datetime.now(MSK),
        trend_regime=trend,
        volatility_regime=vol_regime,
        volume_regime=volume_regime,
        session=session,
        adx=adx_v,
        rv_annualized=rv,
        bb_width=bb_width,
        is_breakout=is_breakout,
    )