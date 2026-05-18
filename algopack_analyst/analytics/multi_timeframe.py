"""Multi-timeframe analysis: согласованность сигналов на 1м/5м/15м/1ч/1д.

Главная идея: сигнал валиден только когда подтверждён минимум на 2 таймфреймах.
Это резко снижает FP-rate (ложные сигналы на шуме мелких ТФ).
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from analytics.technical import detect_trend, ema, rsi

Direction = Literal["bullish", "bearish", "neutral"]


def trend_on_tf(df: pd.DataFrame) -> dict[str, object]:
    """Get trend label on a single timeframe."""
    if df is None or df.empty or len(df) < 30:
        return {"direction": "neutral", "strength": 0.0, "adx": 0}
    d = detect_trend(df)
    mapping = {"uptrend": "bullish", "downtrend": "bearish",
               "sideways": "neutral", "mixed": "neutral", "unknown": "neutral"}
    return {
        "direction": mapping.get(d["trend"], "neutral"),
        "strength": d["strength"],
        "adx": d["adx"],
    }


def mtf_confluence(ohlcv_by_tf: dict[int, pd.DataFrame]) -> dict[str, object]:
    """Aggregate trend signals across timeframes with weights.

    Веса (важнее = больше):
      1m  : 0.05   (шум)
      5m  : 0.15   (краткосрочка)
      15m : 0.20   (intraday-тренд)
      60m : 0.30   (свинг)
      1d  : 0.30   (макро)

    Returns:
      {
        'composite_direction': bullish|bearish|neutral,
        'confluence_score': 0..1  — насколько ТФ согласованы,
        'tf_trends': {1: {...}, 5: {...}, ...},
        'bullish_count': int, 'bearish_count': int,
      }
    """
    weights = {1: 0.05, 5: 0.15, 15: 0.20, 60: 0.30, 24 * 60: 0.30}
    tf_trends: dict[int, dict] = {}
    bull_w = 0.0
    bear_w = 0.0
    total_w = 0.0

    for tf, w in weights.items():
        df = ohlcv_by_tf.get(tf)
        if df is None or df.empty:
            continue
        sorted_df = df.sort_values("ts") if "ts" in df.columns else df
        t = trend_on_tf(sorted_df)
        tf_trends[tf] = t
        total_w += w
        # weight by strength as well — strong trends matter more
        eff_w = w * max(0.3, float(t["strength"]))
        if t["direction"] == "bullish":
            bull_w += eff_w
        elif t["direction"] == "bearish":
            bear_w += eff_w

    if total_w == 0:
        return {
            "composite_direction": "neutral", "confluence_score": 0.0,
            "tf_trends": {}, "bullish_count": 0, "bearish_count": 0,
        }

    net = (bull_w - bear_w) / total_w
    composite = "bullish" if net > 0.15 else ("bearish" if net < -0.15 else "neutral")
    confluence = abs(net)

    return {
        "composite_direction": composite,
        "confluence_score": round(confluence, 3),
        "tf_trends": tf_trends,
        "bullish_count": sum(1 for t in tf_trends.values() if t["direction"] == "bullish"),
        "bearish_count": sum(1 for t in tf_trends.values() if t["direction"] == "bearish"),
        "net_weight": round(net, 3),
    }


def mtf_rsi_divergence(ohlcv_by_tf: dict[int, pd.DataFrame]) -> dict[str, object]:
    """Detect price vs RSI divergence on each TF.

    Bullish divergence: price makes lower low, RSI makes higher low → reversal up.
    Bearish divergence: price higher high, RSI lower high → reversal down.
    """
    results: dict[int, str] = {}
    for tf, df in ohlcv_by_tf.items():
        if df is None or len(df) < 40:
            continue
        df_sorted = df.sort_values("ts") if "ts" in df.columns else df
        c = df_sorted["c"].astype(float)
        r = rsi(c, 14)
        # find last two swing lows/highs over last 30 bars
        tail = c.tail(30).reset_index(drop=True)
        rsi_tail = r.tail(30).reset_index(drop=True)
        if len(tail) < 30:
            continue
        # crude: split into halves
        h1, h2 = tail[:15], tail[15:]
        rh1, rh2 = rsi_tail[:15], rsi_tail[15:]
        # bullish div
        if h2.min() < h1.min() and rh2.min() > rh1.min() and rh2.min() < 40:
            results[tf] = "bullish_divergence"
        # bearish div
        elif h2.max() > h1.max() and rh2.max() < rh1.max() and rh2.max() > 60:
            results[tf] = "bearish_divergence"

    has_bull = any(v == "bullish_divergence" for v in results.values())
    has_bear = any(v == "bearish_divergence" for v in results.values())
    return {
        "divergences": results,
        "any_bullish": has_bull,
        "any_bearish": has_bear,
    }


def golden_cross_check(ohlcv_by_tf: dict[int, pd.DataFrame], fast: int = 50, slow: int = 200) -> dict[str, bool]:
    """Detect golden/death cross on 1d timeframe.

    Golden cross — fast SMA пересекает slow вверх (бычий сигнал).
    Death cross — наоборот.
    """
    df = ohlcv_by_tf.get(24 * 60)
    if df is None or len(df) < slow + 5:
        return {"golden_cross_recent": False, "death_cross_recent": False,
                "above_long_term_ma": False}
    sorted_df = df.sort_values("ts") if "ts" in df.columns else df
    c = sorted_df["c"].astype(float)
    f = c.rolling(fast).mean()
    s = c.rolling(slow).mean()
    if len(f) < 5 or pd.isna(f.iloc[-5]) or pd.isna(s.iloc[-5]):
        return {"golden_cross_recent": False, "death_cross_recent": False,
                "above_long_term_ma": False}
    golden = (f.iloc[-1] > s.iloc[-1]) and (f.iloc[-5] <= s.iloc[-5])
    death = (f.iloc[-1] < s.iloc[-1]) and (f.iloc[-5] >= s.iloc[-5])
    above = c.iloc[-1] > s.iloc[-1]
    return {
        "golden_cross_recent": bool(golden),
        "death_cross_recent": bool(death),
        "above_long_term_ma": bool(above),
    }