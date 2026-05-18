"""Order flow analytics from Super Candles tradestats.

Cumulative Delta = sum(buy_vol - sell_vol). Один из самых сильных
опережающих индикаторов: если цена движется без delta — это слабое движение.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_delta(super_candles: pd.DataFrame) -> pd.Series:
    """Cumulative buy-sell volume difference."""
    if super_candles.empty:
        return pd.Series(dtype=float)
    df = super_candles.sort_values("ts").copy()
    delta = df["buy_vol"].fillna(0) - df["sell_vol"].fillna(0)
    return delta.cumsum()


def delta_divergence(
    super_candles: pd.DataFrame, lookback: int = 20
) -> dict[str, object]:
    """Detect price/delta divergence.

    Bullish: price down, delta up → buyers accumulating on weakness.
    Bearish: price up, delta down → distribution into strength.
    """
    if super_candles.empty or len(super_candles) < lookback:
        return {"type": None, "strength": 0.0}
    df = super_candles.sort_values("ts").tail(lookback).copy()
    cum_d = (df["buy_vol"].fillna(0) - df["sell_vol"].fillna(0)).cumsum()
    price = df["pr_close"].astype(float)

    price_change = price.iloc[-1] - price.iloc[0]
    delta_change = cum_d.iloc[-1] - cum_d.iloc[0]

    if price.iloc[0] == 0:
        return {"type": None, "strength": 0.0}

    p_pct = price_change / price.iloc[0]

    # Bullish divergence
    if p_pct < -0.002 and delta_change > 0:
        return {"type": "bullish_divergence",
                "strength": float(min(1.0, abs(delta_change) / max(1, df["vol"].sum())))}
    # Bearish divergence
    if p_pct > 0.002 and delta_change < 0:
        return {"type": "bearish_divergence",
                "strength": float(min(1.0, abs(delta_change) / max(1, df["vol"].sum())))}
    return {"type": None, "strength": 0.0}


def absorption(super_candles: pd.DataFrame, threshold_ratio: float = 2.0) -> dict[str, object]:
    """Detect absorption — high volume with minimal price movement.

    Returns:
      type: 'absorption_buy'|'absorption_sell'|None
      Если за период объём аномально высокий (>2x avg), но цена почти не двинулась,
      и доминирует buy_vol → крупный покупатель поглощает offers (бычий сигнал).
    """
    if super_candles.empty or len(super_candles) < 10:
        return {"type": None, "strength": 0.0}
    df = super_candles.sort_values("ts").copy()
    last = df.iloc[-1]

    avg_vol = df["vol"].tail(20).mean()
    if avg_vol == 0:
        return {"type": None, "strength": 0.0}
    vol_ratio = last["vol"] / avg_vol

    price_range = (last["pr_high"] - last["pr_low"]) / last["pr_open"] if last["pr_open"] else 0
    avg_range = ((df["pr_high"] - df["pr_low"]) / df["pr_open"].replace(0, np.nan)).tail(20).mean()
    range_ratio = price_range / avg_range if avg_range else 1.0

    if vol_ratio >= threshold_ratio and range_ratio < 0.5:
        buy = last.get("buy_vol", 0) or 0
        sell = last.get("sell_vol", 0) or 0
        if buy > sell * 1.3:
            return {"type": "absorption_buy",
                    "strength": float(min(1.0, vol_ratio / threshold_ratio - 1.0))}
        if sell > buy * 1.3:
            return {"type": "absorption_sell",
                    "strength": float(min(1.0, vol_ratio / threshold_ratio - 1.0))}
    return {"type": None, "strength": 0.0}


def iceberg_score(orderstats: pd.DataFrame) -> dict[str, object]:
    """Score iceberg (hidden order) activity.

    Айсберг: высокий cancel ratio + повторное появление объёма на тех же ценах.
    """
    if orderstats.empty or len(orderstats) < 5:
        return {"side": None, "score": 0.0}
    df = orderstats.sort_values("ts").tail(10)
    put_b = df["put_orders_b"].fillna(0).sum()
    put_s = df["put_orders_s"].fillna(0).sum()
    can_b = df["cancel_orders_b"].fillna(0).sum()
    can_s = df["cancel_orders_s"].fillna(0).sum()

    ratio_b = can_b / put_b if put_b > 0 else 0
    ratio_s = can_s / put_s if put_s > 0 else 0

    # Iceberg buy: many puts on bid, but volume sustained → hidden buying
    if ratio_b > 0.6 and put_b > put_s:
        return {"side": "buy", "score": float(min(1.0, ratio_b))}
    if ratio_s > 0.6 and put_s > put_b:
        return {"side": "sell", "score": float(min(1.0, ratio_s))}
    return {"side": None, "score": 0.0}


def buy_pressure_score(super_candles: pd.DataFrame, n: int = 6) -> dict[str, float]:
    """Composite buy pressure score over last n bars.

    Combines:
      - buy_vol / total_vol ratio
      - VWAP buy vs VWAP sell premium
      - Delta acceleration
    """
    if super_candles.empty:
        return {"score": 0.5, "buy_dominance": 0.5,
                "vwap_premium": 0.0, "delta_accel": 0.0}
    df = super_candles.sort_values("ts").tail(n).copy()

    buy_v = df["buy_vol"].fillna(0).sum()
    sell_v = df["sell_vol"].fillna(0).sum()
    total = buy_v + sell_v
    dom = buy_v / total if total else 0.5

    vwap_b = df["pr_vwap_b"].fillna(0).mean()
    vwap_s = df["pr_vwap_s"].fillna(0).mean()
    premium = (vwap_b - vwap_s) / vwap_s if vwap_s else 0

    # Delta acceleration: latest delta vs avg
    deltas = (df["buy_vol"].fillna(0) - df["sell_vol"].fillna(0)).values
    if len(deltas) >= 3:
        recent_d = deltas[-2:].mean()
        baseline_d = deltas[:-2].mean() if len(deltas) > 2 else 0
        accel = (recent_d - baseline_d) / (abs(baseline_d) + 1e-6)
    else:
        accel = 0.0

    # Normalize composite
    score = 0.5 + (dom - 0.5) * 0.6 + np.clip(premium * 50, -0.2, 0.2) + np.clip(accel * 0.05, -0.1, 0.1)
    return {
        "score": float(np.clip(score, 0, 1)),
        "buy_dominance": float(dom),
        "vwap_premium": float(premium),
        "delta_accel": float(accel),
    }