"""FUTOI-based sentiment (smart money vs retail)."""
from __future__ import annotations

import pandas as pd

from config import STOCK_TO_FUTURE


def net_position(df: pd.DataFrame, group: str = "YUR") -> float:
    """Net position = long - short for a given client group."""
    if df.empty:
        return 0.0
    sub = df[df["clgroup"].astype(str).str.upper() == group.upper()]
    if sub.empty:
        return 0.0
    latest = sub.sort_values("ts").iloc[-1]
    return float((latest.get("pos_long") or 0) - (latest.get("pos_short") or 0))


def position_change(df: pd.DataFrame, group: str = "YUR", lookback: int = 6) -> float:
    """Change in net position over `lookback` snapshots (≈ 30 min if 5-min snaps)."""
    if df.empty:
        return 0.0
    sub = df[df["clgroup"].astype(str).str.upper() == group.upper()].sort_values("ts")
    if len(sub) < 2:
        return 0.0
    tail = sub.tail(lookback)
    net = (tail["pos_long"].fillna(0) - tail["pos_short"].fillna(0)).astype(float)
    return float(net.iloc[-1] - net.iloc[0])


def jur_sentiment(df: pd.DataFrame) -> dict:
    """Compute legal-entity ("smart money") sentiment block."""
    if df.empty:
        return {
            "sentiment": "neutral",
            "net_yur": 0.0, "net_fiz": 0.0,
            "delta_yur": 0.0, "delta_fiz": 0.0,
            "divergence": False,
        }

    net_y = net_position(df, "YUR")
    net_f = net_position(df, "FIZ")
    d_y = position_change(df, "YUR")
    d_f = position_change(df, "FIZ")

    label = "neutral"
    if d_y > 0 and net_y >= 0:
        label = "bullish"
    elif d_y < 0 and net_y <= 0:
        label = "bearish"

    # divergence: smart vs retail moving opposite directions
    divergence = (d_y * d_f) < 0 and (abs(d_y) > 1e-6 and abs(d_f) > 1e-6)

    return {
        "sentiment": label,
        "net_yur": net_y, "net_fiz": net_f,
        "delta_yur": d_y, "delta_fiz": d_f,
        "divergence": bool(divergence),
    }


def map_stock_to_future(ticker: str) -> str | None:
    return STOCK_TO_FUTURE.get(ticker.upper())