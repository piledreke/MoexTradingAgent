"""Mega Alerts aggregation & interpretation."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config import ALERT_IMPACT


def recent_alerts(alerts_df: pd.DataFrame, lookback_min: int = 60) -> list[dict]:
    if alerts_df.empty:
        return []
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=lookback_min)
    df = alerts_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df[df["ts"].notna() & (df["ts"] >= cutoff)]
    return df.sort_values("ts", ascending=False).to_dict(orient="records")


def alert_density(alerts_df: pd.DataFrame, window_min: int = 60) -> float:
    """Alerts per minute over the last window."""
    n = len(recent_alerts(alerts_df, window_min))
    return n / max(1, window_min)


def alerts_score_impact(alerts_df: pd.DataFrame, lookback_min: int = 60) -> dict:
    """Compute net bullish/bearish impact from recent alerts."""
    items = recent_alerts(alerts_df, lookback_min)
    impact = 0.0
    bull, bear = 0, 0
    for a in items:
        at = str(a.get("alert_type") or "").lower()
        w = ALERT_IMPACT.get(at, 0.0)
        impact += w
        if w > 0:
            bull += 1
        elif w < 0:
            bear += 1
    # clamp
    impact = max(-1.0, min(1.0, impact))
    return {
        "impact": impact,
        "bullish_count": bull,
        "bearish_count": bear,
        "total": len(items),
    }