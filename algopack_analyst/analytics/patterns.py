"""Candlestick & chart pattern recognition.

Без TA-Lib (тяжелая C-зависимость). Реализованы топ-15 самых надёжных паттернов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bodies(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Helper: body, upper shadow, lower shadow, range."""
    body = (df["c"] - df["o"]).abs()
    up_shadow = df["h"] - df[["o", "c"]].max(axis=1)
    lo_shadow = df[["o", "c"]].min(axis=1) - df["l"]
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    return body, up_shadow, lo_shadow, rng


def candle_patterns(df: pd.DataFrame) -> dict[str, bool]:
    """Detect single/double/triple candlestick patterns on the latest bars."""
    if df.empty or len(df) < 3:
        return {}
    last3 = df.tail(3).copy().reset_index(drop=True)
    body, up_s, lo_s, rng = _bodies(last3)

    out: dict[str, bool] = {}

    # ─ Single candle ─
    last = last3.iloc[-1]
    last_body = body.iloc[-1]
    last_up = up_s.iloc[-1]
    last_lo = lo_s.iloc[-1]
    last_rng = rng.iloc[-1] or 1

    # Doji: small body
    out["doji"] = bool(last_body / last_rng < 0.1)

    # Hammer: small body at top, long lower shadow, little upper shadow
    out["hammer"] = bool(
        last_lo > 2 * last_body and last_up < 0.3 * last_body and last_body / last_rng < 0.4
    )
    # Shooting star: inverted hammer (bearish)
    out["shooting_star"] = bool(
        last_up > 2 * last_body and last_lo < 0.3 * last_body and last["c"] < last["o"]
    )

    # Marubozu: full body, no shadows
    out["marubozu_bull"] = bool(
        last_body / last_rng > 0.9 and last["c"] > last["o"]
    )
    out["marubozu_bear"] = bool(
        last_body / last_rng > 0.9 and last["c"] < last["o"]
    )

    # ─ Two-candle ─
    if len(last3) >= 2:
        prev = last3.iloc[-2]
        # Bullish engulfing: prev bearish, current bullish, engulfs prev body
        out["bullish_engulfing"] = bool(
            prev["c"] < prev["o"]
            and last["c"] > last["o"]
            and last["o"] <= prev["c"]
            and last["c"] >= prev["o"]
        )
        out["bearish_engulfing"] = bool(
            prev["c"] > prev["o"]
            and last["c"] < last["o"]
            and last["o"] >= prev["c"]
            and last["c"] <= prev["o"]
        )
        # Piercing line
        mid_prev = (prev["o"] + prev["c"]) / 2
        out["piercing"] = bool(
            prev["c"] < prev["o"]
            and last["o"] < prev["l"]
            and last["c"] > mid_prev
            and last["c"] < prev["o"]
        )

    # ─ Three-candle ─
    if len(last3) >= 3:
        c1, c2, c3 = last3.iloc[0], last3.iloc[1], last3.iloc[2]
        # Morning star
        out["morning_star"] = bool(
            c1["c"] < c1["o"]
            and abs(c2["c"] - c2["o"]) / (rng.iloc[1] or 1) < 0.3  # small body
            and c3["c"] > c3["o"]
            and c3["c"] > (c1["o"] + c1["c"]) / 2
        )
        # Evening star
        out["evening_star"] = bool(
            c1["c"] > c1["o"]
            and abs(c2["c"] - c2["o"]) / (rng.iloc[1] or 1) < 0.3
            and c3["c"] < c3["o"]
            and c3["c"] < (c1["o"] + c1["c"]) / 2
        )
        # Three white soldiers
        out["three_white_soldiers"] = bool(
            all(c["c"] > c["o"] for c in [c1, c2, c3])
            and c2["c"] > c1["c"] and c3["c"] > c2["c"]
        )
        out["three_black_crows"] = bool(
            all(c["c"] < c["o"] for c in [c1, c2, c3])
            and c2["c"] < c1["c"] and c3["c"] < c2["c"]
        )

    return out


def bullish_patterns_score(patterns: dict[str, bool]) -> float:
    """Aggregate bullish patterns to [0..1] confidence."""
    weights = {
        "hammer": 0.4, "bullish_engulfing": 0.6, "piercing": 0.4,
        "morning_star": 0.7, "three_white_soldiers": 0.6,
        "marubozu_bull": 0.3,
    }
    s = sum(w for p, w in weights.items() if patterns.get(p))
    return float(min(1.0, s))


def bearish_patterns_score(patterns: dict[str, bool]) -> float:
    weights = {
        "shooting_star": 0.4, "bearish_engulfing": 0.6,
        "evening_star": 0.7, "three_black_crows": 0.6,
        "marubozu_bear": 0.3,
    }
    s = sum(w for p, w in weights.items() if patterns.get(p))
    return float(min(1.0, s))


# ─── Chart patterns ─────────────────────────────────────

def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> dict[str, object]:
    """Detect price breakout from consolidation range."""
    if df.empty or len(df) < lookback + 1:
        return {"breakout": False, "direction": None}
    sub = df.sort_values("ts").tail(lookback + 1)
    prev = sub.iloc[:-1]
    last = sub.iloc[-1]
    hi = prev["h"].max()
    lo = prev["l"].min()
    range_pct = (hi - lo) / lo if lo else 0

    # need tight range for valid breakout
    if range_pct > 0.05:
        return {"breakout": False, "direction": None}

    if last["c"] > hi:
        return {"breakout": True, "direction": "up",
                "magnitude_pct": float((last["c"] - hi) / hi * 100)}
    if last["c"] < lo:
        return {"breakout": True, "direction": "down",
                "magnitude_pct": float((lo - last["c"]) / lo * 100)}
    return {"breakout": False, "direction": None}


def detect_double_bottom(df: pd.DataFrame, lookback: int = 60, tol: float = 0.015) -> bool:
    """Crude double-bottom: two similar lows with a peak between."""
    if df.empty or len(df) < lookback:
        return False
    sub = df.sort_values("ts").tail(lookback).reset_index(drop=True)
    lows = sub["l"].values
    # find local minima
    min_idx = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            min_idx.append(i)
    if len(min_idx) < 2:
        return False
    i1, i2 = min_idx[-2], min_idx[-1]
    if i2 - i1 < 5:
        return False
    if abs(lows[i1] - lows[i2]) / lows[i1] > tol:
        return False
    peak = sub["h"][i1:i2].max()
    if peak < lows[i1] * (1 + tol * 2):
        return False
    return sub["c"].iloc[-1] > peak * 0.98


def detect_volume_climax(df: pd.DataFrame, n: int = 50) -> dict[str, object]:
    """Detect volume climax (capitulation or buying frenzy)."""
    if df.empty or len(df) < n:
        return {"climax": False}
    sub = df.sort_values("ts").tail(n)
    last = sub.iloc[-1]
    avg_v = sub["v"].iloc[:-1].mean()
    if avg_v == 0:
        return {"climax": False}
    ratio = last["v"] / avg_v
    if ratio < 3:
        return {"climax": False}
    return {
        "climax": True,
        "ratio": float(ratio),
        "direction": "buying" if last["c"] > last["o"] else "selling",
    }