"""Orderbook-based analytics."""
from __future__ import annotations

from typing import Any

import pandas as pd


def bid_ask_imbalance(snapshot: dict[str, Any], depth: int = 5) -> float:
    """Compute bid/ask volume imbalance in [-1, +1].

    +1 = всё на bid (buyers dominate), -1 = всё на ask.
    """
    bids = snapshot.get("bids", [])[:depth]
    asks = snapshot.get("asks", [])[:depth]
    bv = sum(q for _, q in bids)
    av = sum(q for _, q in asks)
    if bv + av == 0:
        return 0.0
    return float((bv - av) / (bv + av))


def liquidity_score(snapshot: dict[str, Any], depth: int = 10) -> float:
    """Aggregate liquidity score: total displayed volume on both sides."""
    bids = snapshot.get("bids", [])[:depth]
    asks = snapshot.get("asks", [])[:depth]
    return float(sum(q for _, q in bids) + sum(q for _, q in asks))


def spread_bps(snapshot: dict[str, Any]) -> float | None:
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    if not bids or not asks:
        return None
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if best_bid <= 0:
        return None
    mid = (best_bid + best_ask) / 2
    return float((best_ask - best_bid) / mid * 10_000)


def detect_iceberg_orders(orderstats_df: pd.DataFrame) -> dict[str, Any]:
    """Heuristic detection of iceberg activity from Super Candles orderstats.

    Сигнал: высокий cancel_orders/put_orders ratio при сохранении объёма исполнения.
    """
    if orderstats_df.empty:
        return {"detected": False, "side": None, "confidence": 0.0}

    df = orderstats_df.copy().sort_values("ts").tail(6)
    put_b = df["put_orders_b"].fillna(0).sum()
    put_s = df["put_orders_s"].fillna(0).sum()
    can_b = df["cancel_orders_b"].fillna(0).sum()
    can_s = df["cancel_orders_s"].fillna(0).sum()

    ratio_b = can_b / put_b if put_b > 0 else 0.0
    ratio_s = can_s / put_s if put_s > 0 else 0.0

    if ratio_b > 0.7 and put_b > put_s * 1.5:
        return {"detected": True, "side": "buy", "confidence": min(1.0, ratio_b)}
    if ratio_s > 0.7 and put_s > put_b * 1.5:
        return {"detected": True, "side": "sell", "confidence": min(1.0, ratio_s)}
    return {"detected": False, "side": None, "confidence": 0.0}


def microprice(snapshot: dict[str, Any]) -> float | None:
    """Compute microprice = weighted mid by opposite-side volume."""
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    if not bids or not asks:
        return None
    bp, bv = bids[0]
    ap, av = asks[0]
    total = bv + av
    if total <= 0:
        return None
    return float((bp * av + ap * bv) / total)