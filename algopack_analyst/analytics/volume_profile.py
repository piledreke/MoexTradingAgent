"""Volume Profile analysis — distribution of volume across price levels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_profile(
    df: pd.DataFrame, bins: int = 50, value_area_pct: float = 0.70
) -> dict[str, object]:
    """Compute volume profile for given OHLCV df.

    Args:
        df: DataFrame with h, l, c, v.
        bins: Number of price bins.
        value_area_pct: Fraction of volume defining Value Area (default 70%).

    Returns:
        {
          poc: float,                      # Point of Control — top volume node
          vah: float,                      # Value Area High
          val: float,                      # Value Area Low
          hvn: [float, ...],               # High Volume Nodes
          lvn: [float, ...],               # Low Volume Nodes
          profile: [(price, volume), ...], # full distribution
          location: 'inside_va'|'above_va'|'below_va',
        }
    """
    if df.empty or "v" not in df.columns:
        return {}
    sub = df.copy()
    sub["tp"] = (sub["h"] + sub["l"] + sub["c"]) / 3
    prices = sub["tp"].values
    vols = sub["v"].fillna(0).values

    if vols.sum() == 0:
        return {}

    p_lo, p_hi = float(prices.min()), float(prices.max())
    if p_hi <= p_lo:
        return {"poc": p_lo, "vah": p_lo, "val": p_lo,
                "hvn": [p_lo], "lvn": [], "profile": [(p_lo, float(vols.sum()))],
                "location": "inside_va"}

    edges = np.linspace(p_lo, p_hi, bins + 1)
    profile, _ = np.histogram(prices, bins=edges, weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2

    poc_idx = int(np.argmax(profile))
    poc = float(centers[poc_idx])

    # Value Area: expand from POC until cumulative volume >= 70%
    target = profile.sum() * value_area_pct
    lo, hi = poc_idx, poc_idx
    acc = profile[poc_idx]
    while acc < target and (lo > 0 or hi < len(profile) - 1):
        left = profile[lo - 1] if lo > 0 else -1
        right = profile[hi + 1] if hi < len(profile) - 1 else -1
        if left >= right and lo > 0:
            lo -= 1
            acc += profile[lo]
        elif hi < len(profile) - 1:
            hi += 1
            acc += profile[hi]
        else:
            break
    val = float(centers[lo])
    vah = float(centers[hi])

    # HVN / LVN — top 20% and bottom 20% by volume
    threshold_hi = np.percentile(profile, 80)
    threshold_lo = np.percentile(profile, 20)
    hvn = [float(centers[i]) for i, v in enumerate(profile) if v >= threshold_hi]
    lvn = [float(centers[i]) for i, v in enumerate(profile) if v <= threshold_lo and v > 0]

    last = float(sub["c"].iloc[-1])
    location = "above_va" if last > vah else ("below_va" if last < val else "inside_va")

    return {
        "poc": poc,
        "vah": vah, "val": val,
        "hvn": hvn[:10], "lvn": lvn[:10],
        "profile": [(float(c), float(v)) for c, v in zip(centers, profile)],
        "location": location,
        "last_price": last,
        "dist_to_poc_pct": (last - poc) / poc * 100 if poc else 0,
    }


def nearest_node(price: float, nodes: list[float], side: str = "above") -> float | None:
    """Find nearest volume node above or below price."""
    if not nodes:
        return None
    if side == "above":
        cand = [n for n in nodes if n > price]
        return min(cand) if cand else None
    cand = [n for n in nodes if n < price]
    return max(cand) if cand else None