"""Multi-level concentration risk assessment based on HI2 metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

RiskLevel = Literal["low", "medium", "high", "critical"]
Action = Literal["allow", "downsize", "hold", "avoid"]


@dataclass
class ConcentrationAssessment:
    risk_level: RiskLevel
    action: Action
    hhi_volume: float
    hhi_buy: float
    hhi_sell: float
    bias: str  # 'accumulation' | 'distribution' | 'neutral'
    passive_active_ratio: float | None
    reasons: list[str]
    max_position_pct_cap: float

    def as_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "action": self.action,
            "hhi_volume": self.hhi_volume,
            "hhi_buy": self.hhi_buy,
            "hhi_sell": self.hhi_sell,
            "bias": self.bias,
            "passive_active_ratio": self.passive_active_ratio,
            "reasons": self.reasons,
            "max_position_pct_cap": self.max_position_pct_cap,
        }


def assess_concentration(
    hi2_row: pd.Series | dict | None,
    *,
    intraday_volume: float = 0.0,
    vol_spike_ratio: float = 1.0,
    low_volume_threshold: float = 50_000_000,
) -> ConcentrationAssessment:
    """Multi-level concentration analysis.

    Args:
      hi2_row: Latest HI2 metrics.
      intraday_volume: Today's traded value (rubles).
      vol_spike_ratio: Today's volume / 20-day avg.
      low_volume_threshold: Below this — microcap risk.

    Returns:
      ConcentrationAssessment with overridable recommendation.
    """
    if hi2_row is None or (hasattr(hi2_row, "empty") and hi2_row.empty):
        return ConcentrationAssessment(
            risk_level="medium", action="allow",
            hhi_volume=0, hhi_buy=0, hhi_sell=0,
            bias="neutral", passive_active_ratio=None,
            reasons=["no_hi2_data"],
            max_position_pct_cap=10.0,
        )

    def g(k: str) -> float:
        if isinstance(hi2_row, dict):
            v = hi2_row.get(k, 0)
        else:
            v = hi2_row.get(k, 0) if hasattr(hi2_row, "get") else 0
        try:
            return float(v) if pd.notna(v) else 0.0
        except Exception:
            return 0.0

    hhi_vol = g("hhi_volume")
    hhi_buy = g("hhi_buy")
    hhi_sell = g("hhi_sell")
    passive = g("hhi_passive")
    active = g("hhi_active")
    passive_buy = g("hhi_passive_buy")
    active_buy = g("hhi_active_buy")
    netflow_buy = g("hhi_netflow_buy")
    netflow_sell = g("hhi_netflow_sell")

    pa_ratio = passive / active if active > 0 else None

    reasons: list[str] = []

    # Bias
    if hhi_buy > hhi_sell * 1.3 and hhi_buy > 1800:
        bias = "accumulation"
    elif hhi_sell > hhi_buy * 1.3 and hhi_sell > 1800:
        bias = "distribution"
    else:
        bias = "neutral"

    # ─── Risk classification ───
    risk: RiskLevel = "low"
    action: Action = "allow"
    cap = 15.0

    if hhi_vol < 1500:
        risk = "low"
        cap = 15.0
    elif hhi_vol < 2500:
        risk = "medium"
        cap = 10.0
    elif hhi_vol < 3500:
        risk = "high"
        cap = 5.0
        reasons.append(f"HHI volume high ({hhi_vol:.0f})")
    else:
        risk = "critical"
        action = "avoid"
        cap = 0.0
        reasons.append(f"HHI volume critical ({hhi_vol:.0f}) — likely whale dominance")

    # Override rules ──────────────────────────────────────

    # Rule 1: critical buy-side concentration
    if hhi_buy > 3500:
        risk = "critical"
        action = "avoid"
        cap = 0.0
        reasons.append(f"HHI buy critical ({hhi_buy:.0f}) — single buyer dominates")

    # Rule 2: critical sell-side (distribution)
    if hhi_sell > 3500 and bias == "distribution":
        risk = "critical"
        action = "avoid"
        cap = 0.0
        reasons.append(f"HHI sell critical with distribution bias")

    # Rule 3: high concentration + volume spike → suspicious
    if hhi_vol > 2500 and vol_spike_ratio > 3.0:
        if risk != "critical":
            risk = "critical"
            action = "avoid"
            cap = 0.0
        reasons.append(f"High HHI + volume spike {vol_spike_ratio:.1f}x — manipulation suspect")

    # Rule 4: low intraday volume + high concentration → microcap trap
    if intraday_volume > 0 and intraday_volume < low_volume_threshold and hhi_vol > 2000:
        if risk == "low" or risk == "medium":
            risk = "high"
            action = "hold"
            cap = min(cap, 3.0)
        reasons.append(f"Low liquidity ({intraday_volume/1e6:.1f}M) + high HHI — microcap trap")

    # Rule 5: passive >> active → fake liquidity
    if pa_ratio is not None and pa_ratio > 3.0:
        reasons.append(f"Passive/active ratio {pa_ratio:.1f}x — fake liquidity")
        cap = min(cap, 5.0)

    # Rule 6: positive override — bias accumulation with low concentration
    if bias == "accumulation" and risk == "low":
        reasons.append("Healthy accumulation pattern")

    # Decide final action
    if risk == "high" and action == "allow":
        action = "downsize"
    if risk == "medium" and action == "allow":
        action = "allow"

    return ConcentrationAssessment(
        risk_level=risk,
        action=action,
        hhi_volume=hhi_vol,
        hhi_buy=hhi_buy,
        hhi_sell=hhi_sell,
        bias=bias,
        passive_active_ratio=round(pa_ratio, 2) if pa_ratio else None,
        reasons=reasons,
        max_position_pct_cap=cap,
    )


# Back-compat shim
def interpret_hi2(hi2_row) -> dict:
    """Legacy API."""
    a = assess_concentration(hi2_row)
    return {
        "level": {"low": "low", "medium": "medium",
                  "high": "high", "critical": "high"}[a.risk_level],
        "hhi_volume": a.hhi_volume,
        "hhi_buy": a.hhi_buy,
        "hhi_sell": a.hhi_sell,
        "bias": a.bias,
        "risk": "; ".join(a.reasons),
    }