"""Adaptive weight computation based on market regime.

Идея: разные индикаторы по-разному надёжны в разных условиях.
В экстремальной волатильности — тренд-индикаторы шумят, orderbook надёжнее.
В боковике — momentum-индикаторы дают ложные сигналы.
"""
from __future__ import annotations

from analytics.regime import MarketRegime
from config import settings


DEFAULT_WEIGHTS = {
    "trend": 0.20,
    "momentum": 0.12,
    "volume_profile": 0.13,
    "order_flow": 0.13,
    "orderbook": 0.10,
    "patterns": 0.07,
    "alerts": 0.10,
    "hi2": 0.05,
    "futoi": 0.07,
    "mtf_confluence": 0.03,
}


def adaptive_weights(regime: MarketRegime) -> dict[str, float]:
    """Return weights adjusted to current regime, summing to 1.0."""
    w = dict(DEFAULT_WEIGHTS)

    # ─── Volatility adjustments ─────────────────────────
    if regime.volatility_regime == "extreme":
        w["trend"] *= 0.4
        w["momentum"] *= 0.4
        w["orderbook"] *= 1.6
        w["order_flow"] *= 1.6
        w["alerts"] *= 1.4
        w["patterns"] *= 0.6
    elif regime.volatility_regime == "high":
        w["trend"] *= 0.7
        w["momentum"] *= 0.7
        w["orderbook"] *= 1.3
        w["order_flow"] *= 1.3
        w["alerts"] *= 1.2
    elif regime.volatility_regime == "low":
        w["trend"] *= 1.3
        w["momentum"] *= 1.2
        w["mtf_confluence"] *= 1.5

    # ─── Trend regime adjustments ───────────────────────
    if regime.trend_regime == "sideways":
        # in range — trend indicators give false signals
        w["trend"] *= 0.5
        w["momentum"] *= 1.3
        w["volume_profile"] *= 1.4
        w["patterns"] *= 1.3
    elif regime.trend_regime == "uptrend":
        w["trend"] *= 1.3
        w["mtf_confluence"] *= 1.4
        w["volume_profile"] *= 1.2
    elif regime.trend_regime == "downtrend":
        # Bot is LONG-only — be conservative in downtrend
        w["alerts"] *= 1.5
        w["orderbook"] *= 1.3
        w["patterns"] *= 1.3  # reversal patterns matter

    # ─── Volume regime ──────────────────────────────────
    if regime.volume_regime == "high":
        w["order_flow"] *= 1.4
        w["volume_profile"] *= 1.3
    elif regime.volume_regime == "low":
        # low volume = unreliable order flow
        w["order_flow"] *= 0.6
        w["orderbook"] *= 0.7
        w["hi2"] *= 1.5  # concentration risk amplifies

    # ─── Session adjustments ────────────────────────────
    if regime.session == "premarket":
        for k in w:
            w[k] *= 0.5  # everything less reliable
        w["alerts"] *= 1.5
    elif regime.session == "morning":
        w["orderbook"] *= 1.3
        w["order_flow"] *= 1.3
        w["trend"] *= 0.8
    elif regime.session == "midday":
        w["mtf_confluence"] *= 1.3
        w["volume_profile"] *= 1.2
        w["order_flow"] *= 0.8
    elif regime.session == "close":
        w["orderbook"] *= 1.4
        w["order_flow"] *= 1.4
    elif regime.session in ("evening", "closed"):
        for k in w:
            w[k] *= 0.4

    # ─── Breakout boost ─────────────────────────────────
    if regime.is_breakout:
        w["order_flow"] *= 1.4
        w["volume_profile"] *= 1.3
        w["patterns"] *= 1.4

    # Normalize to sum=1
    total = sum(w.values())
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {k: v / total for k, v in w.items()}


def skip_scoring(regime: MarketRegime) -> tuple[bool, str | None]:
    """Decide whether to skip scoring entirely (return HOLD)."""
    if regime.session in ("closed", "evening", "premarket"):
        return True, f"session_{regime.session}"
    # Extreme volatility + sideways = chop, very dangerous
    if regime.volatility_regime == "extreme" and regime.trend_regime == "sideways":
        return True, "extreme_chop"
    return False, None