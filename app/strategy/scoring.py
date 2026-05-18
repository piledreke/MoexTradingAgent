"""Deterministic technical scorer.

The scorer aggregates intuitive components in fixed ranges so the score is
explainable. Each component is a *pure function* of the feature bundle, which
keeps the system testable and reproducible.

Final ``technical_score`` is clamped to ``[0, 100]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.features.feature_builder import FeatureBundle


def _clamp(value: float, lo: float, hi: float) -> float:
    if value != value:  # NaN
        return lo
    return max(lo, min(hi, value))


def _f(features: Dict[str, Any], key: str) -> Optional[float]:
    v = features.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:  # NaN
        return None
    return f


@dataclass
class ScoreBreakdown:
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volume_flow_score: float = 0.0
    orderbook_liquidity_score: float = 0.0
    alerts_score: float = 0.0
    hi2_risk_adjustment: float = 0.0
    risk_penalty: float = 0.0
    technical_score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)


class TechnicalScorer:
    """Compute :class:`ScoreBreakdown` from a :class:`FeatureBundle`."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    def score(self, bundle: FeatureBundle) -> ScoreBreakdown:
        features = bundle.features
        out = ScoreBreakdown()
        out.trend_score = self._trend(features, out)
        out.momentum_score = self._momentum(features, out)
        out.volume_flow_score = self._volume_flow(features, out)
        out.orderbook_liquidity_score = self._orderbook(features, out)
        out.alerts_score = self._alerts(features, out)
        out.hi2_risk_adjustment = self._hi2_adjustment(features, bundle.hi2, out)
        out.risk_penalty = self._risk_penalty(features, out)

        total = (
            out.trend_score
            + out.momentum_score
            + out.volume_flow_score
            + out.orderbook_liquidity_score
            + out.alerts_score
            + out.hi2_risk_adjustment
            - out.risk_penalty
        )
        out.technical_score = _clamp(total, 0.0, 100.0)
        out.confidence = self._confidence(bundle, out)
        return out

    # ------------------------------------------------------------------
    def _trend(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """0..25 - alignment of EMAs and price vs VWAP."""
        score = 0.0
        ema_above = f.get("ema_9_above_21")
        close_above_21 = f.get("close_above_ema21")
        close_above_9 = f.get("close_above_ema9")
        dist_vwap = _f(f, "distance_to_vwap_pct")
        if ema_above is True:
            score += 8
            out.reasons.append("ema9>ema21 (trend up)")
        elif ema_above is False:
            out.negative_factors.append("ema9<ema21 (trend down)")
        if close_above_21 is True:
            score += 6
            out.reasons.append("close above EMA21")
        elif close_above_21 is False:
            out.negative_factors.append("close below EMA21")
        if close_above_9 is True:
            score += 4
        if dist_vwap is not None:
            if 0.0 <= dist_vwap < 0.6:
                score += 5
                out.reasons.append(f"price slightly above VWAP ({dist_vwap:.2f}%)")
            elif -0.3 <= dist_vwap < 0.0:
                score += 2
            elif dist_vwap >= 0.6:
                # Already extended; be cautious.
                score += 1
                out.negative_factors.append(f"price extended {dist_vwap:.2f}% above VWAP")
            else:
                out.negative_factors.append(f"price {dist_vwap:.2f}% below VWAP")
        return _clamp(score, 0.0, 25.0)

    # ------------------------------------------------------------------
    def _momentum(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """0..20 - short-term returns and RSI."""
        score = 0.0
        r1 = _f(f, "return_1_bar")
        r3 = _f(f, "return_3_bar")
        r12 = _f(f, "return_12_bar")
        rsi14 = _f(f, "rsi14")
        if r1 is not None:
            score += _clamp(r1 * 300.0, -4.0, 4.0)
        if r3 is not None:
            score += _clamp(r3 * 200.0, -5.0, 5.0)
        if r12 is not None:
            score += _clamp(r12 * 80.0, -3.0, 3.0)
            if r12 > 0:
                out.reasons.append(f"positive 12-bar return ({r12*100:.2f}%)")
            elif r12 < -0.005:
                out.negative_factors.append(f"negative 12-bar return ({r12*100:.2f}%)")
        if rsi14 is not None:
            if 45 <= rsi14 <= 65:
                score += 5
                out.reasons.append(f"RSI healthy ({rsi14:.1f})")
            elif 65 < rsi14 <= 75:
                score += 3
            elif rsi14 > 75:
                score -= 3
                out.negative_factors.append(f"RSI overbought ({rsi14:.1f})")
            elif rsi14 < 35:
                score -= 2
                out.negative_factors.append(f"RSI weak ({rsi14:.1f})")
        return _clamp(score, 0.0, 20.0)

    # ------------------------------------------------------------------
    def _volume_flow(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """0..20 - buyer flow strength."""
        score = 0.0
        vol_z = _f(f, "volume_zscore")
        val_z = _f(f, "value_zscore")
        buy_ratio = _f(f, "buy_volume_ratio")
        disb = _f(f, "disb_now")
        agg = _f(f, "aggressive_buy_pressure")
        pvbs = _f(f, "pr_vwap_b_minus_s_pct")

        if vol_z is not None:
            score += _clamp(vol_z * 1.2, -3.0, 4.0)
        if val_z is not None and val_z > 1.0:
            score += min(4.0, val_z)
            out.reasons.append(f"value above rolling mean (z={val_z:.2f})")
        if buy_ratio is not None:
            if buy_ratio >= 0.55:
                score += 4
                out.reasons.append(f"buy volume dominates ({buy_ratio*100:.1f}%)")
            elif buy_ratio < 0.45:
                score -= 3
                out.negative_factors.append(f"sell volume dominates ({(1-buy_ratio)*100:.1f}%)")
        if disb is not None:
            if disb > 0.1:
                score += 3
            elif disb < -0.15:
                score -= 4
                out.negative_factors.append(f"order flow imbalance ({disb:.2f})")
        if agg is not None:
            if agg > 0.1:
                score += 3
                out.reasons.append("aggressive buy pressure")
            elif agg < -0.15:
                score -= 3
                out.negative_factors.append("aggressive sell pressure")
        if pvbs is not None and pvbs > 0:
            score += 1
        return _clamp(score, 0.0, 20.0)

    # ------------------------------------------------------------------
    def _orderbook(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """0..15 - liquidity and order-book imbalance."""
        score = 0.0
        imb_bbo = _f(f, "imbalance_bbo")
        imb_val = _f(f, "imbalance_val")
        liq = _f(f, "liquidity_score")
        slip = _f(f, "slippage_1mio_bps")
        spread_bbo = _f(f, "spread_bbo")
        last_price = _f(f, "last_price") or _f(f, "last_close")
        if last_price and spread_bbo is not None:
            spread_bps = spread_bbo / last_price * 10000.0 if last_price > 0 else None
            if spread_bps is not None:
                if spread_bps < 8:
                    score += 4
                    out.reasons.append(f"tight spread ({spread_bps:.1f} bps)")
                elif spread_bps < 20:
                    score += 2
                elif spread_bps > 60:
                    out.negative_factors.append(f"wide spread ({spread_bps:.0f} bps)")
        if liq is not None:
            score += _clamp(liq * 5.0, 0.0, 5.0)
            if liq > 0.5:
                out.reasons.append("strong order book liquidity")
            elif liq < 0.25:
                out.negative_factors.append("thin order book liquidity")
        if imb_bbo is not None:
            if imb_bbo > 0.1:
                score += 3
                out.reasons.append("BBO imbalance favors buyers")
            elif imb_bbo < -0.15:
                score -= 2
                out.negative_factors.append("BBO imbalance favors sellers")
        if imb_val is not None and imb_val > 0.1:
            score += 1
        if slip is not None and slip < 25:
            score += 1
        return _clamp(score, 0.0, 15.0)

    # ------------------------------------------------------------------
    def _alerts(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """-15..15 - MegaAlerts net contribution."""
        pos = int(f.get("alerts_positive_count") or 0)
        neg = int(f.get("alerts_negative_count") or 0)
        score = (pos - neg) * 3.0
        ref = f.get("alerts_reference_summary")
        if isinstance(ref, dict):
            hit_rate = ref.get("hit_rate")
            mean_change = ref.get("mean_change_15m")
            n_obs = ref.get("n_observations") or 0
            if hit_rate is not None and n_obs >= 5:
                if hit_rate >= 0.65:
                    score += 4
                    out.reasons.append(
                        f"reference stats: hit_rate {hit_rate*100:.0f}% over {n_obs} obs"
                    )
                elif hit_rate <= 0.4:
                    score -= 3
                    out.negative_factors.append(
                        f"poor reference stats: hit_rate {hit_rate*100:.0f}%"
                    )
            if mean_change is not None:
                score += _clamp(mean_change * 2.0, -4.0, 4.0)
        if (f.get("alerts_extreme_high_price") or 0) > 0 and (f.get("return_3_bar") or 0) > 0.02:
            # Beware buying tops.
            score -= 3
            out.negative_factors.append("buying near 90d high after a rally")
        if (f.get("alerts_extreme_low_price") or 0) > 0:
            score -= 4
            out.negative_factors.append("price near 90d low (downside risk)")
        return _clamp(score, -15.0, 15.0)

    # ------------------------------------------------------------------
    def _hi2_adjustment(self, f: Dict[str, Any], hi2: Dict[str, float], out: ScoreBreakdown) -> float:
        """-10..5 - market concentration adjustment."""
        if not hi2:
            return 0.0
        adj = 0.0
        agg_sell = _f(f, "hi2_aggressive_sell")
        agg_buy = _f(f, "hi2_aggressive_buy")
        nf_sell = _f(f, "hi2_netflow_sell")
        nf_buy = _f(f, "hi2_netflow_buy")
        vol = _f(f, "hi2_volume")
        if vol is not None and vol > 2500:
            adj -= 2
            out.negative_factors.append(f"HI2 volume concentrated ({vol:.0f})")
        if agg_sell is not None and agg_sell > 2500:
            adj -= 4
            out.negative_factors.append("HI2 aggressive sell concentrated")
        if nf_sell is not None and nf_sell > 2500:
            adj -= 4
            out.negative_factors.append("HI2 netflow sell concentrated")
        if agg_buy is not None and agg_buy > 1500:
            adj += 3
            out.reasons.append("HI2 aggressive buy elevated (momentum)")
        if nf_buy is not None and nf_buy > 1500:
            adj += 2
        return _clamp(adj, -10.0, 5.0)

    # ------------------------------------------------------------------
    def _risk_penalty(self, f: Dict[str, Any], out: ScoreBreakdown) -> float:
        """0..30 - explicit penalties for risk-off conditions."""
        penalty = 0.0
        spoof = _f(f, "spoof_risk_score")
        net_cancel = _f(f, "net_cancel_pressure")
        vol = _f(f, "realized_vol")
        r1 = _f(f, "return_1_bar")
        spread_1mio = _f(f, "spread_1mio")
        last_price = _f(f, "last_price") or _f(f, "last_close")
        if spoof is not None and spoof > 0.5:
            penalty += spoof * 8.0
            out.negative_factors.append("spoof risk pattern in order stats")
        if net_cancel is not None and net_cancel > 0.4:
            penalty += min(6.0, net_cancel * 10.0)
            out.negative_factors.append("net cancel pressure on sell side")
        if vol is not None and vol > 0.04:
            penalty += min(6.0, (vol - 0.04) * 100.0)
            out.negative_factors.append(f"elevated realized vol ({vol*100:.2f}%)")
        if r1 is not None and r1 < -0.015:
            penalty += min(6.0, (-r1) * 200.0)
            out.negative_factors.append(f"sharp recent drop {r1*100:.2f}%")
        if spread_1mio is not None and last_price and last_price > 0:
            spread_bps = spread_1mio / last_price * 10000.0
            if spread_bps > 80:
                penalty += min(6.0, (spread_bps - 80) / 20.0)
                out.negative_factors.append(f"1mio spread is {spread_bps:.0f} bps")
        return _clamp(penalty, 0.0, 30.0)

    # ------------------------------------------------------------------
    def _confidence(self, bundle: FeatureBundle, out: ScoreBreakdown) -> float:
        f = bundle.features
        dq = bundle.data_quality
        base = float(dq.get("quality_score") or 0.0)
        ref = f.get("alerts_reference_summary")
        if isinstance(ref, dict) and ref.get("n_observations"):
            n = int(ref["n_observations"])
            if n >= 30:
                base += 0.1
            elif n < 5:
                base -= 0.1
        # Consistent direction across pillars adds to confidence.
        consistency = 0.0
        if out.trend_score >= 15:
            consistency += 0.05
        if out.momentum_score >= 10:
            consistency += 0.05
        if out.volume_flow_score >= 12:
            consistency += 0.05
        if out.orderbook_liquidity_score >= 8:
            consistency += 0.05
        if out.risk_penalty <= 5:
            consistency += 0.05
        if out.alerts_score > 0:
            consistency += 0.05
        # Penalise heavy disagreement.
        if out.trend_score >= 15 and (f.get("aggressive_buy_pressure") or 0) < -0.1:
            consistency -= 0.05
        conf = base + consistency
        return _clamp(conf, 0.0, 1.0)
