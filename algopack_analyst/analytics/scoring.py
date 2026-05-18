"""Composite BUY scoring v1.3 — uses atomic snapshot + adaptive weights + MTF."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pandas as pd
import pytz

from analytics.adaptive_scoring import adaptive_weights, skip_scoring
from analytics.anomaly import alerts_score_impact, recent_alerts
from analytics.concentration import assess_concentration
from analytics.correlations import relative_strength_index_vs_market
from analytics.multi_timeframe import (
    golden_cross_check, mtf_confluence, mtf_rsi_divergence,
)
from analytics.order_flow import (
    absorption, buy_pressure_score, delta_divergence, iceberg_score,
)
from analytics.orderbook import bid_ask_imbalance, microprice, spread_bps
from analytics.patterns import (
    bearish_patterns_score, bullish_patterns_score,
    candle_patterns, detect_breakout, detect_volume_climax,
)
from analytics.regime import detect_regime
from analytics.sentiment import jur_sentiment, map_stock_to_future
from analytics.technical import (
    atr, compute_all_features, detect_trend, rsi, support_resistance,
)
from analytics.volume_profile import compute_volume_profile, nearest_node
from config import settings
from storage.repository import get_repo
from storage.snapshot import DataSnapshot, get_snapshot_builder
from utils.logger import logger
from utils.moex_client import get_moex_client

MSK = pytz.timezone("Europe/Moscow")

# In-memory dedup
_recent_recs: dict[str, tuple[float, dict]] = {}


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


async def composite_buy_score(
    ticker: str,
    horizon: str = "intraday",
    context: str | None = None,
) -> dict[str, Any]:
    """v1.3 composite BUY scoring with atomic snapshot and adaptive weights."""
    ticker = ticker.upper()
    now = time.time()

    # ─── Dedup cache ────────────────────────────────────
    if ticker in _recent_recs:
        ts, cached = _recent_recs[ticker]
        if now - ts < settings.RECOMMENDATION_TTL_SEC:
            out = dict(cached)
            out["from_cache"] = True
            return out

    # ─── Atomic snapshot ────────────────────────────────
    builder = get_snapshot_builder()
    snap = await builder.build(ticker)

    # ─── Guards ─────────────────────────────────────────
    if snap.suspended:
        return _build_avoid(ticker, "trading_suspended", snap)
    if not snap.tradable_tqbr:
        return _build_avoid(ticker, "not_in_tqbr", snap)
    if not snap.has_minimal_data:
        return _build_avoid(ticker, "insufficient_data", snap)

    # ─── Detect regime ──────────────────────────────────
    ohlcv_5m = snap.ohlcv.get(5, pd.DataFrame())
    ohlcv_1h = snap.ohlcv.get(60, pd.DataFrame())
    regime = detect_regime(ticker, ohlcv_5m, ohlcv_1h)

    # Skip if conditions unsuitable
    skip, skip_reason = skip_scoring(regime)
    if skip:
        return _build_hold(ticker, f"skip_{skip_reason}", regime, snap)

    if not snap.market_open:
        return _build_hold(ticker, "market_closed", regime, snap)

    # ─── Compute factor scores [0..1] ───────────────────
    factor_scores: dict[str, float] = {}
    signals: dict[str, Any] = {}
    reasons: list[str] = []
    risks: list[str] = []
    confidence = 1.0

    # Apply quality penalties
    if snap.quality.get("super_candles") == "stale":
        confidence -= 0.15
        risks.append("Super Candles устарели")
    if snap.quality.get("orderbook") == "stale":
        confidence -= 0.25
        risks.append("Стакан устарел")

    # ───── 1. TREND (with MTF + ADX) ────────────────────
    ohlcv_1m = snap.ohlcv.get(1, pd.DataFrame())
    if not ohlcv_1m.empty and len(ohlcv_1m) >= 30:
        sorted_1m = ohlcv_1m.sort_values("ts")
        trend_d = detect_trend(sorted_1m)
        rsi_v = float(rsi(sorted_1m["c"], 14).iloc[-1])
        signals["technical"] = {
            "trend": trend_d["trend"],
            "trend_strength": trend_d["strength"],
            "rsi14": round(rsi_v, 1),
            "adx": trend_d.get("adx"),
            "ema20": trend_d.get("ema20"),
            "ema50": trend_d.get("ema50"),
            "macd_hist": trend_d.get("macd_hist"),
        }
        ts_map = {"uptrend": 0.85, "sideways": 0.5,
                  "downtrend": 0.15, "mixed": 0.4, "unknown": 0.5}
        factor_scores["trend"] = ts_map[trend_d["trend"]]

        if trend_d["trend"] == "uptrend":
            reasons.append(
                f"Восходящий тренд (ADX={trend_d.get('adx')}, "
                f"EMA20>{trend_d.get('ema50'):.2f})"
            )
        elif trend_d["trend"] == "downtrend":
            risks.append("Нисходящий тренд")

        # Momentum
        rsi_norm = _norm01(rsi_v, 30, 70)
        if rsi_v > 75:
            rsi_norm = max(0.3, rsi_norm - 0.3)
            risks.append(f"RSI в зоне перекупленности ({rsi_v:.0f})")
        roc = (
            float(sorted_1m["c"].iloc[-1] / sorted_1m["c"].iloc[-10] - 1) * 100
            if len(sorted_1m) >= 10 else 0
        )
        factor_scores["momentum"] = (rsi_norm + _norm01(roc, -2, 2)) / 2
    else:
        factor_scores["trend"] = 0.5
        factor_scores["momentum"] = 0.5
        signals["technical"] = {"available": False}
        confidence *= 0.7

    # ───── 2. MTF CONFLUENCE ────────────────────────────
    mtf = mtf_confluence(snap.ohlcv)
    signals["mtf"] = mtf
    if mtf["composite_direction"] == "bullish":
        factor_scores["mtf_confluence"] = 0.5 + mtf["confluence_score"] * 0.5
        reasons.append(
            f"MTF подтверждение: bullish на {mtf['bullish_count']} ТФ "
            f"(score={mtf['confluence_score']})"
        )
    elif mtf["composite_direction"] == "bearish":
        factor_scores["mtf_confluence"] = 0.5 - mtf["confluence_score"] * 0.5
        risks.append(f"MTF: bearish на {mtf['bearish_count']} ТФ")
    else:
        factor_scores["mtf_confluence"] = 0.5

    # MTF divergence
    div = mtf_rsi_divergence(snap.ohlcv)
    if div["any_bullish"]:
        reasons.append("Бычья дивергенция цена/RSI")
        factor_scores["mtf_confluence"] = min(1.0, factor_scores["mtf_confluence"] + 0.1)
    if div["any_bearish"]:
        risks.append("Медвежья дивергенция цена/RSI")

    # Golden cross
    gc = golden_cross_check(snap.ohlcv)
    if gc["golden_cross_recent"]:
        reasons.append("Недавний golden cross на дневном ТФ")
    if gc["death_cross_recent"]:
        risks.append("Недавний death cross на дневном ТФ")

    # ───── 3. VOLUME PROFILE ────────────────────────────
    vp_source = snap.ohlcv.get(15, snap.ohlcv.get(5, pd.DataFrame()))
    if not vp_source.empty and len(vp_source) >= 30:
        vp = compute_volume_profile(vp_source.sort_values("ts"), bins=50)
        signals["volume_profile"] = {
            "poc": vp.get("poc"),
            "vah": vp.get("vah"),
            "val": vp.get("val"),
            "location": vp.get("location"),
            "dist_to_poc_pct": vp.get("dist_to_poc_pct"),
        }
        loc = vp.get("location")
        if loc == "above_va":
            factor_scores["volume_profile"] = 0.65
            reasons.append("Цена выше Value Area — продолжение тренда")
        elif loc == "below_va":
            factor_scores["volume_profile"] = 0.30
            risks.append("Цена ниже Value Area")
        else:
            factor_scores["volume_profile"] = 0.55
    else:
        factor_scores["volume_profile"] = 0.5
        signals["volume_profile"] = {"available": False}

    # ───── 4. ORDER FLOW (Super Candles) ────────────────
    if not snap.super_candles.empty:
        bp = buy_pressure_score(snap.super_candles)
        dd = delta_divergence(snap.super_candles)
        ab = absorption(snap.super_candles)
        ice = iceberg_score(snap.orderstats) if not snap.orderstats.empty else {"side": None, "score": 0}

        signals["order_flow"] = {
            "buy_pressure": bp,
            "delta_divergence": dd,
            "absorption": ab,
            "iceberg": ice,
        }

        of_score = bp["score"]
        if dd["type"] == "bullish_divergence":
            of_score = min(1.0, of_score + 0.15)
            reasons.append("Бычья дельта-дивергенция (накопление на просадке)")
        elif dd["type"] == "bearish_divergence":
            of_score = max(0.0, of_score - 0.15)
            risks.append("Медвежья дельта-дивергенция (распределение)")

        if ab["type"] == "absorption_buy":
            of_score = min(1.0, of_score + 0.10)
            reasons.append("Поглощение sell-ордеров крупным покупателем")
        elif ab["type"] == "absorption_sell":
            of_score = max(0.0, of_score - 0.10)
            risks.append("Поглощение buy-ордеров продавцом")

        if ice["side"] == "buy":
            of_score = min(1.0, of_score + 0.05)
            reasons.append("Айсберг-активность на bid")
        if ice["side"] == "sell":
            of_score = max(0.0, of_score - 0.05)
            risks.append("Айсберг-активность на ask")

        factor_scores["order_flow"] = of_score

        if bp["buy_dominance"] > 0.6:
            reasons.append(f"Покупатели доминируют ({bp['buy_dominance']*100:.0f}%)")
    else:
        factor_scores["order_flow"] = 0.5
        signals["order_flow"] = {"available": False}
        confidence *= 0.8

    # ───── 5. ORDERBOOK ─────────────────────────────────
    if snap.orderbook:
        imb = bid_ask_imbalance(snap.orderbook)
        sp_bps = spread_bps(snap.orderbook)
        mp = microprice(snap.orderbook)
        signals["orderbook"] = {
            "imbalance": round(imb, 3),
            "spread_bps": round(sp_bps, 2) if sp_bps else None,
            "microprice": mp,
            "best_bid": snap.orderbook["bids"][0][0] if snap.orderbook["bids"] else None,
            "best_ask": snap.orderbook["asks"][0][0] if snap.orderbook["asks"] else None,
        }
        factor_scores["orderbook"] = _norm01(imb, -0.5, 0.5)
        if imb > 0.2:
            reasons.append(f"Стакан в пользу покупателей ({imb*100:+.0f}%)")
        elif imb < -0.2:
            risks.append(f"Давление продавцов в стакане ({imb*100:+.0f}%)")
    else:
        factor_scores["orderbook"] = 0.5
        signals["orderbook"] = {"available": False}
        confidence -= 0.3

    # ───── 6. CANDLESTICK PATTERNS ──────────────────────
    pattern_source = snap.ohlcv.get(15, snap.ohlcv.get(5, pd.DataFrame()))
    if not pattern_source.empty and len(pattern_source) >= 5:
        sorted_p = pattern_source.sort_values("ts")
        patterns = candle_patterns(sorted_p)
        bull_s = bullish_patterns_score(patterns)
        bear_s = bearish_patterns_score(patterns)
        active_patterns = [k for k, v in patterns.items() if v]

        signals["patterns"] = {
            "active": active_patterns,
            "bullish_score": bull_s,
            "bearish_score": bear_s,
        }

        # breakout
        br = detect_breakout(sorted_p)
        signals["patterns"]["breakout"] = br
        if br.get("breakout") and br.get("direction") == "up":
            reasons.append(f"Breakout вверх (+{br['magnitude_pct']:.2f}%)")
            bull_s = min(1.0, bull_s + 0.2)
        elif br.get("breakout") and br.get("direction") == "down":
            risks.append(f"Breakout вниз")
            bear_s = min(1.0, bear_s + 0.2)

        # volume climax
        vc = detect_volume_climax(sorted_p)
        if vc.get("climax"):
            signals["patterns"]["volume_climax"] = vc
            if vc["direction"] == "buying":
                reasons.append(f"Buying climax (объём x{vc['ratio']:.1f})")
            else:
                risks.append(f"Selling climax")

        factor_scores["patterns"] = 0.5 + (bull_s - bear_s) * 0.5
        for p in active_patterns:
            if p in ("hammer", "bullish_engulfing", "morning_star",
                     "piercing", "three_white_soldiers"):
                reasons.append(f"Бычий паттерн: {p}")
            elif p in ("shooting_star", "bearish_engulfing",
                       "evening_star", "three_black_crows"):
                risks.append(f"Медвежий паттерн: {p}")
    else:
        factor_scores["patterns"] = 0.5
        signals["patterns"] = {"available": False}

    # ───── 7. MEGA ALERTS ───────────────────────────────
    alerts_impact = alerts_score_impact(snap.alerts, lookback_min=60)
    signals["alerts_last_hour"] = recent_alerts(snap.alerts, 60)[:5]
    factor_scores["alerts"] = _norm01(alerts_impact["impact"], -0.5, 0.5)
    if alerts_impact["bullish_count"] > 0:
        reasons.append(f"Бычьих Mega Alerts: {alerts_impact['bullish_count']}")
    if alerts_impact["bearish_count"] > 0:
        risks.append(f"Медвежьих Mega Alerts: {alerts_impact['bearish_count']}")

    # ───── 8. HI2 — multi-level ─────────────────────────
    intraday_vol = float(snap.super_candles["val"].sum()) if not snap.super_candles.empty else 0
    vol_spike = 1.0
    if not snap.ohlcv.get(60, pd.DataFrame()).empty:
        daily = snap.ohlcv[60].sort_values("ts")
        if len(daily) > 24:
            today_v = daily.tail(8)["v"].sum()
            avg_v = daily.tail(40)["v"].mean() * 8
            vol_spike = today_v / avg_v if avg_v > 0 else 1.0

    hi2_row = snap.hi2.iloc[0] if not snap.hi2.empty else None
    conc = assess_concentration(
        hi2_row,
        intraday_volume=intraday_vol,
        vol_spike_ratio=vol_spike,
    )
    signals["hi2"] = conc.as_dict()
    risks.extend(conc.reasons[:2])

    # Critical concentration — override
    if conc.action == "avoid":
        return _build_avoid(ticker, "concentration_critical", snap,
                            reasons=conc.reasons, regime=regime)

    conc_score_map = {"low": 0.85, "medium": 0.55, "high": 0.25, "critical": 0.0}
    factor_scores["hi2"] = conc_score_map[conc.risk_level]

    # ───── 9. FUTOI ─────────────────────────────────────
    if not snap.futoi.empty:
        fs = jur_sentiment(snap.futoi)
        signals["futoi"] = fs
        factor_scores["futoi"] = {"bullish": 0.8, "neutral": 0.5, "bearish": 0.2}[fs["sentiment"]]
        if fs["sentiment"] == "bullish":
            reasons.append(f"Юр.лица наращивают long (Δ={fs['delta_yur']:.0f})")
        elif fs["sentiment"] == "bearish":
            risks.append("Юр.лица сокращают long")
        if fs["divergence"]:
            reasons.append("Расхождение настроений smart vs retail")
    else:
        factor_scores["futoi"] = 0.5
        signals["futoi"] = {"available": False}

    # ───── 10. RELATIVE STRENGTH (vs IMOEX) ─────────────
    imoex = snap.index_ohlcv.get("IMOEX", pd.DataFrame())
    asset_for_rs = snap.ohlcv.get(60, snap.ohlcv.get(15, pd.DataFrame()))
    if not imoex.empty and not asset_for_rs.empty:
        rs = relative_strength_index_vs_market(asset_for_rs, imoex, period=20)
        signals["relative_strength"] = rs
        if rs["label"] == "outperforming":
            reasons.append(f"Опережает IMOEX (+{rs['outperformance_pct']:.2f}%)")
            # bonus on momentum
            factor_scores["momentum"] = min(1.0, factor_scores.get("momentum", 0.5) + 0.05)
        elif rs["label"] == "underperforming":
            risks.append(f"Отстаёт от IMOEX ({rs['outperformance_pct']:.2f}%)")
            factor_scores["momentum"] = max(0.0, factor_scores.get("momentum", 0.5) - 0.05)

    # ─── Composite score ────────────────────────────────
    weights = adaptive_weights(regime)
    weighted = sum(factor_scores.get(k, 0.5) * w for k, w in weights.items())
    score_100 = int(round(weighted * 100))

    # ─── Decide recommendation ──────────────────────────
    if score_100 >= settings.SCORE_STRONG_BUY:
        rec = "STRONG_BUY"
    elif score_100 >= settings.SCORE_BUY:
        rec = "BUY"
    elif score_100 >= settings.SCORE_HOLD:
        rec = "HOLD"
    else:
        rec = "AVOID"

    # Downsize from concentration
    if conc.action == "downsize":
        rec = "HOLD" if rec == "BUY" else rec
    if conc.action == "hold":
        rec = "HOLD"

    # ─── Risk management ────────────────────────────────
    entry_zone, stop_loss, take_profit = _compute_levels(snap, signals)

    # Position sizing
    max_pos = min(settings.MAX_POS_PCT_DEFAULT, conc.max_position_pct_cap)
    if alerts_impact["bearish_count"] > 0:
        max_pos = min(max_pos, settings.MAX_POS_PCT_NEGATIVE_ALERTS)
    if score_100 >= settings.SCORE_STRONG_BUY:
        max_pos = max(max_pos, min(settings.MAX_POS_PCT_STRONG_SIGNAL,
                                   conc.max_position_pct_cap))
    sp_bps = signals.get("orderbook", {}).get("spread_bps")
    if sp_bps and sp_bps > 50:
        max_pos = min(max_pos, settings.MAX_POS_PCT_WIDE_SPREAD)
        risks.append(f"Широкий спред {sp_bps:.0f}bps")

    if regime.volatility_regime == "extreme":
        max_pos = min(max_pos, 3.0)
    elif regime.volatility_regime == "high":
        max_pos = min(max_pos, 7.0)

    confidence = max(0.0, min(1.0, confidence))

    # ─── Build response ─────────────────────────────────
    response = {
        "ticker": ticker,
        "snapshot_id": snap.snapshot_id,
        "timestamp": datetime.now(MSK).isoformat(),
        "recommendation": rec,
        "score": score_100,
        "confidence": round(confidence, 3),
        "horizon": horizon,
        "regime": regime.as_dict(),
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "max_position_pct": round(max_pos, 2),
        "signals": signals,
        "reasons": reasons[:10],
        "risks": risks[:10],
        "factor_scores": {k: round(v, 3) for k, v in factor_scores.items()},
        "weights_used": {k: round(v, 3) for k, v in weights.items()},
        "concentration": conc.as_dict(),
        "strategy_version": settings.STRATEGY_VERSION,
        "data_freshness_seconds": snap.freshness,
        "data_quality": snap.quality,
        "from_cache": False,
    }

    _recent_recs[ticker] = (now, response)
    return response


# ───── Helpers ──────────────────────────────────────────

def _compute_levels(snap: DataSnapshot, signals: dict) -> tuple[dict, float | None, float | None]:
    """Compute entry zone, SL and TP using ATR + support/resistance + POC."""
    ohlcv_5m = snap.ohlcv.get(5, pd.DataFrame())
    if ohlcv_5m.empty or len(ohlcv_5m) < 20:
        last = snap.last_price or 0
        return ({"min": last, "max": last}, None, None)

    sorted_df = ohlcv_5m.sort_values("ts")
    last = float(sorted_df["c"].iloc[-1])
    atr_v = float(atr(sorted_df, 14).iloc[-1])
    sr = support_resistance(sorted_df)

    # Entry: use orderbook bbo if available
    ob_sig = signals.get("orderbook", {})
    best_bid = ob_sig.get("best_bid")
    best_ask = ob_sig.get("best_ask")
    if best_bid and best_ask:
        entry_zone = {"min": round(best_bid, 4), "max": round(best_ask * 1.001, 4)}
    else:
        entry_zone = {"min": round(last * 0.999, 4), "max": round(last * 1.003, 4)}

    # Stop loss: max(support - small buffer, last - 1.5*ATR)
    sl_atr = last - 1.5 * atr_v
    sl_sup = sr["support"] * 0.998
    stop_loss = round(max(sl_atr, sl_sup), 4)

    # Take profit: min(resistance, last + 1.5*risk)
    risk_amt = last - stop_loss
    if risk_amt <= 0:
        return entry_zone, stop_loss, None
    tp_rr = last + 1.5 * risk_amt
    tp_res = sr["resistance"] * 0.998
    # Prefer resistance if it gives better R:R
    if tp_res > last + risk_amt:
        take_profit = round(min(tp_rr, tp_res), 4)
    else:
        take_profit = round(tp_rr, 4)

    return entry_zone, stop_loss, take_profit


def _build_avoid(
    ticker: str, reason: str, snap: DataSnapshot | None = None,
    reasons: list[str] | None = None, regime=None,
) -> dict:
    return {
        "ticker": ticker,
        "snapshot_id": snap.snapshot_id if snap else None,
        "timestamp": datetime.now(MSK).isoformat(),
        "recommendation": "AVOID",
        "score": 0,
        "confidence": 1.0,
        "horizon": "intraday",
        "regime": regime.as_dict() if regime else {},
        "entry_zone": {"min": None, "max": None},
        "stop_loss": None,
        "take_profit": None,
        "max_position_pct": 0,
        "signals": {},
        "reasons": [],
        "risks": [reason] + (reasons or []),
        "factor_scores": {},
        "weights_used": {},
        "concentration": {},
        "strategy_version": settings.STRATEGY_VERSION,
        "data_freshness_seconds": snap.freshness if snap else {},
        "data_quality": snap.quality if snap else {},
        "from_cache": False,
    }


def _build_hold(ticker: str, reason: str, regime, snap: DataSnapshot) -> dict:
    return {
        "ticker": ticker,
        "snapshot_id": snap.snapshot_id,
        "timestamp": datetime.now(MSK).isoformat(),
        "recommendation": "HOLD",
        "score": 50,
        "confidence": 0.5,
        "horizon": "intraday",
        "regime": regime.as_dict() if regime else {},
        "entry_zone": {"min": None, "max": None},
        "stop_loss": None,
        "take_profit": None,
        "max_position_pct": 0,
        "signals": {},
        "reasons": [],
        "risks": [reason],
        "factor_scores": {},
        "weights_used": {},
        "concentration": {},
        "strategy_version": settings.STRATEGY_VERSION,
        "data_freshness_seconds": snap.freshness,
        "data_quality": snap.quality,
        "from_cache": False,
    }