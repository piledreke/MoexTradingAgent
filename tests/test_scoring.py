"""Tests for the deterministic scorer."""

from __future__ import annotations

from datetime import datetime

from app.features.feature_builder import FeatureBundle
from app.strategy.scoring import TechnicalScorer


def _bundle(features: dict, hi2: dict | None = None, dq: dict | None = None) -> FeatureBundle:
    return FeatureBundle(
        secid="SBER",
        ts=datetime.utcnow(),
        feature_version="fv-test",
        features=features,
        data_quality=dq or {"quality_score": 0.9, "freshness_ok": True},
        alerts=[],
        hi2=hi2 or {},
        market_snapshot={},
        last_price=features.get("last_price"),
    )


def _strong_buy_features() -> dict:
    return {
        "last_close": 300.0,
        "last_price": 300.0,
        "ema9": 299.0,
        "ema21": 298.0,
        "ema50": 295.0,
        "ema_9_above_21": True,
        "close_above_ema21": True,
        "close_above_ema9": True,
        "distance_to_vwap_pct": 0.3,
        "rsi14": 58.0,
        "bollinger_z": 1.2,
        "realized_vol": 0.012,
        "return_1_bar": 0.002,
        "return_3_bar": 0.006,
        "return_6_bar": 0.012,
        "return_12_bar": 0.015,
        "volume_zscore": 2.5,
        "value_zscore": 2.3,
        "trades_zscore": 1.8,
        "buy_volume_ratio": 0.62,
        "sell_volume_ratio": 0.38,
        "disb_now": 0.25,
        "aggressive_buy_pressure": 0.2,
        "pr_vwap_b_minus_s_pct": 0.05,
        "put_cancel_ratio_b": 1.0,
        "put_cancel_ratio_s": 1.0,
        "net_cancel_pressure": 0.0,
        "spread_bbo": 0.05,
        "spread_1mio": 0.2,
        "imbalance_bbo": 0.25,
        "imbalance_val": 0.2,
        "liquidity_score": 0.7,
        "slippage_1mio_bps": 10.0,
        "alerts_positive_count": 2,
        "alerts_negative_count": 0,
        "alerts_reference_summary": {"mean_change_15m": 0.4, "up": 30, "down": 10, "hit_rate": 0.75, "n_observations": 40},
        "val_b": 5_000_000.0,
    }


def _bearish_features() -> dict:
    f = _strong_buy_features()
    f.update({
        "ema_9_above_21": False,
        "close_above_ema21": False,
        "close_above_ema9": False,
        "distance_to_vwap_pct": -0.8,
        "rsi14": 32.0,
        "return_1_bar": -0.02,
        "return_3_bar": -0.04,
        "return_12_bar": -0.05,
        "buy_volume_ratio": 0.35,
        "sell_volume_ratio": 0.65,
        "disb_now": -0.4,
        "aggressive_buy_pressure": -0.3,
        "alerts_positive_count": 0,
        "alerts_negative_count": 3,
        "alerts_reference_summary": {"mean_change_15m": -0.5, "up": 5, "down": 25, "hit_rate": 0.16, "n_observations": 30},
        "imbalance_bbo": -0.3,
        "liquidity_score": 0.3,
        "realized_vol": 0.05,
    })
    return f


def test_strong_buy_score_is_high() -> None:
    scorer = TechnicalScorer()
    b = _bundle(_strong_buy_features())
    out = scorer.score(b)
    assert out.technical_score >= 60
    assert 0.0 <= out.confidence <= 1.0
    assert out.reasons


def test_bearish_score_is_low() -> None:
    scorer = TechnicalScorer()
    b = _bundle(_bearish_features())
    out = scorer.score(b)
    assert out.technical_score < 40
    assert out.negative_factors


def test_score_clamped_to_range() -> None:
    f = _strong_buy_features()
    f.update({
        "alerts_positive_count": 50,
        "alerts_negative_count": 0,
        "volume_zscore": 50.0,
        "value_zscore": 50.0,
    })
    scorer = TechnicalScorer()
    b = _bundle(f)
    out = scorer.score(b)
    assert 0.0 <= out.technical_score <= 100.0


def test_hi2_high_concentration_penalises() -> None:
    scorer = TechnicalScorer()
    base = _strong_buy_features()
    base.update({
        "hi2_aggressive_sell": 3000.0,
        "hi2_netflow_sell": 2800.0,
        "hi2_volume": 2600.0,
    })
    b = _bundle(base, hi2={"hhi_active_sell": 3000.0, "hhi_volume": 2600.0, "hhi_netflow_sell": 2800.0})
    out_with_hi2 = scorer.score(b)

    base2 = _strong_buy_features()
    b2 = _bundle(base2)
    out_without = scorer.score(b2)
    assert out_with_hi2.technical_score < out_without.technical_score
