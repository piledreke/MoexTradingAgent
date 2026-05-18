"""Tests for hard risk veto + portfolio sizing."""

from __future__ import annotations

from datetime import datetime

from app.config import reload_settings
from app.features.feature_builder import FeatureBundle
from app.strategy.recommendation import CurrentPosition, PortfolioState
from app.strategy.risk import RiskManager


def _bundle(features: dict, dq: dict | None = None, secid: str = "SBER") -> FeatureBundle:
    return FeatureBundle(
        secid=secid,
        ts=datetime.utcnow(),
        feature_version="fv-test",
        features=features,
        data_quality=dq or {"quality_score": 0.9, "freshness_ok": True,
                            "latest_tradestats_age_sec": 60,
                            "latest_marketdata_age_sec": 10,
                            "missing_sources": []},
        alerts=[],
        hi2={},
        market_snapshot={},
        last_price=features.get("last_price"),
    )


def test_ticker_not_in_universe_triggers_hard_veto() -> None:
    settings = reload_settings()
    rm = RiskManager(settings)
    b = _bundle({"last_price": 100, "spread_bbo": 0.05, "val_b": 1e6, "liquidity_score": 0.7}, secid="ZZZZ")
    res = rm.assess_buy(b, intended_cash_rub=50_000, portfolio_state=None, current_position=None)
    assert res.hard_veto is True
    assert "ticker_not_in_universe" in res.risk_flags


def test_stale_data_triggers_hard_veto() -> None:
    rm = RiskManager()
    b = _bundle(
        {"last_price": 100, "spread_bbo": 0.05, "val_b": 1e6, "liquidity_score": 0.7},
        dq={
            "quality_score": 0.0,
            "freshness_ok": False,
            "latest_tradestats_age_sec": 99999,
            "latest_marketdata_age_sec": 99999,
            "missing_sources": [],
        },
    )
    res = rm.assess_buy(b, None, None, None)
    assert res.hard_veto is True
    assert "stale_data" in res.risk_flags


def test_wide_spread_triggers_hard_veto() -> None:
    rm = RiskManager()
    b = _bundle({
        "last_price": 100,
        "spread_bbo": 5.0,
        "val_b": 1e6,
        "liquidity_score": 0.7,
    })
    res = rm.assess_buy(b, None, None, None)
    assert res.hard_veto is True
    assert any(flag.startswith("spread_bbo_too_wide") for flag in res.risk_flags)


def test_low_liquidity_triggers_hard_veto() -> None:
    rm = RiskManager()
    b = _bundle({"last_price": 100, "spread_bbo": 0.05, "val_b": 100, "liquidity_score": 0.7})
    res = rm.assess_buy(b, None, None, None)
    assert "liquidity_too_low" in res.risk_flags
    assert res.hard_veto is True


def test_position_limit_blocks_additional_buys() -> None:
    rm = RiskManager()
    b = _bundle({"last_price": 100, "spread_bbo": 0.05, "val_b": 1e6, "liquidity_score": 0.7})
    ps = PortfolioState(
        cash_rub=500_000,
        equity_rub=1_000_000,
        positions_value_rub=200_000,
        daily_trades_count=10,
        daily_trade_limit=200,
    )
    pos = CurrentPosition(quantity=100, average_price=1000, market_value_rub=110_000, unrealized_pnl_pct=1.0)
    res = rm.assess_buy(b, intended_cash_rub=50_000, portfolio_state=ps, current_position=pos)
    assert res.hard_veto is True
    assert "position_limit_reached" in res.risk_flags


def test_daily_trade_limit_blocks_buy() -> None:
    rm = RiskManager()
    b = _bundle({"last_price": 100, "spread_bbo": 0.05, "val_b": 1e6, "liquidity_score": 0.7})
    ps = PortfolioState(
        cash_rub=500_000,
        equity_rub=1_000_000,
        positions_value_rub=50_000,
        daily_trades_count=200,
        daily_trade_limit=200,
    )
    res = rm.assess_buy(b, intended_cash_rub=10_000, portfolio_state=ps, current_position=None)
    assert res.hard_veto is True
    assert "daily_trade_limit_reached" in res.risk_flags


def test_position_size_multiplier_capped() -> None:
    rm = RiskManager()
    b = _bundle({"last_price": 100, "spread_bbo": 0.05, "val_b": 1e6, "liquidity_score": 0.7})
    ps = PortfolioState(
        cash_rub=900_000,
        equity_rub=1_000_000,
        positions_value_rub=0,
        daily_trades_count=0,
        daily_trade_limit=200,
    )
    res = rm.assess_buy(b, intended_cash_rub=1_000_000, portfolio_state=ps, current_position=None)
    # max_single_order is 5% of equity = 50_000 => multiplier 0.05 of intended_cash
    assert 0.04 <= res.position_size_multiplier <= 0.06
    assert res.max_cash_rub is not None and res.max_cash_rub <= 50_000 + 1
