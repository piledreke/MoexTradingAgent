"""End-to-end tests for TechnicalAdvisor (BUY_CHECK + EXIT_CHECK)."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.config import get_settings
from app.storage.repository import Repository
from app.strategy.advisor import TechnicalAdvisor
from app.strategy.recommendation import (
    AdviceIntent,
    AdviceRequest,
    AdviceResponse,
    BuyAction,
    CurrentPosition,
    ExitAction,
    PortfolioState,
)


def _seed_strong_buy(repo: Repository, secid: str = "SBER") -> None:
    """Populate the DB with strongly bullish synthetic data for ``secid``."""
    now = datetime.utcnow()
    ts_rows = []
    base = 280.0
    for i in range(60):
        ts = now - timedelta(minutes=(60 - i) * 5)
        td = ts.date().isoformat()
        tt = ts.strftime("%H:%M:%S")
        close = base + i * 0.4
        ts_rows.append({
            "secid": secid,
            "tradedate": td,
            "tradetime": tt,
            "ts": ts,
            "pr_open": close - 0.2,
            "pr_high": close + 0.3,
            "pr_low": close - 0.4,
            "pr_close": close,
            "pr_std": 0.05,
            "vol": 5000 + i * 50,
            "val": (5000 + i * 50) * close,
            "trades": 80 + i,
            "pr_vwap": close - 0.05,
            "pr_change": 0.002,
            "trades_b": 55 + i,
            "trades_s": 25 + i // 2,
            "val_b": (3000 + i * 30) * close,
            "val_s": (2000 + i * 20) * close,
            "vol_b": 3000 + i * 30,
            "vol_s": 2000 + i * 20,
            "disb": 0.2,
            "pr_vwap_b": close - 0.02,
            "pr_vwap_s": close - 0.08,
            "systime": ts,
        })
    repo.save_tradestats(ts_rows)

    candle_rows = []
    for i in range(60):
        begin = now - timedelta(minutes=(60 - i) * 5)
        close = base + i * 0.4
        candle_rows.append({
            "secid": secid,
            "begin": begin,
            "end": begin + timedelta(minutes=5),
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 5000 + i * 50,
            "value": (5000 + i * 50) * close,
        })
    repo.save_candles_5m(candle_rows)

    repo.save_marketdata_snapshot([{
        "secid": secid,
        "last": base + 60 * 0.4,
        "bid": base + 60 * 0.4 - 0.05,
        "offer": base + 60 * 0.4 + 0.05,
        "spread": 0.1,
        "open": base,
        "high": base + 60 * 0.4 + 1.0,
        "low": base - 0.5,
        "lastchangeprcnt": 1.2,
        "voltoday": 500_000,
        "valtoday": 500_000 * (base + 24.0),
        "waprice": base + 12.0,
        "numtrades": 4000,
        "updatetime": now.strftime("%H:%M:%S"),
        "systime": now,
        "lotsize": 10,
        "prevprice": base - 1.0,
        "decimals": 2,
    }])

    # Order stats - normal, no spoof.
    os_rows = []
    for i in range(20):
        ts = now - timedelta(minutes=(20 - i) * 5)
        td = ts.date().isoformat()
        tt = ts.strftime("%H:%M:%S")
        os_rows.append({
            "secid": secid,
            "tradedate": td,
            "tradetime": tt,
            "ts": ts,
            "put_orders_b": 500,
            "put_orders_s": 480,
            "put_val_b": 1_500_000,
            "put_val_s": 1_400_000,
            "put_vol_b": 5000,
            "put_vol_s": 4500,
            "put_vwap_b": 290.0,
            "put_vwap_s": 290.5,
            "cancel_orders_b": 400,
            "cancel_orders_s": 410,
            "cancel_val_b": 1_200_000,
            "cancel_val_s": 1_200_000,
            "cancel_vol_b": 4000,
            "cancel_vol_s": 4000,
            "cancel_vwap_b": 290.1,
            "cancel_vwap_s": 290.4,
            "systime": ts,
        })
    repo.save_orderstats(os_rows)

    # OB stats - tight spread, positive imbalance.
    ob_rows = []
    for i in range(20):
        ts = now - timedelta(minutes=(20 - i) * 5)
        td = ts.date().isoformat()
        tt = ts.strftime("%H:%M:%S")
        ob_rows.append({
            "secid": secid,
            "tradedate": td,
            "tradetime": tt,
            "ts": ts,
            "spread_bbo": 0.05,
            "spread_lv10": 0.3,
            "spread_1mio": 0.2,
            "levels_b": 200,
            "levels_s": 180,
            "vol_b": 20_000,
            "vol_s": 15_000,
            "val_b": 6_000_000,
            "val_s": 4_500_000,
            "imbalance_vol_bbo": 0.25,
            "imbalance_val_bbo": 0.22,
            "imbalance_vol": 0.15,
            "imbalance_val": 0.18,
            "vwap_b": 289.95,
            "vwap_s": 290.10,
            "vwap_b_1mio": 289.98,
            "vwap_s_1mio": 290.05,
            "systime": ts,
        })
    repo.save_obstats(ob_rows)


def _seed_bearish(repo: Repository, secid: str = "SBER") -> None:
    """Populate DB with bearish data: falling prices, sell pressure, wide spread."""
    now = datetime.utcnow()
    ts_rows = []
    base = 320.0
    for i in range(60):
        ts = now - timedelta(minutes=(60 - i) * 5)
        td = ts.date().isoformat()
        tt = ts.strftime("%H:%M:%S")
        close = base - i * 0.5
        ts_rows.append({
            "secid": secid,
            "tradedate": td,
            "tradetime": tt,
            "ts": ts,
            "pr_open": close + 0.2,
            "pr_high": close + 0.5,
            "pr_low": close - 0.5,
            "pr_close": close,
            "pr_std": 0.3,
            "vol": 7000,
            "val": 7000 * close,
            "trades": 80,
            "pr_vwap": close + 0.1,
            "pr_change": -0.003,
            "trades_b": 20,
            "trades_s": 60,
            "val_b": 1500 * close,
            "val_s": 5500 * close,
            "vol_b": 1500,
            "vol_s": 5500,
            "disb": -0.5,
            "pr_vwap_b": close + 0.05,
            "pr_vwap_s": close - 0.05,
            "systime": ts,
        })
    repo.save_tradestats(ts_rows)
    candle_rows = []
    for i in range(60):
        begin = now - timedelta(minutes=(60 - i) * 5)
        close = base - i * 0.5
        candle_rows.append({
            "secid": secid,
            "begin": begin,
            "end": begin + timedelta(minutes=5),
            "open": close + 0.2,
            "high": close + 0.5,
            "low": close - 0.6,
            "close": close,
            "volume": 7000,
            "value": 7000 * close,
        })
    repo.save_candles_5m(candle_rows)

    repo.save_marketdata_snapshot([{
        "secid": secid,
        "last": base - 30.0,
        "bid": base - 30.0 - 0.05,
        "offer": base - 30.0 + 0.05,
        "spread": 0.1,
        "open": base,
        "high": base + 1.0,
        "low": base - 30.5,
        "lastchangeprcnt": -3.0,
        "voltoday": 500_000,
        "valtoday": 500_000 * 290.0,
        "waprice": base - 10.0,
        "numtrades": 4000,
        "updatetime": now.strftime("%H:%M:%S"),
        "systime": now,
        "lotsize": 10,
        "prevprice": base + 0.5,
        "decimals": 2,
    }])
    ob_rows = []
    for i in range(20):
        ts = now - timedelta(minutes=(20 - i) * 5)
        td = ts.date().isoformat()
        tt = ts.strftime("%H:%M:%S")
        ob_rows.append({
            "secid": secid,
            "tradedate": td,
            "tradetime": tt,
            "ts": ts,
            "spread_bbo": 0.05,
            "spread_lv10": 0.5,
            "spread_1mio": 0.4,
            "levels_b": 100,
            "levels_s": 200,
            "vol_b": 5000,
            "vol_s": 20_000,
            "val_b": 1_000_000,
            "val_s": 6_000_000,
            "imbalance_vol_bbo": -0.3,
            "imbalance_val_bbo": -0.4,
            "imbalance_vol": -0.4,
            "imbalance_val": -0.5,
            "vwap_b": base - 10.05,
            "vwap_s": base - 9.90,
            "vwap_b_1mio": base - 10.10,
            "vwap_s_1mio": base - 9.85,
            "systime": ts,
        })
    repo.save_obstats(ob_rows)


def test_advisor_buy_check_returns_full_schema(repo) -> None:
    _seed_strong_buy(repo)
    advisor = TechnicalAdvisor(repo=repo)
    req = AdviceRequest(secid="SBER", intent=AdviceIntent.BUY_CHECK, intended_cash_rub=50_000)
    resp = advisor.get_advice(req)
    assert isinstance(resp, AdviceResponse)
    assert resp.secid == "SBER"
    assert resp.intent == AdviceIntent.BUY_CHECK
    assert resp.recommended_action in {a.value for a in BuyAction}
    # Validate every required acceptance criteria field is populated.
    assert isinstance(resp.action.value, str)
    assert isinstance(resp.allow_buy, bool)
    assert 0.0 <= resp.technical_score <= 100.0
    assert 0.0 <= resp.confidence <= 1.0
    assert isinstance(resp.reasons, list)
    assert isinstance(resp.risk_flags, list)
    assert isinstance(resp.data_quality, dict)
    assert resp.strategy_version


def test_advisor_buy_check_unknown_ticker_is_vetoed(repo) -> None:
    advisor = TechnicalAdvisor(repo=repo)
    req = AdviceRequest(secid="UNKN", intent=AdviceIntent.BUY_CHECK)
    resp = advisor.get_advice(req)
    assert resp.allow_buy is False
    assert "ticker_not_in_universe" in resp.risk_flags
    assert resp.recommended_action == BuyAction.DO_NOT_BUY.value


def test_advisor_exit_check_no_position_returns_hold(repo) -> None:
    advisor = TechnicalAdvisor(repo=repo)
    req = AdviceRequest(secid="SBER", intent=AdviceIntent.EXIT_CHECK)
    resp = advisor.get_advice(req)
    assert resp.recommended_action == ExitAction.HOLD_POSITION.value
    assert "no_position_to_exit" in resp.risk_flags
    assert resp.exit_warning is False


def test_advisor_exit_check_bearish_triggers_exit(repo) -> None:
    _seed_bearish(repo)
    advisor = TechnicalAdvisor(repo=repo)
    position = CurrentPosition(
        quantity=100,
        average_price=320.0,
        market_value_rub=29_000,
        unrealized_pnl_pct=-5.0,
    )
    req = AdviceRequest(
        secid="SBER",
        intent=AdviceIntent.EXIT_CHECK,
        position=position,
    )
    resp = advisor.get_advice(req)
    assert resp.recommended_action in (
        ExitAction.EXIT_POSITION.value,
        ExitAction.TRIM_POSITION.value,
    )
    assert resp.exit_warning is True
    assert resp.recommended_sell_quantity is not None
    assert resp.recommended_sell_quantity > 0


def test_advisor_persists_recommendation(repo) -> None:
    _seed_strong_buy(repo)
    advisor = TechnicalAdvisor(repo=repo)
    advisor.get_advice(AdviceRequest(secid="SBER"))
    rec = repo.get_latest_recommendation("SBER")
    assert rec is not None
    assert rec["secid"] == "SBER"
    latest = repo.get_latest_recommendations()
    assert any(r["secid"] == "SBER" for r in latest)
