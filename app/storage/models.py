"""SQLAlchemy ORM models.

The schema is intentionally storage-agnostic (no SQLite-only features) so a
future migration to PostgreSQL only requires swapping the engine URL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ALGOPACK datasets
# ---------------------------------------------------------------------------


class EqTradeStats(Base):
    __tablename__ = "eq_tradestats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    tradedate: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    tradetime: Mapped[Optional[str]] = mapped_column(String(10))
    ts: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    pr_open: Mapped[Optional[float]] = mapped_column(Float)
    pr_high: Mapped[Optional[float]] = mapped_column(Float)
    pr_low: Mapped[Optional[float]] = mapped_column(Float)
    pr_close: Mapped[Optional[float]] = mapped_column(Float)
    pr_std: Mapped[Optional[float]] = mapped_column(Float)
    vol: Mapped[Optional[float]] = mapped_column(Float)
    val: Mapped[Optional[float]] = mapped_column(Float)
    trades: Mapped[Optional[int]] = mapped_column(Integer)
    pr_vwap: Mapped[Optional[float]] = mapped_column(Float)
    pr_change: Mapped[Optional[float]] = mapped_column(Float)
    trades_b: Mapped[Optional[int]] = mapped_column(Integer)
    trades_s: Mapped[Optional[int]] = mapped_column(Integer)
    val_b: Mapped[Optional[float]] = mapped_column(Float)
    val_s: Mapped[Optional[float]] = mapped_column(Float)
    vol_b: Mapped[Optional[float]] = mapped_column(Float)
    vol_s: Mapped[Optional[float]] = mapped_column(Float)
    disb: Mapped[Optional[float]] = mapped_column(Float)
    pr_vwap_b: Mapped[Optional[float]] = mapped_column(Float)
    pr_vwap_s: Mapped[Optional[float]] = mapped_column(Float)
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("secid", "tradedate", "tradetime", name="uq_eq_tradestats"),
        Index("ix_eq_tradestats_secid_ts", "secid", "ts"),
    )


class EqOrderStats(Base):
    __tablename__ = "eq_orderstats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    tradedate: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    tradetime: Mapped[Optional[str]] = mapped_column(String(10))
    ts: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    put_orders_b: Mapped[Optional[float]] = mapped_column(Float)
    put_orders_s: Mapped[Optional[float]] = mapped_column(Float)
    put_val_b: Mapped[Optional[float]] = mapped_column(Float)
    put_val_s: Mapped[Optional[float]] = mapped_column(Float)
    put_vol_b: Mapped[Optional[float]] = mapped_column(Float)
    put_vol_s: Mapped[Optional[float]] = mapped_column(Float)
    put_vwap_b: Mapped[Optional[float]] = mapped_column(Float)
    put_vwap_s: Mapped[Optional[float]] = mapped_column(Float)
    put_vol: Mapped[Optional[float]] = mapped_column(Float)
    put_val: Mapped[Optional[float]] = mapped_column(Float)
    put_orders: Mapped[Optional[float]] = mapped_column(Float)
    cancel_orders_b: Mapped[Optional[float]] = mapped_column(Float)
    cancel_orders_s: Mapped[Optional[float]] = mapped_column(Float)
    cancel_val_b: Mapped[Optional[float]] = mapped_column(Float)
    cancel_val_s: Mapped[Optional[float]] = mapped_column(Float)
    cancel_vol_b: Mapped[Optional[float]] = mapped_column(Float)
    cancel_vol_s: Mapped[Optional[float]] = mapped_column(Float)
    cancel_vwap_b: Mapped[Optional[float]] = mapped_column(Float)
    cancel_vwap_s: Mapped[Optional[float]] = mapped_column(Float)
    cancel_vol: Mapped[Optional[float]] = mapped_column(Float)
    cancel_val: Mapped[Optional[float]] = mapped_column(Float)
    cancel_orders: Mapped[Optional[float]] = mapped_column(Float)
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("secid", "tradedate", "tradetime", name="uq_eq_orderstats"),
        Index("ix_eq_orderstats_secid_ts", "secid", "ts"),
    )


class EqObStats(Base):
    __tablename__ = "eq_obstats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    tradedate: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    tradetime: Mapped[Optional[str]] = mapped_column(String(10))
    ts: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    spread_bbo: Mapped[Optional[float]] = mapped_column(Float)
    spread_lv10: Mapped[Optional[float]] = mapped_column(Float)
    spread_1mio: Mapped[Optional[float]] = mapped_column(Float)
    levels_b: Mapped[Optional[float]] = mapped_column(Float)
    levels_s: Mapped[Optional[float]] = mapped_column(Float)
    vol_b: Mapped[Optional[float]] = mapped_column(Float)
    vol_s: Mapped[Optional[float]] = mapped_column(Float)
    val_b: Mapped[Optional[float]] = mapped_column(Float)
    val_s: Mapped[Optional[float]] = mapped_column(Float)
    imbalance_vol_bbo: Mapped[Optional[float]] = mapped_column(Float)
    imbalance_val_bbo: Mapped[Optional[float]] = mapped_column(Float)
    imbalance_vol: Mapped[Optional[float]] = mapped_column(Float)
    imbalance_val: Mapped[Optional[float]] = mapped_column(Float)
    vwap_b: Mapped[Optional[float]] = mapped_column(Float)
    vwap_s: Mapped[Optional[float]] = mapped_column(Float)
    vwap_b_1mio: Mapped[Optional[float]] = mapped_column(Float)
    vwap_s_1mio: Mapped[Optional[float]] = mapped_column(Float)
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("secid", "tradedate", "tradetime", name="uq_eq_obstats"),
        Index("ix_eq_obstats_secid_ts", "secid", "ts"),
    )


class EqAlerts(Base):
    __tablename__ = "eq_alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    tradedate: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    tradetime: Mapped[Optional[str]] = mapped_column(String(10))
    ts: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    alert_type: Mapped[Optional[str]] = mapped_column(String(96), index=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float)
    value: Mapped[Optional[float]] = mapped_column(Float)
    reference: Mapped[Optional[str]] = mapped_column(Text)
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "secid", "tradedate", "tradetime", "alert_type", name="uq_eq_alerts"
        ),
    )


class EqHi2(Base):
    __tablename__ = "eq_hi2"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    tradedate: Mapped[Optional[str]] = mapped_column(String(10), index=True)
    tradetime: Mapped[Optional[str]] = mapped_column(String(10))
    metric: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    value: Mapped[Optional[float]] = mapped_column(Float)
    reference: Mapped[Optional[str]] = mapped_column(Text)
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("secid", "tradedate", "metric", name="uq_eq_hi2"),
    )


# ---------------------------------------------------------------------------
# Real-time data
# ---------------------------------------------------------------------------


class IntradayCandle(Base):
    """Intraday OHLCV candle. Default fetch interval is 10 minutes.

    MOEX ISS natively supports intervals 1, 10, 60, 24 — there is no native 5m.
    The bar duration is stored in ``interval_min`` so the table can hold mixed
    granularities if needed in the future.
    """

    __tablename__ = "intraday_candles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    begin: Mapped[datetime] = mapped_column(DateTime, index=True)
    end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    interval_min: Mapped[int] = mapped_column(Integer, default=10)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    close: Mapped[Optional[float]] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float)
    value: Mapped[Optional[float]] = mapped_column(Float)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("secid", "begin", "interval_min", name="uq_intraday_candles"),
    )


class MarketData(Base):
    """Latest snapshot of public marketdata for a secid (overwriting)."""

    __tablename__ = "marketdata"
    secid: Mapped[str] = mapped_column(String(32), primary_key=True)
    last: Mapped[Optional[float]] = mapped_column(Float)
    bid: Mapped[Optional[float]] = mapped_column(Float)
    offer: Mapped[Optional[float]] = mapped_column(Float)
    spread: Mapped[Optional[float]] = mapped_column(Float)
    open: Mapped[Optional[float]] = mapped_column(Float)
    high: Mapped[Optional[float]] = mapped_column(Float)
    low: Mapped[Optional[float]] = mapped_column(Float)
    last_change_pct: Mapped[Optional[float]] = mapped_column(Float)
    voltoday: Mapped[Optional[float]] = mapped_column(Float)
    valtoday: Mapped[Optional[float]] = mapped_column(Float)
    waprice: Mapped[Optional[float]] = mapped_column(Float)
    numtrades: Mapped[Optional[int]] = mapped_column(Integer)
    updatetime: Mapped[Optional[str]] = mapped_column(String(10))
    systime: Mapped[Optional[datetime]] = mapped_column(DateTime)
    lotsize: Mapped[Optional[int]] = mapped_column(Integer)
    prevprice: Mapped[Optional[float]] = mapped_column(Float)
    decimals: Mapped[Optional[int]] = mapped_column(Integer)
    raw_json: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Derived data
# ---------------------------------------------------------------------------


class DerivedFeatures(Base):
    __tablename__ = "derived_features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    feature_version: Mapped[str] = mapped_column(String(32))
    features_json: Mapped[str] = mapped_column(Text)


class Recommendations(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    intent: Mapped[str] = mapped_column(String(16), default="BUY_CHECK")
    action: Mapped[str] = mapped_column(String(16))
    recommended_action: Mapped[Optional[str]] = mapped_column(String(24))
    allow_buy: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation_json: Mapped[str] = mapped_column(Text)
    strategy_version: Mapped[str] = mapped_column(String(32))
    llm_used: Mapped[int] = mapped_column(Integer, default=0)


class Anomalies(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    anomaly_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[Optional[str]] = mapped_column(Text)


class AgentEvents(Base):
    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    level: Mapped[str] = mapped_column(String(16))
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[Optional[str]] = mapped_column(Text)
    payload_json: Mapped[Optional[str]] = mapped_column(Text)


class LLMLog(Base):
    __tablename__ = "llm_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    secid: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    prompt_version: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    feature_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text)
    response: Mapped[Optional[str]] = mapped_column(Text)
    usage_json: Mapped[Optional[str]] = mapped_column(Text)
    success: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)


class LLMCache(Base):
    __tablename__ = "llm_cache"
    feature_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    response_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
