"""Pydantic models for API requests / responses."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeTickerRequest(BaseModel):
    ticker: str = Field(..., min_length=2, max_length=12)
    horizon: Literal["intraday", "swing", "position"] = "intraday"
    context: str | None = None


class AnalyzeQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class WatchlistAddRequest(BaseModel):
    tickers: list[str]


class EntryZone(BaseModel):
    min: float | None
    max: float | None


class Recommendation(BaseModel):
    ticker: str
    timestamp: str
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "AVOID", "EXIT_LONG"]
    score: int
    confidence: float
    horizon: str
    entry_zone: EntryZone
    stop_loss: float | None
    take_profit: float | None
    max_position_pct: float
    signals: dict[str, Any]
    reasons: list[str]
    risks: list[str]
    factor_scores: dict[str, float] = Field(default_factory=dict)
    guard_reasons: list[str] = Field(default_factory=list)
    llm_explanation: str | None = None
    strategy_version: str
    data_freshness_seconds: dict[str, int]
    from_cache: bool = False


class HealthResponse(BaseModel):
    status: str
    market_open: bool
    watchlist_size: int
    last_collections: dict[str, str | None]
    db_size_mb: float


class TopSignalsItem(BaseModel):
    ticker: str
    score: int
    recommendation: str
    confidence: float


class AnomalyItem(BaseModel):
    ticker: str
    ts: str
    alert_type: str
    side: str | None
    magnitude: float | None
    description: str | None
    explanation: str | None = None