"""Pydantic schemas for the technical advisor API.

These are the single source of truth for:
* the Python service class ``TechnicalAdvisor.get_advice``;
* the FastAPI endpoint ``POST /advice``;
* the database recommendation payloads.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AdviceIntent(str, Enum):
    BUY_CHECK = "BUY_CHECK"
    EXIT_CHECK = "EXIT_CHECK"


class BuyAction(str, Enum):
    BUY = "BUY"
    WAIT = "WAIT"
    DO_NOT_BUY = "DO_NOT_BUY"


class ExitAction(str, Enum):
    HOLD_POSITION = "HOLD_POSITION"
    TRIM_POSITION = "TRIM_POSITION"
    EXIT_POSITION = "EXIT_POSITION"


class ExitUrgency(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AggregateAction(str, Enum):
    """Legacy compact view for the news-driven agent."""

    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


# ---------------------------------------------------------------------------
class PortfolioState(BaseModel):
    """Snapshot of the trader's portfolio (read-only, from Arenago or caller)."""

    model_config = ConfigDict(extra="ignore")

    cash_rub: float = Field(default=0.0, ge=0.0)
    equity_rub: float = Field(default=0.0, ge=0.0)
    positions_value_rub: float = Field(default=0.0, ge=0.0)
    daily_trades_count: int = Field(default=0, ge=0)
    daily_trade_limit: int = Field(default=200, ge=0)


class CurrentPosition(BaseModel):
    """Snapshot of an existing long position for EXIT_CHECK."""

    model_config = ConfigDict(extra="ignore")

    quantity: int = Field(default=0, ge=0)
    average_price: float = Field(default=0.0, ge=0.0)
    market_value_rub: float = Field(default=0.0, ge=0.0)
    unrealized_pnl_pct: float = Field(default=0.0)


# ---------------------------------------------------------------------------
class AdviceRequest(BaseModel):
    """Request body for ``POST /advice`` and :meth:`TechnicalAdvisor.get_advice`.

    The portfolio can be supplied in two ways:

    * ``portfolio`` as an inline :class:`PortfolioState` dict — used as-is.
    * ``portfolio_name`` as the Arenago portfolio name — looked up live (read-only)
      by the advisor at evaluation time.

    Likewise for the current position: pass ``position`` inline or let the
    advisor pull it from Arenago by portfolio name.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    secid: str
    intent: AdviceIntent = AdviceIntent.BUY_CHECK
    news_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    intended_cash_rub: Optional[float] = Field(default=None, ge=0.0)
    horizon_minutes: int = Field(default=60, ge=1, le=24 * 60)

    portfolio_name: Optional[str] = Field(
        default=None,
        description="Arenago portfolio name for live read-only lookup.",
    )
    portfolio: Optional[PortfolioState] = Field(
        default=None,
        validation_alias=AliasChoices("portfolio", "portfolio_state"),
        description="Inline portfolio snapshot (alternative to portfolio_name).",
    )
    position: Optional[CurrentPosition] = Field(
        default=None,
        validation_alias=AliasChoices("position", "current_position"),
        description="Inline current position for EXIT_CHECK or BUY_CHECK sizing.",
    )

    @field_validator("secid")
    @classmethod
    def _normalize_secid(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("secid is required")
        return v.strip().upper()


# ---------------------------------------------------------------------------
class AdviceResponse(BaseModel):
    """Unified response covering both BUY_CHECK and EXIT_CHECK intents."""

    model_config = ConfigDict(extra="ignore")

    secid: str
    timestamp: datetime
    intent: AdviceIntent
    strategy_version: str

    # Aggregate "legacy" action so the news agent can keep simple BUY/HOLD/AVOID checks.
    action: AggregateAction
    # Detailed intent-specific action.
    recommended_action: str

    allow_buy: bool = False
    allow_action: bool = False
    exit_warning: bool = False
    exit_urgency: ExitUrgency = ExitUrgency.NONE

    technical_score: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    position_size_multiplier: float = Field(default=0.0, ge=0.0, le=1.0)

    horizon_minutes: int = 60
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    max_cash_rub: Optional[float] = None
    recommended_cash_rub: Optional[float] = None
    recommended_quantity: Optional[int] = None
    recommended_sell_quantity: Optional[int] = None

    reasons: List[str] = Field(default_factory=list)
    negative_factors: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)

    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    portfolio_constraints: Dict[str, Any] = Field(default_factory=dict)

    llm_used: bool = False
    llm_comment: Optional[str] = None
