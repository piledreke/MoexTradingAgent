"""Centralised configuration loaded strictly from environment variables.

No secret is ever hard-coded. The same :class:`Settings` instance is shared by
clients, storage, ingestion and strategy modules.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_UNIVERSE = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T", "NVTK", "X5",
    "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "MOEX", "SNGSP", "MTSS", "PIKK",
]


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # MOEX / ALGOPACK
    # ------------------------------------------------------------------
    moex_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("MOEX_API_KEY", "MOEX_TOKEN"),
    )
    moex_algopack_token: Optional[str] = Field(default=None, alias="MOEX_ALGOPACK_TOKEN")
    moex_base_url: str = Field(default="https://apim.moex.com/iss", alias="MOEX_BASE_URL")
    moex_public_base_url: str = Field(default="https://iss.moex.com/iss", alias="MOEX_PUBLIC_BASE_URL")
    moex_request_timeout: float = Field(default=10.0, alias="MOEX_REQUEST_TIMEOUT")
    moex_max_retries: int = Field(default=4, alias="MOEX_MAX_RETRIES")

    # ------------------------------------------------------------------
    # Polza AI
    # ------------------------------------------------------------------
    polza_ai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("POLZA_AI_API_KEY", "POLZA_API_KEY"),
    )
    polza_base_url: str = Field(default="https://polza.ai/api/v1", alias="POLZA_BASE_URL")
    polza_model: str = Field(
        default="openai/gpt-4o",
        validation_alias=AliasChoices("POLZA_MODEL", "LLM_MODEL"),
    )
    polza_timeout: float = Field(default=20.0, alias="POLZA_TIMEOUT")
    enable_llm: bool = Field(default=True, alias="ENABLE_LLM")

    # ------------------------------------------------------------------
    # Arenago (read-only)
    # ------------------------------------------------------------------
    arenago_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ARENAGO_TOKEN", "API_BEARER_TOKEN"),
    )
    arenago_base_url: str = Field(default="https://arenago.ru", alias="ARENAGO_BASE_URL")
    arenago_timeout: float = Field(default=10.0, alias="ARENAGO_TIMEOUT")
    bot_name: Optional[str] = Field(default=None, alias="BOT_NAME")

    # ------------------------------------------------------------------
    # Storage / runtime
    # ------------------------------------------------------------------
    db_path: str = Field(default="/app/data/tech_agent.sqlite3", alias="DB_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")
    raw_cache_retention_days: int = Field(default=7, alias="RAW_CACHE_RETENTION_DAYS")

    # ------------------------------------------------------------------
    # HTTP API
    # ------------------------------------------------------------------
    enable_http: bool = Field(default=True, alias="ENABLE_HTTP")
    http_host: str = Field(default="0.0.0.0", alias="HTTP_HOST")
    http_port: int = Field(default=8080, alias="HTTP_PORT")

    # ------------------------------------------------------------------
    # Universe & strategy
    # ------------------------------------------------------------------
    universe_raw: str = Field(
        default=",".join(DEFAULT_UNIVERSE),
        validation_alias=AliasChoices("UNIVERSE", "WATCHLIST"),
    )
    strategy_version: str = Field(default="det-1.0", alias="STRATEGY_VERSION")

    backfill_days: int = Field(default=30, alias="BACKFILL_DAYS")
    poll_interval_seconds: int = Field(default=60, alias="POLL_INTERVAL_SECONDS")
    super_candle_interval_seconds: int = Field(default=300, alias="SUPER_CANDLE_INTERVAL_SECONDS")
    hi2_hour_msk: int = Field(default=18, alias="HI2_HOUR_MSK")
    hi2_minute_msk: int = Field(default=45, alias="HI2_MINUTE_MSK")

    # ------------------------------------------------------------------
    # Risk limits
    # ------------------------------------------------------------------
    initial_capital_rub: float = Field(default=1_000_000.0, alias="INITIAL_CAPITAL_RUB")
    daily_trade_limit: int = Field(default=200, alias="DAILY_TRADE_LIMIT")
    max_position_pct: float = Field(default=0.10, alias="MAX_POSITION_PCT")
    max_single_order_pct: float = Field(default=0.05, alias="MAX_SINGLE_ORDER_PCT")
    max_portfolio_exposure_pct: float = Field(default=0.95, alias="MAX_PORTFOLIO_EXPOSURE_PCT")
    reserve_cash_pct: float = Field(default=0.05, alias="RESERVE_CASH_PCT")
    min_order_cash_rub: float = Field(default=5_000.0, alias="MIN_ORDER_CASH_RUB")

    # ------------------------------------------------------------------
    # Decision thresholds
    # ------------------------------------------------------------------
    buy_score_threshold: float = Field(default=70.0, alias="BUY_SCORE_THRESHOLD")
    min_confidence_buy: float = Field(default=0.55, alias="MIN_CONFIDENCE_BUY")
    hold_score_threshold: float = Field(default=45.0, alias="HOLD_SCORE_THRESHOLD")

    max_spread_bbo_bps: float = Field(default=80.0, alias="MAX_SPREAD_BBO_BPS")
    max_spread_1mio_bps: float = Field(default=200.0, alias="MAX_SPREAD_1MIO_BPS")
    max_stale_seconds: float = Field(default=900.0, alias="MAX_STALE_SECONDS")
    min_liquidity_val_b: float = Field(default=50_000.0, alias="MIN_LIQUIDITY_VAL_B")

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def universe(self) -> List[str]:
        items = [s.strip().upper() for s in self.universe_raw.split(",") if s.strip()]
        # preserve order, deduplicate
        seen = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    @property
    def moex_token(self) -> Optional[str]:
        """Return effective MOEX bearer token from any of the env aliases."""
        return self.moex_api_key or self.moex_algopack_token

    @property
    def has_moex_token(self) -> bool:
        return bool(self.moex_token)

    @property
    def has_polza_token(self) -> bool:
        return bool(self.polza_ai_api_key)

    @property
    def has_arenago_token(self) -> bool:
        return bool(self.arenago_token)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        if not isinstance(value, str):
            return "INFO"
        return value.upper().strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor used across the codebase."""
    return Settings()


def reload_settings() -> Settings:
    """Force-reload settings (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()


def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "y", "t")
