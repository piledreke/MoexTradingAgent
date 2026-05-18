"""Centralized configuration via Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── Versioning ─────────────────────────────────────
    STRATEGY_VERSION: str = "v1.2.0"
    SCORING_VERSION: str = "v1.2.0"
    RANDOM_SEED: int = 42

    # ─── MOEX ───────────────────────────────────────────
    MOEX_ALGOPACK_TOKEN: str | None = None
    MOEX_ALGOPACK_BASE: str = "https://apim.moex.com/iss"
    MOEX_PUBLIC_BASE: str = "https://iss.moex.com/iss"
    MOEX_MAX_CONCURRENT: int = 10
    MOEX_TIMEOUT: int = 30
    MOEX_CACHE_TTL: int = 5

    # ─── LLM ────────────────────────────────────────────
    POLZA_API_KEY: str | None = None
    POLZA_BASE_URL: str = "https://polza.ai/api/v1"
    LLM_MODEL: str = "openai/gpt-4o"
    LLM_TIMEOUT: int = 30
    LLM_MAX_TOKENS: int = 1024

    # ─── API ────────────────────────────────────────────
    API_BEARER_TOKEN: str = "change_me"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ─── Storage ────────────────────────────────────────
    DUCKDB_PATH: str = "./data/analyst.duckdb"
    PARQUET_COLD_DIR: str = "./data/cold"
    HOT_DATA_RETENTION_DAYS: int = 3
    DB_SIZE_THRESHOLD_GB: float = 7.0

    # ─── Logging ────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_ROTATION_MB: int = 100
    LOG_RETENTION_COUNT: int = 5

    # ─── Collectors ─────────────────────────────────────
    ENABLE_COLLECTORS: bool = True
    WATCHLIST: str = (
        "SBER,GAZP,LKOH,GMKN,ROSN,YNDX,VTBR,TATN,MGNT,"
        "MTSS,ALRS,CHMF,NVTK,PLZL,POLY,SNGS,AFLT,MOEX"
    )
    MAX_WATCHLIST_SIZE: int = 50

    # Intervals (seconds)
    SUPER_CANDLES_INTERVAL: int = 300
    FUTOI_INTERVAL: int = 300
    MEGA_ALERTS_INTERVAL: int = 60
    OHLCV_INTERVAL: int = 60
    ORDERBOOK_INTERVAL: int = 10
    HI2_CRON_HOUR: int = 19
    HI2_CRON_MINUTE: int = 5

    # ─── Scoring weights ────────────────────────────────
    W_TREND: float = 0.25
    W_MOMENTUM: float = 0.15
    W_VOLUME_PROFILE: float = 0.20
    W_ORDERBOOK_IMBALANCE: float = 0.10
    W_MEGA_ALERTS: float = 0.15
    W_HI2: float = 0.05
    W_FUTOI: float = 0.10

    # Score thresholds
    SCORE_STRONG_BUY: int = 80
    SCORE_BUY: int = 60
    SCORE_HOLD: int = 40

    # Risk caps
    MAX_POS_PCT_DEFAULT: float = 10.0
    MAX_POS_PCT_HIGH_HI2: float = 5.0
    MAX_POS_PCT_NEGATIVE_ALERTS: float = 3.0
    MAX_POS_PCT_STRONG_SIGNAL: float = 15.0
    MAX_POS_PCT_WIDE_SPREAD: float = 3.0

    # Freshness thresholds (sec)
    FRESH_SUPER_CANDLES: int = 600
    FRESH_ORDERBOOK: int = 30
    FRESH_ALERTS: int = 120

    # Dedup
    RECOMMENDATION_TTL_SEC: int = 300

    # ─── Helpers ────────────────────────────────────────
    @property
    def watchlist_list(self) -> list[str]:
        return [t.strip().upper() for t in self.WATCHLIST.split(",") if t.strip()]

    @property
    def use_algopack(self) -> bool:
        return bool(self.MOEX_ALGOPACK_TOKEN)

    def ensure_dirs(self) -> None:
        for path in (self.DUCKDB_PATH, self.PARQUET_COLD_DIR, self.LOG_DIR):
            p = Path(path)
            if p.suffix:  # file path
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)


# ─── Static mappings ────────────────────────────────────
STOCK_TO_FUTURE: dict[str, str] = {
    "SBER": "SBRF", "GAZP": "GAZR", "LKOH": "LKOH", "GMKN": "GMKN",
    "ROSN": "ROSN", "YNDX": "YNDX", "VTBR": "VTBR", "TATN": "TATN",
    "MGNT": "MGNT", "MTSS": "MTSI", "ALRS": "ALRS", "CHMF": "CHMF",
    "NVTK": "NOTK", "PLZL": "PLZL", "POLY": "POLY", "SNGS": "SNGR",
    "AFLT": "AFLT", "MOEX": "MOEX",
}

FUTURE_UNDERLYING: dict[str, str] = {
    "Si": "USD/RUB", "Eu": "EUR/RUB", "CNY": "CNY/RUB",
    "RI": "RTSI", "MX": "IMOEX",
    "BR": "Brent", "GD": "Gold", "SV": "Silver", "NG": "Natural Gas",
}

ALERT_IMPACT: dict[str, float] = {
    "big_buy_order": +0.15,
    "big_sell_order": -0.20,
    "price_spike_up": +0.10,
    "price_spike_down": -0.15,
    "unusual_volume": 0.0,
    "iceberg_buy": +0.10,
    "iceberg_sell": -0.10,
    "wash_trade": -0.05,
    "absorption_buy": +0.12,
    "absorption_sell": -0.12,
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
settings.ensure_dirs()

# Добавить в Settings:

# Volatility regime thresholds (annualized %)
LOW_VOLATILITY_THRESHOLD: float = 12.0
NORMAL_VOLATILITY_THRESHOLD: float = 20.0
HIGH_VOLATILITY_THRESHOLD: float = 35.0

# Concentration thresholds
HHI_CRITICAL_THRESHOLD: float = 3500
HHI_HIGH_THRESHOLD: float = 2500
LOW_VOLUME_THRESHOLD_RUB: float = 50_000_000

# MTF collection intervals (seconds)
MTF_INTERVAL: int = 60          # collect 1m every minute
MTF_15M_INTERVAL: int = 900     # 15m every 15min
MTF_1H_INTERVAL: int = 3600     # 1h every hour
MTF_1D_INTERVAL: int = 21600    # 1d every 6 hours
TAPE_INTERVAL: int = 30         # tape every 30 sec
INDEX_INTERVAL: int = 300       # index data every 5 min

# Feature store
FEATURE_STORE_ENABLED: bool = True