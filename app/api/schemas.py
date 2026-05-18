"""FastAPI response schemas (mostly re-exports from :mod:`app.strategy.recommendation`)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    db_ok: bool
    db_size_bytes: Optional[int] = None
    moex_token: bool
    polza_token: bool
    arenago_token: bool
    universe_size: int
    last_ingestion_age_sec: Dict[str, Optional[float]]
    strategy_version: str


class AnomalyOut(BaseModel):
    id: int
    secid: str
    ts: datetime
    anomaly_type: str
    severity: float
    source: str
    payload: Optional[Dict[str, Any]] = None
