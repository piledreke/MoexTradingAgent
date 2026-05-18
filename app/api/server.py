"""FastAPI HTTP server.

Routes are thin: they validate JSON, call the corresponding service method on
:class:`app.strategy.advisor.TechnicalAdvisor` and return the result. Strategy
logic never lives inside route handlers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import AnomalyOut, HealthResponse
from app.clients.arena_client import ArenagoClient
from app.config import Settings, get_settings
from app.logging_config import get_logger, setup_logging
from app.storage.db import get_database
from app.storage.repository import Repository
from app.strategy.advisor import TechnicalAdvisor
from app.strategy.recommendation import AdviceRequest, AdviceResponse

_LOG = get_logger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, json_output=settings.log_json)

    db = get_database(settings)
    repo = Repository(db)
    arena_client = ArenagoClient(settings) if settings.has_arenago_token else None
    advisor = TechnicalAdvisor(repo=repo, settings=settings, arena_client=arena_client)

    app = FastAPI(
        title="MOEX Technical Advisor",
        description="Advisory-only technical agent for MOEX intraday long-only trading.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if db.healthcheck() else "degraded",
            timestamp=datetime.utcnow(),
            db_ok=db.healthcheck(),
            db_size_bytes=db.size_bytes(),
            moex_token=settings.has_moex_token,
            polza_token=settings.has_polza_token,
            arenago_token=settings.has_arenago_token,
            universe_size=len(settings.universe),
            last_ingestion_age_sec=repo.last_ingestion_age(),
            strategy_version=settings.strategy_version,
        )

    # ------------------------------------------------------------------
    @app.get("/recommendations")
    def list_recommendations() -> List[Dict[str, Any]]:
        return repo.get_latest_recommendations()

    @app.get("/recommendations/{secid}")
    def get_recommendation(secid: str) -> Dict[str, Any]:
        secid = secid.upper()
        rec = repo.get_latest_recommendation(secid)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no recommendation for {secid}")
        return rec

    # ------------------------------------------------------------------
    @app.post("/advice", response_model=AdviceResponse)
    def post_advice(payload: AdviceRequest) -> AdviceResponse:
        if payload.secid not in set(settings.universe):
            raise HTTPException(
                status_code=400,
                detail=f"secid {payload.secid} is not in configured UNIVERSE",
            )
        try:
            return advisor.get_advice(payload)
        except Exception as exc:  # pragma: no cover
            _LOG.error("advice_failed", extra={"secid": payload.secid, "error": str(exc)})
            raise HTTPException(status_code=500, detail=f"advice failure: {exc}") from exc

    # ------------------------------------------------------------------
    @app.get("/anomalies", response_model=List[AnomalyOut])
    def list_anomalies(limit: int = 100) -> List[AnomalyOut]:
        rows = repo.get_recent_anomalies(limit=max(1, min(limit, 1000)))
        return [AnomalyOut(**r) for r in rows]

    @app.get("/features/{secid}")
    def get_features(secid: str) -> Dict[str, Any]:
        secid = secid.upper()
        f = repo.get_latest_features(secid)
        if f is None:
            raise HTTPException(status_code=404, detail=f"no features for {secid}")
        return f

    # ------------------------------------------------------------------
    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "moex-tech-agent",
            "strategy_version": settings.strategy_version,
            "universe_size": len(settings.universe),
            "advisory_only": True,
            "places_trades": False,
        }

    return app


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    import uvicorn

    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=settings.log_json)
    uvicorn.run(
        create_app(settings),
        host=host or settings.http_host,
        port=int(port or settings.http_port),
        log_level=settings.log_level.lower(),
    )
