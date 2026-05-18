"""FastAPI application entry point."""
from __future__ import annotations
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings
from data_collectors.scheduler import shutdown_scheduler, start_scheduler
from storage.db import close_db, get_conn
from utils.logger import logger
from utils.moex_client import get_moex_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 ALGOPACK Analyst starting")
    get_conn()
    get_moex_client()
    if settings.ENABLE_COLLECTORS:
        start_scheduler()
    yield
    logger.info("🛑 Shutting down")
    await shutdown_scheduler()
    client = get_moex_client()
    await client.close()
    close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ALGOPACK Analyst Agent",
        version=settings.STRATEGY_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    # Let Uvicorn install and manage signal handlers for graceful shutdown.
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )