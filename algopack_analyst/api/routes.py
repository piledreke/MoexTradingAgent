"""FastAPI routes — entry point for the main trading agent."""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from analytics.scoring import composite_buy_score
from api.schemas import (
    AnalyzeQueryRequest,
    AnalyzeTickerRequest,
    AnomalyItem,
    HealthResponse,
    Recommendation,
    MoexHealthResponse,
    TopSignalsItem,
    WatchlistAddRequest,
)
from config import settings
from data_collectors.scheduler import add_to_watchlist, get_watchlist
from llm.client import explain_anomaly, explain_recommendation, parse_intent
from storage.repository import get_repo
from utils.logger import logger
from utils.moex_client import get_moex_client

security = HTTPBearer(auto_error=False)
router = APIRouter()


async def require_token(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    if settings.API_BEARER_TOKEN == "change_me":
        # dev mode
        return
    if creds is None or creds.credentials != settings.API_BEARER_TOKEN:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


@router.post("/analyze/ticker", response_model=Recommendation,
             dependencies=[Depends(require_token)])
async def analyze_ticker(req: AnalyzeTickerRequest) -> Recommendation:
    t0 = time.perf_counter()
    rec = await composite_buy_score(req.ticker, req.horizon, req.context)
    if rec["recommendation"] in {"BUY", "STRONG_BUY", "HOLD"}:
        rec["llm_explanation"] = await explain_recommendation(rec)
    latency = (time.perf_counter() - t0) * 1000
    get_repo().log_agent_query(
        query_text=f"analyze:{req.ticker}", response=rec,
        recommendation=rec["recommendation"], latency_ms=latency,
    )
    return Recommendation(**rec)


@router.post("/analyze/query", dependencies=[Depends(require_token)])
async def analyze_query(req: AnalyzeQueryRequest) -> dict[str, Any]:
    intent = await parse_intent(req.query)
    intent_type = intent.get("intent", "unknown")

    if intent_type == "analyze_ticker" and intent.get("ticker"):
        rec = await composite_buy_score(
            intent["ticker"], intent.get("horizon") or "intraday"
        )
        rec["llm_explanation"] = await explain_recommendation(rec)
        return {"intent": intent, "result": rec}

    if intent_type == "top_signals":
        return {"intent": intent, "result": await _top_signals(
            intent.get("limit") or 5, intent.get("direction") or "bullish"
        )}

    if intent_type == "anomalies" and intent.get("ticker"):
        return {"intent": intent, "result": await _anomalies(
            intent["ticker"], intent.get("lookback_minutes") or 60
        )}

    if intent_type == "history" and intent.get("ticker"):
        repo = get_repo()
        return {
            "intent": intent,
            "result": {
                "super_candles": repo.latest_super_candles(intent["ticker"], 20)
                .to_dict(orient="records"),
            },
        }

    return {"intent": intent, "result": None, "error": "unknown intent"}


@router.get("/ticker/{ticker}/history", dependencies=[Depends(require_token)])
async def ticker_history(
    ticker: str,
    type: str = Query("candles"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
) -> dict[str, Any]:
    repo = get_repo()
    ticker = ticker.upper()
    if type == "candles":
        df = repo.latest_ohlcv(ticker, timeframe=1, n=500)
    elif type == "alerts":
        df = repo.alerts_window(ticker, minutes=24 * 60)
    elif type == "futoi":
        df = repo.latest_futoi(ticker, lookback_min=24 * 60)
    elif type == "hi2":
        df = repo.latest_hi2(ticker)
    else:
        raise HTTPException(400, f"unknown type: {type}")
    return {"ticker": ticker, "type": type, "data": df.to_dict(orient="records")}


@router.get("/ticker/{ticker}/anomalies", dependencies=[Depends(require_token)])
async def ticker_anomalies(
    ticker: str, lookback_minutes: int = Query(60, ge=1, le=24 * 60)
) -> dict[str, Any]:
    return {"ticker": ticker.upper(),
            "anomalies": await _anomalies(ticker.upper(), lookback_minutes)}


@router.get("/ticker/{ticker}/features", dependencies=[Depends(require_token)])
async def ticker_features(ticker: str, timeframe: int = 5):
    """Latest computed feature snapshot."""
    from storage.feature_store import get_feature_store
    snap = get_feature_store().get_latest(ticker.upper(), timeframe)
    if not snap:
        raise HTTPException(404, "no features yet")
    return {"ticker": ticker, "timeframe": timeframe,
            "ts": snap.ts.isoformat(), "features": snap.features}


@router.get("/ticker/{ticker}/regime", dependencies=[Depends(require_token)])
async def ticker_regime(ticker: str):
    """Current market regime detection."""
    from analytics.regime import detect_regime
    from storage.snapshot import get_snapshot_builder
    snap = await get_snapshot_builder().build(ticker.upper(), include_cross_asset=False)
    regime = detect_regime(ticker.upper(), snap.ohlcv.get(5), snap.ohlcv.get(60))
    return regime.as_dict()


@router.get("/ticker/{ticker}/volume_profile", dependencies=[Depends(require_token)])
async def ticker_volume_profile(ticker: str, timeframe: int = 15, bins: int = 50):
    from analytics.volume_profile import compute_volume_profile
    df = get_repo().latest_ohlcv(ticker.upper(), timeframe, n=500)
    if df.empty:
        raise HTTPException(404, "no data")
    return compute_volume_profile(df.sort_values("ts"), bins=bins)


@router.get("/ticker/{ticker}/order_flow", dependencies=[Depends(require_token)])
async def ticker_order_flow(ticker: str):
    from analytics.order_flow import (
        absorption, buy_pressure_score, cumulative_delta, delta_divergence
    )
    sc = get_repo().latest_super_candles(ticker.upper(), n=30)
    if sc.empty:
        raise HTTPException(404, "no super candles")
    cd = cumulative_delta(sc)
    return {
        "buy_pressure": buy_pressure_score(sc),
        "delta_divergence": delta_divergence(sc),
        "absorption": absorption(sc),
        "cumulative_delta_latest": float(cd.iloc[-1]) if not cd.empty else 0,
    }


@router.get("/ticker/{ticker}/snapshot", dependencies=[Depends(require_token)])
async def ticker_snapshot(ticker: str):
    """Full atomic snapshot (for debugging / audit)."""
    from storage.snapshot import get_snapshot_builder
    snap = await get_snapshot_builder().build(ticker.upper())
    return {
        "ticker": snap.ticker,
        "snapshot_id": snap.snapshot_id,
        "captured_at": snap.captured_at.isoformat(),
        "market_open": snap.market_open,
        "suspended": snap.suspended,
        "freshness": snap.freshness,
        "quality": snap.quality,
        "available_data": {
            "ohlcv_timeframes": list(snap.ohlcv.keys()),
            "super_candles_rows": len(snap.super_candles),
            "obstats_rows": len(snap.obstats),
            "orderbook": snap.orderbook is not None,
            "alerts_rows": len(snap.alerts),
            "futoi_rows": len(snap.futoi),
            "hi2_rows": len(snap.hi2),
            "indices": list(snap.index_ohlcv.keys()),
        },
    }


@router.get("/market/top_signals", dependencies=[Depends(require_token)])
async def market_top_signals(
    limit: int = Query(10, ge=1, le=30),
    direction: str = Query("bullish"),
) -> dict[str, Any]:
    return {"items": await _top_signals(limit, direction)}


@router.post("/watch/add", dependencies=[Depends(require_token)])
async def watch_add(req: WatchlistAddRequest) -> dict[str, Any]:
    client = get_moex_client()
    valid: list[str] = []
    for t in req.tickers:
        if await client.is_tradable_tqbr(t):
            valid.append(t.upper())
    new_list = add_to_watchlist(valid)
    return {"added": valid, "watchlist_size": len(new_list)}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    client = get_moex_client()
    try:
        market_open = await client.is_market_open()
    except Exception:
        market_open = False

    repo = get_repo()
    last: dict[str, str | None] = {}
    for table in ("super_candles_eq", "mega_alerts", "futoi",
                  "orderbook_snapshots", "ohlcv"):
        try:
            conn = repo  # reuse via get_conn
            from storage.db import get_conn

            row = get_conn().execute(
                f"SELECT MAX(ts) FROM {table}"
            ).fetchone()
            last[table] = str(row[0]) if row and row[0] else None
        except Exception:
            last[table] = None

    try:
        size_mb = Path(settings.DUCKDB_PATH).stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = 0.0

    return HealthResponse(
        status="ok",
        market_open=market_open,
        watchlist_size=len(get_watchlist()),
        last_collections=last,
        db_size_mb=round(size_mb, 2),
    )


@router.get("/health/moex", response_model=MoexHealthResponse)
async def moex_health() -> MoexHealthResponse:
    client = get_moex_client()
    public_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities/LKOH/orderbook.json"
    auth_url = "https://apim.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities/LKOH/orderbook.json"

    async def _probe(url: str, token: str | None = None) -> tuple[str, str | None, str | None, str | None]:
        import aiohttp

        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                content_type = resp.headers.get("Content-Type")
                text = await resp.text()
                sample = text[:160].replace("\n", " ")
                if resp.status >= 400:
                    return ("error", content_type, sample, f"HTTP {resp.status}")
                if "application/json" not in (content_type or "").lower():
                    return ("html", content_type, sample, None)
                return ("ok", content_type, sample, None)

    public_status, public_ct, public_sample, public_err = await _probe(public_url)
    auth_status = None
    auth_ct = None
    auth_sample = None

    if client._token:
        auth_status, auth_ct, auth_sample, _ = await _probe(auth_url, client._token)

    explanation = (
        "Public ISS orderbook endpoint returns HTML denial pages in this environment; "
        "ALGOPACK with a valid token returns JSON."
        if public_status == "html" and auth_status == "ok"
        else "Orderbook health probe completed; inspect content type and sample for the active mode."
    )

    return MoexHealthResponse(
        status="ok" if auth_status == "ok" or public_status == "ok" else "degraded",
        algopack_token_present=bool(client._token),
        public_orderbook_status=public_status if public_status != "error" else f"error: {public_err}",
        algopack_orderbook_status=auth_status,
        public_orderbook_content_type=public_ct,
        algopack_orderbook_content_type=auth_ct,
        public_orderbook_sample=public_sample,
        algopack_orderbook_sample=auth_sample,
        explanation=explanation,
    )


# ─── internal helpers ──────────────────────────────────
async def _top_signals(limit: int, direction: str) -> list[dict]:
    tickers = get_watchlist()
    results = []
    for t in tickers:
        try:
            rec = await composite_buy_score(t)
            results.append(rec)
        except Exception as e:
            logger.warning(f"top_signals score for {t} failed: {e}")
    if direction == "bullish":
        results.sort(key=lambda r: r["score"], reverse=True)
    else:
        results.sort(key=lambda r: r["score"])
    out = []
    for r in results[:limit]:
        out.append({
            "ticker": r["ticker"],
            "score": r["score"],
            "recommendation": r["recommendation"],
            "confidence": r["confidence"],
        })
    return out


async def _anomalies(ticker: str, lookback_minutes: int) -> list[dict]:
    repo = get_repo()
    df = repo.alerts_window(ticker, lookback_minutes)
    items: list[dict] = []
    for _, row in df.iterrows():
        item = {
            "ticker": ticker,
            "ts": str(row.get("ts")),
            "alert_type": row.get("alert_type"),
            "side": row.get("side"),
            "magnitude": float(row.get("magnitude") or 0),
            "description": row.get("description"),
        }
        try:
            item["explanation"] = await explain_anomaly(item)
        except Exception:
            item["explanation"] = None
        items.append(item)
    return items