import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path for test discovery/imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add package directory so top-level imports like `api.routes` resolve
PKG_DIR = os.path.abspath(os.path.join(ROOT, "algopack_analyst"))
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from algopack_analyst.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_db(monkeypatch):
    """Prevent tests from touching the real DuckDB / repository."""
    class FakeCursor:
        def fetchone(self):
            return (None,)

        def fetch_df(self):
            import pandas as _pd

            return _pd.DataFrame()

    class FakeConn:
        def execute(self, *args, **kwargs):
            return FakeCursor()

    def fake_get_conn():
        return FakeConn()

    class FakeRepo:
        def log_agent_query(self, *args, **kwargs):
            return None

        def latest_super_candles(self, *args, **kwargs):
            import pandas as _pd

            return _pd.DataFrame()

        def latest_ohlcv(self, *args, **kwargs):
            import pandas as _pd

            return _pd.DataFrame()

        def alerts_window(self, *args, **kwargs):
            import pandas as _pd

            return _pd.DataFrame()

    # Patch both db.get_conn and repository.get_conn (repository imported get_conn at module import)
    monkeypatch.setattr("algopack_analyst.storage.db.get_conn", fake_get_conn)
    monkeypatch.setattr("algopack_analyst.storage.repository.get_conn", fake_get_conn)
    monkeypatch.setattr("algopack_analyst.api.routes.get_repo", lambda: FakeRepo())


@pytest.mark.smoke
def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "market_open" in data
    assert "watchlist_size" in data


@pytest.mark.asyncio
def test_analyze_ticker_with_mock(monkeypatch):
    async def fake_composite_buy_score(ticker, horizon="intraday", context=None):
        return {
            "ticker": ticker.upper(),
            "recommendation": "BUY",
            "score": 72,
            "confidence": 0.81,
            "timestamp": "2026-05-17T14:30:00+03:00",
            "horizon": horizon,
            "signals": {},
            "entry_zone": {"min": 312.5, "max": 313.8},
            "stop_loss": 310.2,
            "take_profit": 318.5,
            "max_position_pct": 10,
            "reasons": ["test"],
            "risks": [],
            "strategy_version": "v1.3.0",
            "regime": {},
            "data_freshness_seconds": {},
            "data_quality": {},
        }

    async def fake_explain_recommendation(rec):
        return "explanation"

    # Patch both package-qualified and top-level module names (imported either way at runtime)
    monkeypatch.setattr(
        "algopack_analyst.api.routes.composite_buy_score", fake_composite_buy_score
    )
    monkeypatch.setattr("api.routes.composite_buy_score", fake_composite_buy_score)
    monkeypatch.setattr(
        "algopack_analyst.api.routes.explain_recommendation", fake_explain_recommendation
    )
    monkeypatch.setattr("api.routes.explain_recommendation", fake_explain_recommendation)

    payload = {"ticker": "SBER", "horizon": "intraday", "context": "test"}
    r = client.post("/analyze/ticker", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == "SBER"
    assert data["recommendation"] == "BUY"
    assert "llm_explanation" in data


def test_market_top_signals_with_mocks(monkeypatch):
    # Provide a small watchlist and fake scoring
    # Patch both api module paths
    monkeypatch.setattr(
        "algopack_analyst.api.routes.get_watchlist",
        lambda: ["SBER", "GAZP"],
    )
    monkeypatch.setattr(
        "api.routes.get_watchlist",
        lambda: ["SBER", "GAZP"],
    )

    async def fake_score(ticker, *args, **kwargs):
        return {"ticker": ticker, "score": 50 + (0 if ticker == "GAZP" else 20),
                "recommendation": "BUY", "confidence": 0.7}

    monkeypatch.setattr(
        "algopack_analyst.api.routes.composite_buy_score",
        fake_score,
    )
    monkeypatch.setattr("api.routes.composite_buy_score", fake_score)

    r = client.get("/market/top_signals?limit=2&direction=bullish")
    assert r.status_code == 200
    items = r.json().get("items", [])
    assert len(items) == 2
    assert any(i["ticker"] == "SBER" for i in items)
