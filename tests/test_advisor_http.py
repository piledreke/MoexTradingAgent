"""Integration test that POST /advice mirrors TechnicalAdvisor.get_advice."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import create_app
from tests.test_recommendation import _seed_strong_buy


def test_health_endpoint(repo) -> None:
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "strategy_version" in body
    assert isinstance(body["last_ingestion_age_sec"], dict)


def test_advice_endpoint_buy_check(repo) -> None:
    _seed_strong_buy(repo)
    app = create_app()
    client = TestClient(app)
    payload = {"secid": "SBER", "intent": "BUY_CHECK", "intended_cash_rub": 50000}
    r = client.post("/advice", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["secid"] == "SBER"
    assert body["intent"] == "BUY_CHECK"
    assert body["recommended_action"] in {"BUY", "WAIT", "DO_NOT_BUY"}
    assert "data_quality" in body
    assert "risk_flags" in body


def test_advice_endpoint_rejects_unknown_secid() -> None:
    app = create_app()
    client = TestClient(app)
    r = client.post("/advice", json={"secid": "ZZZZ", "intent": "BUY_CHECK"})
    assert r.status_code == 400


def test_recommendations_endpoints(repo) -> None:
    _seed_strong_buy(repo)
    app = create_app()
    client = TestClient(app)
    client.post("/advice", json={"secid": "SBER", "intent": "BUY_CHECK"})
    r = client.get("/recommendations")
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list)
    assert any(item.get("secid") == "SBER" for item in items)
    r = client.get("/recommendations/SBER")
    assert r.status_code == 200
    assert r.json()["secid"] == "SBER"
