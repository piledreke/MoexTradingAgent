"""Sanity tests: the technical agent must never trade via Arenago."""

from __future__ import annotations

import pytest

from app.clients.arena_client import ArenagoClient, ArenagoForbiddenError


def test_submit_order_is_always_forbidden() -> None:
    client = ArenagoClient()
    with pytest.raises(ArenagoForbiddenError):
        client.submit_order(secid="SBER", quantity=1, price=100)


def test_get_with_forbidden_path_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENAGO_TOKEN", "fake-test-token")
    from app.config import reload_settings

    reload_settings()
    client = ArenagoClient()
    with pytest.raises(ArenagoForbiddenError):
        client._get("/api/submit_order")
