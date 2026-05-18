"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force every test to use a temporary SQLite file and no real tokens."""
    tmp = tempfile.mkdtemp(prefix="moex-tech-agent-test-")
    db_path = os.path.join(tmp, "tech.sqlite3")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setenv("ENABLE_HTTP", "false")
    monkeypatch.delenv("MOEX_API_KEY", raising=False)
    monkeypatch.delenv("MOEX_ALGOPACK_TOKEN", raising=False)
    monkeypatch.delenv("POLZA_AI_API_KEY", raising=False)
    monkeypatch.delenv("ARENAGO_TOKEN", raising=False)
    # Force reload of cached settings + DB singleton.
    from app.config import reload_settings
    from app.storage.db import reset_database

    reload_settings()
    reset_database(db_path)
    yield
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def repo():
    from app.config import get_settings
    from app.storage.db import get_database
    from app.storage.repository import Repository

    db = get_database(get_settings())
    return Repository(db)
