"""Tests for TechnicalAdvisor._resolve_portfolio_context against the real
Arenago payload shape (secid/position/average_price/direction)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from app.storage.repository import Repository
from app.strategy.advisor import TechnicalAdvisor


class FakeArena:
    enabled = True

    def __init__(self, bots: List[Dict[str, Any]], positions: List[Dict[str, Any]], trades: Optional[List[Dict[str, Any]]] = None):
        self._bots = bots
        self._positions = positions
        self._trades = trades or []

    def get_bots(self) -> List[Dict[str, Any]]:
        return self._bots

    def get_positions(self, _name: str) -> List[Dict[str, Any]]:
        return self._positions

    def get_trades(self, _name: str) -> List[Dict[str, Any]]:
        return self._trades


def _make_advisor(repo: Repository, arena: FakeArena) -> TechnicalAdvisor:
    advisor = TechnicalAdvisor(repo=repo, arena_client=arena)
    return advisor


def test_resolver_parses_real_arenago_payload(repo: Repository) -> None:
    """Replicates the exact shape returned by arenago.ru today."""
    bots = [{"cash_balance": 497839.07, "name": "Байкальск"}]
    positions = [
        {
            "average_price": 89.285,
            "bot": "Байкальск",
            "direction": "B",
            "id": 11200,
            "nickname": "piledreke",
            "position": 2220,
            "secid": "VTBR",
            "updatedate": "2026-05-15",
            "updatetime": "15:24:50.648972",
        },
        {
            "average_price": 539.172073677956,
            "bot": "Байкальск",
            "direction": "B",
            "id": 11182,
            "nickname": "piledreke",
            "position": 367,
            "secid": "PIKK",
            "updatedate": "2026-05-15",
            "updatetime": "15:24:33.203000",
        },
        {
            "average_price": 413.2,
            "bot": "Байкальск",
            "direction": "B",
            "id": 11170,
            "nickname": "piledreke",
            "position": 240,
            "secid": "ROSN",
            "updatedate": "2026-05-15",
            "updatetime": "09:35:28.573992",
        },
    ]
    arena = FakeArena(bots=bots, positions=positions)
    advisor = _make_advisor(repo, arena)

    # Sanity: querying for VTBR returns the matching position.
    ps, cur = advisor._resolve_portfolio_context(
        portfolio_state=None,
        current_position=None,
        portfolio_name="Байкальск",
        secid="VTBR",
    )
    assert ps is not None
    assert abs(ps.cash_rub - 497839.07) < 0.01
    assert ps.daily_trades_count == 0
    # positions_value comes from qty*ref_price; with no marketdata in DB ref_price
    # falls back to average_price → 2220*89.285 + 367*539.172… + 240*413.2 ≈ 496,985
    assert ps.positions_value_rub > 400_000
    assert cur is not None
    assert cur.quantity == 2220
    assert abs(cur.average_price - 89.285) < 1e-6

    # Querying for a non-held ticker → no current position.
    _, cur_sber = advisor._resolve_portfolio_context(
        portfolio_state=None,
        current_position=None,
        portfolio_name="Байкальск",
        secid="SBER",
    )
    assert cur_sber is None


def test_resolver_ignores_short_positions_for_safety(repo: Repository) -> None:
    """Any direction != 'B' MUST NOT register as a buyable / holdable long."""
    bots = [{"cash_balance": 100_000.0, "name": "T1"}]
    positions = [
        {"secid": "VTBR", "position": 1000, "average_price": 90.0, "direction": "S"},  # short -- ignored
        {"secid": "SBER", "position": 100, "average_price": 300.0, "direction": "B"},
    ]
    arena = FakeArena(bots=bots, positions=positions)
    advisor = _make_advisor(repo, arena)

    ps, cur = advisor._resolve_portfolio_context(
        portfolio_state=None,
        current_position=None,
        portfolio_name="T1",
        secid="VTBR",
    )
    # No long VTBR position must be reported.
    assert cur is None
    # positions_value reflects only the SBER long (100 * 300 = 30k).
    assert ps is not None
    assert abs(ps.positions_value_rub - 30000.0) < 1e-3


def test_resolver_uses_marketdata_for_current_price_and_pnl(repo: Repository) -> None:
    """If we have live marketdata, PnL is computed against the live price."""
    repo.save_marketdata_snapshot([
        {
            "secid": "VTBR",
            "last": 95.0,
            "open": 93.0,
            "low": 92.0,
            "high": 96.0,
            "updated_at": datetime.utcnow(),
        }
    ])
    bots = [{"cash_balance": 50_000.0, "name": "T2"}]
    positions = [
        {"secid": "VTBR", "position": 100, "average_price": 90.0, "direction": "B"},
    ]
    arena = FakeArena(bots=bots, positions=positions)
    advisor = _make_advisor(repo, arena)

    ps, cur = advisor._resolve_portfolio_context(
        portfolio_state=None,
        current_position=None,
        portfolio_name="T2",
        secid="VTBR",
    )
    assert cur is not None
    # PnL ≈ (95-90)/90 * 100 ≈ 5.555
    assert 5.0 < cur.unrealized_pnl_pct < 6.0
    # market value should use live price = 100 * 95 = 9500
    assert abs(ps.positions_value_rub - 9500.0) < 1e-3


def test_resolver_returns_none_when_arena_disabled(repo: Repository) -> None:
    """When no Arenago token is configured, the resolver returns ``None,None``."""
    arena = MagicMock()
    arena.enabled = False
    advisor = TechnicalAdvisor(repo=repo, arena_client=arena)
    ps, cur = advisor._resolve_portfolio_context(
        portfolio_state=None,
        current_position=None,
        portfolio_name="Байкальск",
        secid="VTBR",
    )
    assert ps is None and cur is None
