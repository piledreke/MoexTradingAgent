"""CLI entrypoint.

Run ``python -m app.main --help`` for the full command list. The CLI is built
on Typer for nice UX, but all commands also work in plain argparse-style.
"""

from __future__ import annotations

import json
import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.backtest import Backtester, parse_date
from app.clients.arena_client import ArenagoClient
from app.clients.moex_client import MoexClient
from app.clients.polza_client import PolzaClient
from app.config import get_settings
from app.ingestion.backfill import Backfiller
from app.ingestion.scheduler import IngestionService
from app.logging_config import get_logger, setup_logging
from app.storage.db import get_database
from app.storage.repository import Repository
from app.strategy.advisor import TechnicalAdvisor

cli = typer.Typer(add_completion=False, help="MOEX Technical Advisor – advisory-only.")
console = Console()
_LOG = get_logger(__name__)


def _init_components():
    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=settings.log_json)
    db = get_database(settings)
    repo = Repository(db)
    moex = MoexClient(settings)
    arena = ArenagoClient(settings) if settings.has_arenago_token else None
    advisor = TechnicalAdvisor(repo=repo, settings=settings, arena_client=arena)
    return settings, db, repo, moex, arena, advisor


# ---------------------------------------------------------------------------
@cli.command()
def once() -> None:
    """Run one ingestion + recommendation cycle."""
    settings, db, repo, moex, arena, advisor = _init_components()
    service = IngestionService(repo=repo, client=moex, settings=settings)
    if settings.backfill_days > 0:
        service.lightweight_backfill_if_empty(days=min(5, settings.backfill_days))
    result = service.run_once(refresh_candles=True)
    recs = advisor.generate_for_universe()
    console.print(
        f"[bold green]Cycle done.[/] fetched={result.fetched} errors={result.errors} "
        f"recommendations={len(recs)}"
    )
    moex.close()
    if arena:
        arena.close()


# ---------------------------------------------------------------------------
@cli.command()
def run() -> None:
    """Endless ingest + recommend loop (blocks)."""
    settings, db, repo, moex, arena, advisor = _init_components()
    service = IngestionService(repo=repo, client=moex, settings=settings)
    if settings.backfill_days > 0:
        service.lightweight_backfill_if_empty(days=min(5, settings.backfill_days))

    http_thread = None
    if settings.enable_http:
        import threading

        from app.api.server import run_server

        http_thread = threading.Thread(target=run_server, daemon=True)
        http_thread.start()
        _LOG.info("http_thread_started", extra={"host": settings.http_host, "port": settings.http_port})

    def _on_cycle(_result) -> None:
        try:
            advisor.generate_for_universe()
        except Exception as exc:
            _LOG.warning("generate_universe_failed", extra={"error": str(exc)})

    def _shutdown(*_args) -> None:
        _LOG.info("scheduler_stopping")
        service.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        service.run_forever(on_cycle=_on_cycle)
    finally:
        moex.close()
        if arena:
            arena.close()


# ---------------------------------------------------------------------------
@cli.command()
def backfill(
    days: int = typer.Option(30, "--days", help="How many days of history to load."),
) -> None:
    """Historical backfill for the universe."""
    settings, db, repo, moex, arena, _advisor = _init_components()
    bf = Backfiller(repo=repo, client=moex, settings=settings)
    res = bf.backfill_universe(settings.universe, days=days)
    table = Table(title=f"Backfill ({days} days)")
    table.add_column("secid")
    cols = ("tradestats", "orderstats", "obstats", "alerts", "hi2", "candles")
    for k in cols:
        table.add_column(k)
    for secid, counts in res.items():
        table.add_row(secid, *[str(counts.get(k, 0)) for k in cols])
    console.print(table)
    moex.close()
    if arena:
        arena.close()


# ---------------------------------------------------------------------------
@cli.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
) -> None:
    """Start the FastAPI HTTP server (foreground)."""
    from app.api.server import run_server

    run_server(host=host, port=port)


# ---------------------------------------------------------------------------
@cli.command()
def doctor() -> None:
    """Verify environment, network, DB. Never raises."""
    settings, db, repo, moex, arena, _advisor = _init_components()
    table = Table(title="Doctor")
    table.add_column("Check")
    table.add_column("OK")
    table.add_column("Details", overflow="fold")

    def _check(name: str, ok: bool, details: str) -> None:
        table.add_row(name, "✓" if ok else "✗", details)

    # ENV.
    _check(
        "MOEX_API_KEY",
        settings.has_moex_token,
        "set" if settings.has_moex_token else "missing - ALGOPACK endpoints will be skipped",
    )
    _check(
        "POLZA_AI_API_KEY",
        settings.has_polza_token,
        "set (LLM enabled)" if settings.has_polza_token else "missing - deterministic-only mode",
    )
    _check(
        "ARENAGO_TOKEN",
        settings.has_arenago_token,
        "set (read-only)" if settings.has_arenago_token else "missing - portfolio context disabled",
    )

    # DB.
    db_ok = db.healthcheck()
    _check("DB", db_ok, f"path={db.db_path} size_bytes={db.size_bytes()}")

    # MOEX reachability.
    moex_ok = False
    moex_detail = ""
    try:
        md = moex.fetch_marketdata(secids=["SBER"])
        moex_ok = bool(md)
        moex_detail = f"marketdata rows for SBER: {len(md)}"
    except Exception as exc:
        moex_detail = f"failed: {exc}"
    _check("MOEX public ISS", moex_ok, moex_detail)

    if settings.has_moex_token:
        algopack_ok = False
        details = ""
        try:
            rows = moex.fetch_tradestats(secid="SBER", latest=True)
            algopack_ok = True
            details = f"tradestats rows: {len(rows)}"
        except Exception as exc:
            details = f"failed: {exc}"
        _check("MOEX ALGOPACK", algopack_ok, details)

    # Polza.
    if settings.has_polza_token:
        polza_ok = False
        details = ""
        try:
            client = PolzaClient(settings)
            polza_ok = client.enabled
            details = f"enabled={polza_ok} model={settings.polza_model}"
        except Exception as exc:
            details = f"failed: {exc}"
        _check("Polza AI", polza_ok, details)

    # Arenago.
    if arena is not None and arena.enabled:
        ar_ok = True
        details = "client initialised (read-only)"
        try:
            bots = arena.get_bots()
            details = f"bots fetched: {0 if bots is None else len(bots)}"
        except Exception as exc:
            ar_ok = False
            details = f"failed: {exc}"
        _check("Arenago read-only", ar_ok, details)

    console.print(table)

    # Universe summary.
    console.print(f"[bold]Universe:[/] {', '.join(settings.universe)}")
    console.print(f"[bold]Strategy version:[/] {settings.strategy_version}")
    moex.close()
    if arena:
        arena.close()


# ---------------------------------------------------------------------------
@cli.command(name="backtest")
def backtest_cmd(
    date_from: str = typer.Option(..., "--from", help="Start date YYYY-MM-DD."),
    till: str = typer.Option(..., "--till", help="End date YYYY-MM-DD."),
    take_profit_pct: float = typer.Option(1.5, "--tp"),
    stop_loss_pct: float = typer.Option(0.8, "--sl"),
    max_bars_hold: int = typer.Option(12, "--max-bars"),
) -> None:
    """Run a simple historical buy-only simulation."""
    settings, db, repo, moex, arena, _advisor = _init_components()
    bt = Backtester(repo=repo, settings=settings)
    result = bt.run(
        date_from=parse_date(date_from),
        date_till=parse_date(till),
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        max_bars_hold=max_bars_hold,
    )
    console.print_json(json.dumps(result.summary, default=str))
    moex.close()
    if arena:
        arena.close()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
