# MOEX Technical Advisor (advisory only)

A production-ready **technical advisory** micro-service for the Moscow Exchange
trading competition on [arenago.ru](https://arenago.ru).

The agent never places trades. It analyses MOEX ISS, ALGOPACK SuperCandles
(`tradestats`, `orderstats`, `obstats`), MegaAlerts (`eq/alerts`), HI2
concentration metrics (`eq/hi2`), live `marketdata`, and intraday 5m candles,
then produces a **structured recommendation** for the main news-driven trading
agent.

## Why advisory-only?

The competition's main agent is news-driven. This service is a *second
opinion* layer:

1. The news agent receives a positive headline and decides "I want to buy
   SBER for 50 000 RUB."
2. It calls `POST /advice` (or the in-process `TechnicalAdvisor.get_advice`
   method).
3. The advisor returns:
    * `recommended_action`: `BUY` / `WAIT` / `DO_NOT_BUY` (for `BUY_CHECK`),
      or `HOLD_POSITION` / `TRIM_POSITION` / `EXIT_POSITION` (for `EXIT_CHECK`).
    * `allow_buy` boolean and a `position_size_multiplier` between 0 and 1.
    * Concrete `recommended_cash_rub` and `recommended_quantity` once
      portfolio limits and liquidity are factored in.
    * Hard `risk_flags`, plus a textual `reasons` / `negative_factors`
      breakdown for traceability.
4. The news agent multiplies its intended size by `position_size_multiplier`
   and respects `allow_buy`. The advisor itself **never** submits orders.

The competition allows **long-only** stock trades. Selling advice is only
issued for existing long positions (TRIM / EXIT) — shorts are never
recommended.

## Architecture

```
app/
  main.py                 # Typer CLI: once | run | backfill | serve | doctor | backtest
  config.py               # pydantic-settings, env-driven
  logging_config.py       # JSON logs to stdout
  clients/
    moex_client.py        # ISS + ALGOPACK (Bearer auth), universal parser, retries
    arena_client.py       # Arenago read-only (submit_order forbidden)
    polza_client.py       # Polza AI (OpenAI-compatible) wrapper
  storage/
    db.py                 # SQLite, WAL, abstraction-ready for PostgreSQL
    models.py             # SQLAlchemy ORM models
    repository.py         # All upserts / reads / retention
  ingestion/
    backfill.py           # 30-day backfill
    scheduler.py          # market data 60s, super candles 5m, hi2 daily
    normalizers.py        # MOEX raw rows -> DB-ready dicts
  features/
    indicators.py         # EMA/RSI/ATR/zscore/etc
    feature_builder.py    # Per-ticker feature bundle
    anomaly_detector.py   # Custom anomalies in addition to MegaAlerts
  strategy/
    scoring.py            # Deterministic 0..100 score with explicit breakdown
    risk.py               # Hard veto + portfolio sizing (long-only)
    llm_advisor.py        # Polza adjustment, strict JSON schema, cached
    recommendation.py     # Pydantic schemas (AdviceRequest, AdviceResponse)
    advisor.py            # TechnicalAdvisor: single source of truth
  api/
    server.py             # FastAPI thin wrapper around TechnicalAdvisor
    schemas.py            # HTTP response models
  backtest.py             # Skeleton historical simulator
  utils/                  # time, retry helpers
tests/                    # pytest suite covering parser/scoring/risk/recommendation/LLM
Dockerfile
requirements.txt / pyproject.toml
.env.example
```

The Python service class **`app.strategy.advisor.TechnicalAdvisor`** is the
single source of truth. The HTTP layer is a thin wrapper that only validates
JSON, calls `TechnicalAdvisor.get_advice()`, and returns the response — no
strategy logic lives in the route handlers.

## Storage

Default backend is **embedded SQLite in WAL mode** at
`/app/data/tech_agent.sqlite3` (configurable via `DB_PATH`). Single-container
deployment without docker-compose is the explicit constraint:

* WAL + `synchronous=NORMAL` give safe concurrent read access to the API
  while the scheduler writes.
* The storage layer goes through a `Database` abstraction so swapping to
  PostgreSQL later is just an URL change.
* Raw API caches (tradestats / orderstats / obstats / alerts / candles /
  agent_events / derived_features) are auto-pruned every 6 hours according to
  `RAW_CACHE_RETENTION_DAYS` (default 7).

## Configuration

All secrets and tunables are env-only. See [`.env.example`](.env.example) for
the full reference. The most important ones:

| Variable | Purpose |
| --- | --- |
| `MOEX_API_KEY` / `MOEX_ALGOPACK_TOKEN` | Bearer token for ALGOPACK endpoints. Without it the agent runs on public ISS data only. |
| `POLZA_AI_API_KEY` | Enables the LLM advisory layer. Without it the deterministic recommendation is returned unchanged. |
| `ARENAGO_TOKEN` | Read-only portfolio context (positions / trades / bots). |
| `UNIVERSE` | Comma-separated tickers. Defaults to the 20-name competition list. |
| `STRATEGY_VERSION` | Embedded into every persisted recommendation for reproducibility. |
| `DB_PATH` | SQLite file path. |
| `MAX_POSITION_PCT` / `MAX_SINGLE_ORDER_PCT` / `MAX_PORTFOLIO_EXPOSURE_PCT` / `RESERVE_CASH_PCT` / `MIN_ORDER_CASH_RUB` | Risk limits used by the sizing engine. |
| `BUY_SCORE_THRESHOLD` / `MIN_CONFIDENCE_BUY` / `HOLD_SCORE_THRESHOLD` | Decision thresholds. |
| `MAX_SPREAD_BBO_BPS` / `MAX_SPREAD_1MIO_BPS` / `MAX_STALE_SECONDS` / `MIN_LIQUIDITY_VAL_B` | Hard veto thresholds. |

Never put secrets in code or in the Docker image — pass them through
`docker run -e ...` or a runtime secrets manager.

## Running locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit
export $(grep -v '^#' .env | xargs) # or use direnv

python -m app.main doctor       # sanity-check env / DB / network
python -m app.main backfill --days 30
python -m app.main once
python -m app.main serve        # FastAPI on http://0.0.0.0:8080
python -m app.main run          # ingest + recommend forever, also serves HTTP
```

Run the test suite with:

```bash
pip install pytest
pytest -q
```

## Running in Docker (single container)

```bash
docker build -t moex-tech-agent .

mkdir -p data
docker run --rm \
    -e MOEX_API_KEY=... \
    -e POLZA_AI_API_KEY=... \
    -e ARENAGO_TOKEN=... \
    -e BOT_NAME=MyTradingBot \
    -p 8080:8080 \
    -v "$(pwd)/data:/app/data" \
    moex-tech-agent
```

The default `CMD` is `python -m app.main run`, which:

* starts the FastAPI server in a background thread (`ENABLE_HTTP=true`),
* runs a lightweight 5-day backfill if the DB is empty,
* polls `marketdata` and `alerts` every 60s,
* fetches `tradestats` / `orderstats` / `obstats` and 5m candles every 5min
  with a small lag,
* fetches `hi2` once a day after `18:45 MSK`,
* generates a recommendation for every ticker in the universe after each
  cycle.

## CLI reference

```text
python -m app.main once       # one ingestion + recommendation pass
python -m app.main run        # endless loop (also serves HTTP if ENABLE_HTTP=true)
python -m app.main backfill --days 30
python -m app.main serve      # FastAPI only
python -m app.main doctor     # config / network / DB sanity report
python -m app.main backtest --from 2026-04-01 --till 2026-04-30
```

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness + last-ingestion age per dataset. |
| GET | `/recommendations` | Latest recommendation per ticker (one row each). |
| GET | `/recommendations/{secid}` | Latest recommendation for `secid`. |
| POST | `/advice` | The main integration point (see below). |
| GET | `/anomalies?limit=200` | Recent anomalies. |
| GET | `/features/{secid}` | Latest computed features for `secid`. |

### `POST /advice`

```json
{
  "secid": "SBER",
  "intent": "BUY_CHECK",
  "news_score": 0.75,
  "intended_cash_rub": 50000,
  "portfolio": "MyTradingBot",
  "horizon_minutes": 60,
  "portfolio_state": {
    "cash_rub": 800000,
    "equity_rub": 1000000,
    "positions_value_rub": 200000,
    "daily_trades_count": 23,
    "daily_trade_limit": 200
  },
  "current_position": {
    "quantity": 0,
    "average_price": 0.0,
    "market_value_rub": 0.0,
    "unrealized_pnl_pct": 0.0
  }
}
```

Response (full schema, every field always present):

```json
{
  "secid": "SBER",
  "timestamp": "2026-05-18T10:00:00Z",
  "intent": "BUY_CHECK",
  "strategy_version": "det-1.0",
  "action": "BUY",
  "recommended_action": "BUY",
  "allow_buy": true,
  "allow_action": true,
  "exit_warning": false,
  "exit_urgency": "NONE",
  "technical_score": 78.2,
  "confidence": 0.71,
  "position_size_multiplier": 1.0,
  "horizon_minutes": 60,
  "stop_loss_pct": 0.7,
  "take_profit_pct": 1.5,
  "max_cash_rub": 50000.0,
  "recommended_cash_rub": 50000.0,
  "recommended_quantity": 100,
  "recommended_sell_quantity": null,
  "reasons": ["ema9>ema21 (trend up)", "buy volume dominates (62.0%)", ...],
  "negative_factors": [],
  "risk_flags": [],
  "feature_snapshot": { ... },
  "data_quality": { ... },
  "portfolio_constraints": { ... },
  "llm_used": true,
  "llm_comment": "Positive momentum, but spread is widening."
}
```

For `intent=EXIT_CHECK`, the same envelope is reused but `recommended_action`
is one of `HOLD_POSITION` / `TRIM_POSITION` / `EXIT_POSITION` and
`recommended_sell_quantity` is populated.

## Integrating with the main (news) agent

```python
import httpx

resp = httpx.post(
    "http://moex-tech-agent:8080/advice",
    json={
        "secid": "SBER",
        "intent": "BUY_CHECK",
        "news_score": 0.8,
        "intended_cash_rub": intended_size_rub,
        "portfolio": bot_name,
    },
    timeout=10.0,
).json()

if not resp["allow_buy"]:
    return  # never buy if the technical agent forbids it

target_cash = intended_size_rub * resp["position_size_multiplier"]
# now place the order via Arenago using your existing logic
```

Or, if the technical agent is colocated in the same Python process:

```python
from app.strategy import TechnicalAdvisor, AdviceRequest, AdviceIntent

advice = advisor.get_advice(AdviceRequest(
    secid="SBER",
    intent=AdviceIntent.BUY_CHECK,
    intended_cash_rub=50_000,
))
if not advice.allow_buy:
    return
```

## Safety guarantees

* `ArenagoClient.submit_order(...)` always raises `ArenagoForbiddenError`,
  and the client refuses to make any request whose path starts with
  `/api/submit_order` (or any of the other write endpoints). There is no code
  path inside the project that places orders.
* The LLM cannot override a hard risk veto. When `hard_veto=True`, LLM
  adjustments may only *lower* the score / confidence, never raise them.
* Hard vetoes include: stale data, ticker not in universe, wide spread,
  insufficient liquidity, extreme volatility, sharp drop with sell
  pressure, spoof patterns, daily trade limit reached, position limit
  reached, portfolio exposure limit reached.
* All prompts and responses sent to Polza are stored in the `llm_log` table
  with `prompt_version` / `model` / token usage — no secrets are stored, and
  every recommendation is reproducible.

## Reproducibility

* `STRATEGY_VERSION` is stamped onto every recommendation row.
* Feature bundles are persisted with `feature_version`.
* LLM responses are cached by SHA-256 of the compact feature pack +
  deterministic decision + prompt version, so identical inputs return
  identical advice and no additional LLM tokens are spent.
* All inbound rows, computed features and final recommendations stay in
  SQLite for at least `RAW_CACHE_RETENTION_DAYS` (default 7).

## Tests

```bash
pytest -q
```

Covers:

* the universal ISS JSON parser (single block, envelope, ticker alias),
* indicator math (EMA / RSI / ATR / rolling z-score / volatility),
* scoring thresholds for bullish and bearish synthetic data,
* hard risk veto rules and position sizing,
* recommendation schema for both `BUY_CHECK` and `EXIT_CHECK`,
* LLM fallback when Polza is disabled, plus strict schema enforcement,
* the `submit_order` tripwire on the Arenago client,
* a FastAPI integration test for `/advice`, `/health`, `/recommendations`.
