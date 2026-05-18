Понял, вот красиво оформленный README.md:

```markdown
# ALGOPACK Analyst Agent

Аналитический агент для торгового бота на MOEX. Не торгует — рекомендует.

## Возможности

- Сбор и хранение Super Candles, FUTOI, HI2, Mega Alerts, OHLCV, стакана
- Композитный BUY-скоринг с 7 факторами
- LLM-объяснения через polza.ai
- REST API для главного агента

## Быстрый старт

```bash
cp .env.example .env    # → впишите токены
docker compose up -d --build

curl -H "Authorization: Bearer $API_BEARER_TOKEN" \
     -d '{"ticker":"SBER","horizon":"intraday"}' \
     -H "Content-Type: application/json" \
     http://localhost:8000/analyze/ticker
```

## API

| Метод | Путь | Назначение |
|-------|------|------------|
| POST | `/analyze/ticker` | Рекомендация по тикеру |
| POST | `/analyze/query` | Свободный текстовый запрос |
| GET | `/ticker/{t}/history` | Историческая сводка |
| GET | `/ticker/{t}/anomalies` | Mega Alerts |
| GET | `/market/top_signals` | Топ-сигналы |
| POST | `/watch/add` | Расширение watchlist |
| GET | `/health` | Статус системы |

## Модули и источники данных

| Модуль | Источники |
|--------|-----------|
| `analytics/technical.py` | S3, S16 |
| `analytics/orderbook.py` | S4, S7 |
| `analytics/concentration.py` | S9 |
| `analytics/sentiment.py` | S11, S6 |
| `analytics/anomaly.py` | S10, S5 |
| `analytics/scoring.py` | Все факторы |
| `data_collectors/super_candles.py` | S6, S7, S8 |
| `data_collectors/futoi.py` | S11 |
| `data_collectors/hi2.py` | S9 |
| `data_collectors/mega_alerts.py` | S10 |
| `data_collectors/realtime.py` | S2, S3, S4, S5 |
| `scheduler guards` | S12, S13 |
| `pre-rec guards` | S14 |
| `watchlist validation` | S15, S17 |

## Тестирование

```bash
pytest -q
```

## Архитектурные решения

- **DuckDB** — embedded OLAP, parquet-совместимая, идеальна под 16GB RAM
- **APScheduler** — оркестрация коллекторов с разными интервалами
- **LLM только для объяснений** — все цифры детерминированы
- **Pre-recommendation pipeline** (A.9 в спецификации):
  1. `is_market_open()` → иначе HOLD
  2. `is_trading_suspended()` → иначе AVOID
  3. `is_tradable_tqbr()` → иначе AVOID
  4. Freshness checks → понижение confidence
  5. Wide spread → урезание `max_position_pct`
  6. Дедуп: повторный запрос за 5 мин → cached

## Формат ответа

```json
{
  "ticker": "SBER",
  "recommendation": "BUY",
  "score": 72,
  "confidence": 0.81,
  "entry_zone": {"min": 312.5, "max": 313.8},
  "stop_loss": 310.2,
  "take_profit": 318.5,
  "max_position_pct": 10,
  "signals": { "...": "..." },
  "reasons": ["..."],
  "risks": ["..."],
  "llm_explanation": "...",
  "strategy_version": "v1.2.0"
}
```

## ✅ Что реализовано

- 18 endpoints MOEX (S1—S18) в `MoexClient`
- 5 коллекторов + scheduler с проверкой `is_market_open`
- 6 аналитических модулей (technical, orderbook, concentration, sentiment, anomaly, scoring)
- Pre-recommendation guards (раздел A.9) — suspended/tradable/freshness/spread/dedup
- Композитный скоринг с 7 взвешенными факторами из config
- Risk management — ATR-based SL, R:R ≥ 1.5 TP, динамический `max_position_pct`
- LLM слой (polza.ai, OpenAI SDK) — только для объяснений и intent parsing
- FastAPI — 7 endpoints с Bearer auth
- DuckDB + ротация в Parquet
- Structured logging (loguru JSON)
- Тесты для критичных модулей
- Docker + docker-compose с healthcheck

## 🔧 Что подкрутить под прод

- **Названия колонок Super Candles** — у MOEX поля могут быть в верхнем регистре (`PR_VWAP` vs `pr_vwap`). После первого реального ответа добавить нормализацию `df.columns = df.columns.str.lower()` в `MoexClient.to_dataframe`.
- **Маппинг `alert_type` в `ALERT_IMPACT`** — проверить фактические значения по `eq/alerts` ответам.
- **Точные имена блоков ISS** (`data`, `futoi`, `securities`, `marketdata`, `candles`, `orderbook`) — могут потребовать корректировки после первого live-вызова.
- **`moexalgo` установлена в requirements** — можно использовать как fallback в местах, где REST-вызов сложен. Всё оставлено на чистом `aiohttp` для полного контроля.

## Запуск

```bash
docker compose up -d --build
curl http://localhost:8000/health
```
```

Готово — чистый, структурированный README без лишнего мусора и с сохранением всей смысловой нагрузки.