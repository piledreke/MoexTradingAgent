"""Versioned prompt templates for LLM."""
from __future__ import annotations

EXPLAIN_PROMPT_V1 = """\
Ты — опытный sell-side аналитик MOEX. Получив структурированные метрики по
тикеру, составь связное, лаконичное (5-8 предложений) обоснование рекомендации
на русском языке. Не выдумывай данные. Цитируй конкретные значения из входа.
Формат: связный текст без буллетов.

ВХОД:
{payload}

ОБОСНОВАНИЕ:"""


INTENT_PARSE_PROMPT_V1 = """\
Ты парсер торговых запросов. На входе — свободный вопрос торгового бота.
Верни СТРОГО валидный JSON со следующими полями:
{{
  "intent": "analyze_ticker" | "top_signals" | "anomalies" | "history" | "unknown",
  "ticker": "<тикер или null>",
  "horizon": "intraday" | "swing" | "position" | null,
  "lookback_minutes": <int или null>,
  "limit": <int или null>,
  "direction": "bullish" | "bearish" | null
}}

ЗАПРОС: {query}

JSON:"""


ANOMALY_EXPLAIN_PROMPT_V1 = """\
Ты объясняешь причины аномалии на бирже. Получив один Mega Alert,
дай контекстуальную интерпретацию (2-3 предложения) на русском.

АЛЕРТ:
{alert}

ОБЪЯСНЕНИЕ:"""


PROMPTS = {
    "explain_v1": EXPLAIN_PROMPT_V1,
    "intent_v1": INTENT_PARSE_PROMPT_V1,
    "anomaly_v1": ANOMALY_EXPLAIN_PROMPT_V1,
}