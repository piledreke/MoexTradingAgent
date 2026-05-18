"""OpenAI-compatible client for polza.ai with logging."""
from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI

from config import settings
from llm.prompts import PROMPTS
from utils.logger import logger

_client: AsyncOpenAI | None = None


def get_llm() -> AsyncOpenAI | None:
    global _client
    if _client is None:
        if not settings.POLZA_API_KEY:
            logger.warning("POLZA_API_KEY missing — LLM disabled")
            return None
        _client = AsyncOpenAI(
            api_key=settings.POLZA_API_KEY,
            base_url=settings.POLZA_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    return _client


async def llm_complete(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Send completion request, log every call."""
    client = get_llm()
    if client is None:
        return ""
    mdl = model or settings.LLM_MODEL
    kwargs: dict[str, Any] = {
        "model": mdl,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        dt = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        logger.info(
            "LLM call",
            extra={
                "model": mdl,
                "tokens": getattr(usage, "total_tokens", None),
                "latency_ms": round(dt, 1),
                "prompt_len": len(prompt),
                "response_len": len(text),
            },
        )
        return text
    except Exception as e:
        logger.exception(f"LLM call failed: {e}")
        return ""


async def explain_recommendation(recommendation: dict) -> str:
    """Generate human-readable explanation for a recommendation."""
    # Strip nested large fields for prompt brevity
    slim = {
        k: v for k, v in recommendation.items()
        if k not in ("signals",)
    }
    slim["signals_summary"] = _slim_signals(recommendation.get("signals", {}))
    prompt = PROMPTS["explain_v1"].format(payload=json.dumps(slim, ensure_ascii=False))
    return await llm_complete(prompt, temperature=0.4)


async def parse_intent(query: str) -> dict:
    """Parse free-form query to structured intent."""
    prompt = PROMPTS["intent_v1"].format(query=query)
    raw = await llm_complete(prompt, temperature=0.0, json_mode=True)
    if not raw:
        return {"intent": "unknown"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"intent parse JSON decode failed: {raw[:200]}")
        return {"intent": "unknown"}


async def explain_anomaly(alert: dict) -> str:
    prompt = PROMPTS["anomaly_v1"].format(alert=json.dumps(alert, ensure_ascii=False))
    return await llm_complete(prompt, temperature=0.4)


def _slim_signals(sig: dict) -> dict:
    return {
        "technical": sig.get("technical"),
        "orderbook": {
            k: v for k, v in (sig.get("orderbook") or {}).items()
            if k in ("imbalance", "spread_bps", "liquidity")
        },
        "super_candles": sig.get("super_candles"),
        "hi2": sig.get("hi2"),
        "futoi": sig.get("futoi"),
        "alerts_count": len(sig.get("alerts_last_hour") or []),
    }