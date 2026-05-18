"""Polza AI OpenAI-compatible client wrapper.

The Polza endpoint is OpenAI-compatible, so we re-use the official ``openai``
library with a custom ``base_url`` and ``api_key`` from env.

The client is intentionally thin: prompt formatting and JSON validation live
in :mod:`app.strategy.llm_advisor`. We expose just ``chat_json`` which returns
the raw assistant text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.logging_config import get_logger

_LOG = get_logger(__name__)


class PolzaClient:
    """Wrapper around the OpenAI Python SDK for Polza AI."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.polza_model
        self.timeout = self.settings.polza_timeout
        self._enabled = self.settings.enable_llm and bool(self.settings.polza_ai_api_key)
        self._client = None
        if self._enabled:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                self._client = OpenAI(
                    base_url=self.settings.polza_base_url,
                    api_key=self.settings.polza_ai_api_key,
                    timeout=self.timeout,
                )
            except Exception as exc:  # pragma: no cover - import error path
                _LOG.warning("polza_init_failed", extra={"error": str(exc)})
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    # ------------------------------------------------------------------
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        request_json: bool = True,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return ``{text, usage}``.

        Raises :class:`RuntimeError` if the client is disabled / fails. The
        caller (LLMAdvisor) is responsible for handling errors and falling
        back to deterministic-only output.
        """
        if not self.enabled:
            raise RuntimeError("Polza client is disabled (missing token or ENABLE_LLM=false)")

        assert self._client is not None  # narrow for type checker
        kwargs: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if request_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except TypeError:
            # Older OpenAI SDKs do not understand response_format.
            kwargs.pop("response_format", None)
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some OpenAI-compatible servers (e.g. moonshot via Polza) reject
            # `response_format=json_object`. Retry once without it before
            # bubbling the error up to the LLMAdvisor.
            if "response_format" in kwargs and _looks_like_format_error(exc):
                kwargs.pop("response_format", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise

        text = ""
        reasoning_text = ""
        finish_reason: Optional[str] = None
        try:
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            msg = choice.message
            text = (getattr(msg, "content", None) or "") or ""
            # Reasoning models (kimi-k2.x, deepseek-r1, etc) may place output in
            # a non-standard `reasoning` field; keep it as a fallback so the
            # LLMAdvisor can still attempt JSON extraction.
            reasoning_text = getattr(msg, "reasoning", "") or ""
        except Exception as exc:  # pragma: no cover
            _LOG.error("polza_response_parse_failed", extra={"error": str(exc)})
        usage: Dict[str, Any] = {}
        if getattr(resp, "usage", None) is not None:
            for f in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = getattr(resp.usage, f, None)
                if v is not None:
                    usage[f] = v
        return {
            "text": text,
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
            "usage": usage,
            "model": kwargs["model"],
        }


def _looks_like_format_error(exc: BaseException) -> bool:
    """Heuristic: did the server complain about ``response_format`` support?"""
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "response_format",
            "json_object",
            "unsupported",
            "not supported",
            "invalid_request_error",
        )
    )
