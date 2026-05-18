"""Optional LLM advisory layer (Polza AI).

The LLM only *adjusts* the deterministic recommendation. It cannot override a
hard risk veto, cannot turn AVOID into BUY when the veto is active, and never
recommends shorts. If the LLM is disabled or fails, the deterministic output
is used unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.polza_client import PolzaClient
from app.config import Settings, get_settings
from app.features.feature_builder import FeatureBundle
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.strategy.scoring import ScoreBreakdown

_LOG = get_logger(__name__)

PROMPT_VERSION = "p-1.0"

SYSTEM_PROMPT = (
    "Ты риск-ориентированный технический аналитик MOEX. "
    "Тебе ЗАПРЕЩЕНО рекомендовать шорт. "
    "Тебе ЗАПРЕЩЕНО игнорировать hard risk veto, переданный в feature pack. "
    "Используй только предоставленные признаки, не выдумывай новости и не выходи за рамки данных. "
    "Твоя задача — выдать аккуратный комментарий и небольшую коррекцию score/confidence "
    "детерминированного советника. Ответ строго в формате JSON по схеме."
)

USER_PROMPT_TEMPLATE = (
    "## Тикер: {secid}\n"
    "## Намерение: {intent}\n"
    "## Текущая детерминированная рекомендация\n"
    "{deterministic_block}\n\n"
    "## Feature pack\n"
    "{features_block}\n\n"
    "## Схема ответа (верни ровно эти ключи, ничего лишнего)\n"
    "{schema_block}\n"
)


class LLMResponse(BaseModel):
    """Strict schema returned by the LLM."""

    model_config = ConfigDict(extra="ignore")

    score_adjustment: int = Field(default=0, ge=-10, le=10)
    confidence_adjustment: float = Field(default=0.0, ge=-0.15, le=0.15)
    final_comment: str = Field(default="")
    risks: List[str] = Field(default_factory=list)
    supporting_reasons: List[str] = Field(default_factory=list)
    disagreement: bool = False


# ---------------------------------------------------------------------------
class LLMAdvisor:
    """Computes the LLM-side adjustment to ``technical_score`` / ``confidence``."""

    def __init__(
        self,
        repo: Repository,
        client: Optional[PolzaClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repo
        self.client = client or PolzaClient(self.settings)

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.client.enabled

    # ------------------------------------------------------------------
    def evaluate(
        self,
        bundle: FeatureBundle,
        breakdown: ScoreBreakdown,
        deterministic_decision: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        """Return ``{used: bool, response: LLMResponse|None, comment: str|None}``.

        Always safe: on any failure, returns ``used=False`` and the caller keeps
        the deterministic recommendation.
        """
        if not self.enabled:
            return {"used": False, "response": None, "comment": None}

        feature_pack = self._compact_feature_pack(bundle, breakdown, deterministic_decision, intent)
        feature_hash = self._hash(feature_pack)
        cached = self.repo.get_llm_cache(feature_hash)
        if cached is not None:
            try:
                parsed = LLMResponse.model_validate(cached)
                return {"used": True, "response": parsed, "comment": parsed.final_comment, "cached": True}
            except ValidationError:
                pass

        user_prompt = USER_PROMPT_TEMPLATE.format(
            secid=bundle.secid,
            intent=intent,
            deterministic_block=json.dumps(deterministic_decision, ensure_ascii=False, indent=2),
            features_block=json.dumps(feature_pack["features"], ensure_ascii=False, indent=2),
            schema_block=_SCHEMA_TEXT,
        )

        try:
            resp = self.client.chat_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=2000,
                request_json=True,
            )
        except Exception as exc:
            _LOG.warning("llm_call_failed", extra={"secid": bundle.secid, "error": str(exc)})
            self.repo.log_llm_call(
                secid=bundle.secid,
                prompt_version=PROMPT_VERSION,
                model=self.client.model,
                feature_hash=feature_hash,
                prompt=user_prompt,
                response=None,
                usage=None,
                success=False,
                error=str(exc),
            )
            return {"used": False, "response": None, "comment": None}

        text = (resp.get("text") or "").strip()
        reasoning_text = (resp.get("reasoning") or "").strip()
        # Prefer `content`; fall back to `reasoning` text for reasoning models
        # whose JSON payload ends up there.
        parsed_payload = self._parse_strict_json(text)
        if parsed_payload is None and reasoning_text:
            parsed_payload = self._parse_strict_json(reasoning_text)
        success = False
        parsed: Optional[LLMResponse] = None
        error: Optional[str] = None
        if parsed_payload is not None:
            try:
                parsed = LLMResponse.model_validate(parsed_payload)
                success = True
                self.repo.put_llm_cache(feature_hash, parsed.model_dump())
            except ValidationError as exc:
                error = f"validation: {exc}"
        else:
            finish = resp.get("finish_reason")
            error = (
                f"non-json response (finish_reason={finish}, "
                f"content_len={len(text)}, reasoning_len={len(reasoning_text)})"
            )

        self.repo.log_llm_call(
            secid=bundle.secid,
            prompt_version=PROMPT_VERSION,
            model=resp.get("model") or self.client.model,
            feature_hash=feature_hash,
            prompt=user_prompt,
            response=text,
            usage=resp.get("usage"),
            success=success,
            error=error,
        )
        if not success or parsed is None:
            return {"used": False, "response": None, "comment": None}
        return {
            "used": True,
            "response": parsed,
            "comment": parsed.final_comment,
            "cached": False,
        }

    # ------------------------------------------------------------------
    def _compact_feature_pack(
        self,
        bundle: FeatureBundle,
        breakdown: ScoreBreakdown,
        deterministic_decision: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        """Return a tiny, stable feature pack used both as prompt + cache key."""
        keep_feature_keys = (
            "return_1_bar", "return_3_bar", "return_6_bar", "return_12_bar",
            "intraday_return_from_open", "ema_9_above_21", "close_above_ema21",
            "distance_to_vwap_pct", "rsi14", "bollinger_z", "realized_vol",
            "atr_5m", "volume_zscore", "value_zscore", "buy_volume_ratio",
            "sell_volume_ratio", "disb_now", "aggressive_buy_pressure",
            "put_cancel_ratio_b", "put_cancel_ratio_s", "net_cancel_pressure",
            "spread_bbo", "spread_1mio", "imbalance_bbo", "imbalance_vol",
            "liquidity_score", "slippage_1mio_bps", "alerts_positive_count",
            "alerts_negative_count", "alerts_reference_summary",
            "hi2_volume", "hi2_aggressive_buy", "hi2_aggressive_sell",
            "hi2_netflow_sell", "hi2_risk_level", "last_price",
        )
        features: Dict[str, Any] = {}
        for k in keep_feature_keys:
            features[k] = bundle.features.get(k)
        breakdown_block = {
            "trend_score": round(breakdown.trend_score, 2),
            "momentum_score": round(breakdown.momentum_score, 2),
            "volume_flow_score": round(breakdown.volume_flow_score, 2),
            "orderbook_liquidity_score": round(breakdown.orderbook_liquidity_score, 2),
            "alerts_score": round(breakdown.alerts_score, 2),
            "hi2_risk_adjustment": round(breakdown.hi2_risk_adjustment, 2),
            "risk_penalty": round(breakdown.risk_penalty, 2),
            "technical_score": round(breakdown.technical_score, 2),
            "confidence": round(breakdown.confidence, 3),
        }
        return {
            "secid": bundle.secid,
            "intent": intent,
            "strategy_version": self.settings.strategy_version,
            "prompt_version": PROMPT_VERSION,
            "deterministic_decision": deterministic_decision,
            "breakdown": breakdown_block,
            "data_quality": bundle.data_quality,
            "features": features,
        }

    @staticmethod
    def _hash(payload: Dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_strict_json(text: str) -> Optional[Dict[str, Any]]:
        r"""Tolerant JSON extractor.

        Handles three common output styles from OpenAI-compatible servers:

        1. Bare JSON object.
        2. Triple-backtick fenced JSON block (markdown).
        3. Object embedded in surrounding commentary or reasoning text.
        """
        if not text:
            return None
        text = text.strip()

        # 1) Bare JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

        # 2) Fenced block ```json ... ```
        if "```" in text:
            for fence_start in ("```json", "```JSON", "```"):
                idx = text.find(fence_start)
                if idx < 0:
                    continue
                rest = text[idx + len(fence_start):]
                end = rest.find("```")
                candidate = rest[:end] if end >= 0 else rest
                candidate = candidate.strip()
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue

        # 3) Find the first balanced {...} object via brace counting.
        start = text.find("{")
        while start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            start = text.find("{", start + 1)
        return None


_SCHEMA_TEXT = (
    "{\n"
    '  "score_adjustment": integer between -10 and 10,\n'
    '  "confidence_adjustment": number between -0.15 and 0.15,\n'
    '  "final_comment": string,\n'
    '  "risks": [string],\n'
    '  "supporting_reasons": [string],\n'
    '  "disagreement": boolean\n'
    "}"
)
