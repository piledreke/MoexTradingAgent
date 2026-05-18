"""Tests for LLM advisor fallback + safety contract."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from app.features.feature_builder import FeatureBundle
from app.strategy.llm_advisor import LLMAdvisor, LLMResponse
from app.strategy.scoring import ScoreBreakdown


def _bundle() -> FeatureBundle:
    return FeatureBundle(
        secid="SBER",
        ts=datetime.utcnow(),
        feature_version="fv-test",
        features={"last_price": 100.0, "rsi14": 60.0},
        data_quality={"quality_score": 0.9},
        alerts=[],
        hi2={},
        market_snapshot={},
        last_price=100.0,
    )


def test_llm_disabled_returns_unused(repo) -> None:
    advisor = LLMAdvisor(repo=repo)
    assert advisor.enabled is False
    out = advisor.evaluate(
        bundle=_bundle(),
        breakdown=ScoreBreakdown(technical_score=70, confidence=0.6),
        deterministic_decision={"recommended_action": "BUY"},
        intent="BUY_CHECK",
    )
    assert out["used"] is False
    assert out["response"] is None


def test_llm_cache_is_used_when_available(repo) -> None:
    advisor = LLMAdvisor(repo=repo)
    breakdown = ScoreBreakdown(technical_score=72.0, confidence=0.6)
    feat_pack = advisor._compact_feature_pack(_bundle(), breakdown, {"recommended_action": "BUY"}, "BUY_CHECK")
    h = advisor._hash(feat_pack)
    repo.put_llm_cache(h, {
        "score_adjustment": -3,
        "confidence_adjustment": -0.05,
        "final_comment": "cached",
        "risks": ["risk1"],
        "supporting_reasons": ["good trend"],
        "disagreement": False,
    })
    # Even when client is disabled, the cache hit ALONE should not be returned
    # since enabled=False short-circuits. We force-enable to validate the cache path.
    advisor.client._enabled = True
    advisor.client._client = object()  # truthy
    out = advisor.evaluate(
        bundle=_bundle(),
        breakdown=breakdown,
        deterministic_decision={"recommended_action": "BUY"},
        intent="BUY_CHECK",
    )
    assert out["used"] is True
    assert out["response"].final_comment == "cached"


def test_llm_response_schema_enforces_bounds() -> None:
    # Out-of-range integers are rejected by pydantic.
    import pytest

    with pytest.raises(Exception):
        LLMResponse.model_validate({
            "score_adjustment": 100,
            "confidence_adjustment": 0.0,
            "final_comment": "x",
            "risks": [],
            "supporting_reasons": [],
            "disagreement": False,
        })

    with pytest.raises(Exception):
        LLMResponse.model_validate({
            "score_adjustment": 0,
            "confidence_adjustment": 1.0,
            "final_comment": "x",
            "risks": [],
            "supporting_reasons": [],
            "disagreement": False,
        })


def test_parse_strict_json_handles_various_shapes() -> None:
    pj = LLMAdvisor._parse_strict_json
    # bare object
    obj = pj('{"score_adjustment": 3, "confidence_adjustment": 0.05}')
    assert obj is not None and obj["score_adjustment"] == 3

    # fenced
    fenced = '```json\n{"score_adjustment": -2, "confidence_adjustment": 0.0, "final_comment": "ok"}\n```'
    obj = pj(fenced)
    assert obj is not None and obj["score_adjustment"] == -2

    # embedded with surrounding commentary (typical reasoning model output)
    embedded = (
        "Some reasoning here.\n"
        "Final answer:\n"
        '{"score_adjustment": 1, "confidence_adjustment": 0.02, '
        '"final_comment": "trend OK", "risks": ["spread"], '
        '"supporting_reasons": ["ema crossover"], "disagreement": false}\n'
        "End of response."
    )
    obj = pj(embedded)
    assert obj is not None
    assert obj["final_comment"] == "trend OK"
    assert obj["risks"] == ["spread"]

    # garbage
    assert pj("") is None
    assert pj("nothing useful here") is None
