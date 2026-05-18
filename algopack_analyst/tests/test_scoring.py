import pytest

from analytics.scoring import _build_avoid, _norm01


def test_norm01_clamps():
    assert _norm01(-5, 0, 1) == 0.0
    assert _norm01(5, 0, 1) == 1.0
    assert _norm01(0.5, 0, 1) == 0.5


def test_build_avoid_structure():
    r = _build_avoid("SBER", "suspended")
    assert r["recommendation"] == "AVOID"
    assert r["score"] == 0
    assert "suspended" in r["risks"]