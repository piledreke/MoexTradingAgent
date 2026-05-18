"""Tests for the universal ISS JSON parser."""

from __future__ import annotations

from app.clients.moex_client import parse_iss_block, parse_iss_envelope


def test_parse_iss_block_single_block_dict() -> None:
    payload = {
        "columns": ["secid", "price", "vol"],
        "data": [["SBER", 100.0, 1500], ["GAZP", 200.5, 700]],
    }
    rows = parse_iss_block(payload)
    assert rows == [
        {"secid": "SBER", "price": 100.0, "vol": 1500},
        {"secid": "GAZP", "price": 200.5, "vol": 700},
    ]


def test_parse_iss_block_with_envelope_and_block_arg() -> None:
    payload = {
        "data": {
            "columns": ["tradedate", "secid", "value"],
            "data": [["2024-04-12", "SBER", 99.0]],
        }
    }
    rows = parse_iss_block(payload, block="data")
    assert rows == [{"tradedate": "2024-04-12", "secid": "SBER", "value": 99.0}]


def test_parse_iss_block_renames_ticker_to_secid() -> None:
    payload = {
        "columns": ["TICKER", "BOARD", "PRICE"],
        "data": [["SBER", "TQBR", 250.5]],
    }
    rows = parse_iss_block(payload)
    assert rows == [{"secid": "SBER", "boardid": "TQBR", "price": 250.5}]


def test_parse_iss_envelope_multi_blocks() -> None:
    payload = {
        "securities": {
            "columns": ["SECID", "LOTSIZE"],
            "data": [["SBER", 10]],
        },
        "marketdata": {
            "columns": ["SECID", "LAST"],
            "data": [["SBER", 300.0]],
        },
    }
    blocks = parse_iss_envelope(payload)
    assert "securities" in blocks and "marketdata" in blocks
    assert blocks["securities"][0]["secid"] == "SBER"
    assert blocks["marketdata"][0]["last"] == 300.0


def test_parse_iss_block_empty_inputs() -> None:
    assert parse_iss_block(None) == []
    assert parse_iss_block({}) == []
    assert parse_iss_block({"columns": ["a"], "data": []}) == []
    assert parse_iss_block({"unknown": {}}) == []


def test_parse_iss_block_ragged_rows_are_truncated() -> None:
    payload = {
        "columns": ["a", "b", "c"],
        "data": [["x", "y"], ["x", "y", "z", "extra"]],
    }
    rows = parse_iss_block(payload)
    assert rows[0] == {"a": "x", "b": "y"}
    assert rows[1] == {"a": "x", "b": "y", "c": "z"}
