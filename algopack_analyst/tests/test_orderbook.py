import pytest

from analytics.orderbook import bid_ask_imbalance, microprice, spread_bps


def test_imbalance_buy_side():
    snap = {"bids": [(100, 100), (99, 50)], "asks": [(101, 20), (102, 10)]}
    assert bid_ask_imbalance(snap) > 0.5


def test_imbalance_sell_side():
    snap = {"bids": [(100, 10), (99, 5)], "asks": [(101, 100), (102, 50)]}
    assert bid_ask_imbalance(snap) < -0.5


def test_spread_bps():
    snap = {"bids": [(100, 1)], "asks": [(100.5, 1)]}
    bps = spread_bps(snap)
    assert 49 < bps < 51  # ~50 bps


def test_microprice_between_bid_ask():
    snap = {"bids": [(100, 10)], "asks": [(101, 90)]}
    mp = microprice(snap)
    assert 100 < mp < 101