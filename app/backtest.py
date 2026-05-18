"""Lightweight historical backtester.

Walks through stored 5m candles + tradestats for one or several tickers and
simulates BUY-only entries triggered when the deterministic scorer says BUY.
Exits are time- / take-profit- / stop-loss-based for analysis only.

This is intentionally a thin skeleton – the goal is reproducibility of the
strategy thresholds, not a full quant backtesting framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.features.feature_builder import FeatureBundle, FeatureBuilder
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.strategy.scoring import TechnicalScorer

_LOG = get_logger(__name__)


@dataclass
class Trade:
    secid: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    score_at_entry: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    total_return: float = 0.0
    max_drawdown: float = 0.0
    winrate: float = 0.0
    avg_trade_return: float = 0.0
    exposure: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


class Backtester:
    """Simulate buy-only entries on historical 5m candles."""

    def __init__(
        self,
        repo: Repository,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repo
        self.feature_builder = FeatureBuilder(repo)
        self.scorer = TechnicalScorer()

    # ------------------------------------------------------------------
    def run(
        self,
        date_from: date,
        date_till: date,
        universe: Optional[Iterable[str]] = None,
        take_profit_pct: float = 1.5,
        stop_loss_pct: float = 0.8,
        max_bars_hold: int = 12,
    ) -> BacktestResult:
        universe = list(universe or self.settings.universe)
        trades: List[Trade] = []
        exposure_bars = 0
        total_bars = 0

        for secid in universe:
            df = self.repo.get_intraday_candles_df(secid, limit=5000)
            if df.empty or "close" not in df.columns:
                continue
            df = df[(df["begin"] >= datetime.combine(date_from, datetime.min.time()))
                    & (df["begin"] <= datetime.combine(date_till, datetime.max.time()))]
            df = df.reset_index(drop=True)
            if len(df) < 20:
                continue
            total_bars += len(df)
            in_pos = False
            entry_idx = 0
            entry_price = 0.0
            entry_score = 0.0
            entry_time = None
            for i in range(20, len(df)):
                close = float(df["close"].iloc[i])
                # Approximate "current" snapshot by truncating tradestats df.
                bundle = self._make_synthetic_bundle(secid, df.iloc[: i + 1])
                breakdown = self.scorer.score(bundle)
                if not in_pos:
                    if (
                        breakdown.technical_score >= self.settings.buy_score_threshold
                        and breakdown.confidence >= self.settings.min_confidence_buy
                    ):
                        in_pos = True
                        entry_idx = i
                        entry_price = close
                        entry_time = df["begin"].iloc[i]
                        entry_score = breakdown.technical_score
                else:
                    exposure_bars += 1
                    bars_held = i - entry_idx
                    ret_pct = (close - entry_price) / entry_price * 100.0
                    if (
                        ret_pct >= take_profit_pct
                        or ret_pct <= -stop_loss_pct
                        or bars_held >= max_bars_hold
                    ):
                        trade = Trade(
                            secid=secid,
                            entry_time=entry_time,  # type: ignore[arg-type]
                            entry_price=entry_price,
                            exit_time=df["begin"].iloc[i],
                            exit_price=close,
                            pnl_pct=ret_pct,
                            score_at_entry=entry_score,
                            bars_held=bars_held,
                        )
                        trades.append(trade)
                        in_pos = False

        result = self._summarize(trades, exposure_bars, total_bars)
        return result

    # ------------------------------------------------------------------
    def _make_synthetic_bundle(self, secid: str, candles_slice: pd.DataFrame) -> FeatureBundle:
        """Build a FeatureBundle using a truncated candles slice and stored stats.

        For simplicity we reuse the live builder but with a *current view* of
        candles - the rest of the data (tradestats etc.) is taken as-is from
        the DB. This is good enough for threshold-tuning sanity checks.
        """
        bundle = self.feature_builder.build(secid)
        try:
            close = candles_slice["close"].astype(float)
            bundle.features["last_close"] = float(close.iloc[-1])
            bundle.last_price = float(close.iloc[-1])
            if len(close) > 12:
                bundle.features["return_12_bar"] = float(
                    (close.iloc[-1] - close.iloc[-13]) / close.iloc[-13]
                )
            if len(close) > 3:
                bundle.features["return_3_bar"] = float(
                    (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]
                )
            if len(close) > 1:
                bundle.features["return_1_bar"] = float(
                    (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
                )
        except Exception:
            pass
        return bundle

    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(trades: List[Trade], exposure_bars: int, total_bars: int) -> BacktestResult:
        if not trades:
            return BacktestResult(trades=[], summary={"trades": 0})
        returns = np.array([t.pnl_pct / 100.0 for t in trades])
        equity_curve = (1.0 + returns).cumprod()
        total_return = float(equity_curve[-1] - 1.0)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0
        winrate = float(np.mean(returns > 0))
        avg_return = float(np.mean(returns))
        exposure = exposure_bars / total_bars if total_bars else 0.0
        return BacktestResult(
            trades=trades,
            total_return=total_return,
            max_drawdown=max_dd,
            winrate=winrate,
            avg_trade_return=avg_return,
            exposure=float(exposure),
            summary={
                "trades": len(trades),
                "total_return_pct": total_return * 100.0,
                "max_drawdown_pct": max_dd * 100.0,
                "winrate_pct": winrate * 100.0,
                "avg_trade_return_pct": avg_return * 100.0,
                "exposure_pct": exposure * 100.0,
            },
        )


def parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()
