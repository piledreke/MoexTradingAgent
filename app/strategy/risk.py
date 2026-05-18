"""Hard risk-veto + portfolio sizing rules.

This module is intentionally *deterministic and conservative*: nothing here
can be overridden by the LLM. If a hard veto triggers, BUY is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.features.feature_builder import FeatureBundle
from app.strategy.recommendation import CurrentPosition, PortfolioState


@dataclass
class RiskAssessment:
    hard_veto: bool = False
    risk_flags: List[str] = field(default_factory=list)
    portfolio_constraints: Dict[str, Any] = field(default_factory=dict)
    max_cash_rub: Optional[float] = None
    position_size_multiplier: float = 0.0


class RiskManager:
    """Encapsulates universe / freshness / spread / portfolio veto rules."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._universe = set(self.settings.universe)

    # ------------------------------------------------------------------
    def assess_buy(
        self,
        bundle: FeatureBundle,
        intended_cash_rub: Optional[float],
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
    ) -> RiskAssessment:
        s = self.settings
        out = RiskAssessment()
        features = bundle.features

        # 1. Universe gate.
        if bundle.secid not in self._universe:
            out.risk_flags.append("ticker_not_in_universe")
            out.hard_veto = True

        # 2. Freshness gate.
        dq = bundle.data_quality
        ts_age = dq.get("latest_tradestats_age_sec")
        md_age = dq.get("latest_marketdata_age_sec")
        if (ts_age is None and md_age is None) or (
            (ts_age is not None and ts_age > s.max_stale_seconds)
            and (md_age is None or md_age > s.max_stale_seconds)
        ):
            out.risk_flags.append("stale_data")
            out.hard_veto = True
        if not dq.get("freshness_ok", False):
            out.risk_flags.append("freshness_low")

        # 3. Spread gates.
        last_price = features.get("last_price") or features.get("last_close")
        spread_bbo = features.get("spread_bbo")
        spread_1mio = features.get("spread_1mio")
        if last_price and last_price > 0:
            if spread_bbo is not None:
                spread_bps = spread_bbo / last_price * 10000.0
                if spread_bps > s.max_spread_bbo_bps:
                    out.risk_flags.append(f"spread_bbo_too_wide_{spread_bps:.0f}bps")
                    out.hard_veto = True
            if spread_1mio is not None:
                spread_bps_1mio = spread_1mio / last_price * 10000.0
                if spread_bps_1mio > s.max_spread_1mio_bps:
                    out.risk_flags.append(f"spread_1mio_too_wide_{spread_bps_1mio:.0f}bps")
                    out.hard_veto = True

        # 4. Liquidity gate.
        val_b = features.get("val_b")
        liq = features.get("liquidity_score")
        if val_b is not None and val_b < s.min_liquidity_val_b:
            out.risk_flags.append("liquidity_too_low")
            out.hard_veto = True
        if liq is not None and liq < 0.1:
            out.risk_flags.append("liquidity_score_critical")
            out.hard_veto = True

        # 5. Volatility gate.
        rv = features.get("realized_vol")
        if rv is not None and rv > 0.08:
            out.risk_flags.append("extreme_volatility")
            out.hard_veto = True

        # 6. Sell pressure / sharp drop without recovery.
        r1 = features.get("return_1_bar")
        r3 = features.get("return_3_bar")
        agg = features.get("aggressive_buy_pressure")
        if r1 is not None and r1 < -0.025 and (r3 is None or r3 < -0.02):
            out.risk_flags.append("sharp_drop_no_recovery")
            out.hard_veto = True
        if agg is not None and agg < -0.35:
            out.risk_flags.append("extreme_sell_pressure")
            out.hard_veto = True

        # 7. Spoof.
        spoof = features.get("spoof_risk_score")
        if spoof is not None and spoof >= 0.8:
            out.risk_flags.append("spoof_risk_high")
            out.hard_veto = True

        # 8. Portfolio constraints.
        max_cash_rub, mult, port_flags, hard = self._portfolio_constraints(
            last_price=last_price,
            intended_cash_rub=intended_cash_rub,
            portfolio_state=portfolio_state,
            current_position=current_position,
        )
        out.max_cash_rub = max_cash_rub
        out.position_size_multiplier = mult
        out.risk_flags.extend(port_flags)
        if hard:
            out.hard_veto = True

        out.portfolio_constraints = self._constraints_summary(
            last_price=last_price,
            max_cash_rub=max_cash_rub,
            portfolio_state=portfolio_state,
            current_position=current_position,
        )
        return out

    # ------------------------------------------------------------------
    def _portfolio_constraints(
        self,
        last_price: Optional[float],
        intended_cash_rub: Optional[float],
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
    ) -> tuple[Optional[float], float, List[str], bool]:
        """Return ``(max_cash_rub, multiplier, flags, hard_veto)``."""
        s = self.settings
        flags: List[str] = []
        hard = False
        equity = (
            portfolio_state.equity_rub
            if portfolio_state and portfolio_state.equity_rub > 0
            else s.initial_capital_rub
        )
        if portfolio_state is None:
            # No portfolio context - we use defaults.
            max_cash = equity * s.max_single_order_pct
            return max_cash, 1.0, flags, False

        if portfolio_state.daily_trades_count >= portfolio_state.daily_trade_limit:
            flags.append("daily_trade_limit_reached")
            hard = True

        # Soft warning approaching the limit.
        if (
            portfolio_state.daily_trade_limit > 0
            and portfolio_state.daily_trades_count
            >= int(0.9 * portfolio_state.daily_trade_limit)
        ):
            flags.append("daily_trade_limit_near")

        # Existing exposure check.
        existing_value = current_position.market_value_rub if current_position else 0.0
        max_position_rub = equity * s.max_position_pct
        if existing_value >= max_position_rub:
            flags.append("position_limit_reached")
            hard = True

        room_in_ticker = max(0.0, max_position_rub - existing_value)
        per_order_cap = equity * s.max_single_order_pct
        cash_cap = max(
            0.0,
            portfolio_state.cash_rub - equity * s.reserve_cash_pct,
        )
        exposure_after_target = equity * s.max_portfolio_exposure_pct
        exposure_room = max(0.0, exposure_after_target - portfolio_state.positions_value_rub)

        max_cash = min(per_order_cap, room_in_ticker, cash_cap, exposure_room)
        if intended_cash_rub is not None:
            max_cash = min(max_cash, intended_cash_rub)
        if max_cash < s.min_order_cash_rub:
            flags.append("insufficient_cash_after_limits")
            hard = True
            max_cash = 0.0

        if exposure_room <= 0:
            flags.append("portfolio_exposure_limit_reached")
            hard = True

        # Determine multiplier of intended_cash_rub.
        if intended_cash_rub and intended_cash_rub > 0:
            multiplier = max_cash / intended_cash_rub
        else:
            multiplier = max_cash / per_order_cap if per_order_cap > 0 else 0.0
        multiplier = max(0.0, min(1.0, multiplier))
        return max_cash, multiplier, flags, hard

    # ------------------------------------------------------------------
    def _constraints_summary(
        self,
        last_price: Optional[float],
        max_cash_rub: Optional[float],
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
    ) -> Dict[str, Any]:
        s = self.settings
        equity = (
            portfolio_state.equity_rub
            if portfolio_state and portfolio_state.equity_rub > 0
            else s.initial_capital_rub
        )
        summary = {
            "equity_used_rub": equity,
            "max_position_pct": s.max_position_pct,
            "max_single_order_pct": s.max_single_order_pct,
            "max_portfolio_exposure_pct": s.max_portfolio_exposure_pct,
            "reserve_cash_pct": s.reserve_cash_pct,
            "min_order_cash_rub": s.min_order_cash_rub,
            "daily_trade_limit": (
                portfolio_state.daily_trade_limit if portfolio_state else s.daily_trade_limit
            ),
            "daily_trades_count": (
                portfolio_state.daily_trades_count if portfolio_state else None
            ),
            "max_cash_rub": max_cash_rub,
            "last_price": last_price,
            "current_position_value_rub": (
                current_position.market_value_rub if current_position else 0.0
            ),
        }
        return summary
