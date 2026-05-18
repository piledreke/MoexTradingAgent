"""TechnicalAdvisor - the public service class.

This is the ONLY place where the BUY_CHECK / EXIT_CHECK decision logic lives.
The FastAPI route ``POST /advice`` is a thin wrapper that validates JSON, calls
:meth:`TechnicalAdvisor.get_advice` and returns the response unchanged.

Hard guarantees:
* The advisor NEVER opens a short.
* The advisor NEVER calls Arenago ``submit_order``.
* For EXIT_CHECK the worst case is "EXIT_POSITION" - this is advice to close
  an existing long, not a short signal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.clients.arena_client import ArenagoClient
from app.config import Settings, get_settings
from app.features.anomaly_detector import AnomalyDetector
from app.features.feature_builder import FeatureBuilder, FeatureBundle
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.strategy.llm_advisor import LLMAdvisor
from app.strategy.recommendation import (
    AdviceIntent,
    AdviceRequest,
    AdviceResponse,
    AggregateAction,
    BuyAction,
    CurrentPosition,
    ExitAction,
    ExitUrgency,
    PortfolioState,
)
from app.strategy.risk import RiskManager
from app.strategy.scoring import ScoreBreakdown, TechnicalScorer

_LOG = get_logger(__name__)


class TechnicalAdvisor:
    """High-level orchestrator that the news agent / FastAPI calls."""

    def __init__(
        self,
        repo: Repository,
        settings: Optional[Settings] = None,
        scorer: Optional[TechnicalScorer] = None,
        risk_manager: Optional[RiskManager] = None,
        llm_advisor: Optional[LLMAdvisor] = None,
        arena_client: Optional[ArenagoClient] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = repo
        self.scorer = scorer or TechnicalScorer()
        self.risk_manager = risk_manager or RiskManager(self.settings)
        self.llm_advisor = llm_advisor or LLMAdvisor(repo, settings=self.settings)
        self.arena_client = arena_client
        self.feature_builder = feature_builder or FeatureBuilder(repo)
        self.anomaly_detector = anomaly_detector or AnomalyDetector(repo)

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def get_advice(self, request: AdviceRequest) -> AdviceResponse:
        secid = request.secid.upper()
        bundle = self.feature_builder.build(secid)
        # Side effect (fire-and-forget): record anomalies derived from the
        # freshly-built features. This is cheap and safe to do here too.
        try:
            self.anomaly_detector.detect(bundle)
        except Exception as exc:  # pragma: no cover
            _LOG.warning("anomaly_detect_failed", extra={"secid": secid, "error": str(exc)})

        breakdown = self.scorer.score(bundle)
        portfolio_state, current_position = self._resolve_portfolio_context(
            request.portfolio, request.position, request.portfolio_name, secid
        )

        if request.intent == AdviceIntent.EXIT_CHECK:
            response = self._build_exit_check(
                request=request,
                bundle=bundle,
                breakdown=breakdown,
                portfolio_state=portfolio_state,
                current_position=current_position,
            )
        else:
            response = self._build_buy_check(
                request=request,
                bundle=bundle,
                breakdown=breakdown,
                portfolio_state=portfolio_state,
                current_position=current_position,
            )

        # Persist the recommendation for audit + GET /recommendations.
        self._persist_recommendation(bundle, response)
        return response

    # ==================================================================
    # BUY_CHECK
    # ==================================================================
    def _build_buy_check(
        self,
        request: AdviceRequest,
        bundle: FeatureBundle,
        breakdown: ScoreBreakdown,
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
    ) -> AdviceResponse:
        risk = self.risk_manager.assess_buy(
            bundle=bundle,
            intended_cash_rub=request.intended_cash_rub,
            portfolio_state=portfolio_state,
            current_position=current_position,
        )

        det_decision = self._deterministic_buy_decision(
            score=breakdown.technical_score,
            confidence=breakdown.confidence,
            hard_veto=risk.hard_veto,
        )

        llm_result = self.llm_advisor.evaluate(
            bundle=bundle,
            breakdown=breakdown,
            deterministic_decision={
                "intent": "BUY_CHECK",
                "secid": bundle.secid,
                "technical_score": round(breakdown.technical_score, 2),
                "confidence": round(breakdown.confidence, 3),
                "hard_veto": risk.hard_veto,
                "risk_flags": risk.risk_flags,
                "recommended_action": det_decision,
            },
            intent="BUY_CHECK",
        )
        adj_score, adj_conf, llm_comment = self._apply_llm_adjustments(
            breakdown=breakdown, llm_result=llm_result, allow_relax=(not risk.hard_veto)
        )

        final_action = self._final_buy_action(
            score=adj_score, confidence=adj_conf, hard_veto=risk.hard_veto
        )
        allow_buy = final_action == BuyAction.BUY and not risk.hard_veto

        position_size_multiplier = (
            risk.position_size_multiplier if allow_buy else 0.0
        )
        # Soften size for WAIT recommendation.
        if final_action == BuyAction.WAIT:
            position_size_multiplier = min(position_size_multiplier, 0.4)

        recommended_cash_rub = (
            (request.intended_cash_rub or risk.max_cash_rub or 0.0)
            * position_size_multiplier
            if allow_buy
            else 0.0
        )
        recommended_quantity = self._compute_quantity(
            cash_rub=recommended_cash_rub,
            last_price=bundle.last_price,
            lot_size=bundle.lot_size,
        ) if allow_buy else 0

        aggregate = self._buy_to_aggregate(final_action)
        stop_loss_pct, take_profit_pct = self._suggest_levels(bundle, breakdown)

        # Combine deterministic reasons + llm reasons (no duplicate, conservative).
        reasons = list(dict.fromkeys(breakdown.reasons))
        negatives = list(dict.fromkeys(breakdown.negative_factors))
        if llm_result.get("response"):
            llm_resp = llm_result["response"]
            for r in (llm_resp.supporting_reasons or [])[:5]:
                if r and r not in reasons:
                    reasons.append(f"llm: {r}")
            for r in (llm_resp.risks or [])[:5]:
                if r and r not in negatives:
                    negatives.append(f"llm: {r}")

        return AdviceResponse(
            secid=bundle.secid,
            timestamp=datetime.utcnow(),
            intent=AdviceIntent.BUY_CHECK,
            strategy_version=self.settings.strategy_version,
            action=aggregate,
            recommended_action=final_action.value,
            allow_buy=allow_buy,
            allow_action=allow_buy,
            exit_warning=False,
            exit_urgency=ExitUrgency.NONE,
            technical_score=round(adj_score, 2),
            confidence=round(adj_conf, 3),
            position_size_multiplier=round(position_size_multiplier, 3),
            horizon_minutes=request.horizon_minutes,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_cash_rub=risk.max_cash_rub,
            recommended_cash_rub=round(recommended_cash_rub, 2) if allow_buy else 0.0,
            recommended_quantity=recommended_quantity,
            recommended_sell_quantity=None,
            reasons=reasons,
            negative_factors=negatives,
            risk_flags=risk.risk_flags,
            feature_snapshot=bundle.as_dict(),
            data_quality=bundle.data_quality,
            portfolio_constraints=risk.portfolio_constraints,
            llm_used=bool(llm_result.get("used")),
            llm_comment=llm_comment,
        )

    # ==================================================================
    # EXIT_CHECK
    # ==================================================================
    def _build_exit_check(
        self,
        request: AdviceRequest,
        bundle: FeatureBundle,
        breakdown: ScoreBreakdown,
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
    ) -> AdviceResponse:
        risk_flags: List[str] = []
        # If no position, just return safe HOLD_POSITION.
        if current_position is None or current_position.quantity <= 0:
            risk_flags.append("no_position_to_exit")
            return AdviceResponse(
                secid=bundle.secid,
                timestamp=datetime.utcnow(),
                intent=AdviceIntent.EXIT_CHECK,
                strategy_version=self.settings.strategy_version,
                action=AggregateAction.HOLD,
                recommended_action=ExitAction.HOLD_POSITION.value,
                allow_buy=False,
                allow_action=False,
                exit_warning=False,
                exit_urgency=ExitUrgency.NONE,
                technical_score=round(breakdown.technical_score, 2),
                confidence=round(breakdown.confidence, 3),
                position_size_multiplier=0.0,
                horizon_minutes=request.horizon_minutes,
                reasons=["No long position is currently held; nothing to exit."],
                negative_factors=[],
                risk_flags=risk_flags,
                feature_snapshot=bundle.as_dict(),
                data_quality=bundle.data_quality,
                portfolio_constraints={"current_position_quantity": 0},
                llm_used=False,
                llm_comment=None,
            )

        exit_signals = self._exit_signals(bundle.features, current_position)
        severity = exit_signals["severity"]
        reasons_pos: List[str] = list(breakdown.reasons)
        negatives = list(dict.fromkeys(breakdown.negative_factors + exit_signals["reasons"]))

        # Map severity -> action.
        if exit_signals["hard_exit"]:
            action = ExitAction.EXIT_POSITION
            urgency = ExitUrgency.HIGH
        elif severity >= 0.6:
            action = ExitAction.EXIT_POSITION
            urgency = ExitUrgency.HIGH
        elif severity >= 0.35:
            action = ExitAction.TRIM_POSITION
            urgency = ExitUrgency.MEDIUM
        elif severity >= 0.15:
            action = ExitAction.HOLD_POSITION
            urgency = ExitUrgency.LOW
        else:
            action = ExitAction.HOLD_POSITION
            urgency = ExitUrgency.NONE

        det_decision = {
            "intent": "EXIT_CHECK",
            "secid": bundle.secid,
            "technical_score": round(breakdown.technical_score, 2),
            "confidence": round(breakdown.confidence, 3),
            "exit_severity": severity,
            "exit_action": action.value,
            "exit_urgency": urgency.value,
        }
        llm_result = self.llm_advisor.evaluate(
            bundle=bundle,
            breakdown=breakdown,
            deterministic_decision=det_decision,
            intent="EXIT_CHECK",
        )
        adj_score, adj_conf, llm_comment = self._apply_llm_adjustments(
            breakdown=breakdown, llm_result=llm_result, allow_relax=True
        )

        # Recommend partial sell quantity for TRIM.
        sell_qty = 0
        if action == ExitAction.EXIT_POSITION:
            sell_qty = current_position.quantity
        elif action == ExitAction.TRIM_POSITION:
            sell_qty = max(1, int(current_position.quantity * 0.5))

        # Compose aggregate / risk_flags.
        aggregate = (
            AggregateAction.AVOID if action == ExitAction.EXIT_POSITION else AggregateAction.HOLD
        )
        return AdviceResponse(
            secid=bundle.secid,
            timestamp=datetime.utcnow(),
            intent=AdviceIntent.EXIT_CHECK,
            strategy_version=self.settings.strategy_version,
            action=aggregate,
            recommended_action=action.value,
            allow_buy=False,
            allow_action=(action != ExitAction.HOLD_POSITION),
            exit_warning=(action != ExitAction.HOLD_POSITION),
            exit_urgency=urgency,
            technical_score=round(adj_score, 2),
            confidence=round(adj_conf, 3),
            position_size_multiplier=0.0,
            horizon_minutes=request.horizon_minutes,
            stop_loss_pct=None,
            take_profit_pct=None,
            max_cash_rub=None,
            recommended_cash_rub=None,
            recommended_quantity=None,
            recommended_sell_quantity=sell_qty if sell_qty > 0 else None,
            reasons=reasons_pos,
            negative_factors=negatives,
            risk_flags=risk_flags + exit_signals["risk_flags"],
            feature_snapshot=bundle.as_dict(),
            data_quality=bundle.data_quality,
            portfolio_constraints={
                "current_position_quantity": current_position.quantity,
                "current_position_value_rub": current_position.market_value_rub,
                "unrealized_pnl_pct": current_position.unrealized_pnl_pct,
            },
            llm_used=bool(llm_result.get("used")),
            llm_comment=llm_comment,
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _deterministic_buy_decision(self, score: float, confidence: float, hard_veto: bool) -> str:
        s = self.settings
        if hard_veto or score < s.hold_score_threshold:
            return BuyAction.DO_NOT_BUY.value
        if score >= s.buy_score_threshold and confidence >= s.min_confidence_buy:
            return BuyAction.BUY.value
        return BuyAction.WAIT.value

    def _final_buy_action(self, score: float, confidence: float, hard_veto: bool) -> BuyAction:
        s = self.settings
        if hard_veto:
            return BuyAction.DO_NOT_BUY
        if score < s.hold_score_threshold:
            return BuyAction.DO_NOT_BUY
        if score >= s.buy_score_threshold and confidence >= s.min_confidence_buy:
            return BuyAction.BUY
        return BuyAction.WAIT

    @staticmethod
    def _buy_to_aggregate(action: BuyAction) -> AggregateAction:
        if action == BuyAction.BUY:
            return AggregateAction.BUY
        if action == BuyAction.WAIT:
            return AggregateAction.HOLD
        return AggregateAction.AVOID

    @staticmethod
    def _apply_llm_adjustments(
        breakdown: ScoreBreakdown,
        llm_result: Dict[str, Any],
        allow_relax: bool,
    ) -> tuple[float, float, Optional[str]]:
        """Apply *additive* LLM tweaks; hard veto cannot be bypassed."""
        score = breakdown.technical_score
        conf = breakdown.confidence
        comment: Optional[str] = None
        resp = llm_result.get("response")
        if not llm_result.get("used") or resp is None:
            return score, conf, comment
        sa = int(getattr(resp, "score_adjustment", 0) or 0)
        ca = float(getattr(resp, "confidence_adjustment", 0.0) or 0.0)
        comment = getattr(resp, "final_comment", None)
        sa = max(-10, min(10, sa))
        ca = max(-0.15, min(0.15, ca))
        if not allow_relax:
            # Only allow LLM to *lower* score/confidence when hard veto active.
            sa = min(0, sa)
            ca = min(0.0, ca)
        score = max(0.0, min(100.0, score + sa))
        conf = max(0.0, min(1.0, conf + ca))
        return score, conf, comment

    def _compute_quantity(
        self,
        cash_rub: float,
        last_price: Optional[float],
        lot_size: Optional[int],
    ) -> int:
        if cash_rub <= 0 or not last_price or last_price <= 0:
            return 0
        lot = int(lot_size or 1)
        if lot <= 0:
            lot = 1
        # Buy in whole lots only.
        max_lots = int(cash_rub // (last_price * lot))
        return max(0, max_lots * lot)

    @staticmethod
    def _suggest_levels(
        bundle: FeatureBundle, breakdown: ScoreBreakdown
    ) -> tuple[Optional[float], Optional[float]]:
        rv = bundle.features.get("realized_vol")
        atr = bundle.features.get("atr_5m")
        last_price = bundle.last_price or bundle.features.get("last_close")
        if not last_price or last_price <= 0:
            return None, None
        # Default tight intraday targets.
        stop_loss_pct = 0.7
        take_profit_pct = 1.5
        if isinstance(atr, (int, float)) and atr > 0:
            atr_pct = atr / last_price * 100.0
            stop_loss_pct = max(0.4, min(2.0, atr_pct * 1.5))
            take_profit_pct = max(0.8, min(3.5, atr_pct * 2.5))
        if isinstance(rv, (int, float)) and rv > 0.03:
            stop_loss_pct = min(2.5, stop_loss_pct * 1.3)
            take_profit_pct = min(4.0, take_profit_pct * 1.2)
        return round(stop_loss_pct, 2), round(take_profit_pct, 2)

    # ------------------------------------------------------------------
    def _resolve_portfolio_context(
        self,
        portfolio_state: Optional[PortfolioState],
        current_position: Optional[CurrentPosition],
        portfolio_name: Optional[str],
        secid: str,
    ) -> tuple[Optional[PortfolioState], Optional[CurrentPosition]]:
        """If caller did not supply portfolio info, optionally read from Arenago."""
        if portfolio_state is not None or current_position is not None:
            return portfolio_state, current_position
        if not portfolio_name or self.arena_client is None or not self.arena_client.enabled:
            return portfolio_state, current_position

        # 1) Cash balance — Arenago exposes this on /api/bots, NOT on /api/positions.
        cash_rub = 0.0
        bots = self.arena_client.get_bots() or []
        for b in bots:
            if (b.get("name") or "").strip() == portfolio_name.strip():
                cash_rub = _to_float(b.get("cash_balance") or b.get("cash") or 0) or 0.0
                break

        # 2) Today's trade count — for daily limit awareness.
        trades_today = 0
        trades = self.arena_client.get_trades(portfolio_name) or []
        trades_today = len(trades)

        # 3) Positions — Arenago payload shape:
        #    {"secid": "VTBR", "position": 2220, "average_price": 89.285,
        #     "direction": "B", "bot": "...", "nickname": "...",
        #     "updatedate": "2026-05-15", "updatetime": "15:24:50.648972"}
        positions = self.arena_client.get_positions(portfolio_name) or []
        positions_value = 0.0
        cur_pos: Optional[CurrentPosition] = None
        for row in positions:
            sid = (row.get("secid") or row.get("ticker") or "").upper()
            direction = (row.get("direction") or "B").upper()
            qty = _to_int(
                row.get("position")
                or row.get("quantity")
                or row.get("qty")
                or 0
            )
            avg_px = _to_float(
                row.get("average_price")
                or row.get("avg_price")
                or row.get("price")
                or 0
            ) or 0.0
            # Try to get the current price from cached MOEX marketdata.
            last_price = self._latest_price(sid) if sid else None
            ref_price = last_price if last_price is not None and last_price > 0 else avg_px
            mv = _to_float(
                row.get("market_value")
                or row.get("market_value_rub")
                or (qty * ref_price if qty and ref_price else 0)
            ) or 0.0
            unrl: Optional[float] = None
            if avg_px > 0 and last_price is not None and last_price > 0 and qty:
                # For long positions PnL = (last - avg) / avg * 100
                unrl = (last_price - avg_px) / avg_px * 100.0
                if direction == "S":  # safety: sign-flip just in case
                    unrl = -unrl
            unrl = unrl if unrl is not None else (
                _to_float(row.get("unrealized_pnl_pct") or row.get("pnl_pct") or 0) or 0.0
            )
            if sid in ("RUB", "CASH"):
                cash_rub += mv
                continue
            # Only consider long ("B") positions — by design we never short.
            if direction == "B" and qty > 0:
                positions_value += mv
            if sid == secid and direction == "B" and qty > 0:
                cur_pos = CurrentPosition(
                    quantity=qty,
                    average_price=avg_px,
                    market_value_rub=mv,
                    unrealized_pnl_pct=unrl,
                )

        equity = cash_rub + positions_value
        ps = PortfolioState(
            cash_rub=cash_rub,
            equity_rub=equity if equity > 0 else self.settings.initial_capital_rub,
            positions_value_rub=positions_value,
            daily_trades_count=trades_today,
            daily_trade_limit=self.settings.daily_trade_limit,
        )
        _LOG.info(
            "portfolio_resolved",
            extra={
                "portfolio": portfolio_name,
                "cash_rub": round(cash_rub, 2),
                "positions_value_rub": round(positions_value, 2),
                "equity_rub": round(equity, 2),
                "n_positions": sum(1 for r in positions if (r.get("direction") or "B").upper() == "B"),
                "daily_trades_count": trades_today,
                "has_current_position": cur_pos is not None,
            },
        )
        return ps, cur_pos

    def _latest_price(self, secid: str) -> Optional[float]:
        """Best-effort latest price from cached MOEX marketdata."""
        try:
            row = self.repo.get_marketdata(secid)
            if not row:
                return None
            if not isinstance(row, dict):
                row = {c: getattr(row, c, None) for c in ("last", "lastvalue", "last_price")}
            for k in ("last", "lastvalue", "last_price", "close"):
                v = row.get(k)
                if v is None:
                    continue
                try:
                    f = float(v)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    continue
        except Exception:  # pragma: no cover
            return None
        return None

    # ------------------------------------------------------------------
    def _exit_signals(
        self,
        features: Dict[str, Any],
        position: CurrentPosition,
    ) -> Dict[str, Any]:
        """Compute severity (0..1) and reasons for an EXIT_CHECK."""
        severity = 0.0
        risk_flags: List[str] = []
        reasons: List[str] = []
        hard_exit = False

        below_ema21 = features.get("close_above_ema21") is False
        below_ema9 = features.get("close_above_ema9") is False
        if below_ema9 and below_ema21:
            severity += 0.2
            reasons.append("price below EMA9 and EMA21")

        r1 = _f(features, "return_1_bar")
        r3 = _f(features, "return_3_bar")
        if r1 is not None and r1 < -0.01:
            severity += 0.15
            reasons.append(f"sharp recent drop {r1*100:.2f}%")
        if r3 is not None and r3 < -0.025:
            severity += 0.15
            reasons.append(f"multi-bar drop {r3*100:.2f}%")

        rsi = _f(features, "rsi14")
        if rsi is not None and rsi < 40:
            severity += 0.1
            reasons.append(f"RSI breakdown ({rsi:.1f})")
            if rsi < 30:
                severity += 0.1

        sell_ratio = _f(features, "sell_volume_ratio")
        if sell_ratio is not None and sell_ratio > 0.6:
            severity += 0.1
            reasons.append(f"sell volume dominance ({sell_ratio*100:.0f}%)")

        imb = _f(features, "imbalance_bbo")
        if imb is not None and imb < -0.2:
            severity += 0.1
            reasons.append("negative orderbook imbalance")

        if int(features.get("alerts_negative_count") or 0) > 0:
            severity += 0.1
            reasons.append("negative MegaAlerts present")

        rv = _f(features, "realized_vol")
        if rv is not None and rv > 0.05:
            severity += 0.05
            reasons.append(f"elevated realized vol ({rv*100:.2f}%)")

        # Position-level stop loss (advisory).
        if position.unrealized_pnl_pct is not None and position.unrealized_pnl_pct <= -1.5:
            severity += 0.2
            reasons.append(f"unrealized PnL {position.unrealized_pnl_pct:.2f}% past stop")
            if position.unrealized_pnl_pct <= -3.0:
                hard_exit = True
                risk_flags.append("stop_loss_breach")

        # Liquidity collapse on existing long is an exit trigger.
        liq = _f(features, "liquidity_score")
        if liq is not None and liq < 0.15:
            severity += 0.15
            reasons.append("liquidity deterioration on the book")
            risk_flags.append("liquidity_deterioration")

        if r1 is not None and r1 < -0.03 and sell_ratio and sell_ratio > 0.65:
            hard_exit = True
            risk_flags.append("hard_exit_drop_with_sell_pressure")

        severity = min(1.0, severity)
        return {
            "severity": severity,
            "reasons": reasons,
            "risk_flags": risk_flags,
            "hard_exit": hard_exit,
        }

    # ------------------------------------------------------------------
    def _persist_recommendation(self, bundle: FeatureBundle, response: AdviceResponse) -> None:
        try:
            payload = response.model_dump(mode="json")
        except Exception:
            payload = response.model_dump()
        try:
            self.repo.save_features(
                secid=bundle.secid,
                ts=bundle.ts,
                feature_version=bundle.feature_version,
                features=bundle.features,
            )
        except Exception as exc:
            _LOG.warning("save_features_failed", extra={"secid": bundle.secid, "error": str(exc)})
        try:
            self.repo.save_recommendation(
                secid=response.secid,
                ts=response.timestamp,
                action=response.action.value,
                recommended_action=response.recommended_action,
                allow_buy=response.allow_buy,
                score=response.technical_score,
                confidence=response.confidence,
                recommendation_json=payload,
                strategy_version=response.strategy_version,
                llm_used=response.llm_used,
                intent=response.intent.value,
            )
        except Exception as exc:  # pragma: no cover
            _LOG.warning(
                "save_recommendation_failed",
                extra={"secid": response.secid, "error": str(exc)},
            )

    # ==================================================================
    # Convenience helpers for the scheduler / CLI
    # ==================================================================
    def generate_for_universe(self) -> List[AdviceResponse]:
        """Generate a BUY_CHECK recommendation for every ticker in the universe."""
        out: List[AdviceResponse] = []
        for secid in self.settings.universe:
            try:
                req = AdviceRequest(secid=secid, intent=AdviceIntent.BUY_CHECK)
                out.append(self.get_advice(req))
            except Exception as exc:
                _LOG.warning("advice_failed", extra={"secid": secid, "error": str(exc)})
        return out


# ---------------------------------------------------------------------------
def _f(features: Dict[str, Any], key: str) -> Optional[float]:
    v = features.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:
        return None
    return f


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except Exception:
        return None
