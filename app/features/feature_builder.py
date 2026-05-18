"""Build a feature bundle per ticker from local SQLite data.

Pure read-only with respect to repository – it never writes new state itself,
so the same code can be used in production and inside the backtester.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.features.indicators import (
    atr_like,
    bollinger_zscore,
    ema,
    pct_distance,
    realized_volatility,
    rolling_zscore,
    rsi,
    safe_last,
)
from app.logging_config import get_logger
from app.storage.repository import Repository
from app.utils.time import age_seconds, now_msk, parse_moex_datetime

_LOG = get_logger(__name__)

FEATURE_VERSION = "fv-1.0"


@dataclass
class FeatureBundle:
    """Compact feature pack consumed by the scorer/LLM/advisor."""

    secid: str
    ts: datetime
    feature_version: str = FEATURE_VERSION
    features: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    hi2: Dict[str, float] = field(default_factory=dict)
    market_snapshot: Dict[str, Any] = field(default_factory=dict)
    last_price: Optional[float] = None
    lot_size: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "secid": self.secid,
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else str(self.ts),
            "feature_version": self.feature_version,
            "features": self.features,
            "data_quality": self.data_quality,
            "alerts": self.alerts,
            "hi2": self.hi2,
            "market": _sanitize(self.market_snapshot),
            "last_price": self.last_price,
            "lot_size": self.lot_size,
        }


def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (datetime,)):
            out[k] = v.isoformat()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
class FeatureBuilder:
    """Construct a :class:`FeatureBundle` from the repository."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    # ------------------------------------------------------------------
    def build(self, secid: str) -> FeatureBundle:
        secid = secid.upper()
        ts_now = datetime.utcnow()

        ts_df = self.repo.get_tradestats_df(secid, limit=200)
        os_df = self.repo.get_orderstats_df(secid, limit=200)
        ob_df = self.repo.get_obstats_df(secid, limit=200)
        candles_df = self.repo.get_intraday_candles_df(secid, limit=200)
        market = self.repo.get_marketdata(secid) or {}
        hi2 = self.repo.get_latest_hi2(secid)
        alerts_raw = self.repo.get_recent_alerts(
            secid,
            since=datetime.utcnow() - timedelta(hours=12),
            limit=20,
        )

        feats: Dict[str, Any] = {}
        feats.update(self._price_features(ts_df, candles_df))
        feats.update(self._volume_flow_features(ts_df))
        feats.update(self._order_stats_features(os_df))
        feats.update(self._orderbook_features(ob_df))
        feats.update(self._alerts_features(alerts_raw))
        feats.update(self._hi2_features(hi2))

        last_price = (
            market.get("last")
            or feats.get("last_close")
            or (safe_last(candles_df["close"]) if "close" in candles_df.columns else None)
        )
        feats["last_price"] = last_price
        feats["bid"] = market.get("bid")
        feats["offer"] = market.get("offer")
        feats["bid_offer_spread"] = market.get("spread")
        feats["voltoday"] = market.get("voltoday")
        feats["valtoday"] = market.get("valtoday")

        dq = self._data_quality(ts_df, os_df, ob_df, candles_df, market, alerts_raw)

        return FeatureBundle(
            secid=secid,
            ts=ts_now,
            feature_version=FEATURE_VERSION,
            features=_sanitize(feats),
            data_quality=dq,
            alerts=_format_alerts(alerts_raw),
            hi2=hi2,
            market_snapshot=market,
            last_price=last_price,
            lot_size=market.get("lotsize"),
        )

    # ------------------------------------------------------------------
    def _price_features(self, ts_df: pd.DataFrame, candles_df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if ts_df is not None and not ts_df.empty and "pr_close" in ts_df.columns:
            close = ts_df["pr_close"].astype(float)
            out["last_close"] = safe_last(close)
            out["pr_open"] = safe_last(ts_df["pr_open"]) if "pr_open" in ts_df.columns else None
            out["pr_vwap"] = safe_last(ts_df["pr_vwap"]) if "pr_vwap" in ts_df.columns else None
            for lag in (1, 3, 6, 12):
                col = f"return_{lag}_bar"
                if len(close) > lag:
                    diff = (close.iloc[-1] - close.iloc[-1 - lag]) / close.iloc[-1 - lag]
                    out[col] = float(diff) if not math.isnan(diff) else None
                else:
                    out[col] = None
            ema9 = ema(close, 9)
            ema21 = ema(close, 21)
            ema50 = ema(close, 50)
            out["ema9"] = safe_last(ema9)
            out["ema21"] = safe_last(ema21)
            out["ema50"] = safe_last(ema50)
            out["ema_9_above_21"] = (
                None
                if out["ema9"] is None or out["ema21"] is None
                else bool(out["ema9"] > out["ema21"])
            )
            out["close_above_ema21"] = (
                None
                if out["last_close"] is None or out["ema21"] is None
                else bool(out["last_close"] > out["ema21"])
            )
            out["close_above_ema9"] = (
                None
                if out["last_close"] is None or out["ema9"] is None
                else bool(out["last_close"] > out["ema9"])
            )
            out["distance_to_vwap_pct"] = pct_distance(out["last_close"], out["pr_vwap"])
            out["rsi14"] = safe_last(rsi(close, 14))
            out["bollinger_z"] = safe_last(bollinger_zscore(close, 20))
            out["realized_vol"] = safe_last(realized_volatility(close, 20))
        if candles_df is not None and not candles_df.empty:
            for col in ("high", "low", "close"):
                if col not in candles_df.columns:
                    return out
            out["atr_intraday"] = safe_last(
                atr_like(candles_df["high"], candles_df["low"], candles_df["close"], 14)
            )
            # Backwards-compatible alias used elsewhere.
            out["atr_5m"] = out["atr_intraday"]
            close_intra = candles_df["close"].astype(float)
            if len(close_intra) >= 12:
                out["return_intraday_12_bars"] = float(
                    (close_intra.iloc[-1] - close_intra.iloc[-13]) / close_intra.iloc[-13]
                )
            else:
                out["return_intraday_12_bars"] = None
            day_open = None
            if "open" in candles_df.columns and len(candles_df) > 0:
                # First candle of the latest tradedate.
                last_begin = candles_df["begin"].iloc[-1]
                if isinstance(last_begin, datetime):
                    same_day = candles_df[candles_df["begin"] >= last_begin.replace(hour=0, minute=0, second=0)]
                    if not same_day.empty:
                        day_open = float(same_day["open"].iloc[0])
            if day_open is not None and out.get("last_close") is not None:
                out["intraday_return_from_open"] = (out["last_close"] - day_open) / day_open
            else:
                out["intraday_return_from_open"] = None
        return out

    # ------------------------------------------------------------------
    def _volume_flow_features(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "volume_zscore": None,
            "value_zscore": None,
            "trades_zscore": None,
            "buy_volume_ratio": None,
            "sell_volume_ratio": None,
            "disb_now": None,
            "disb_rolling_mean": None,
            "aggressive_buy_pressure": None,
            "pr_vwap_b_minus_s_pct": None,
        }
        if ts_df is None or ts_df.empty:
            return out
        if "vol" in ts_df.columns:
            out["volume_zscore"] = safe_last(rolling_zscore(ts_df["vol"].astype(float), 30))
        if "val" in ts_df.columns:
            out["value_zscore"] = safe_last(rolling_zscore(ts_df["val"].astype(float), 30))
        if "trades" in ts_df.columns:
            out["trades_zscore"] = safe_last(rolling_zscore(ts_df["trades"].astype(float), 30))

        if "vol_b" in ts_df.columns and "vol" in ts_df.columns:
            vol = ts_df["vol"].astype(float).replace(0.0, np.nan)
            out["buy_volume_ratio"] = safe_last((ts_df["vol_b"].astype(float) / vol))
        if "vol_s" in ts_df.columns and "vol" in ts_df.columns:
            vol = ts_df["vol"].astype(float).replace(0.0, np.nan)
            out["sell_volume_ratio"] = safe_last((ts_df["vol_s"].astype(float) / vol))
        if "disb" in ts_df.columns:
            out["disb_now"] = safe_last(ts_df["disb"].astype(float))
            out["disb_rolling_mean"] = safe_last(ts_df["disb"].astype(float).rolling(20, min_periods=3).mean())
        if {"val_b", "val_s"}.issubset(ts_df.columns):
            tot = (ts_df["val_b"].astype(float) + ts_df["val_s"].astype(float)).replace(0.0, np.nan)
            pressure = (ts_df["val_b"].astype(float) - ts_df["val_s"].astype(float)) / tot
            out["aggressive_buy_pressure"] = safe_last(pressure)
        if {"pr_vwap_b", "pr_vwap_s"}.issubset(ts_df.columns):
            pvb = ts_df["pr_vwap_b"].astype(float)
            pvs = ts_df["pr_vwap_s"].astype(float)
            denom = ((pvb + pvs) / 2.0).replace(0.0, np.nan)
            out["pr_vwap_b_minus_s_pct"] = safe_last(((pvb - pvs) / denom) * 100.0)
        return out

    # ------------------------------------------------------------------
    def _order_stats_features(self, os_df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "put_cancel_ratio_b": None,
            "put_cancel_ratio_s": None,
            "net_put_pressure": None,
            "net_cancel_pressure": None,
            "spoof_risk_score": None,
        }
        if os_df is None or os_df.empty:
            return out
        if {"put_val_b", "cancel_val_b"}.issubset(os_df.columns):
            cb = os_df["cancel_val_b"].astype(float).replace(0.0, np.nan)
            out["put_cancel_ratio_b"] = safe_last(os_df["put_val_b"].astype(float) / cb)
        if {"put_val_s", "cancel_val_s"}.issubset(os_df.columns):
            cs = os_df["cancel_val_s"].astype(float).replace(0.0, np.nan)
            out["put_cancel_ratio_s"] = safe_last(os_df["put_val_s"].astype(float) / cs)
        if {"put_val_b", "put_val_s"}.issubset(os_df.columns):
            out["net_put_pressure"] = safe_last(
                (os_df["put_val_b"].astype(float) - os_df["put_val_s"].astype(float))
                / (os_df["put_val_b"].astype(float) + os_df["put_val_s"].astype(float)).replace(0.0, np.nan)
            )
        if {"cancel_val_b", "cancel_val_s"}.issubset(os_df.columns):
            out["net_cancel_pressure"] = safe_last(
                (os_df["cancel_val_s"].astype(float) - os_df["cancel_val_b"].astype(float))
                / (os_df["cancel_val_b"].astype(float) + os_df["cancel_val_s"].astype(float)).replace(0.0, np.nan)
            )
        # Spoof risk: high cancel/put ratio on both sides AND strong imbalance.
        pcr_b = out["put_cancel_ratio_b"]
        pcr_s = out["put_cancel_ratio_s"]
        net_put = out["net_put_pressure"]
        if pcr_b is not None and pcr_s is not None and net_put is not None:
            spoof = 0.0
            if pcr_b > 1.5:
                spoof += 0.4
            if pcr_s > 1.5:
                spoof += 0.4
            if abs(net_put) > 0.4:
                spoof += 0.2
            out["spoof_risk_score"] = min(1.0, spoof)
        return out

    # ------------------------------------------------------------------
    def _orderbook_features(self, ob_df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "spread_bbo": None,
            "spread_1mio": None,
            "imbalance_vol": None,
            "imbalance_val": None,
            "imbalance_bbo": None,
            "liquidity_score": None,
            "levels_b": None,
            "levels_s": None,
            "slippage_1mio_bps": None,
        }
        if ob_df is None or ob_df.empty:
            return out
        last = ob_df.iloc[-1]
        for f in (
            "spread_bbo", "spread_lv10", "spread_1mio",
            "imbalance_vol", "imbalance_val", "imbalance_vol_bbo",
            "vol_b", "vol_s", "val_b", "val_s",
            "vwap_b_1mio", "vwap_s_1mio", "levels_b", "levels_s",
        ):
            if f in last.index:
                v = last.get(f)
                if pd.isna(v):
                    v = None
                if f in out:
                    out[f] = v if v is None else float(v)
                else:
                    out[f] = v if v is None else float(v)
        out["imbalance_bbo"] = out.pop("imbalance_vol_bbo", None) if "imbalance_vol_bbo" in out else last.get("imbalance_vol_bbo")
        if out.get("imbalance_bbo") is not None:
            try:
                out["imbalance_bbo"] = float(out["imbalance_bbo"])
            except Exception:
                out["imbalance_bbo"] = None
        # liquidity_score: simple proxy on log(val_b + val_s) * (1 - spread_1mio_bps/1000)
        vol_liquidity = None
        if out.get("val_b") is not None and out.get("val_s") is not None:
            tot = float(out["val_b"]) + float(out["val_s"])
            vol_liquidity = math.log10(max(tot, 1.0))
        spread_bps_penalty = 0.0
        if out.get("spread_1mio") is not None and out.get("last_price") not in (None, 0):
            # spread is in price units; convert to bps via best-effort: leave as raw.
            pass
        out["liquidity_score"] = (
            None if vol_liquidity is None else max(0.0, min(1.0, (vol_liquidity - 5.0) / 4.0))
        )
        # slippage_1mio approximated as (vwap_s_1mio - vwap_b_1mio) / mid * 10000 bps
        if out.get("vwap_s_1mio") is not None and out.get("vwap_b_1mio") is not None:
            mid = (float(out["vwap_s_1mio"]) + float(out["vwap_b_1mio"])) / 2.0
            if mid > 0:
                out["slippage_1mio_bps"] = (float(out["vwap_s_1mio"]) - float(out["vwap_b_1mio"])) / mid * 10000.0
        return out

    # ------------------------------------------------------------------
    def _alerts_features(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "alerts_positive_count": 0,
            "alerts_negative_count": 0,
            "alerts_extreme_high_price": 0,
            "alerts_extreme_low_price": 0,
            "alerts_recent_types": [],
            "alerts_reference_summary": None,
        }
        if not alerts:
            return out
        pos_types = {"vol_b_99_9_pctl", "vol_b_max", "net_vol_99_9_pctl+", "net_vol_max",
                     "pr_change_99_9_pctl+", "pr_change_max"}
        neg_types = {"vol_s_99_9_pctl", "vol_s_max", "net_vol_99_9_pctl-", "net_vol_min",
                     "pr_change_99_9_pctl-", "pr_change_min", "pr_low_min"}
        types: List[str] = []
        refs_avg_changes: List[float] = []
        refs_up: List[int] = []
        refs_down: List[int] = []
        for a in alerts:
            at = (a.get("alert_type") or "").strip()
            if not at:
                continue
            types.append(at)
            if at in pos_types:
                out["alerts_positive_count"] += 1
            if at in neg_types:
                out["alerts_negative_count"] += 1
            if at == "pr_high_max":
                out["alerts_extreme_high_price"] += 1
            if at == "pr_low_min":
                out["alerts_extreme_low_price"] += 1
            ref = a.get("reference")
            parsed = _parse_alert_reference(ref)
            if parsed:
                # Prefer 15m statistic.
                key = "m_15" if "m_15" in parsed else next(iter(parsed.keys()))
                stat = parsed[key]
                if stat.get("mean_change") is not None:
                    refs_avg_changes.append(stat["mean_change"])
                if stat.get("up") is not None:
                    refs_up.append(stat["up"])
                if stat.get("down") is not None:
                    refs_down.append(stat["down"])
        out["alerts_recent_types"] = types[:10]
        if refs_avg_changes:
            mean_change = float(np.mean(refs_avg_changes))
            tot_up = int(sum(refs_up)) if refs_up else 0
            tot_down = int(sum(refs_down)) if refs_down else 0
            tot_obs = tot_up + tot_down
            hit_rate = tot_up / tot_obs if tot_obs > 0 else None
            out["alerts_reference_summary"] = {
                "mean_change_15m": mean_change,
                "up": tot_up,
                "down": tot_down,
                "hit_rate": hit_rate,
                "n_observations": tot_obs,
            }
        return out

    # ------------------------------------------------------------------
    def _hi2_features(self, hi2: Dict[str, float]) -> Dict[str, Any]:
        if not hi2:
            return {
                "hi2_available": False,
                "hi2_volume": None,
                "hi2_aggressive_buy": None,
                "hi2_aggressive_sell": None,
                "hi2_netflow_buy": None,
                "hi2_netflow_sell": None,
                "hi2_risk_level": "unknown",
            }

        def _g(name: str) -> Optional[float]:
            v = hi2.get(name)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        vol = _g("hhi_volume")
        agg_buy = _g("hhi_active_buy")
        agg_sell = _g("hhi_active_sell")
        nf_buy = _g("hhi_netflow_buy")
        nf_sell = _g("hhi_netflow_sell")
        risk = "low"
        for v in (vol, agg_buy, agg_sell, nf_sell):
            if v is not None and v > 2500:
                risk = "high"
                break
            if v is not None and v > 1500:
                risk = "medium"
        return {
            "hi2_available": True,
            "hi2_volume": vol,
            "hi2_aggressive_buy": agg_buy,
            "hi2_aggressive_sell": agg_sell,
            "hi2_netflow_buy": nf_buy,
            "hi2_netflow_sell": nf_sell,
            "hi2_risk_level": risk,
        }

    # ------------------------------------------------------------------
    def _data_quality(
        self,
        ts_df: pd.DataFrame,
        os_df: pd.DataFrame,
        ob_df: pd.DataFrame,
        candles_df: pd.DataFrame,
        market: Dict[str, Any],
        alerts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _last_age(df: pd.DataFrame, col: str = "ts") -> Optional[float]:
            if df is None or df.empty or col not in df.columns:
                return None
            v = df[col].iloc[-1]
            if isinstance(v, str):
                v = parse_moex_datetime(v)
            return age_seconds(v)

        ts_age = _last_age(ts_df)
        os_age = _last_age(os_df)
        ob_age = _last_age(ob_df)
        cd_age = _last_age(candles_df, "begin")

        market_age = None
        sys_t = market.get("systime") or market.get("updated_at")
        if isinstance(sys_t, str):
            sys_t = parse_moex_datetime(sys_t)
        if sys_t is not None:
            market_age = age_seconds(sys_t)

        alerts_age = None
        if alerts:
            t = alerts[0].get("ts")
            if isinstance(t, str):
                t = parse_moex_datetime(t)
            alerts_age = age_seconds(t)

        missing: List[str] = []
        for name, val in (
            ("tradestats", ts_age),
            ("orderstats", os_age),
            ("obstats", ob_age),
            ("candles_5m", cd_age),
            ("marketdata", market_age),
        ):
            if val is None:
                missing.append(name)

        # Quality score combines freshness and completeness.
        ages = [a for a in (ts_age, os_age, ob_age, cd_age, market_age) if a is not None]
        if ages:
            staleness = min(1.0, np.mean(ages) / (15 * 60.0))
            completeness = 1.0 - len(missing) / 5.0
            quality = max(0.0, min(1.0, completeness * (1.0 - 0.5 * staleness)))
        else:
            quality = 0.0
        freshness_ok = (
            (ts_age is not None and ts_age < 30 * 60)
            and (market_age is None or market_age < 15 * 60)
        )
        return {
            "latest_tradestats_age_sec": ts_age,
            "latest_orderstats_age_sec": os_age,
            "latest_obstats_age_sec": ob_age,
            "latest_candle_age_sec": cd_age,
            "latest_alerts_age_sec": alerts_age,
            "latest_marketdata_age_sec": market_age,
            "missing_sources": missing,
            "freshness_ok": bool(freshness_ok),
            "quality_score": float(quality),
        }


# ---------------------------------------------------------------------------
def _format_alerts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "alert_type": r.get("alert_type"),
            "ts": r.get("ts").isoformat() if isinstance(r.get("ts"), datetime) else r.get("ts"),
            "threshold": r.get("threshold"),
            "value": r.get("value"),
            "reference": r.get("reference"),
        })
    return out


def _parse_alert_reference(raw: Any) -> Optional[Dict[str, Dict[str, Optional[float]]]]:
    """Parse the JSON-encoded ``reference`` field of an alert.

    Spec example:
        ``{"m_5":["0.318","-0.224","33","24","0.09"], ...}``
    Where the array is approximately ``[mean_change, threshold?, up, down, ???]``.
    We are conservative – we only trust ``mean_change`` and the up/down counts.
    """
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key, arr in raw.items():
        if not isinstance(arr, (list, tuple)):
            continue

        def _f(v: Any) -> Optional[float]:
            try:
                return float(v) if v not in (None, "", "null") else None
            except Exception:
                return None

        def _i(v: Any) -> Optional[int]:
            try:
                return int(float(v))
            except Exception:
                return None

        mean_change = _f(arr[0]) if len(arr) > 0 else None
        up = _i(arr[2]) if len(arr) > 2 else None
        down = _i(arr[3]) if len(arr) > 3 else None
        out[str(key)] = {"mean_change": mean_change, "up": up, "down": down}
    return out or None
