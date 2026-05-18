"""High-level data access for ingestion, features and recommendations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type

import pandas as pd
from sqlalchemy import delete, desc, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.storage.db import Database
from app.storage.models import (
    AgentEvents,
    Anomalies,
    Base,
    DerivedFeatures,
    EqAlerts,
    EqHi2,
    EqObStats,
    EqOrderStats,
    EqTradeStats,
    IntradayCandle,
    LLMCache,
    LLMLog,
    MarketData,
    Recommendations,
)
from app.utils.time import merge_date_time, parse_moex_datetime

_LOG = get_logger(__name__)


_TS_FIELDS = ("ts", "begin", "end", "systime", "updated_at", "ingested_at", "created_at")


def _coerce_row(row: Mapping[str, Any], columns: set[str]) -> Dict[str, Any]:
    """Return a dict containing only known columns with normalised values."""
    out: Dict[str, Any] = {}
    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if val is None or val == "":
            out[col] = None
            continue
        if col in _TS_FIELDS and isinstance(val, str):
            out[col] = parse_moex_datetime(val)
        else:
            out[col] = val
    return out


def _model_columns(model: Type[Base]) -> set[str]:
    return {c.name for c in model.__table__.columns}


class Repository:
    """All write/read helpers used by ingestion + strategy modules."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.engine: Engine = db.engine

    # ------------------------------------------------------------------
    # Generic upsert helpers (SQLite uses ON CONFLICT)
    # ------------------------------------------------------------------
    def _upsert_many(
        self,
        model: Type[Base],
        rows: Sequence[Mapping[str, Any]],
        conflict_cols: Sequence[str],
    ) -> int:
        if not rows:
            return 0
        cols = _model_columns(model)
        prepared: List[Dict[str, Any]] = []
        for r in rows:
            normalised = _coerce_row(r, cols)
            # Auto-fill ``ts`` from tradedate+tradetime if missing.
            if "ts" in cols and not normalised.get("ts"):
                td = normalised.get("tradedate") or r.get("tradedate")
                tt = normalised.get("tradetime") or r.get("tradetime")
                if td:
                    normalised["ts"] = merge_date_time(str(td), str(tt) if tt else None)
            prepared.append(normalised)
        if not prepared:
            return 0
        stmt = sqlite_insert(model).values(prepared)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in model.__table__.columns
            if c.name not in conflict_cols and not c.primary_key
        }
        stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=update_cols)
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount or len(prepared)

    # ------------------------------------------------------------------
    # ALGOPACK ingestion
    # ------------------------------------------------------------------
    def save_tradestats(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(EqTradeStats, rows, ("secid", "tradedate", "tradetime"))

    def save_orderstats(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(EqOrderStats, rows, ("secid", "tradedate", "tradetime"))

    def save_obstats(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(EqObStats, rows, ("secid", "tradedate", "tradetime"))

    def save_alerts(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(EqAlerts, rows, ("secid", "tradedate", "tradetime", "alert_type"))

    def save_hi2(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(EqHi2, rows, ("secid", "tradedate", "metric"))

    def save_intraday_candles(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self._upsert_many(IntradayCandle, rows, ("secid", "begin", "interval_min"))

    # Backwards-compatible alias used by older callers/tests.
    def save_candles_5m(self, rows: Sequence[Mapping[str, Any]]) -> int:
        return self.save_intraday_candles(rows)

    def save_marketdata_snapshot(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        cols = _model_columns(MarketData)
        prepared = []
        now = datetime.utcnow()
        for r in rows:
            d = _coerce_row(r, cols)
            d.setdefault("updated_at", now)
            if "raw_json" not in d:
                d["raw_json"] = json.dumps(dict(r), default=str)
            prepared.append(d)
        stmt = sqlite_insert(MarketData).values(prepared)
        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in MarketData.__table__.columns
            if c.name != "secid"
        }
        stmt = stmt.on_conflict_do_update(index_elements=["secid"], set_=update_cols)
        with self.engine.begin() as conn:
            conn.execute(stmt)
        return len(prepared)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def _df_for(self, model: Type[Base], secid: str, limit: int = 500) -> pd.DataFrame:
        stmt = (
            select(model)
            .where(model.secid == secid)  # type: ignore[attr-defined]
            .order_by(desc(model.ts))  # type: ignore[attr-defined]
            .limit(limit)
        )
        records: List[Dict[str, Any]] = []
        with self.db.session() as s:
            res = s.execute(stmt).scalars().all()
            records = [
                {c.name: getattr(obj, c.name) for c in model.__table__.columns}
                for obj in res
            ]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)
        return df

    def get_tradestats_df(self, secid: str, limit: int = 500) -> pd.DataFrame:
        return self._df_for(EqTradeStats, secid, limit)

    def get_orderstats_df(self, secid: str, limit: int = 500) -> pd.DataFrame:
        return self._df_for(EqOrderStats, secid, limit)

    def get_obstats_df(self, secid: str, limit: int = 500) -> pd.DataFrame:
        return self._df_for(EqObStats, secid, limit)

    def get_recent_alerts(self, secid: str, since: Optional[datetime] = None, limit: int = 50) -> List[Dict[str, Any]]:
        stmt = select(EqAlerts).where(EqAlerts.secid == secid)
        if since:
            stmt = stmt.where(EqAlerts.ts >= since)
        stmt = stmt.order_by(desc(EqAlerts.ts)).limit(limit)
        with self.db.session() as s:
            rows = s.execute(stmt).scalars().all()
            return [
                {c.name: getattr(r, c.name) for c in EqAlerts.__table__.columns}
                for r in rows
            ]

    def get_latest_hi2(self, secid: str) -> Dict[str, float]:
        """Return ``{metric: value}`` for the latest tradedate available."""
        with self.db.session() as s:
            latest = s.execute(
                select(EqHi2.tradedate)
                .where(EqHi2.secid == secid)
                .order_by(desc(EqHi2.tradedate))
                .limit(1)
            ).scalar_one_or_none()
            if not latest:
                return {}
            rows = s.execute(
                select(EqHi2).where(EqHi2.secid == secid, EqHi2.tradedate == latest)
            ).scalars().all()
            return {r.metric: r.value for r in rows if r.metric is not None}

    def get_marketdata(self, secid: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as s:
            row = s.execute(select(MarketData).where(MarketData.secid == secid)).scalar_one_or_none()
            if row is None:
                return None
            return {c.name: getattr(row, c.name) for c in MarketData.__table__.columns}

    def get_intraday_candles_df(
        self, secid: str, limit: int = 200, interval_min: Optional[int] = None
    ) -> pd.DataFrame:
        stmt = select(IntradayCandle).where(IntradayCandle.secid == secid)
        if interval_min is not None:
            stmt = stmt.where(IntradayCandle.interval_min == int(interval_min))
        stmt = stmt.order_by(desc(IntradayCandle.begin)).limit(limit)
        records: List[Dict[str, Any]] = []
        with self.db.session() as s:
            res = s.execute(stmt).scalars().all()
            records = [{c.name: getattr(o, c.name) for c in IntradayCandle.__table__.columns} for o in res]
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records).sort_values("begin").reset_index(drop=True)
        return df

    # Backwards-compatible alias used by older callers/tests.
    def get_candles_5m_df(self, secid: str, limit: int = 200) -> pd.DataFrame:
        return self.get_intraday_candles_df(secid, limit=limit)

    # ------------------------------------------------------------------
    # Derived features / recommendations / anomalies / events
    # ------------------------------------------------------------------
    def save_features(self, secid: str, ts: datetime, feature_version: str, features: Dict[str, Any]) -> None:
        with self.db.session() as s:
            s.add(DerivedFeatures(
                secid=secid,
                ts=ts,
                feature_version=feature_version,
                features_json=json.dumps(features, default=str),
            ))

    def get_latest_features(self, secid: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as s:
            row = s.execute(
                select(DerivedFeatures)
                .where(DerivedFeatures.secid == secid)
                .order_by(desc(DerivedFeatures.ts))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "secid": row.secid,
                "ts": row.ts,
                "feature_version": row.feature_version,
                "features": json.loads(row.features_json),
            }

    def save_recommendation(
        self,
        secid: str,
        ts: datetime,
        action: str,
        recommended_action: str,
        allow_buy: bool,
        score: float,
        confidence: float,
        recommendation_json: Dict[str, Any],
        strategy_version: str,
        llm_used: bool,
        intent: str = "BUY_CHECK",
    ) -> None:
        with self.db.session() as s:
            s.add(Recommendations(
                secid=secid,
                ts=ts,
                intent=intent,
                action=action,
                recommended_action=recommended_action,
                allow_buy=1 if allow_buy else 0,
                score=float(score),
                confidence=float(confidence),
                recommendation_json=json.dumps(recommendation_json, default=str),
                strategy_version=strategy_version,
                llm_used=1 if llm_used else 0,
            ))

    def get_latest_recommendations(self) -> List[Dict[str, Any]]:
        """One latest recommendation per secid."""
        sql = (
            "SELECT r.* FROM recommendations r INNER JOIN ("
            "  SELECT secid, MAX(ts) AS max_ts FROM recommendations GROUP BY secid"
            ") last ON last.secid = r.secid AND last.max_ts = r.ts"
            " ORDER BY r.ts DESC"
        )
        with self.engine.connect() as conn:
            rows = conn.exec_driver_sql(sql).mappings().all()
        return [self._row_to_recommendation(r) for r in rows]

    def get_latest_recommendation(self, secid: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as s:
            row = s.execute(
                select(Recommendations)
                .where(Recommendations.secid == secid)
                .order_by(desc(Recommendations.ts))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            payload = {
                "secid": row.secid,
                "ts": row.ts,
                "action": row.action,
                "recommended_action": row.recommended_action,
                "allow_buy": bool(row.allow_buy),
                "score": row.score,
                "confidence": row.confidence,
                "strategy_version": row.strategy_version,
                "llm_used": bool(row.llm_used),
                "intent": row.intent,
                "recommendation_json": row.recommendation_json,
            }
            return self._row_to_recommendation(payload)

    @staticmethod
    def _row_to_recommendation(row: Mapping[str, Any]) -> Dict[str, Any]:
        payload = row.get("recommendation_json")
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except Exception:
                return {"secid": row.get("secid"), "raw": payload}
        return dict(row)

    def save_anomaly(
        self,
        secid: str,
        ts: datetime,
        anomaly_type: str,
        severity: float,
        source: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.db.session() as s:
            s.add(Anomalies(
                secid=secid,
                ts=ts,
                anomaly_type=anomaly_type,
                severity=severity,
                source=source,
                payload_json=json.dumps(payload, default=str) if payload else None,
            ))

    def get_recent_anomalies(self, limit: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self.db.session() as s:
            rows = s.execute(
                select(Anomalies).order_by(desc(Anomalies.ts)).limit(limit)
            ).scalars().all()
            for r in rows:
                payload = None
                if r.payload_json:
                    try:
                        payload = json.loads(r.payload_json)
                    except Exception:
                        payload = {"raw": r.payload_json}
                out.append({
                    "id": r.id,
                    "secid": r.secid,
                    "ts": r.ts,
                    "anomaly_type": r.anomaly_type,
                    "severity": r.severity,
                    "source": r.source,
                    "payload": payload,
                })
        return out

    def log_event(
        self,
        level: str,
        event_type: str,
        message: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            with self.db.session() as s:
                s.add(AgentEvents(
                    level=level,
                    event_type=event_type,
                    message=message,
                    payload_json=json.dumps(payload, default=str) if payload else None,
                ))
        except Exception as exc:  # pragma: no cover
            _LOG.error("log_event_failed", extra={"error": str(exc), "event": event_type})

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def get_llm_cache(self, feature_hash: str) -> Optional[Dict[str, Any]]:
        with self.db.session() as s:
            row = s.execute(select(LLMCache).where(LLMCache.feature_hash == feature_hash)).scalar_one_or_none()
            if row is None:
                return None
            try:
                return json.loads(row.response_json)
            except Exception:
                return None

    def put_llm_cache(self, feature_hash: str, payload: Dict[str, Any]) -> None:
        stmt = sqlite_insert(LLMCache).values(
            feature_hash=feature_hash,
            response_json=json.dumps(payload, default=str),
            created_at=datetime.utcnow(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["feature_hash"],
            set_={
                "response_json": stmt.excluded.response_json,
                "created_at": stmt.excluded.created_at,
            },
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def log_llm_call(
        self,
        secid: Optional[str],
        prompt_version: str,
        model: str,
        feature_hash: str,
        prompt: Optional[str],
        response: Optional[str],
        usage: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        try:
            with self.db.session() as s:
                s.add(LLMLog(
                    secid=secid,
                    prompt_version=prompt_version,
                    model=model,
                    feature_hash=feature_hash,
                    prompt=prompt,
                    response=response,
                    usage_json=json.dumps(usage, default=str) if usage else None,
                    success=1 if success else 0,
                    error=error,
                ))
        except Exception as exc:  # pragma: no cover
            _LOG.error("log_llm_call_failed", extra={"error": str(exc)})

    # ------------------------------------------------------------------
    # Retention / housekeeping
    # ------------------------------------------------------------------
    def apply_retention(self, retention_days: int) -> Dict[str, int]:
        """Delete raw API cache older than ``retention_days`` (best effort)."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        cutoff_date = cutoff.date().isoformat()
        removed: Dict[str, int] = {}
        with self.engine.begin() as conn:
            for model in (EqTradeStats, EqOrderStats, EqObStats, EqAlerts, IntradayCandle):
                col = "begin" if model is IntradayCandle else "ts"
                res = conn.execute(delete(model).where(getattr(model, col) < cutoff))
                removed[model.__tablename__] = res.rowcount or 0
            # HI2 is keyed by tradedate text.
            res = conn.execute(delete(EqHi2).where(EqHi2.tradedate < cutoff_date))
            removed[EqHi2.__tablename__] = res.rowcount or 0
            res = conn.execute(delete(AgentEvents).where(AgentEvents.ts < cutoff))
            removed[AgentEvents.__tablename__] = res.rowcount or 0
            res = conn.execute(delete(DerivedFeatures).where(DerivedFeatures.ts < cutoff))
            removed[DerivedFeatures.__tablename__] = res.rowcount or 0
        return removed

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def last_ingestion_age(self, secid: Optional[str] = None) -> Dict[str, Optional[float]]:
        """Return seconds since last row for each main dataset."""
        from app.utils.time import age_seconds
        out: Dict[str, Optional[float]] = {}
        targets: Iterable[tuple[str, Type[Base], str]] = [
            ("tradestats", EqTradeStats, "ts"),
            ("orderstats", EqOrderStats, "ts"),
            ("obstats", EqObStats, "ts"),
            ("alerts", EqAlerts, "ts"),
            ("hi2", EqHi2, "tradedate"),
            ("intraday_candles", IntradayCandle, "begin"),
            ("marketdata", MarketData, "updated_at"),
        ]
        with self.db.session() as s:
            for name, model, col in targets:
                stmt = select(getattr(model, col)).order_by(desc(getattr(model, col))).limit(1)
                if secid is not None and hasattr(model, "secid"):
                    stmt = stmt.where(model.secid == secid)  # type: ignore[attr-defined]
                row = s.execute(stmt).scalar_one_or_none()
                if row is None:
                    out[name] = None
                    continue
                if isinstance(row, str):
                    dt = parse_moex_datetime(row)
                else:
                    dt = row
                out[name] = age_seconds(dt) if dt else None
        return out
