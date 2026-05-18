"""Lightweight anomaly detector that augments MOEX MegaAlerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.features.feature_builder import FeatureBundle
from app.logging_config import get_logger
from app.storage.repository import Repository

_LOG = get_logger(__name__)


@dataclass
class AnomalyEvent:
    secid: str
    ts: datetime
    anomaly_type: str
    severity: float
    source: str = "internal"
    payload: Optional[Dict[str, Any]] = None


class AnomalyDetector:
    """Inspect a freshly-built feature bundle and persist anomalies."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    # ------------------------------------------------------------------
    def detect(self, bundle: FeatureBundle, persist: bool = True) -> List[AnomalyEvent]:
        f = bundle.features
        events: List[AnomalyEvent] = []
        secid = bundle.secid
        ts = bundle.ts

        def _add(at: str, severity: float, payload: Dict[str, Any]) -> None:
            severity = max(0.0, min(1.0, severity))
            events.append(AnomalyEvent(secid, ts, at, severity, "internal", payload))

        vol_z = _safe(f.get("volume_zscore"))
        val_z = _safe(f.get("value_zscore"))
        bbz = _safe(f.get("bollinger_z"))
        disb = _safe(f.get("disb_now"))
        spread_bbo = _safe(f.get("spread_bbo"))
        spread_1mio = _safe(f.get("spread_1mio"))
        imb_bbo = _safe(f.get("imbalance_bbo"))
        ret_1 = _safe(f.get("return_1_bar"))
        ret_3 = _safe(f.get("return_3_bar"))
        liquidity = _safe(f.get("liquidity_score"))
        spoof = _safe(f.get("spoof_risk_score"))

        if vol_z is not None and vol_z > 3.5:
            _add("volume_zscore_extreme", min(1.0, (vol_z - 3.5) / 3.0), {"vol_z": vol_z})
        if val_z is not None and val_z > 3.5:
            _add("value_zscore_extreme", min(1.0, (val_z - 3.5) / 3.0), {"val_z": val_z})
        if bbz is not None:
            if bbz > 3.0:
                _add("price_jump_z", min(1.0, (bbz - 3.0) / 2.0), {"bollinger_z": bbz})
            elif bbz < -3.0:
                _add("price_drop_z", min(1.0, (-bbz - 3.0) / 2.0), {"bollinger_z": bbz})
        if ret_1 is not None and ret_1 < -0.02:
            _add("intraday_drop", min(1.0, (-ret_1) / 0.05), {"return_1_bar": ret_1})
        if ret_3 is not None and ret_3 < -0.04:
            _add("multi_bar_drop", min(1.0, (-ret_3) / 0.08), {"return_3_bar": ret_3})
        if disb is not None and disb < -0.5:
            _add("strong_sell_disbalance", min(1.0, (-disb - 0.5) * 2.0), {"disb": disb})
        if disb is not None and disb > 0.7:
            _add("strong_buy_disbalance", min(1.0, (disb - 0.7) * 3.0), {"disb": disb})
        if imb_bbo is not None and imb_bbo < -0.4:
            _add("orderbook_sell_pressure", min(1.0, (-imb_bbo - 0.4) * 2.0), {"imbalance_bbo": imb_bbo})
        if spread_bbo is not None and spread_1mio is not None and spread_1mio > 5 * max(spread_bbo, 1e-9):
            _add("spread_spike", 0.8, {"spread_bbo": spread_bbo, "spread_1mio": spread_1mio})
        if liquidity is not None and liquidity < 0.2:
            _add("liquidity_drop", 0.7, {"liquidity_score": liquidity})
        if spoof is not None and spoof > 0.6:
            _add("spoof_risk", spoof, {"spoof_risk_score": spoof})

        if persist:
            for ev in events:
                try:
                    self.repo.save_anomaly(
                        secid=ev.secid,
                        ts=ev.ts,
                        anomaly_type=ev.anomaly_type,
                        severity=ev.severity,
                        source=ev.source,
                        payload=ev.payload,
                    )
                except Exception as exc:  # pragma: no cover
                    _LOG.warning("anomaly_save_failed", extra={"error": str(exc), "type": ev.anomaly_type})
        return events


def _safe(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:  # NaN
        return None
    return f
