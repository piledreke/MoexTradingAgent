"""Extended technical indicators library.

Все функции — чистые, vectorized (pandas/numpy), testable.
Включает: тренд, импульс, волатильность, объёмные, осцилляторы,
Ichimoku, Donchian, Keltner, ADX, OBV, MFI, CCI, Stochastic, Williams%R.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ═══════════════ TREND ═══════════════════════════════════

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    """Linearly weighted MA."""
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull MA — отзывчивая, но сглаженная."""
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)


def adx(df: pd.DataFrame, n: int = 14) -> dict[str, pd.Series]:
    """Average Directional Index — измеряет СИЛУ тренда (не направление).

    ADX < 20: нет тренда (боковик).
    ADX 20-40: тренд набирает силу.
    ADX > 40: сильный тренд.
    """
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean()

    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)

    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_ = dx.ewm(alpha=1 / n, adjust=False).mean()
    return {"adx": adx_.fillna(0), "plus_di": plus_di.fillna(0), "minus_di": minus_di.fillna(0)}


# ═══════════════ MOMENTUM ════════════════════════════════

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.Series]:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return {"macd": line, "signal": sig, "hist": line - sig}


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> dict[str, pd.Series]:
    """Stochastic oscillator. %K — fast, %D — slow."""
    ll = df["l"].rolling(k, min_periods=1).min()
    hh = df["h"].rolling(k, min_periods=1).max()
    raw_k = 100 * (df["c"] - ll) / (hh - ll).replace(0, np.nan)
    k_smooth = raw_k.rolling(smooth, min_periods=1).mean()
    d_ = k_smooth.rolling(d, min_periods=1).mean()
    return {"k": k_smooth.fillna(50), "d": d_.fillna(50)}


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R: -100 (oversold) to 0 (overbought)."""
    hh = df["h"].rolling(n, min_periods=1).max()
    ll = df["l"].rolling(n, min_periods=1).min()
    return (-100 * (hh - df["c"]) / (hh - ll).replace(0, np.nan)).fillna(-50)


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity Channel Index. >+100 overbought, <-100 oversold."""
    tp = (df["h"] + df["l"] + df["c"]) / 3
    sma_tp = tp.rolling(n, min_periods=1).mean()
    md = tp.rolling(n, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - sma_tp) / (0.015 * md.replace(0, np.nan))).fillna(0)


def rate_of_change(s: pd.Series, n: int = 10) -> pd.Series:
    return ((s / s.shift(n) - 1) * 100).fillna(0)


# ═══════════════ VOLATILITY ══════════════════════════════

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def realized_volatility(s: pd.Series, n: int = 20, annualize: int = 252 * 8 * 60) -> pd.Series:
    """Annualized realized vol of log-returns (default for 1m bars on stocks)."""
    ret = np.log(s / s.shift(1))
    return ret.rolling(n, min_periods=2).std() * np.sqrt(annualize)


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> dict[str, pd.Series]:
    mid = sma(s, n)
    std = s.rolling(n, min_periods=1).std(ddof=0)
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return {"mid": mid, "upper": upper, "lower": lower,
            "width": width.fillna(0), "pct_b": pct_b.fillna(0.5)}


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> dict[str, pd.Series]:
    """Keltner channels — EMA ± mult*ATR."""
    mid = ema(df["c"], n)
    a = atr(df, n)
    return {"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a}


def donchian(df: pd.DataFrame, n: int = 20) -> dict[str, pd.Series]:
    """Donchian — каналы экстремумов. Breakout-индикатор (Turtle traders)."""
    up = df["h"].rolling(n, min_periods=1).max()
    dn = df["l"].rolling(n, min_periods=1).min()
    return {"upper": up, "lower": dn, "mid": (up + dn) / 2}


def squeeze_momentum(df: pd.DataFrame, n: int = 20) -> dict[str, pd.Series]:
    """TTM Squeeze: Bollinger inside Keltner = volatility compression (часто breakout)."""
    bb = bollinger(df["c"], n, 2.0)
    kc = keltner(df, n, 1.5)
    in_squeeze = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    return {"squeeze": in_squeeze.astype(int), "bb_width": bb["width"]}


# ═══════════════ VOLUME ══════════════════════════════════

def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from start of df."""
    tp = (df["h"] + df["l"] + df["c"]) / 3
    return (tp * df["v"]).cumsum() / df["v"].cumsum().replace(0, np.nan)


def session_vwap(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """VWAP reset on each trading day. Adds vwap, vwap_upper, vwap_lower bands (±1σ)."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out["_date"] = out[ts_col].dt.date
    tp = (out["h"] + out["l"] + out["c"]) / 3
    out["_pv"] = tp * out["v"]
    out["vwap"] = (out.groupby("_date")["_pv"].cumsum()
                   / out.groupby("_date")["v"].cumsum().replace(0, np.nan))
    # rolling std for bands
    out["_dev"] = (tp - out["vwap"]) ** 2 * out["v"]
    var = (out.groupby("_date")["_dev"].cumsum()
           / out.groupby("_date")["v"].cumsum().replace(0, np.nan))
    sd = np.sqrt(var)
    out["vwap_upper"] = out["vwap"] + sd
    out["vwap_lower"] = out["vwap"] - sd
    return out.drop(columns=["_date", "_pv", "_dev"])


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(df["c"].diff().fillna(0))
    return (sign * df["v"]).cumsum()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Money Flow Index — RSI с учётом объёма."""
    tp = (df["h"] + df["l"] + df["c"]) / 3
    mf = tp * df["v"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(n, min_periods=1).sum()
    neg_sum = neg.rolling(n, min_periods=1).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - 100 / (1 + mr)).fillna(50)


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    mfm = ((df["c"] - df["l"]) - (df["h"] - df["c"])) / (df["h"] - df["l"]).replace(0, np.nan)
    mfv = mfm * df["v"]
    return (mfv.rolling(n, min_periods=1).sum() / df["v"].rolling(n, min_periods=1).sum().replace(0, np.nan)).fillna(0)


def adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    mfm = ((df["c"] - df["l"]) - (df["h"] - df["c"])) / (df["h"] - df["l"]).replace(0, np.nan)
    return (mfm * df["v"]).cumsum().fillna(0)


# ═══════════════ ICHIMOKU ════════════════════════════════

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict[str, pd.Series]:
    """Ichimoku Cloud."""
    def mid(n):
        return (df["h"].rolling(n, min_periods=1).max() + df["l"].rolling(n, min_periods=1).min()) / 2

    tk = mid(tenkan)
    kj = mid(kijun)
    span_a = ((tk + kj) / 2).shift(kijun)
    span_b = mid(senkou_b).shift(kijun)
    chikou = df["c"].shift(-kijun)
    return {"tenkan": tk, "kijun": kj, "span_a": span_a, "span_b": span_b, "chikou": chikou}


# ═══════════════ TREND DETECTION (COMPOSITE) ═════════════

def detect_trend(df: pd.DataFrame) -> dict[str, object]:
    """Multi-indicator trend detection: EMA20/50, ADX, MACD."""
    if df.empty or len(df) < 50:
        return {"trend": "unknown", "strength": 0.0,
                "adx": None, "ema20": None, "ema50": None}
    c = df["c"].astype(float)
    e20, e50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
    adx_d = adx(df, 14)
    adx_v = float(adx_d["adx"].iloc[-1])
    plus_di = float(adx_d["plus_di"].iloc[-1])
    minus_di = float(adx_d["minus_di"].iloc[-1])
    macd_d = macd(c)
    hist = float(macd_d["hist"].iloc[-1])

    # Composite logic
    bullish = (e20 > e50) and (hist > 0) and (plus_di > minus_di)
    bearish = (e20 < e50) and (hist < 0) and (minus_di > plus_di)

    if adx_v < 20:
        trend = "sideways"
    elif bullish:
        trend = "uptrend"
    elif bearish:
        trend = "downtrend"
    else:
        trend = "mixed"

    strength = min(1.0, adx_v / 50.0)
    return {
        "trend": trend, "strength": round(strength, 3),
        "adx": round(adx_v, 1),
        "plus_di": round(plus_di, 1), "minus_di": round(minus_di, 1),
        "ema20": float(e20), "ema50": float(e50),
        "macd_hist": float(hist),
    }


def support_resistance(df: pd.DataFrame, lookback: int = 100, bins: int = 30) -> dict[str, float]:
    """Find support/resistance via volume profile peaks."""
    if df.empty or len(df) < 10:
        last = float(df["c"].iloc[-1]) if not df.empty else 0.0
        return {"support": last * 0.98, "resistance": last * 1.02, "poc": last}

    sub = df.tail(lookback).copy()
    last = float(sub["c"].iloc[-1])
    prices = sub["c"].values
    vols = sub["v"].values
    if vols.sum() == 0:
        return {"support": float(sub["l"].min()),
                "resistance": float(sub["h"].max()), "poc": last}

    edges = np.linspace(prices.min(), prices.max(), bins + 1)
    profile, _ = np.histogram(prices, bins=edges, weights=vols)
    centers = (edges[:-1] + edges[1:]) / 2

    poc_idx = int(np.argmax(profile))
    poc = float(centers[poc_idx])

    # Top 5 volume nodes
    top_idx = np.argsort(profile)[-5:]
    levels = sorted(float(centers[i]) for i in top_idx)
    sups = [x for x in levels if x < last]
    ress = [x for x in levels if x > last]
    return {
        "support": max(sups) if sups else float(sub["l"].min()),
        "resistance": min(ress) if ress else float(sub["h"].max()),
        "poc": poc,
    }


def relative_strength(asset: pd.Series, index: pd.Series, period: int = 20) -> float:
    if len(asset) < period or len(index) < period:
        return 1.0
    a = asset.iloc[-1] / asset.iloc[-period] - 1
    i = index.iloc[-1] / index.iloc[-period] - 1
    if abs(i) < 1e-9:
        return 1.0 + a  # treat as raw return
    return float(a / i)


# ═══════════════ FEATURE BUNDLE ══════════════════════════

def compute_all_features(df: pd.DataFrame) -> dict[str, float]:
    """Compute snapshot of latest values of all indicators.

    Returns flat dict {feature_name: latest_value}. Used by feature store
    and scoring.
    """
    if df.empty or len(df) < 30:
        return {}
    out: dict[str, float] = {}
    c = df["c"]

    # Trend
    trend = detect_trend(df)
    out["ema20"] = trend.get("ema20") or 0
    out["ema50"] = trend.get("ema50") or 0
    out["ema20_above_50"] = float(out["ema20"] > out["ema50"])
    out["adx"] = trend.get("adx") or 0
    out["trend_strength"] = trend.get("strength") or 0
    out["trend_label"] = trend.get("trend")

    # Momentum
    out["rsi14"] = float(rsi(c, 14).iloc[-1])
    md = macd(c)
    out["macd_hist"] = float(md["hist"].iloc[-1])
    out["macd_cross_up"] = float(md["hist"].iloc[-1] > 0 and md["hist"].iloc[-2] <= 0) if len(md["hist"]) >= 2 else 0
    st = stochastic(df)
    out["stoch_k"] = float(st["k"].iloc[-1])
    out["stoch_d"] = float(st["d"].iloc[-1])
    out["williams_r"] = float(williams_r(df).iloc[-1])
    out["cci"] = float(cci(df).iloc[-1])
    out["roc10"] = float(rate_of_change(c, 10).iloc[-1])

    # Volatility
    a = atr(df, 14)
    out["atr"] = float(a.iloc[-1])
    out["atr_pct"] = float(a.iloc[-1] / c.iloc[-1] * 100) if c.iloc[-1] else 0
    bb = bollinger(c)
    out["bb_width"] = float(bb["width"].iloc[-1])
    out["bb_pct_b"] = float(bb["pct_b"].iloc[-1])
    out["realized_vol"] = float(realized_volatility(c).iloc[-1])
    sq = squeeze_momentum(df)
    out["in_squeeze"] = float(sq["squeeze"].iloc[-1])

    # Channels
    don = donchian(df, 20)
    out["donchian_upper"] = float(don["upper"].iloc[-1])
    out["donchian_lower"] = float(don["lower"].iloc[-1])
    out["donchian_breakout_up"] = float(c.iloc[-1] >= don["upper"].iloc[-2]) if len(don["upper"]) >= 2 else 0

    # Volume
    out["obv"] = float(obv(df).iloc[-1])
    out["mfi"] = float(mfi(df).iloc[-1])
    out["cmf"] = float(cmf(df).iloc[-1])

    # Ichimoku
    ich = ichimoku(df)
    out["ich_tenkan"] = float(ich["tenkan"].iloc[-1])
    out["ich_kijun"] = float(ich["kijun"].iloc[-1])
    out["above_cloud"] = float(
        c.iloc[-1] > max(ich["span_a"].iloc[-1] or 0, ich["span_b"].iloc[-1] or 0)
    ) if pd.notna(ich["span_a"].iloc[-1]) else 0

    # S/R
    sr = support_resistance(df)
    out["support"] = sr["support"]
    out["resistance"] = sr["resistance"]
    out["poc"] = sr["poc"]
    out["dist_to_resistance_pct"] = (sr["resistance"] - c.iloc[-1]) / c.iloc[-1] * 100
    out["dist_to_support_pct"] = (c.iloc[-1] - sr["support"]) / c.iloc[-1] * 100

    out["close"] = float(c.iloc[-1])
    return out