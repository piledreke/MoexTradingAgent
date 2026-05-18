"""Cross-asset analysis: correlation with IMOEX, beta, relative strength."""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(s: pd.Series, log: bool = True) -> pd.Series:
    if log:
        return np.log(s / s.shift(1))
    return s.pct_change()


def correlation(asset: pd.Series, index: pd.Series, window: int = 60) -> float:
    """Rolling correlation. Returns latest value."""
    a_r = returns(asset)
    i_r = returns(index)
    df = pd.concat([a_r, i_r], axis=1).dropna()
    if len(df) < window:
        return 0.0
    return float(df.iloc[-window:].corr().iloc[0, 1])


def beta(asset: pd.Series, index: pd.Series, window: int = 60) -> float:
    """Rolling beta of asset vs index."""
    a_r = returns(asset).iloc[-window:]
    i_r = returns(index).iloc[-window:]
    df = pd.concat([a_r, i_r], axis=1).dropna()
    if len(df) < 10:
        return 1.0
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    return float(cov / var) if var > 0 else 1.0


def relative_strength_index_vs_market(
    asset_df: pd.DataFrame, index_df: pd.DataFrame, period: int = 20
) -> dict[str, float]:
    """Compute relative strength + outperformance signal."""
    if asset_df.empty or index_df.empty or len(asset_df) < period or len(index_df) < period:
        return {"rs": 1.0, "outperformance_pct": 0.0, "label": "neutral"}

    a = asset_df.sort_values("ts")["c"] if "ts" in asset_df.columns else asset_df["c"]
    i = index_df.sort_values("ts")["c"] if "ts" in index_df.columns else index_df["c"]
    a_ret = float(a.iloc[-1] / a.iloc[-period] - 1)
    i_ret = float(i.iloc[-1] / i.iloc[-period] - 1)
    out_pct = (a_ret - i_ret) * 100

    if abs(i_ret) < 1e-9:
        rs = 1.0 + a_ret
    else:
        rs = a_ret / i_ret

    if out_pct > 1.5:
        label = "outperforming"
    elif out_pct < -1.5:
        label = "underperforming"
    else:
        label = "neutral"

    return {
        "rs": round(rs, 3),
        "outperformance_pct": round(out_pct, 2),
        "label": label,
        "asset_return_pct": round(a_ret * 100, 2),
        "index_return_pct": round(i_ret * 100, 2),
    }