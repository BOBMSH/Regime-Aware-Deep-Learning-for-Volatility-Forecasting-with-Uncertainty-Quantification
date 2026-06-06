"""Realized-volatility estimators.

The dissertation's **primary** RV target is the intraday Andersen-Bollerslev-
Diebold-Labys (2003) realized variance, sourced pre-computed from the Oxford-Man
Realized Library (see ``src.data.ingest.OxfordManIngestor``). This module owns
two roles in support of that:

1. **Yang-Zhang OHLC RV** (Yang & Zhang 2000) -- the sensitivity estimator used
   in Phase 1 to sanity-check the Oxford-Man series against an OHLC-derived
   alternative, and the fallback target if the curated source disappears.
2. **Squared-return RV** -- the noisy diagnostic baseline. Never the headline
   target, but it is the simplest possible proxy and worth carrying for plots.

All estimators return *daily* realized **volatility** (sqrt of variance) by
default, indexed by the trading day. Pass ``squared=True`` for variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC_REQUIRED = ("open", "high", "low", "close")
TRADING_DAYS_PER_YEAR = 252


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in _OHLC_REQUIRED if c not in cols]
    if missing:
        raise ValueError(
            f"OHLC frame missing columns {missing}; got {list(df.columns)}"
        )
    out = df[[cols[c] for c in _OHLC_REQUIRED]].copy()
    out.columns = list(_OHLC_REQUIRED)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns. First row is NaN by construction."""
    prices = prices.astype(float)
    return np.log(prices).diff().rename("log_return")


def squared_returns_rv(
    close: pd.Series, *, squared: bool = False, annualize: bool = False
) -> pd.Series:
    """Squared-return RV. Diagnostic only -- noisy and biased downward at the
    daily scale, but useful to show alongside cleaner estimators.
    """
    r = log_returns(close.astype(float))
    rv2 = r.pow(2)
    if annualize:
        rv2 = rv2 * TRADING_DAYS_PER_YEAR
    if squared:
        return rv2.rename("rv_sqret_var")
    return np.sqrt(rv2).rename("rv_sqret")


def parkinson_rv(
    df: pd.DataFrame, *, squared: bool = False, annualize: bool = False
) -> pd.Series:
    """Parkinson (1980) high-low estimator. Carried as an internal helper for
    Yang-Zhang; not a headline output of this module.
    """
    df = _validate_ohlc(df)
    hl = np.log(df["high"] / df["low"])
    rv2 = (hl.pow(2)) / (4.0 * np.log(2.0))
    if annualize:
        rv2 = rv2 * TRADING_DAYS_PER_YEAR
    return rv2.rename("rv_parkinson_var") if squared else np.sqrt(rv2).rename("rv_parkinson")


def yang_zhang_rv(
    df: pd.DataFrame,
    *,
    window: int = 21,
    squared: bool = False,
    annualize: bool = False,
) -> pd.Series:
    """Yang-Zhang (2000) OHLC realized-variance estimator.

    Combines overnight, open-to-close, and Rogers-Satchell components into a
    drift-independent, open-jump-robust estimator. Computed as a rolling
    average over ``window`` trading days; ``window=21`` is the literature
    convention (one trading month). Returned as **volatility** (sqrt of
    variance) unless ``squared=True``.

    Parameters
    ----------
    df : OHLC DataFrame indexed by date.
    window : rolling window in trading days. Must be >= 2.
    squared : if True, return variance; default is volatility.
    annualize : multiply by 252 (variance) / sqrt(252) (volatility).

    Returns
    -------
    pd.Series indexed by date; the first ``window`` rows are NaN.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    df = _validate_ohlc(df)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    c_prev = c.shift(1)

    # Overnight return: log(O_t / C_{t-1}).
    log_oc = np.log(o / c_prev)
    # Open-to-close return: log(C_t / O_t).
    log_co = np.log(c / o)
    # Rogers-Satchell drift-free intraday term.
    rs = (np.log(h / c) * np.log(h / o)) + (np.log(l / c) * np.log(l / o))

    n = window
    # Yang-Zhang's k tunes the weight of the open-to-close variance; the
    # original derivation gives a specific functional form below.
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    sigma2_o = log_oc.rolling(window=n, min_periods=n).var(ddof=1)
    sigma2_c = log_co.rolling(window=n, min_periods=n).var(ddof=1)
    sigma2_rs = rs.rolling(window=n, min_periods=n).mean()

    yz_var = sigma2_o + k * sigma2_c + (1 - k) * sigma2_rs
    yz_var = yz_var.clip(lower=0.0)  # tiny negatives can occur numerically

    if annualize:
        yz_var = yz_var * TRADING_DAYS_PER_YEAR
    out = yz_var if squared else np.sqrt(yz_var)
    return out.rename("rv_yang_zhang_var" if squared else "rv_yang_zhang")


def realized_vol_from_intraday_rv(
    rv_variance: pd.Series, *, annualize: bool = False
) -> pd.Series:
    """Helper for Oxford-Man-style daily realized **variance** -> volatility.

    The Oxford-Man library publishes 5-minute realized variance ``rv5``; the
    canonical Andersen-Bollerslev-Diebold-Labys target is its square root.
    """
    rv2 = rv_variance.astype(float).clip(lower=0.0)
    if annualize:
        rv2 = rv2 * TRADING_DAYS_PER_YEAR
    return np.sqrt(rv2).rename("rv_intraday")
