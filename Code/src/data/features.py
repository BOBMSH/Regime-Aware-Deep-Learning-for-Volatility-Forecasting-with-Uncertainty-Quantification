"""Feature construction.

Phase 1 only needs the bare minimum: aligned log returns, lagged RV (for the
HAR-RV baseline in Phase 2) and the VIX. Richer feature engineering -- regime
posteriors etc. -- is introduced from Phase 5 onwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.realized_vol import log_returns


def har_lagged_rv(rv: pd.Series) -> pd.DataFrame:
    """Daily, weekly, monthly lagged RV averages used by HAR-RV.

    Following Corsi (2009):
      - ``rv_d`` = RV_{t-1}
      - ``rv_w`` = mean of RV_{t-1} ... RV_{t-5}
      - ``rv_m`` = mean of RV_{t-1} ... RV_{t-22}

    All three look strictly into the past (``shift(1)`` is applied first), so
    this function is leakage-safe by construction.
    """
    rv = rv.astype(float).rename("rv")
    rv_lag1 = rv.shift(1)
    out = pd.DataFrame(
        {
            "rv_d": rv_lag1,
            "rv_w": rv_lag1.rolling(window=5, min_periods=5).mean(),
            "rv_m": rv_lag1.rolling(window=22, min_periods=22).mean(),
        }
    )
    return out


def build_phase1_features(
    close: pd.Series, rv: pd.Series, *, vix_close: pd.Series | None = None
) -> pd.DataFrame:
    """Assemble the minimal Phase 1 feature frame.

    Columns:
      - ``log_return``       : daily log return on the primary asset
      - ``rv``               : daily realized volatility (target)
      - ``rv_d``, ``rv_w``, ``rv_m`` : HAR-RV lag features
      - ``vix_close``        : optional, included if provided

    Rows containing any NaN (typically the first ~22 rows because of the
    monthly HAR lag) are dropped at the end; downstream code can rely on a
    fully populated frame.
    """
    parts: list[pd.DataFrame | pd.Series] = [
        log_returns(close),
        rv.rename("rv"),
        har_lagged_rv(rv),
    ]
    if vix_close is not None:
        parts.append(vix_close.rename("vix_close"))
    df = pd.concat(parts, axis=1, join="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df
