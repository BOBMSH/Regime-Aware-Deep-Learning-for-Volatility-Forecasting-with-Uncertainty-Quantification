"""Assemble the Phase 2 modelling frame (roadmap §5, the "RV proxies + features"
stage of the engine).

Every econometric baseline consumes the *same* aligned frame, indexed by trading
date, with these columns:

* ``log_return`` -- daily decimal log return of the primary asset (GARCH/EGARCH).
* ``rv``         -- the realized-**variance** target (all models are scored here).
* ``rv_d``, ``rv_w``, ``rv_m`` -- HAR daily/weekly/monthly lags of ``rv``,
  built leakage-safe (``shift(1)`` then rolling mean).
* ``vix_close``  -- optional, carried for later phases (VIX is a feature, §1.4).

The realized-variance target is pluggable so the *same* pipeline runs on any RV
proxy with a one-line config change -- the whole point of decoupling models from
the target (roadmap §5). Three targets are supported:

* ``"yang_zhang"``       -- Yang-Zhang OHLC realized variance from yfinance daily
  bars, 2000-2024. A single consistent estimator that covers the full 2022-2024
  test window without any splice.
* ``"oxfordman_rv5"``    -- Oxford-Man intraday 5-minute realized variance, the
  canonical ABDL target, 2000 -> 2022-02-25.
* ``"oxfordman_spliced"``-- Oxford-Man extended to 2022-2024 with reconstructed
  intraday RV (see :mod:`src.data.intraday`); used once a validated splice exists.

Realized variance -- not volatility -- is returned, because QLIKE/MSE are defined
on variance (Patton 2011) and GARCH forecasts variance natively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.features import har_lagged_rv
from src.data.ingest import load_oxfordman_symbol
from src.data.realized_vol import log_returns, yang_zhang_rv
from src.utils.config import repo_path
from src.utils.io import from_parquet
from src.utils.logging import get_logger

log = get_logger("datasets")

VALID_TARGETS = ("yang_zhang", "oxfordman_rv5", "oxfordman_spliced")


def load_ohlc(cfg: DictConfig, alias: str = "GSPC") -> pd.DataFrame:
    """Load cached daily OHLC(V) for one yfinance alias as a date-indexed frame."""
    path = repo_path(cfg.paths.yfinance, f"{alias}.parquet")
    df = from_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def realized_variance_target(
    cfg: DictConfig, which: str, *, alias: str = "GSPC", yz_window: int = 21
) -> pd.Series:
    """Return the daily realized-**variance** series for the requested proxy.

    Parameters
    ----------
    which : one of :data:`VALID_TARGETS`.
    alias : yfinance alias for OHLC-based proxies (default the primary ^GSPC).
    yz_window : rolling window for the Yang-Zhang estimator.
    """
    if which not in VALID_TARGETS:
        raise ValueError(f"unknown target {which!r}; choose from {VALID_TARGETS}")

    if which == "yang_zhang":
        ohlc = load_ohlc(cfg, alias)
        rv = yang_zhang_rv(ohlc, window=yz_window, squared=True)  # variance
        return rv.rename("rv").dropna()

    if which == "oxfordman_rv5":
        csv = repo_path(cfg.paths.oxfordman, cfg.oxfordman.filename)
        sym = cfg.oxfordman.get("primary_symbol", ".SPX")
        rv = load_oxfordman_symbol(csv, sym)["rv5"].astype(float)
        return rv.rename("rv")[rv > 0].dropna()

    # oxfordman_spliced
    path = repo_path(cfg.paths.interim, "spx_rv5_spliced.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"spliced target not found at {path}. Build it first:\n"
            "  python -m src.data.intraday --intraday-file <your 2022-2024 intraday file>\n"
            "(see src/data/intraday.py for the data-source requirement)."
        )
    rv = from_parquet(path)["rv5"].astype(float)
    return rv.rename("rv")[rv > 0].dropna()


def build_econometric_frame(
    cfg: DictConfig,
    *,
    target: str = "yang_zhang",
    alias: str = "GSPC",
    include_vix: bool = False,
    yz_window: int = 21,
) -> pd.DataFrame:
    """Build the aligned Phase 2 modelling frame for one target proxy.

    Returns a frame indexed by trading date with columns ``log_return``, ``rv``
    (variance target), ``rv_d/rv_w/rv_m`` (HAR lags) and optionally ``vix_close``.
    Rows with any NaN (the ~22-day HAR warm-up at the start) are dropped, so every
    model downstream sees a fully populated frame over an identical index.
    """
    ohlc = load_ohlc(cfg, alias)
    close = ohlc["close"].rename("close")

    rv = realized_variance_target(cfg, target, alias=alias, yz_window=yz_window)

    # Open-to-close log return. Offered alongside the (default) close-to-close
    # return so GARCH/EGARCH can be matched to an *open-to-close* intraday RV
    # target (e.g. Oxford-Man rv5) if desired: close-to-close returns embed the
    # overnight gap and therefore carry ~40% more variance than an open-to-close
    # RV, a level mismatch that QLIKE (being ratio-based, not scale-invariant to
    # the forecast) penalises. Selectable via the model's ``return_col``.
    oc_return = np.log(close / ohlc["open"].astype(float)).rename("oc_log_return")

    parts: list[pd.Series | pd.DataFrame] = [
        log_returns(close),          # decimal close-to-close log returns (default)
        oc_return,                   # open-to-close log returns (fair-match option)
        rv,                          # realized-variance target
        har_lagged_rv(rv),           # HAR daily/weekly/monthly lags
    ]
    if include_vix:
        vix = load_ohlc(cfg, "VIX")["close"].rename("vix_close")
        parts.append(vix)

    frame = pd.concat(parts, axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    frame.index.name = "date"

    log.info(
        "econometric frame [target=%s, asset=%s]: %d rows %s -> %s",
        target, alias, len(frame), frame.index.min().date(), frame.index.max().date(),
    )
    return frame
