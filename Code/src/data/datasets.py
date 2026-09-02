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

from dataclasses import dataclass

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


# --------------------------------------------------------------------------- #
# Asset resolution (configs/data.yaml -> assets.registry)                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssetSpec:
    """Everything the pipeline needs to know about one tradable asset.

    An asset has three identifiers that are easy to confuse and were previously
    held in three unreconciled places: the yfinance ticker (``^RUT``), the local
    OHLC cache alias (``RUT.parquet``), and the Oxford-Man realized-measure
    symbol (``.RUT``). It also carries two facts that decide what may legitimately
    be run on it -- the trading ``session``, and whether the CBOE VIX is a valid
    auxiliary feature for it. Resolving all five together, from one config block,
    is what stops a Phase-8 profile from pairing one asset's returns with another
    asset's realized variance.
    """

    key: str
    yfinance: str
    cache_alias: str
    oxfordman: str | None
    session: str
    vix_feature: bool
    description: str = ""


#: Behaviour for a config predating ``assets.registry`` (2026-08-30). Every
#: pre-Phase-8 call resolved to exactly this, so an old config still runs.
_LEGACY_SPX = AssetSpec(
    key="SPX", yfinance="^GSPC", cache_alias="GSPC", oxfordman=".SPX",
    session="US", vix_feature=True, description="legacy default (no registry in config)",
)


def resolve_asset(cfg: DictConfig, key: str | None = None) -> AssetSpec:
    """Resolve one asset key against ``assets.registry`` in ``configs/data.yaml``.

    ``key=None`` returns the configured primary asset (``assets.primary``), which
    is ``SPX`` -- so every existing call site keeps its exact behaviour. A config
    with no ``assets.registry`` block falls back to :data:`_LEGACY_SPX`.

    Raises ``KeyError`` naming the available keys when an asset is requested that
    the registry does not define, rather than silently forecasting ``.SPX``.
    """
    assets = cfg.get("assets", None)
    registry = None if assets is None else assets.get("registry", None)
    if not registry:
        if key not in (None, _LEGACY_SPX.key):
            raise KeyError(
                f"asset {key!r} requested but configs/data.yaml has no "
                "'assets.registry' block to resolve it against"
            )
        return _LEGACY_SPX

    name = str(key if key is not None else assets.get("primary", _LEGACY_SPX.key))
    if name not in registry:
        raise KeyError(
            f"unknown asset {name!r}; assets.registry defines {sorted(registry)}"
        )
    e = registry[name]
    om = e.get("oxfordman", None)
    return AssetSpec(
        key=name,
        yfinance=str(e["yfinance"]),
        cache_alias=str(e["cache_alias"]),
        oxfordman=None if om is None else str(om),
        session=str(e.get("session", "US")),
        vix_feature=bool(e.get("vix_feature", False)),
        description=str(e.get("description", "")),
    )


def load_ohlc(cfg: DictConfig, alias: str = "GSPC") -> pd.DataFrame:
    """Load cached daily OHLC(V) for one yfinance alias as a date-indexed frame."""
    path = repo_path(cfg.paths.yfinance, f"{alias}.parquet")
    df = from_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def realized_variance_target(
    cfg: DictConfig, which: str, *, alias: str = "GSPC", yz_window: int = 21,
    symbol: str | None = None,
) -> pd.Series:
    """Return the daily realized-**variance** series for the requested proxy.

    Parameters
    ----------
    which : one of :data:`VALID_TARGETS`.
    alias : yfinance cache alias for OHLC-based proxies (default the primary GSPC).
    yz_window : rolling window for the Yang-Zhang estimator.
    symbol : Oxford-Man realized-measure symbol for ``oxfordman_rv5``. ``None``
        (the default) uses ``cfg.oxfordman.primary_symbol``, which is what every
        pre-Phase-8 caller got. Pass it explicitly -- or, better, let
        :func:`build_econometric_frame` pass it from :func:`resolve_asset` -- to
        forecast a different index. Before 2026-08-30 this was read from the
        config unconditionally, so the target was pinned to ``.SPX`` no matter
        whose returns and OHLC the rest of the frame carried: a Phase-8 profile
        would have paired Russell 2000 returns with S&P 500 realized variance and
        produced a plausible, meaningless table.
    """
    if which not in VALID_TARGETS:
        raise ValueError(f"unknown target {which!r}; choose from {VALID_TARGETS}")

    if which == "yang_zhang":
        ohlc = load_ohlc(cfg, alias)
        rv = yang_zhang_rv(ohlc, window=yz_window, squared=True)  # variance
        return rv.rename("rv").dropna()

    if which == "oxfordman_rv5":
        csv = repo_path(cfg.paths.oxfordman, cfg.oxfordman.filename)
        sym = symbol if symbol is not None else cfg.oxfordman.get("primary_symbol", ".SPX")
        rv = load_oxfordman_symbol(csv, str(sym))["rv5"].astype(float)
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
    alias: str | None = None,
    asset: str | None = None,
    include_vix: bool = False,
    yz_window: int = 21,
) -> pd.DataFrame:
    """Build the aligned modelling frame for one asset and one target proxy.

    Returns a frame indexed by trading date with columns ``log_return``,
    ``oc_log_return``, ``rv`` (variance target), ``rv_d/rv_w/rv_m`` (HAR lags)
    and optionally ``vix_close``. Rows with any NaN (the ~22-day HAR warm-up at
    the start) are dropped, so every model downstream sees a fully populated
    frame over an identical index.

    Parameters
    ----------
    asset : registry key (see :func:`resolve_asset`). ``None`` resolves the
        configured primary asset, so every pre-Phase-8 call keeps its behaviour.
        Naming an asset resolves its OHLC alias, its Oxford-Man symbol and its
        VIX eligibility together -- one decision instead of three that can drift.
    alias : explicit OHLC cache alias, overriding the resolved one. Kept for
        callers that want a proxy built from one asset's OHLC while naming
        another; leave it unset in normal use.
    include_vix : request the ``vix_close`` auxiliary column. It is joined only
        when the resolved asset's ``vix_feature`` is true. For an asset where it
        is false (a non-US index, where the CBOE VIX prices a different market's
        options) the column is dropped with a warning rather than joined, because
        the alternative is worse than a missing feature: the inner join below
        would silently intersect two national trading calendars and shrink the
        sample with no error anywhere.

    Notes
    -----
    The frame is assembled with an inner join across sources, so the row count is
    the *intersection* of every input's calendar. That is correct, and on ``.SPX``
    it is invisible because all the sources share the NYSE calendar. On any other
    asset it is a live risk, so the attrition is logged and recorded in
    ``frame.attrs["join_attrition"]`` -- a sample silently 40 days short is the
    kind of thing that is only ever noticed after the chapter is written.
    """
    spec = resolve_asset(cfg, asset)
    ohlc_alias = alias if alias is not None else spec.cache_alias
    ohlc = load_ohlc(cfg, ohlc_alias)
    close = ohlc["close"].rename("close")

    rv = realized_variance_target(
        cfg, target, alias=ohlc_alias, yz_window=yz_window, symbol=spec.oxfordman
    )

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
    vix_joined = False
    if include_vix:
        if spec.vix_feature:
            vix_spec = resolve_asset(cfg, cfg.get("assets", {}).get("vix", "VIX"))
            parts.append(load_ohlc(cfg, vix_spec.cache_alias)["close"].rename("vix_close"))
            vix_joined = True
        else:
            log.warning(
                "VIX column requested for asset %s (%s session) but "
                "assets.registry marks it vix_feature=false -- not joining. The "
                "CBOE VIX prices S&P 500 options, so it is not an auxiliary "
                "feature for this market, and joining it would intersect two "
                "trading calendars and shrink the sample silently.",
                spec.key, spec.session,
            )

    # Row counts BEFORE the join, so calendar attrition is separable from the
    # HAR warm-up. `rv` is the reference: it is the target, and every other part
    # is only useful on days the target exists.
    n_rv = int(len(rv))
    frame = pd.concat(parts, axis=1, join="inner")
    n_joined = int(len(frame))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    frame.index.name = "date"
    n_final = int(len(frame))

    frame.attrs["asset"] = spec.key
    frame.attrs["oxfordman_symbol"] = spec.oxfordman
    frame.attrs["ohlc_alias"] = ohlc_alias
    frame.attrs["session"] = spec.session
    frame.attrs["vix_joined"] = vix_joined
    frame.attrs["join_attrition"] = {
        "n_target_rows": n_rv,
        "n_after_join": n_joined,
        "n_after_dropna": n_final,
        "lost_to_join": n_rv - n_joined,
        "lost_to_warmup_or_nan": n_joined - n_final,
    }

    log.info(
        "modelling frame [asset=%s target=%s ohlc=%s om=%s vix=%s]: "
        "%d rows %s -> %s  (target had %d; %d lost to the calendar join, "
        "%d to warm-up/NaN)",
        spec.key, target, ohlc_alias, spec.oxfordman, vix_joined,
        n_final, frame.index.min().date(), frame.index.max().date(),
        n_rv, n_rv - n_joined, n_joined - n_final,
    )
    if n_rv and (n_rv - n_joined) / n_rv > 0.02:
        log.warning(
            "calendar join dropped %.1f%% of %s's target days (%d of %d). Check "
            "that every source shares this asset's trading calendar before "
            "reporting any sample size.",
            100.0 * (n_rv - n_joined) / n_rv, spec.key, n_rv - n_joined, n_rv,
        )
    return frame


def frame_for_profile(
    cfg: DictConfig, profile, *, include_vix: bool = False, yz_window: int = 21
) -> pd.DataFrame:
    """Build the modelling frame an evaluation *profile* asks for.

    Every experiment runner takes the same two things from a profile -- the
    target proxy and, from 2026-09-01, the **asset** -- so they take them
    through here rather than each spelling out the call. The asset key is the
    reason this exists: :func:`resolve_asset` and the ``asset=`` parameter of
    :func:`build_econometric_frame` were added on 2026-08-30 and, until this
    helper, *no runner passed either*. A Phase-8 profile naming ``RUT`` would
    have been silently run on ``.SPX`` -- the registry would have been a fourth
    place holding the asset's identity with no consumer, which is precisely the
    failure the registry was built to end.

    ``profile.asset`` is optional and absent from every pre-Phase-8 config, so
    omitting it resolves the configured primary asset and reproduces the
    previous behaviour exactly.

    Parameters
    ----------
    profile : the profile node (needs ``target``; may carry ``asset``).
    include_vix : join the auxiliary implied-volatility column. Subject to the
        resolved asset's ``vix_feature`` flag -- an asset whose options trade on
        another market never receives it, whatever the caller asks for.
    """
    asset = profile.get("asset", None) if hasattr(profile, "get") else None
    return build_econometric_frame(
        cfg,
        target=profile.target,
        asset=(str(asset) if asset is not None else None),
        include_vix=include_vix,
        yz_window=yz_window,
    )


def attrition_record(frame: pd.DataFrame, *, profile_name: str = "") -> dict:
    """Flatten ``frame.attrs`` into one reportable row.

    Chapter 3 §3.9 commits to reporting the rows an asset loses to the calendar
    join rather than absorbing them silently, and :func:`build_econometric_frame`
    has recorded them in ``frame.attrs["join_attrition"]`` since 2026-08-30 --
    but nothing read that key, so the commitment had no consumer either. Phase 8
    writes this row per asset.
    """
    att = dict(frame.attrs.get("join_attrition", {}) or {})
    n_target = int(att.get("n_target_rows", 0) or 0)
    lost = int(att.get("lost_to_join", 0) or 0)
    return {
        "profile": profile_name,
        "asset": frame.attrs.get("asset", ""),
        "oxfordman_symbol": frame.attrs.get("oxfordman_symbol", ""),
        "session": frame.attrs.get("session", ""),
        "vix_joined": bool(frame.attrs.get("vix_joined", False)),
        "n_target_rows": n_target,
        "n_after_join": int(att.get("n_after_join", 0) or 0),
        "n_modelling_rows": int(att.get("n_after_dropna", len(frame))),
        "lost_to_calendar_join": lost,
        "lost_to_warmup_or_nan": int(att.get("lost_to_warmup_or_nan", 0) or 0),
        "pct_lost_to_join": (100.0 * lost / n_target) if n_target else float("nan"),
        "first_date": str(frame.index.min().date()) if len(frame) else "",
        "last_date": str(frame.index.max().date()) if len(frame) else "",
    }
