"""Tests for the modelling-frame assembler and the asset registry.

Why this module exists (added 2026-08-30)
-----------------------------------------
``src/data/datasets.py`` is the single entry point every phase uses to build its
modelling frame, and until now it had no tests at all. Two defects survived
because of that:

* the Oxford-Man symbol was read from ``cfg.oxfordman.primary_symbol``
  unconditionally, so the *target* was pinned to ``.SPX`` regardless of whose
  returns and OHLC the rest of the frame carried. Nothing would have failed on a
  Phase-8 profile -- it would simply have regressed Russell 2000 returns on S&P
  500 realized variance and printed a plausible table;
* ``assets:`` in ``configs/data.yaml`` had no consumer whatsoever, so it drifted
  out of date silently (it still named AAPL and TSLA a week after the RQ4 asset
  decision replaced them).

Both are now behaviours with tests behind them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.conftest import log_text

from src.data.datasets import (
    AssetSpec,
    build_econometric_frame,
    realized_variance_target,
    resolve_asset,
)

BDAYS = pd.bdate_range("2015-01-01", periods=400)


def _write_ohlc(path, index, *, level=100.0, seed=0):
    rng = np.random.default_rng(seed)
    close = level * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    frame = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(index))),
            "high": close * (1 + abs(rng.normal(0, 0.004, len(index)))),
            "low": close * (1 - abs(rng.normal(0, 0.004, len(index)))),
            "close": close,
            "volume": rng.integers(1e6, 5e6, len(index)),
        },
        index=index,
    )
    frame.index.name = "date"
    frame.to_parquet(path)
    return frame


def _write_oxfordman(path, per_symbol: dict[str, pd.DatetimeIndex], seed=0):
    """Long-format CSV in the shape ``load_oxfordman_symbol`` expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for sym, idx in per_symbol.items():
        rv = np.exp(rng.normal(-9.2, 0.6, len(idx)))
        rows.append(pd.DataFrame({"Unnamed: 0": idx.strftime("%Y-%m-%d %H:%M:%S%z"),
                                  "Symbol": sym, "rv5": rv, "bv": rv * 0.98}))
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


@pytest.fixture()
def cfg(tmp_path):
    """A config with a three-asset registry and matching cached data on disk.

    ``UK`` deliberately trades on a calendar offset from ``US`` so the inner-join
    attrition the frame builder now reports is actually exercised.
    """
    raw = tmp_path / "data" / "raw"
    (raw / "yfinance").mkdir(parents=True)
    (raw / "oxfordman").mkdir(parents=True)

    us_index = BDAYS
    uk_index = BDAYS[BDAYS.dayofweek != 0]  # drop every Monday: a different calendar

    _write_ohlc(raw / "yfinance" / "GSPC.parquet", us_index, seed=1)
    _write_ohlc(raw / "yfinance" / "RUT.parquet", us_index, level=1500.0, seed=2)
    _write_ohlc(raw / "yfinance" / "FTSE.parquet", uk_index, level=7000.0, seed=3)
    _write_ohlc(raw / "yfinance" / "VIX.parquet", us_index, level=18.0, seed=4)
    _write_oxfordman(
        raw / "oxfordman" / "om.csv",
        {".SPX": us_index, ".RUT": us_index, ".FTSE": uk_index},
    )

    return OmegaConf.create(
        {
            "paths": {
                "yfinance": str(raw / "yfinance"),
                "oxfordman": str(raw / "oxfordman"),
            },
            "oxfordman": {"filename": "om.csv", "primary_symbol": ".SPX"},
            "assets": {
                "primary": "SPX",
                "vix": "VIX",
                "robustness": ["RUT", "FTSE"],
                "registry": {
                    "SPX": {"yfinance": "^GSPC", "cache_alias": "GSPC",
                            "oxfordman": ".SPX", "session": "US", "vix_feature": True},
                    "RUT": {"yfinance": "^RUT", "cache_alias": "RUT",
                            "oxfordman": ".RUT", "session": "US", "vix_feature": True},
                    "FTSE": {"yfinance": "^FTSE", "cache_alias": "FTSE",
                             "oxfordman": ".FTSE", "session": "UK", "vix_feature": False},
                    "VIX": {"yfinance": "^VIX", "cache_alias": "VIX",
                            "oxfordman": None, "session": "US", "vix_feature": False},
                },
            },
        }
    )


# --------------------------------------------------------------------------- #
# resolve_asset                                                                #
# --------------------------------------------------------------------------- #


def test_resolve_asset_defaults_to_the_configured_primary(cfg):
    spec = resolve_asset(cfg)
    assert spec == AssetSpec(
        key="SPX", yfinance="^GSPC", cache_alias="GSPC", oxfordman=".SPX",
        session="US", vix_feature=True, description="",
    )


def test_resolve_asset_returns_all_three_identifiers_together(cfg):
    """The point of the registry: one lookup, no chance of a mismatched pair."""
    rut = resolve_asset(cfg, "RUT")
    assert (rut.yfinance, rut.cache_alias, rut.oxfordman) == ("^RUT", "RUT", ".RUT")


def test_resolve_asset_rejects_an_unknown_key_instead_of_falling_back(cfg):
    with pytest.raises(KeyError, match="unknown asset"):
        resolve_asset(cfg, "NIKKEI")


def test_resolve_asset_falls_back_to_spx_for_a_pre_registry_config():
    """A config written before 2026-08-30 still resolves, with old behaviour."""
    legacy = OmegaConf.create({"oxfordman": {"primary_symbol": ".SPX"}})
    spec = resolve_asset(legacy)
    assert (spec.key, spec.cache_alias, spec.oxfordman) == ("SPX", "GSPC", ".SPX")
    with pytest.raises(KeyError, match="no 'assets.registry' block"):
        resolve_asset(legacy, "RUT")


def test_vix_is_never_a_forecast_target(cfg):
    assert resolve_asset(cfg, "VIX").oxfordman is None


# --------------------------------------------------------------------------- #
# symbol threading -- the defect this was written for                          #
# --------------------------------------------------------------------------- #


def test_realized_variance_target_defaults_to_the_config_symbol(cfg):
    """Omitting ``symbol`` must reproduce the pre-2026-08-30 behaviour exactly."""
    default = realized_variance_target(cfg, "oxfordman_rv5")
    explicit = realized_variance_target(cfg, "oxfordman_rv5", symbol=".SPX")
    pd.testing.assert_series_equal(default, explicit)


def test_realized_variance_target_honours_an_explicit_symbol(cfg):
    spx = realized_variance_target(cfg, "oxfordman_rv5", symbol=".SPX")
    rut = realized_variance_target(cfg, "oxfordman_rv5", symbol=".RUT")
    assert not np.allclose(spx.to_numpy(), rut.to_numpy())


def test_frame_pairs_each_asset_with_its_own_realized_measure(cfg):
    """Regression guard for the defect: returns and target must be one asset.

    Before the symbol was threaded through, ``asset="RUT"`` produced Russell
    returns against the S&P's rv5 -- a frame that looks entirely normal.
    """
    spx = build_econometric_frame(cfg, target="oxfordman_rv5", asset="SPX")
    rut = build_econometric_frame(cfg, target="oxfordman_rv5", asset="RUT")

    assert spx.attrs["oxfordman_symbol"] == ".SPX"
    assert rut.attrs["oxfordman_symbol"] == ".RUT"
    assert not np.allclose(spx["rv"].to_numpy(), rut["rv"].to_numpy())
    assert not np.allclose(spx["log_return"].to_numpy(), rut["log_return"].to_numpy())


def test_omitting_asset_reproduces_the_primary_frame(cfg):
    """Every pre-Phase-8 call site omits ``asset``; it must not have moved."""
    implicit = build_econometric_frame(cfg, target="oxfordman_rv5")
    explicit = build_econometric_frame(cfg, target="oxfordman_rv5", asset="SPX")
    pd.testing.assert_frame_equal(implicit, explicit)


# --------------------------------------------------------------------------- #
# the VIX guard and calendar attrition                                         #
# --------------------------------------------------------------------------- #


def test_vix_is_joined_for_a_us_asset(cfg):
    frame = build_econometric_frame(cfg, target="oxfordman_rv5", asset="RUT",
                                    include_vix=True)
    assert "vix_close" in frame.columns
    assert frame.attrs["vix_joined"] is True


def test_vix_is_refused_for_a_non_us_asset(cfg, project_logs):
    """Requesting the VIX for the FTSE must drop the column, not join it.

    Joining would be worse than useless: the CBOE VIX prices S&P 500 options, and
    the inner join would silently intersect the LSE and NYSE holiday calendars.

    Uses ``project_logs`` rather than ``caplog``: the project's loggers do not
    propagate, so root-attached capture is pytest-version dependent (see the
    fixture's docstring).
    """
    records = project_logs("datasets")
    frame = build_econometric_frame(cfg, target="oxfordman_rv5", asset="FTSE",
                                    include_vix=True)
    assert "vix_close" not in frame.columns
    assert frame.attrs["vix_joined"] is False
    assert "vix_feature=false" in log_text(records)


def test_frame_records_calendar_attrition(cfg):
    """A UK asset joined against UK sources loses nothing to the calendar."""
    frame = build_econometric_frame(cfg, target="oxfordman_rv5", asset="FTSE")
    att = frame.attrs["join_attrition"]
    assert att["lost_to_join"] == 0
    assert att["n_after_dropna"] == att["n_after_join"] - att["lost_to_warmup_or_nan"]
    assert att["lost_to_warmup_or_nan"] > 0  # the ~22-day HAR warm-up


def test_attrition_is_visible_when_calendars_disagree(cfg, project_logs):
    """Force the mismatch the FTSE guard exists to prevent, and check it is loud.

    ``alias="GSPC"`` pairs the FTSE's target with NYSE OHLC -- exactly the shape
    of the error a hand-written Phase-8 profile could make. The frame builder must
    report the loss rather than quietly returning a short sample.
    """
    records = project_logs("datasets")
    frame = build_econometric_frame(
        cfg, target="oxfordman_rv5", asset="FTSE", alias="GSPC"
    )
    assert frame.attrs["join_attrition"]["lost_to_join"] == 0
    # The FTSE target is a strict subset of the NYSE calendar here, so nothing is
    # lost from the target side; the guard fires on the reverse pairing.
    frame2 = build_econometric_frame(
        cfg, target="oxfordman_rv5", asset="SPX", alias="FTSE"
    )
    att = frame2.attrs["join_attrition"]
    assert att["lost_to_join"] > 0
    assert "calendar join dropped" in log_text(records)
    # The attrs are the contract; the warning is how a human finds out. Assert
    # both, so a silent regression in either is caught.
    assert att["lost_to_join"] / att["n_target_rows"] > 0.02
