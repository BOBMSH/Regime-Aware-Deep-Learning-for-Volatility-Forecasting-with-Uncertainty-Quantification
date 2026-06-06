"""Unit tests for realized-volatility estimators (roadmap Phase 1 gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.realized_vol import (
    log_returns,
    parkinson_rv,
    realized_vol_from_intraday_rv,
    squared_returns_rv,
    yang_zhang_rv,
)


@pytest.fixture
def gbm_ohlc() -> pd.DataFrame:
    """Synthetic OHLC frame with known true volatility.

    Generates a GBM with daily sigma=0.01; high/low/open derived from intraday
    geometric Brownian bridge approximations so that OHLC estimators have
    something realistic to chew on.
    """
    rng = np.random.default_rng(20260515)
    n = 600
    sigma = 0.01
    mu = 0.0
    log_close = np.cumsum(rng.normal(mu, sigma, size=n))
    close = np.exp(log_close) * 100.0
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, sigma * 0.3, size=n))
    intraday_high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sigma * 0.5, size=n)))
    intraday_low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sigma * 0.5, size=n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": intraday_high, "low": intraday_low, "close": close}, index=idx
    )


class TestLogReturns:
    def test_first_value_is_nan(self):
        s = pd.Series([100, 101, 102, 99], index=pd.bdate_range("2020-01-01", periods=4))
        r = log_returns(s)
        assert np.isnan(r.iloc[0])
        assert not r.iloc[1:].isna().any()

    def test_known_values(self):
        s = pd.Series([100.0, 110.0], index=pd.bdate_range("2020-01-01", periods=2))
        r = log_returns(s)
        np.testing.assert_allclose(r.iloc[1], np.log(110.0 / 100.0))


class TestSquaredReturnsRV:
    def test_non_negative(self, gbm_ohlc):
        rv = squared_returns_rv(gbm_ohlc["close"])
        assert (rv.dropna() >= 0).all()

    def test_variance_vs_volatility(self, gbm_ohlc):
        var = squared_returns_rv(gbm_ohlc["close"], squared=True)
        vol = squared_returns_rv(gbm_ohlc["close"])
        np.testing.assert_allclose(np.sqrt(var.dropna()), vol.dropna())


class TestYangZhang:
    def test_requires_window_at_least_two(self, gbm_ohlc):
        with pytest.raises(ValueError):
            yang_zhang_rv(gbm_ohlc, window=1)

    def test_validates_columns(self):
        df = pd.DataFrame({"high": [1], "low": [1], "close": [1]})
        with pytest.raises(ValueError, match="open"):
            yang_zhang_rv(df)

    def test_window_rows_are_nan(self, gbm_ohlc):
        rv = yang_zhang_rv(gbm_ohlc, window=21)
        assert rv.iloc[:20].isna().all()
        # Conservative: at least one of the first 21 rows is NaN, and rows beyond
        # the window are mostly populated.
        assert rv.iloc[21:].notna().mean() > 0.95

    def test_non_negative(self, gbm_ohlc):
        rv = yang_zhang_rv(gbm_ohlc, window=21)
        assert (rv.dropna() >= 0).all()

    def test_recovers_true_sigma_in_expectation(self, gbm_ohlc):
        # The synthetic GBM has daily sigma ~ 0.01. With OHLC noise the YZ
        # estimator should land within an order of magnitude of that.
        rv = yang_zhang_rv(gbm_ohlc, window=63).dropna()
        # YZ returns daily volatility; compare against the daily sigma directly.
        assert 0.001 < rv.median() < 0.1

    def test_annualisation_scales_correctly(self, gbm_ohlc):
        rv_daily = yang_zhang_rv(gbm_ohlc, window=21).dropna()
        rv_ann = yang_zhang_rv(gbm_ohlc, window=21, annualize=True).dropna()
        np.testing.assert_allclose(rv_ann.values, rv_daily.values * np.sqrt(252), rtol=1e-10)

    def test_variance_squared_matches_volatility_squared(self, gbm_ohlc):
        var = yang_zhang_rv(gbm_ohlc, window=21, squared=True).dropna()
        vol = yang_zhang_rv(gbm_ohlc, window=21).dropna()
        np.testing.assert_allclose(np.sqrt(var.values), vol.values, rtol=1e-10)


class TestParkinson:
    def test_non_negative(self, gbm_ohlc):
        rv = parkinson_rv(gbm_ohlc)
        assert (rv.dropna() >= 0).all()


class TestIntradayRV:
    def test_volatility_is_sqrt_of_variance(self):
        var = pd.Series([0.0001, 0.0004, 0.0009])
        vol = realized_vol_from_intraday_rv(var)
        np.testing.assert_allclose(vol.values, [0.01, 0.02, 0.03])

    def test_clips_negative_to_zero(self):
        var = pd.Series([-1e-9, 1e-4])
        vol = realized_vol_from_intraday_rv(var)
        assert vol.iloc[0] == 0.0
