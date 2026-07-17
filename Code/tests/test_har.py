"""Unit tests for HAR-RV (roadmap Phase 2 gate; §4 test_har).

Covers the leakage-safety of the HAR lag construction and the estimator itself:
on data generated from a known HAR process, OLS must recover the true
coefficients, and the fold forecaster must be exactly leakage-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import har_lagged_rv
from src.data.splits import Fold, SplitConfig, walk_forward_folds
from src.models.econometric.har import HARForecaster


@pytest.fixture
def rv_series() -> pd.Series:
    """A positive, persistent synthetic realized-variance series."""
    rng = np.random.default_rng(11)
    n = 1500
    idx = pd.bdate_range("2015-01-01", periods=n)
    # AR(1)-ish positive process in log space -> lognormal RV, realistic autocorr.
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 0.3)
    return pd.Series(np.exp(-9 + x), index=idx, name="rv")


class TestHarLags:
    def test_lags_are_strictly_backward(self, rv_series):
        lags = har_lagged_rv(rv_series)
        # rv_d on day t must equal rv on day t-1 (pure shift, no peeking).
        aligned = pd.concat([rv_series.shift(1).rename("shift1"), lags["rv_d"]], axis=1).dropna()
        np.testing.assert_allclose(aligned["shift1"].values, aligned["rv_d"].values)

    def test_weekly_monthly_are_trailing_means(self, rv_series):
        lags = har_lagged_rv(rv_series)
        t = 100
        day = rv_series.index[t]
        exp_w = rv_series.shift(1).iloc[t - 4 : t + 1].mean()  # 5-day trailing of shifted
        assert lags.loc[day, "rv_w"] == pytest.approx(exp_w)

    def test_first_rows_are_nan(self, rv_series):
        lags = har_lagged_rv(rv_series)
        assert lags["rv_m"].iloc[:22].isna().all()


class TestHarEstimation:
    def _make_har_frame(self, seed=3, n=1600, betas=(1e-5, 0.4, 0.3, 0.25)):
        """Generate RV from a known HAR data-generating process, then rebuild the
        modelling frame from it so the test exercises the real feature code."""
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2010-01-01", periods=n)
        b0, bd, bw, bm = betas
        rv = np.empty(n)
        rv[:22] = 1e-4
        for t in range(22, n):
            d = rv[t - 1]
            w = rv[t - 5 : t].mean()
            m = rv[t - 22 : t].mean()
            mu = b0 + bd * d + bw * w + bm * m
            rv[t] = max(mu + rng.normal(0, 2e-6), 1e-8)  # small noise, stay positive
        rv = pd.Series(rv, index=idx, name="rv")
        lags = har_lagged_rv(rv)
        frame = pd.concat([rv, lags], axis=1).dropna()
        return frame, betas

    def test_recovers_known_coefficients(self):
        frame, (b0, bd, bw, bm) = self._make_har_frame()
        model = HARForecaster()
        # Single "fold" spanning the whole sample for a clean coefficient read.
        fold = Fold(
            fold_idx=0,
            refit_date=frame.index[-10],
            train_start=frame.index[0],
            train_end=frame.index[-11],
            predict_start=frame.index[-10],
            predict_end=frame.index[-1],
        )
        model.forecast_fold(frame, fold)
        p = model.fold_params_[0]
        # OLS should recover the DGP coefficients closely.
        assert p["beta_d"] == pytest.approx(bd, abs=0.08)
        assert p["beta_w"] == pytest.approx(bw, abs=0.10)
        assert p["beta_m"] == pytest.approx(bm, abs=0.10)
        assert p["r2"] > 0.5

    def test_forecast_is_positive_and_aligned(self):
        frame, _ = self._make_har_frame()
        model = HARForecaster()
        fold = Fold(
            fold_idx=0, refit_date=frame.index[-30], train_start=frame.index[0],
            train_end=frame.index[-31], predict_start=frame.index[-30],
            predict_end=frame.index[-1],
        )
        pred = model.forecast_fold(frame, fold)
        assert (pred > 0).all()
        # Predictions are indexed exactly on the fold's prediction window.
        assert pred.index.min() == fold.predict_start
        assert pred.index.max() == fold.predict_end

    def test_no_leakage_across_walk_forward(self):
        frame, _ = self._make_har_frame(n=1700)
        cfg = SplitConfig(
            train_end="2013-12-31", val_start="2014-01-01", val_end="2014-12-31",
            test_start="2015-01-01", test_end="2016-06-30", refit_frequency_days=21,
        )
        model = HARForecaster()
        for fold in walk_forward_folds(frame.index, cfg, eval_segment="test"):
            pred = model.forecast_fold(frame, fold)
            # Every forecast date is strictly after the training window.
            assert (pred.index > fold.train_end).all()

    def test_log_transform_runs_and_stays_positive(self):
        frame, _ = self._make_har_frame()
        model = HARForecaster(transform="log")
        fold = Fold(
            fold_idx=0, refit_date=frame.index[-20], train_start=frame.index[0],
            train_end=frame.index[-21], predict_start=frame.index[-20],
            predict_end=frame.index[-1],
        )
        pred = model.forecast_fold(frame, fold)
        assert (pred > 0).all()

    def test_rejects_bad_transform(self):
        with pytest.raises(ValueError):
            HARForecaster(transform="sqrt")
