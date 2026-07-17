"""Unit tests for the arch-backed GARCH / EGARCH forecasters (roadmap Phase 2).

These are lightweight behavioural tests -- correctness of the fit/forecast/rescale
cycle and leakage-safety -- not a re-test of ``arch`` itself. A synthetic GARCH
series is used so the fitted persistence is known to be high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import har_lagged_rv
from src.data.splits import Fold, SplitConfig, walk_forward_folds
from src.models.econometric import EGARCHForecaster, GARCHForecaster


@pytest.fixture
def garch_frame() -> pd.DataFrame:
    """Simulate a GARCH(1,1) return path and build the modelling frame from it."""
    rng = np.random.default_rng(7)
    n = 2200
    omega, alpha, beta = 2e-6, 0.08, 0.90  # persistence 0.98, realistic for equity
    eps = np.zeros(n)
    h = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = rng.normal(0, np.sqrt(h[t]))
    idx = pd.bdate_range("2012-01-02", periods=n)
    ret = pd.Series(eps, index=idx, name="log_return")
    # A realized-variance proxy (squared returns) so the frame has an `rv` target.
    rv = ret.pow(2).rename("rv").clip(lower=1e-10)
    frame = pd.concat([ret, rv, har_lagged_rv(rv)], axis=1).dropna()
    return frame


def _one_fold(frame, n_predict=40) -> Fold:
    return Fold(
        fold_idx=0,
        refit_date=frame.index[-n_predict],
        train_start=frame.index[0],
        train_end=frame.index[-n_predict - 1],
        predict_start=frame.index[-n_predict],
        predict_end=frame.index[-1],
    )


class TestGarch:
    def test_forecasts_positive_and_aligned(self, garch_frame):
        m = GARCHForecaster()
        fold = _one_fold(garch_frame)
        pred = m.forecast_fold(garch_frame, fold)
        assert (pred > 0).all()
        assert pred.index.min() == fold.predict_start
        assert pred.index.max() == fold.predict_end
        assert len(pred) == 40

    def test_recovers_high_persistence(self, garch_frame):
        m = GARCHForecaster()
        m.forecast_fold(garch_frame, _one_fold(garch_frame))
        pers = m.fold_params_[0]["persistence"]
        # True alpha+beta = 0.98; estimate should be clearly persistent.
        assert 0.85 < pers < 1.0

    def test_forecast_scale_matches_variance_target(self, garch_frame):
        # The unscaled (decimal) variance forecast must be the same order of
        # magnitude as the realized variance -- catches a bad scale**2 factor.
        m = GARCHForecaster(scale=100.0)
        pred = m.forecast_fold(garch_frame, _one_fold(garch_frame))
        rv = garch_frame["rv"]
        assert 0.1 < pred.mean() / rv.mean() < 10.0

    def test_forecasts_vary_within_fold(self, garch_frame):
        # Fixed params but the variance recursion is filtered by realized returns,
        # so within-fold forecasts must not be constant.
        m = GARCHForecaster()
        pred = m.forecast_fold(garch_frame, _one_fold(garch_frame))
        assert pred.std() > 0

    def test_no_leakage_train_end_before_first_forecast(self, garch_frame):
        cfg = SplitConfig(
            train_end="2017-12-31", val_start="2018-01-01", val_end="2018-12-31",
            test_start="2019-01-01", test_end="2019-12-31", refit_frequency_days=21,
        )
        m = GARCHForecaster()
        for fold in walk_forward_folds(garch_frame.index, cfg, eval_segment="test"):
            pred = m.forecast_fold(garch_frame, fold)
            assert (pred.index > fold.train_end).all()
            assert pred.index.min() == fold.predict_start


class TestEgarch:
    def test_forecasts_positive_and_aligned(self, garch_frame):
        m = EGARCHForecaster()
        fold = _one_fold(garch_frame)
        pred = m.forecast_fold(garch_frame, fold)
        assert (pred > 0).all()
        assert len(pred) == 40

    def test_distinct_from_garch(self, garch_frame):
        # EGARCH and GARCH should not produce identical forecasts.
        fold = _one_fold(garch_frame)
        g = GARCHForecaster().forecast_fold(garch_frame, fold)
        e = EGARCHForecaster().forecast_fold(garch_frame, fold)
        assert not np.allclose(g.values, e.values)
