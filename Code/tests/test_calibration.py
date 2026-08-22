"""Unit tests for the Phase 6 calibration helpers (roadmap Phase 6; RQ3).

Pure numpy/pandas -- no torch. Checks the log-normal interval algebra, the
interval-metric bundle, the reliability curve on a genuinely calibrated sample,
and the per-regime aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import (
    interval_metrics,
    lognormal_interval,
    lognormal_mean,
    lognormal_quantile,
    per_regime_interval_metrics,
    reliability_curve,
)


class TestLogNormalInterval:
    def test_ordering_and_level_monotonicity(self):
        mu = np.array([-9.0, -8.0, -7.0])
        sigma = np.array([0.3, 0.5, 0.4])
        lo90, hi90 = lognormal_interval(mu, sigma, level=0.90)
        lo95, hi95 = lognormal_interval(mu, sigma, level=0.95)
        assert (lo90 < hi90).all()
        # a higher confidence level is a wider interval on both ends
        assert (lo95 <= lo90).all() and (hi95 >= hi90).all()
        # endpoints are positive (variance scale)
        assert (lo95 > 0).all()

    def test_median_is_exp_mu(self):
        mu = np.array([-9.0, -8.5])
        sigma = np.array([0.4, 0.6])
        np.testing.assert_allclose(lognormal_quantile(mu, sigma, 0.5), np.exp(mu), rtol=1e-9)

    def test_mean_exceeds_median(self):
        mu = np.array([-9.0]); sigma = np.array([0.5])
        # lognormal mean exp(mu+sigma^2/2) > median exp(mu)
        assert lognormal_mean(mu, sigma)[0] > np.exp(mu)[0]

    def test_symmetric_tails_in_log_space(self):
        mu = np.array([-9.0]); sigma = np.array([0.5])
        lo, hi = lognormal_interval(mu, sigma, level=0.90)
        # in log space the interval is symmetric about mu
        np.testing.assert_allclose(np.log(lo)[0] - mu[0], -(np.log(hi)[0] - mu[0]), rtol=1e-6)

    def test_level_bounds_validated(self):
        with pytest.raises(ValueError):
            lognormal_interval(np.array([0.0]), np.array([1.0]), level=1.0)


class TestIntervalMetrics:
    def test_bundle_keys_and_perfect_coverage(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        lo = y - 1
        hi = y + 1
        m = interval_metrics(y, lo, hi, level=0.90)
        assert set(m) == {"n", "picp", "mpiw", "winkler", "coverage_error"}
        assert m["n"] == 5
        assert m["picp"] == 1.0
        assert m["mpiw"] == pytest.approx(2.0)
        assert m["coverage_error"] == pytest.approx(1.0 - 0.90)

    def test_undercoverage_negative_error(self):
        y = np.array([0.0, 5.0, 10.0])
        lo = np.array([-1.0, 6.0, 9.0])
        hi = np.array([1.0, 7.0, 11.0])  # 2/3 covered
        m = interval_metrics(y, lo, hi, level=0.90)
        assert m["picp"] == pytest.approx(2 / 3)
        assert m["coverage_error"] < 0


class TestReliabilityCurve:
    def test_calibrated_normal_sample_is_near_diagonal(self):
        # y ~ N(0,1); the predicted tau-quantile is the true Phi^{-1}(tau) (constant
        # across obs). Empirical coverage should match tau to within sampling error.
        rng = np.random.default_rng(1)
        n = 20000
        y = pd.Series(rng.standard_normal(n))
        from scipy import stats as st
        curve = reliability_curve(y, lambda tau: pd.Series(np.full(n, st.norm.ppf(tau)),
                                                           index=y.index),
                                  [0.1, 0.25, 0.5, 0.75, 0.9])
        assert (curve["gap"].abs() < 0.02).all()
        assert list(curve["tau"]) == sorted(curve["tau"])

    def test_gap_definition(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        # q_tau that puts exactly half at/below -> empirical 0.5
        curve = reliability_curve(y, lambda tau: pd.Series([2.5] * 4, index=y.index), [0.5])
        assert curve.loc[0, "empirical"] == pytest.approx(0.5)
        assert curve.loc[0, "gap"] == pytest.approx(0.0)


# These fixtures score a short synthetic state series over its own full extent, so
# the leading row is unlabelled by construction and there is no earlier history that
# could have been supplied. That is the one case where the reach check added in
# 2026-08-19 (iv) is noise rather than signal; see
# tests/test_regime_timing.py::TestLabelReach for the case it does catch.
@pytest.mark.filterwarnings(
    "ignore::src.evaluation.regime_timing.RegimeLabelReachWarning"
)
class TestPerRegime:
    def test_split_and_all_row_expost(self):
        idx = pd.date_range("2020-01-01", periods=6, freq="B")
        y = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
        lo = pd.Series([0, 0, 0, 0, 0, 0], index=idx, dtype=float)
        hi = pd.Series([2, 2, 4, 4, 6, 6], index=idx, dtype=float)  # covers 1,_,3,4,5,6
        state = pd.Series([0, 0, 0, 1, 1, 1], index=idx)            # two regimes of 3
        out = per_regime_interval_metrics(y, lo, hi, state, ["calm", "crisis"],
                                          level=0.90, shift=0)
        assert list(out["regime"]) == ["calm", "crisis", "all"]
        assert int(out.loc[out["regime"] == "all", "n"].iloc[0]) == 6
        # calm: y=1 in [0,2]? yes; y=2 in [0,2]? yes; y=3 in [0,4]? yes -> 3/3
        assert out.loc[out["regime"] == "calm", "picp"].iloc[0] == pytest.approx(1.0)
        assert out.attrs["regime_shift"] == 0

    def test_default_shift_is_one_and_lags_the_label(self):
        """The default bucketing must use the regime known at t-1.

        This is the fix for the outcome-conditioning defect: the filtered state
        s_t is inferred using the day-t observation, so bucketing day t by s_t
        sorts days by the outcome being scored. Here the state switches at row 3,
        so under the default the switch day itself must still be scored in the
        OLD regime -- that is the whole point.
        """
        idx = pd.date_range("2020-01-01", periods=6, freq="B")
        y = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
        lo = y - 1.0
        hi = y + 1.0
        state = pd.Series([0, 0, 0, 1, 1, 1], index=idx)
        out = per_regime_interval_metrics(y, lo, hi, state, ["calm", "crisis"], level=0.90)
        assert out.attrs["regime_shift"] == 1
        # Row 0 has no t-1 label and is dropped; the switch day (row 3) carries the
        # previous day's label, so calm gets rows 1,2,3 and crisis rows 4,5.
        assert int(out.loc[out["regime"] == "calm", "n"].iloc[0]) == 3
        assert int(out.loc[out["regime"] == "crisis", "n"].iloc[0]) == 2
        # Buckets must sum to the 'all' row -- a mismatch is how the previous
        # convention hid that buckets and pooled tests used different samples.
        assert int(out.loc[out["regime"] == "all", "n"].iloc[0]) == 5

    def test_shift_changes_the_answer(self):
        """A regression guard: the two timings are not interchangeable."""
        idx = pd.date_range("2020-01-01", periods=8, freq="B")
        y = pd.Series([1, 1, 1, 9, 1, 1, 1, 1], index=idx, dtype=float)
        lo = pd.Series(0.0, index=idx)
        hi = pd.Series(2.0, index=idx)          # the spike at row 3 is a miss
        # The filter only learns about the spike on the day it happens.
        state = pd.Series([0, 0, 0, 1, 0, 0, 0, 0], index=idx)
        ex = per_regime_interval_metrics(y, lo, hi, state, ["calm", "crisis"],
                                         level=0.90, shift=0)
        lag = per_regime_interval_metrics(y, lo, hi, state, ["calm", "crisis"], level=0.90)
        # Ex-post: the miss is attributed to 'crisis' -> crisis PICP 0, calm 1.
        assert ex.loc[ex["regime"] == "crisis", "picp"].iloc[0] == pytest.approx(0.0)
        assert ex.loc[ex["regime"] == "calm", "picp"].iloc[0] == pytest.approx(1.0)
        # Lagged: the miss lands in 'calm', because calm is what was known at t-1.
        assert lag.loc[lag["regime"] == "calm", "picp"].iloc[0] < 1.0

    def test_empty_regime_is_nan_not_crash(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="B")
        y = pd.Series([1.0, 2.0, 3.0], index=idx)
        lo = y - 1; hi = y + 1
        state = pd.Series([0, 0, 0], index=idx)
        out = per_regime_interval_metrics(y, lo, hi, state, ["calm", "crisis"], level=0.90)
        crisis = out[out["regime"] == "crisis"].iloc[0]
        assert crisis["n"] == 0 and pd.isna(crisis["picp"])
