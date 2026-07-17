"""Unit tests for the evaluation losses (roadmap Phase 2 gate; §4 test_metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    mae,
    mpiw,
    mse,
    picp,
    point_metrics,
    qlike,
    rmse,
    winkler_score,
)


class TestPointLosses:
    def test_zero_error_is_zero(self):
        y = np.array([1e-4, 2e-4, 3e-4])
        assert mse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert rmse(y, y) == 0.0
        # QLIKE is a Bregman divergence: exactly 0 when forecast == target.
        assert qlike(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_mse_known_value(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0, 5.0])  # errors 0,0,2 -> mean(0,0,4)=4/3
        assert mse(y, f) == pytest.approx(4.0 / 3.0)
        assert rmse(y, f) == pytest.approx(np.sqrt(4.0 / 3.0))

    def test_mae_known_value(self):
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.5, 2.0, 1.0])  # abs errors 0.5,0,2 -> mean = 2.5/3
        assert mae(y, f) == pytest.approx(2.5 / 3.0)

    def test_qlike_positive_and_minimised_at_truth(self):
        # QLIKE(sigma2, h) is minimised uniquely at h == sigma2; perturbing the
        # forecast either way must strictly increase the loss.
        rng = np.random.default_rng(0)
        sigma2 = rng.uniform(5e-5, 5e-4, size=500)
        at_truth = qlike(sigma2, sigma2)
        over = qlike(sigma2, sigma2 * 1.3)
        under = qlike(sigma2, sigma2 * 0.7)
        assert at_truth < over
        assert at_truth < under
        assert over > 0 and under > 0

    def test_qlike_penalises_underprediction_more(self):
        # A symmetric proportional miss should cost more when under-predicting
        # (Patton 2011 / Chapter 2 §2.3: QLIKE's asymmetry).
        sigma2 = np.full(1000, 1e-4)
        over = qlike(sigma2, sigma2 * 1.5)   # forecast 50% too high
        under = qlike(sigma2, sigma2 / 1.5)  # forecast 50% too low
        assert under > over

    def test_series_alignment_by_index(self):
        # Metrics on Series align on the index, not by position.
        idx = pd.date_range("2022-01-03", periods=5, freq="B")
        y = pd.Series([1e-4, 2e-4, 3e-4, 4e-4, 5e-4], index=idx)
        f = y.copy().iloc[::-1]  # reversed values but SAME index labels
        # Aligned by index => compares each date to itself under both orderings.
        assert mse(y, f) == pytest.approx(mse(y.values, f.reindex(idx).values))

    def test_drops_nan_pairs(self):
        y = np.array([1e-4, np.nan, 3e-4])
        f = np.array([1e-4, 2e-4, 3e-4])
        assert mse(y, f) == pytest.approx(0.0)  # only the two finite pairs, both exact

    def test_qlike_handles_nonpositive_forecast_via_floor(self):
        y = np.array([1e-4, 2e-4])
        f = np.array([0.0, 2e-4])  # a zero forecast is floored, not a crash
        val = qlike(y, f)
        assert np.isfinite(val) and val > 0

    def test_point_metrics_bundle(self):
        y = np.array([1e-4, 2e-4, 3e-4])
        f = np.array([1.1e-4, 1.9e-4, 3.2e-4])
        m = point_metrics(y, f)
        assert set(m) == {"n", "mse", "rmse", "mae", "qlike"}
        assert m["n"] == 3
        assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))


class TestIntervalLosses:
    def test_picp_full_and_zero_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        assert picp(y, y - 1, y + 1) == 1.0
        assert picp(y, y + 10, y + 20) == 0.0

    def test_picp_partial(self):
        y = np.array([0.0, 5.0, 10.0])
        lo = np.array([-1.0, 6.0, 9.0])
        hi = np.array([1.0, 7.0, 11.0])  # inside, outside, inside -> 2/3
        assert picp(y, lo, hi) == pytest.approx(2.0 / 3.0)

    def test_mpiw(self):
        lo = np.array([0.0, 1.0])
        hi = np.array([2.0, 5.0])  # widths 2 and 4
        assert mpiw(lo, hi) == pytest.approx(3.0)

    def test_winkler_reduces_to_width_when_covered(self):
        y = np.array([1.0, 2.0])
        lo = np.array([0.0, 1.0])
        hi = np.array([2.0, 3.0])  # both covered -> score == mean width == 2
        assert winkler_score(y, lo, hi, alpha=0.05) == pytest.approx(2.0)

    def test_winkler_penalises_misses(self):
        y = np.array([5.0])
        lo = np.array([0.0])
        hi = np.array([2.0])  # miss above by 3 -> width 2 + (2/alpha)*3
        expected = 2.0 + (2.0 / 0.05) * 3.0
        assert winkler_score(y, lo, hi, alpha=0.05) == pytest.approx(expected)


class TestGuards:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([np.nan]), np.array([np.nan]))
