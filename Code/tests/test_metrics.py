"""Unit tests for the evaluation losses (roadmap Phase 2 gate; §4 test_metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    coverage_error,
    crps_from_quantiles,
    crps_lognormal,
    mae,
    mean_pinball_loss,
    mpiw,
    mse,
    picp,
    pinball_loss,
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


class TestPinballLoss:
    def test_median_equals_half_mae(self):
        # pinball at tau=0.5 is exactly 0.5 * MAE.
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        f = rng.normal(size=200)
        assert pinball_loss(y, f, tau=0.5) == pytest.approx(0.5 * mae(y, f))

    def test_asymmetry_matches_tau(self):
        # under-prediction (y > q) is weighted tau; over-prediction weighted (1-tau).
        y = np.zeros(1)
        assert pinball_loss(y, np.array([-1.0]), tau=0.9) == pytest.approx(0.9)   # y-q=+1
        assert pinball_loss(y, np.array([1.0]), tau=0.9) == pytest.approx(0.1)    # y-q=-1

    def test_zero_at_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(y, y, tau=0.05) == pytest.approx(0.0)
        assert pinball_loss(y, y, tau=0.95) == pytest.approx(0.0)

    def test_tau_bounds_validated(self):
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0]), np.array([1.0]), tau=0.0)
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0]), np.array([1.0]), tau=1.0)

    def test_mean_pinball_averages(self):
        y = np.array([1.0, 2.0, 3.0])
        preds = {0.1: y + 0.1, 0.9: y - 0.1}
        expected = 0.5 * (pinball_loss(y, y + 0.1, tau=0.1) + pinball_loss(y, y - 0.1, tau=0.9))
        assert mean_pinball_loss(y, preds) == pytest.approx(expected)


class TestCoverageError:
    def test_sign_and_magnitude(self):
        y = np.array([0.0, 5.0, 10.0])
        lo = np.array([-1.0, 6.0, 9.0])
        hi = np.array([1.0, 7.0, 11.0])  # 2/3 covered
        # picp 2/3, nominal 0.90 -> negative (under-coverage)
        assert coverage_error(y, lo, hi, level=0.90) == pytest.approx(2 / 3 - 0.90)

    def test_zero_when_calibrated(self):
        y = np.arange(10.0)
        lo = y - 1
        hi = y + 1  # full coverage
        assert coverage_error(y, lo, hi, level=1.0) == pytest.approx(0.0)


class TestCRPS:
    """The CRPS closed form is verified against its definition, not against itself.

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'| for X, X' iid from F. Checking the
    log-normal expression against a Monte-Carlo evaluation of that definition is
    the only test that would actually catch a transcription error in the formula.
    """

    @staticmethod
    def _crps_mc(y: float, mu: float, sigma: float, n: int = 400_000, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        x = np.exp(rng.normal(mu, sigma, n))
        xp = np.exp(rng.normal(mu, sigma, n))
        return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x - xp)))

    @pytest.mark.parametrize(
        "mu,sigma,y",
        [(-8.0, 0.6, 3e-4),        # the dissertation's own scale (log-RV)
         (0.0, 1.0, 1.0),
         (0.5, 0.3, 2.0)],
    )
    def test_lognormal_closed_form_matches_definition(self, mu, sigma, y):
        closed = crps_lognormal(np.array([y]), np.array([mu]), np.array([sigma]))
        mc = self._crps_mc(y, mu, sigma)
        assert closed == pytest.approx(mc, rel=0.01)

    def test_lognormal_is_minimised_at_the_truth(self):
        """A proper score must prefer the data-generating law to any other."""
        rng = np.random.default_rng(7)
        mu, sigma = -8.0, 0.6
        y = np.exp(rng.normal(mu, sigma, 20_000))
        best = crps_lognormal(y, np.full_like(y, mu), np.full_like(y, sigma))
        for bad_mu, bad_sigma in [(mu + 0.5, sigma), (mu - 0.5, sigma),
                                  (mu, sigma * 2), (mu, sigma / 2)]:
            worse = crps_lognormal(y, np.full_like(y, bad_mu), np.full_like(y, bad_sigma))
            assert worse > best

    def test_lognormal_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be strictly positive"):
            crps_lognormal(np.array([1.0]), np.array([0.0]), np.array([0.0]))

    def test_quantile_crps_converges_to_the_closed_form(self):
        from scipy import stats

        rng = np.random.default_rng(11)
        mu, sigma = -8.0, 0.6
        y = np.exp(rng.normal(mu, sigma, 5_000))
        exact = crps_lognormal(y, np.full_like(y, mu), np.full_like(y, sigma))
        prev_err = np.inf
        for k in (11, 51, 201):
            taus = np.linspace(0.0, 1.0, k + 2)[1:-1]
            qp = {float(t): np.full_like(y, float(np.exp(mu + sigma * stats.norm.ppf(t))))
                  for t in taus}
            err = abs(crps_from_quantiles(y, qp) - exact)
            assert err < prev_err          # monotone improvement with grid density
            prev_err = err
        assert prev_err / exact < 0.01     # within 1% by k=201

    def test_quantile_crps_is_not_mean_pinball(self):
        """Guard against the two being conflated again (they differ by ~2x)."""
        y = np.array([1.0, 2.0, 3.0])
        qp = {0.25: y - 0.5, 0.5: y, 0.75: y + 0.5}
        assert crps_from_quantiles(y, qp) != pytest.approx(mean_pinball_loss(y, qp))

    def test_quantile_crps_needs_two_levels(self):
        with pytest.raises(ValueError, match="at least two quantile levels"):
            crps_from_quantiles(np.array([1.0]), {0.5: np.array([1.0])})


class TestGuards:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            mse(np.array([np.nan]), np.array([np.nan]))


class TestIntervalNaNPolicy:
    """The interval metrics must share the point losses' NaN contract.

    Before 2026-08-30 they did not. ``picp`` compared ``NaN >= lower``, which is
    ``False``, so a missing realisation or a missing endpoint was silently scored
    as an interval *miss* — coverage understated by a plausible-looking amount,
    with no warning anywhere. It never fired on the complete ``.SPX`` panel, but
    RQ3 is entirely a coverage question and Phase 8 adds assets on other trading
    calendars, which is exactly where a gap appears.
    """

    def test_picp_excludes_a_missing_truth_rather_than_scoring_it_a_miss(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        lo = np.array([0.5, 1.5, 3.5, 3.5])
        hi = np.array([1.5, 2.5, 4.5, 4.5])
        # All three observable days are covered: 1.0, and the 3 real rows.
        assert picp(y, lo, hi) == pytest.approx(1.0)

    def test_picp_excludes_a_missing_endpoint(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.5, 1.5, np.nan, 3.5])
        hi = np.array([1.5, 2.5, 4.5, 4.5])
        assert picp(y, lo, hi) == pytest.approx(1.0)

    def test_mpiw_and_winkler_ignore_non_finite_rows(self):
        lo = np.array([0.0, 1.0, np.nan])
        hi = np.array([2.0, 3.0, 5.0])
        y = np.array([1.0, 2.0, 4.0])
        assert mpiw(lo, hi) == pytest.approx(2.0)
        assert np.isfinite(winkler_score(y, lo, hi, alpha=0.1))

    def test_coverage_error_is_safe_on_gappy_input(self):
        y = np.array([1.0, np.nan, 3.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([2.0, 2.0, 2.0])
        # One of the two observable days is covered -> picp 0.5.
        assert coverage_error(y, lo, hi, level=0.9) == pytest.approx(0.5 - 0.9)

    def test_all_missing_raises_rather_than_returning_zero(self):
        nan3 = np.array([np.nan] * 3)
        with pytest.raises(ValueError, match="no finite observations"):
            picp(nan3, nan3, nan3)

    def test_shape_mismatch_still_raises(self):
        with pytest.raises(ValueError, match="identical shape"):
            picp(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0, 4.0]))
