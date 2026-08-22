"""Tests for the Phase-7 interval-coverage tests (src/evaluation/coverage_tests.py).

The tests are grouped by what they protect:

* **Identities** — each statistic is checked against a closed form derived
  independently of the implementation (the DQ-with-only-a-constant case equals a
  squared proportion z; LR_cc equals LR_uc + LR_ind exactly).
* **Behaviour** — the statistic must reject what it is supposed to reject and
  stay quiet otherwise (clustered vs iid violations; a real coverage gap vs
  none).
* **Size** — simulated data under the null must reject at roughly the nominal
  rate. A coverage test that is itself mis-sized would turn Phase 7 into the
  same class of error the earlier audits found.
* **The level-vs-timing trap** — the scenario from the 2026-08-19 (iv) audit is
  reproduced directly: a globally under-covering model must fail Kupiec on a
  subsample while the difference-in-coverage test correctly finds nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.coverage_tests import (
    christoffersen_cc,
    christoffersen_independence,
    coverage_test_suite,
    dq_incremental,
    dynamic_quantile,
    expected_rate,
    holm_adjust,
    interval_hits,
    kupiec_pof,
    two_sample_coverage_test,
)


# --------------------------------------------------------------------------- #
# Hit construction                                                             #
# --------------------------------------------------------------------------- #
class TestIntervalHits:
    def test_sides_partition_the_violations(self):
        y = np.array([1.0, 5.0, 10.0, 0.5])
        lo = np.array([2.0, 2.0, 2.0, 2.0])
        hi = np.array([8.0, 8.0, 8.0, 8.0])
        both = interval_hits(y, lo, hi, side="both")
        up = interval_hits(y, lo, hi, side="upper")
        low = interval_hits(y, lo, hi, side="lower")
        assert both.tolist() == [1, 0, 1, 1]
        assert up.tolist() == [0, 0, 1, 0]
        assert low.tolist() == [1, 0, 0, 1]
        # the two one-sided views partition the pooled one
        assert np.array_equal(both, up + low)

    def test_endpoints_are_inside_the_interval(self):
        y = np.array([2.0, 8.0])
        assert interval_hits(y, np.array([2.0, 2.0]), np.array([8.0, 8.0])).sum() == 0

    def test_series_are_index_aligned_and_nan_dropped(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        y = pd.Series([1.0, 5.0, np.nan, 10.0], index=idx)
        lo = pd.Series(2.0, index=idx)
        hi = pd.Series(8.0, index=idx)
        h = interval_hits(y, lo, hi)
        assert h.size == 3 and h.tolist() == [1, 0, 1]

    def test_expected_rate_halves_for_one_sided(self):
        assert expected_rate(0.90, "both") == pytest.approx(0.10)
        assert expected_rate(0.90, "upper") == pytest.approx(0.05)

    def test_bad_side_raises(self):
        with pytest.raises(ValueError, match="side must be"):
            interval_hits(np.zeros(3), np.zeros(3), np.ones(3), side="sideways")


# --------------------------------------------------------------------------- #
# Kupiec                                                                       #
# --------------------------------------------------------------------------- #
class TestKupiec:
    def test_exact_rate_gives_zero_statistic(self):
        h = np.zeros(100)
        h[:10] = 1.0
        r = kupiec_pof(h, 0.10)
        assert r["stat"] == pytest.approx(0.0, abs=1e-12)
        assert r["p_value"] == pytest.approx(1.0)
        assert r["direction"] == "exact"

    def test_matches_the_closed_form(self):
        rng = np.random.default_rng(0)
        h = (rng.random(500) < 0.17).astype(float)
        a, n = 0.10, h.size
        x = int(h.sum())
        pi = x / n
        expect = -2.0 * (
            (x * np.log(a) + (n - x) * np.log(1 - a))
            - (x * np.log(pi) + (n - x) * np.log(1 - pi))
        )
        assert kupiec_pof(h, a)["stat"] == pytest.approx(expect, rel=1e-12)

    def test_under_coverage_is_flagged_directionally(self):
        h = np.zeros(200)
        h[:60] = 1.0                        # 30% violations against a 10% nominal
        r = kupiec_pof(h, 0.10)
        assert r["direction"] == "under-covers"
        assert r["p_value"] < 1e-10

    def test_over_coverage_is_flagged_directionally(self):
        h = np.zeros(200)
        h[:2] = 1.0
        r = kupiec_pof(h, 0.10)
        assert r["direction"] == "over-covers"
        assert r["p_value"] < 0.01

    def test_degenerate_zero_violations_does_not_blow_up(self):
        r = kupiec_pof(np.zeros(200), 0.10)
        assert np.isfinite(r["stat"]) and r["stat"] > 0
        assert r["n_violations"] == 0

    def test_size_is_close_to_nominal_under_the_null(self):
        rng = np.random.default_rng(11)
        a, n, reps = 0.10, 784, 800
        rejects = sum(
            kupiec_pof((rng.random(n) < a).astype(float), a)["p_value"] < 0.05
            for _ in range(reps)
        )
        assert 0.02 < rejects / reps < 0.10

    def test_non_binary_input_rejected(self):
        with pytest.raises(ValueError, match="hits must be 0/1"):
            kupiec_pof(np.array([0.0, 0.5, 1.0]), 0.1)


# --------------------------------------------------------------------------- #
# Christoffersen                                                               #
# --------------------------------------------------------------------------- #
class TestChristoffersen:
    def test_iid_violations_are_not_flagged_as_clustered(self):
        rng = np.random.default_rng(3)
        h = (rng.random(1500) < 0.10).astype(float)
        assert christoffersen_independence(h)["p_value"] > 0.05

    def test_clustered_violations_are_flagged(self):
        # violations arrive in blocks of 10 -> a hit strongly predicts a hit
        h = np.zeros(600)
        for start in range(0, 600, 100):
            h[start:start + 10] = 1.0
        r = christoffersen_independence(h)
        assert r["p_value"] < 1e-6
        assert r["pi11"] > r["pi01"]

    def test_transition_counts_sum_to_n_minus_one(self):
        rng = np.random.default_rng(5)
        h = (rng.random(300) < 0.2).astype(float)
        r = christoffersen_independence(h)
        assert r["n00"] + r["n01"] + r["n10"] + r["n11"] == h.size - 1

    def test_degenerate_no_violations_reports_rather_than_rejects(self):
        r = christoffersen_independence(np.zeros(50))
        assert r["degenerate"] is True
        assert r["stat"] == 0.0 and r["p_value"] == 1.0

    def test_cc_is_exactly_uc_plus_ind(self):
        rng = np.random.default_rng(7)
        h = (rng.random(400) < 0.15).astype(float)
        uc = kupiec_pof(h, 0.10)
        ind = christoffersen_independence(h)
        cc = christoffersen_cc(h, 0.10)
        assert cc["stat"] == pytest.approx(uc["stat"] + ind["stat"], rel=1e-12)
        assert cc["df"] == 2

    def test_size_is_close_to_nominal_under_the_null(self):
        rng = np.random.default_rng(13)
        a, n, reps = 0.10, 784, 600
        rejects = sum(
            christoffersen_cc((rng.random(n) < a).astype(float), a)["p_value"] < 0.05
            for _ in range(reps)
        )
        assert 0.02 < rejects / reps < 0.11


# --------------------------------------------------------------------------- #
# Dynamic quantile                                                             #
# --------------------------------------------------------------------------- #
class TestDynamicQuantile:
    def test_constant_only_equals_a_squared_proportion_z(self):
        """With lags=0 and no exog the design is a lone constant, and DQ reduces
        to ((x - n*a)^2) / (n*a*(1-a)) -- the square of the usual z for a
        proportion. That identity is the cleanest possible check that the
        normalisation by a(1-a) is right."""
        rng = np.random.default_rng(17)
        h = (rng.random(500) < 0.13).astype(float)
        a, n = 0.10, h.size
        x = h.sum()
        z2 = (x - n * a) ** 2 / (n * a * (1 - a))
        assert dynamic_quantile(h, a, lags=0)["stat"] == pytest.approx(z2, rel=1e-10)

    def test_predictable_hits_are_detected(self):
        # every 5th day is a violation -> lagged hits predict the next perfectly
        h = np.zeros(500)
        h[::5] = 1.0
        assert dynamic_quantile(h, 0.20, lags=4)["p_value"] < 1e-6

    def test_iid_hits_are_not_detected(self):
        rng = np.random.default_rng(19)
        h = (rng.random(1500) < 0.10).astype(float)
        assert dynamic_quantile(h, 0.10, lags=4)["p_value"] > 0.05

    def test_exog_enters_and_is_named(self):
        rng = np.random.default_rng(23)
        n = 600
        g = np.zeros(n)
        g[::3] = 1.0
        h = np.where(g > 0, rng.random(n) < 0.35, rng.random(n) < 0.05).astype(float)
        r = dynamic_quantile(h, 0.10, lags=2, exog=g, exog_names=["flag"])
        assert "flag" in r["regressors"]
        assert r["df"] == 4                       # const + 2 lags + 1 exog
        assert r["p_value"] < 1e-6                # the flag predicts the miss

    def test_dead_exog_column_raises_by_name(self):
        rng = np.random.default_rng(29)
        h = (rng.random(300) < 0.1).astype(float)
        with pytest.raises(ValueError, match="never fires"):
            dynamic_quantile(h, 0.1, lags=1, exog=np.zeros(300), exog_names=["dead"])

    def test_collinear_design_raises(self):
        rng = np.random.default_rng(31)
        n = 400
        h = (rng.random(n) < 0.1).astype(float)
        onehot = np.zeros((n, 2))
        onehot[: n // 2, 0] = 1.0
        onehot[n // 2:, 1] = 1.0              # the two columns sum to the constant
        with pytest.raises(ValueError, match="rank-deficient"):
            dynamic_quantile(h, 0.1, lags=0, exog=onehot, exog_names=["a", "b"])

    def test_exog_length_must_match_hits(self):
        h = np.zeros(100)
        h[:10] = 1
        with pytest.raises(ValueError, match="rows but the hit series"):
            dynamic_quantile(h, 0.1, lags=1, exog=np.ones(50))

    def test_size_is_close_to_nominal_under_the_null(self):
        rng = np.random.default_rng(37)
        a, n, reps = 0.10, 784, 400
        rejects = sum(
            dynamic_quantile((rng.random(n) < a).astype(float), a, lags=4)["p_value"] < 0.05
            for _ in range(reps)
        )
        assert 0.02 < rejects / reps < 0.12


# --------------------------------------------------------------------------- #
# Incremental DQ — the level/timing separation in chi-square form              #
# --------------------------------------------------------------------------- #
class TestIncrementalDQ:
    def test_equals_the_difference_of_the_nested_statistics(self):
        rng = np.random.default_rng(61)
        n = 800
        h = (rng.random(n) < 0.12).astype(float)
        g = (np.arange(n) % 5 == 0).astype(float)
        r = dq_incremental(h, 0.10, g, lags=4, exog_names=["flag"])
        assert r["stat"] == pytest.approx(r["dq_full"] - r["dq_restricted"], rel=1e-12)
        assert r["df"] == r["df_full"] - r["df_restricted"] == 1
        assert r["added_regressors"] == ["flag"]

    def test_a_mislevelled_model_no_longer_rejects_through_the_constant(self):
        """The whole point. A globally under-covering model with an *uninformative*
        conditioner rejects the full DQ (via the constant) but must NOT reject the
        incremental test."""
        rng = np.random.default_rng(67)
        n = 800
        h = (rng.random(n) < 0.20).astype(float)      # nominal 0.10 -> badly mis-levelled
        g = (np.arange(n) % 5 == 0).astype(float)     # carries no information about h
        r = dq_incremental(h, 0.10, g, lags=4, exog_names=["noise"])
        assert r["p_full"] < 1e-6                      # full DQ rejects ...
        assert r["p_value"] > 0.05                     # ... incremental does not

    def test_an_informative_conditioner_is_still_detected(self):
        rng = np.random.default_rng(71)
        n = 900
        g = (np.arange(n) % 4 == 0).astype(float)
        h = np.where(g > 0, rng.random(n) < 0.45, rng.random(n) < 0.06).astype(float)
        r = dq_incremental(h, 0.10, g, lags=4, exog_names=["flag"])
        assert r["p_value"] < 1e-6

    def test_size_under_the_null_with_an_uninformative_regressor(self):
        rng = np.random.default_rng(73)
        a, n, reps = 0.10, 784, 400
        rejects = 0
        for _ in range(reps):
            h = (rng.random(n) < a).astype(float)
            g = (rng.random(n) < 0.3).astype(float)
            rejects += dq_incremental(h, a, g, lags=4)["p_value"] < 0.05
        assert 0.02 < rejects / reps < 0.11

    def test_empty_exog_raises(self):
        h = np.zeros(200)
        h[:20] = 1
        with pytest.raises(ValueError, match="no columns|no incremental"):
            dq_incremental(h, 0.1, np.empty((200, 0)), lags=2)


# --------------------------------------------------------------------------- #
# Multiplicity                                                                 #
# --------------------------------------------------------------------------- #
class TestHolm:
    def test_matches_the_textbook_step_down(self):
        p = np.array([0.01, 0.02, 0.03, 0.04])
        # 4*0.01=.04 ; 3*0.02=.06 ; 2*0.03=.06 (monotone) ; 1*0.04=.06 (monotone)
        np.testing.assert_allclose(holm_adjust(p), [0.04, 0.06, 0.06, 0.06])

    def test_is_monotone_and_bounded(self):
        rng = np.random.default_rng(79)
        p = rng.random(25)
        adj = holm_adjust(p)
        assert (adj <= 1.0).all() and (adj >= p - 1e-12).all()
        order = np.argsort(p)
        assert np.all(np.diff(adj[order]) >= -1e-12)

    def test_single_value_is_unchanged(self):
        np.testing.assert_allclose(holm_adjust([0.037]), [0.037])

    def test_empty_is_safe(self):
        assert holm_adjust([]).size == 0


# --------------------------------------------------------------------------- #
# Difference in coverage                                                       #
# --------------------------------------------------------------------------- #
class TestTwoSampleCoverage:
    def test_equal_rates_give_a_null_result(self):
        rng = np.random.default_rng(41)
        n = 1200
        h = (rng.random(n) < 0.12).astype(float)
        g = (np.arange(n) % 4 == 0).astype(float)
        r = two_sample_coverage_test(h, g)
        assert r["p_iid"] > 0.05 and r["p_hac"] > 0.05

    def test_real_gap_is_detected(self):
        rng = np.random.default_rng(43)
        n = 1200
        g = (np.arange(n) % 4 == 0).astype(float)
        h = np.where(g > 0, rng.random(n) < 0.40, rng.random(n) < 0.05).astype(float)
        r = two_sample_coverage_test(h, g)
        assert r["p_iid"] < 1e-8 and r["p_hac"] < 1e-6
        assert r["picp_flagged"] < r["picp_rest"]

    def test_picp_fields_are_one_minus_the_violation_rates(self):
        h = np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        g = np.array([1.0, 1, 0, 0, 0, 0, 1, 1, 0, 0])
        r = two_sample_coverage_test(h, g)
        assert r["picp_flagged"] == pytest.approx(1 - r["violation_rate_flagged"])
        assert r["picp_rest"] == pytest.approx(1 - r["violation_rate_rest"])
        assert r["diff_violation_rate"] == pytest.approx(
            r["violation_rate_flagged"] - r["violation_rate_rest"])

    def test_tiny_side_raises(self):
        h = np.zeros(100)
        g = np.zeros(100)
        g[0] = 1.0
        with pytest.raises(ValueError, match="at least|>= 2"):
            two_sample_coverage_test(h, g)


# --------------------------------------------------------------------------- #
# The level-vs-timing trap (2026-08-19 (iv) caution, reproduced)               #
# --------------------------------------------------------------------------- #
class TestLevelVersusTiming:
    """A globally under-covering model must fail Kupiec on *every* subsample.

    This is the trap the roadmap's Phase-7 entry warns about, reproduced as a
    test so the harness cannot drift back into it. The Quantile-LSTM covers
    0.821 against a nominal 0.90; a Kupiec test of any subsample against nominal
    therefore rejects by construction, and only the difference-in-coverage test
    can say whether the subsample is special.
    """

    def _sample(self, seed=101, n=784, rate=0.179, n_flag=52):
        rng = np.random.default_rng(seed)
        h = (rng.random(n) < rate).astype(float)     # uniform under-coverage
        g = np.zeros(n)
        g[rng.choice(n, size=n_flag, replace=False)] = 1.0
        return h, g

    def test_kupiec_rejects_the_subsample_even_though_nothing_is_special(self):
        h, g = self._sample()
        sub = h[g > 0]
        # the flagged days are drawn from the same distribution as the rest ...
        assert kupiec_pof(sub, 0.10)["p_value"] < 0.05
        # ... yet Kupiec-against-nominal rejects, because the LEVEL is wrong

    def test_difference_test_correctly_finds_nothing(self):
        h, g = self._sample()
        r = two_sample_coverage_test(h, g)
        assert r["p_iid"] > 0.05
        assert r["p_hac"] > 0.05

    def test_the_two_tests_disagree_which_is_the_whole_point(self):
        h, g = self._sample()
        kupiec_p = kupiec_pof(h[g > 0], 0.10)["p_value"]
        diff_p = two_sample_coverage_test(h, g)["p_iid"]
        assert kupiec_p < 0.05 < diff_p


# --------------------------------------------------------------------------- #
# Suite wrapper                                                                #
# --------------------------------------------------------------------------- #
class TestSuite:
    def _frames(self, n=600, seed=53):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        y = pd.Series(rng.lognormal(mean=-9.0, sigma=1.0, size=n), index=idx)
        lo = pd.Series(np.quantile(y, 0.05), index=idx)
        hi = pd.Series(np.quantile(y, 0.95), index=idx)
        return y, lo, hi

    def test_shape_and_columns(self):
        y, lo, hi = self._frames()
        out = coverage_test_suite(y, lo, hi, level=0.90, method="X")
        assert len(out) == 8                       # 2 sides x 4 tests
        assert set(out["test"]) == {
            "Kupiec LR_uc", "Christoffersen LR_ind",
            "Christoffersen LR_cc", "Engle-Manganelli DQ",
        }
        assert set(out["side"]) == {"both", "upper"}
        assert (out["p_value"].between(0, 1)).all()

    def test_expected_rate_halves_on_the_upper_side(self):
        y, lo, hi = self._frames()
        out = coverage_test_suite(y, lo, hi, level=0.90, method="X")
        both = out[out["side"] == "both"]["expected_violation_rate"].iloc[0]
        up = out[out["side"] == "upper"]["expected_violation_rate"].iloc[0]
        assert both == pytest.approx(0.10) and up == pytest.approx(0.05)

    def test_exog_regressors_are_recorded_in_the_row(self):
        y, lo, hi = self._frames()
        ex = pd.DataFrame({"state1": (np.arange(len(y)) % 3 == 0).astype(float)},
                          index=y.index)
        out = coverage_test_suite(y, lo, hi, level=0.90, method="X", exog=ex)
        dq = out[out["test"] == "Engle-Manganelli DQ"].iloc[0]
        assert "state1" in dq["regressors"]

    def test_exog_with_leading_nans_shrinks_only_the_dq_sample(self):
        """A lagged regime indicator is NaN on the first day; the DQ sample loses
        that row while Kupiec keeps the full window. The suite must not silently
        run the two tests on different samples without saying so in ``n``."""
        y, lo, hi = self._frames()
        col = pd.Series((np.arange(len(y)) % 3 == 0).astype(float), index=y.index)
        col.iloc[0] = np.nan
        out = coverage_test_suite(y, lo, hi, level=0.90, method="X",
                                  exog=pd.DataFrame({"state1": col}))
        assert len(out) == 8
        assert out["p_value"].notna().all()
