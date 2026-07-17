"""Unit tests for intraday realized-variance reconstruction and the splice gate.

The reconstruction (``realized_variance_intraday``) is the estimator that would
extend Oxford-Man's ``rv5`` past 2022-02-25. It is validated on synthetic
intraday data with a *known* integrated variance, and the overnight-exclusion and
splice-gating behaviour is checked explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.intraday import OverlapReport, splice, validate_overlap
from src.data.realized_vol import realized_variance_intraday

BARS_PER_DAY = 78          # 6.5h session at 5-minute bars
STEP_SIGMA = 1.132e-3      # per-bar sd -> E[daily RV] = 78 * sigma^2 ~ 1e-4


def _synthetic_intraday(n_days=120, overnight_sd=0.0, seed=0) -> tuple[pd.Series, float]:
    """5-minute price bars with known within-day return variance.

    Returns the price Series and the theoretical daily realized variance
    ``BARS_PER_DAY * STEP_SIGMA**2``. ``overnight_sd`` injects a close-to-open gap
    that a correct open-to-close RV must *exclude*.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=n_days)
    level = 0.0
    stamps: list[pd.Timestamp] = []
    prices: list[float] = []
    for d in days:
        level += rng.normal(0, overnight_sd)  # overnight jump (across-day; excluded)
        times = pd.date_range(d + pd.Timedelta(hours=9, minutes=30),
                              periods=BARS_PER_DAY + 1, freq="5min")
        incr = rng.normal(0, STEP_SIGMA, size=BARS_PER_DAY + 1)
        incr[0] = 0.0  # first point of the day is the open
        logp = level + np.cumsum(incr)
        level = logp[-1]  # carry the close forward
        stamps.extend(times)
        prices.extend(np.exp(logp))
    s = pd.Series(prices, index=pd.DatetimeIndex(stamps), name="price")
    return s, BARS_PER_DAY * STEP_SIGMA**2


class TestReconstruction:
    def test_recovers_integrated_variance(self):
        prices, theory = _synthetic_intraday(n_days=200, overnight_sd=0.0, seed=1)
        rv = realized_variance_intraday(prices, resample_freq=None)
        # Mean daily RV should be within ~15% of the theoretical value.
        assert rv.mean() == pytest.approx(theory, rel=0.15)
        assert (rv > 0).all()
        assert rv.index.equals(rv.index.normalize())  # indexed by trading date

    def test_overnight_gap_excluded(self):
        # Huge overnight jumps must NOT inflate an open-to-close RV.
        prices, theory = _synthetic_intraday(n_days=200, overnight_sd=0.05, seed=2)
        rv = realized_variance_intraday(prices, resample_freq=None)
        assert rv.mean() == pytest.approx(theory, rel=0.15)

    def test_resample_from_one_minute(self):
        # Feed 1-minute bars; resampling to 5min should still recover the level
        # (coarser sampling, larger but fewer returns).
        rng = np.random.default_rng(3)
        days = pd.bdate_range("2021-03-01", periods=60)
        per_min = STEP_SIGMA / np.sqrt(5.0)  # 5 one-min steps make one 5-min return
        stamps, prices, level = [], [], 0.0
        for d in days:
            times = pd.date_range(d + pd.Timedelta(hours=9, minutes=30),
                                  periods=BARS_PER_DAY * 5 + 1, freq="1min")
            incr = rng.normal(0, per_min, size=len(times))
            incr[0] = 0.0
            logp = level + np.cumsum(incr)
            level = logp[-1]
            stamps.extend(times)
            prices.extend(np.exp(logp))
        s = pd.Series(prices, index=pd.DatetimeIndex(stamps))
        rv5 = realized_variance_intraday(s, resample_freq="5min")
        theory = BARS_PER_DAY * STEP_SIGMA**2
        assert rv5.mean() == pytest.approx(theory, rel=0.25)

    def test_subsample_grids_runs(self):
        prices, theory = _synthetic_intraday(n_days=80, seed=4)
        rv = realized_variance_intraday(prices, resample_freq="5min", subsample_grids=3)
        assert rv.mean() == pytest.approx(theory, rel=0.3)

    def test_volatility_option_is_sqrt(self):
        prices, _ = _synthetic_intraday(n_days=40, seed=5)
        var = realized_variance_intraday(prices, resample_freq=None, squared=True)
        vol = realized_variance_intraday(prices, resample_freq=None, squared=False)
        np.testing.assert_allclose(np.sqrt(var.values), vol.values, rtol=1e-10)

    def test_rejects_bad_subsample(self):
        prices, _ = _synthetic_intraday(n_days=10)
        with pytest.raises(ValueError):
            realized_variance_intraday(prices, subsample_grids=0)


class TestSpliceGate:
    def _series(self, n=300, k=1.1, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2021-06-01", periods=n)
        om = pd.Series(np.exp(rng.normal(-9, 0.4, size=n)), index=idx, name="rv5")
        our = (om * k).rename("rv5")  # perfectly correlated, level ratio k
        return om, our, idx

    def test_overlap_report_metrics(self):
        om, our, _ = self._series(k=1.1)
        rep = validate_overlap(our, om)
        assert rep.n == len(om)
        assert rep.pearson == pytest.approx(1.0, abs=1e-6)
        assert rep.median_ratio == pytest.approx(1.1, rel=1e-6)
        assert rep.passes()  # within pearson>=0.9 and |ratio-1|<=0.15

    def test_gate_blocks_level_mismatch(self):
        om, our, _ = self._series(k=2.0)  # 2x level -> should fail the gate
        rep = validate_overlap(our, om)
        assert not rep.passes()
        with pytest.raises(RuntimeError):
            splice(om.iloc[:200], our, cutoff=om.index[150], report=rep)

    def test_splice_level_adjusts_and_joins(self):
        om, our, idx = self._series(k=1.1)
        cutoff = idx[150]
        om_head = om.loc[:cutoff]
        rep = validate_overlap(our.loc[:cutoff], om_head)
        spliced = splice(om_head, our, cutoff=cutoff, report=rep, level_adjust=True)
        # Head is untouched Oxford-Man; tail is our data rescaled to OM's level.
        pd.testing.assert_series_equal(spliced.loc[:cutoff], om_head, check_names=False)
        post = spliced.loc[spliced.index > cutoff]
        expected = (our.loc[our.index > cutoff] / rep.median_ratio)
        np.testing.assert_allclose(post.values, expected.values, rtol=1e-9)

    def test_force_bypasses_gate(self):
        om, our, idx = self._series(k=2.0)
        rep = validate_overlap(our, om)
        # force=True splices despite the failed gate (documented-override path).
        spliced = splice(om.iloc[:200], our, cutoff=idx[150], report=rep, force=True)
        assert len(spliced) > 150

    def test_passes_requires_minimum_overlap(self):
        rep = OverlapReport(n=10, pearson=0.99, spearman=0.99, median_ratio=1.0,
                            mean_abs_log_ratio=0.0, start=None, end=None)
        assert not rep.passes()  # n < 60
