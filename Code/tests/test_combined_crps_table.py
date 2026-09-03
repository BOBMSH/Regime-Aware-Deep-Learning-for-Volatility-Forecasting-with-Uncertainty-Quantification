"""Tests for the Phase-7 CRPS table (``run_combined.crps_table``).

Why this file exists
--------------------
Section 3.7 commits to scoring every method a second time on the quantile head's
own grid, comparing *those* figures across methods, and quoting the closed form
only within the parametric family. Phase 6 delivered that for its three methods
in ``m06_*_crps.csv``. Phase 7's combined model -- ``MC-Dropout-Regime-LSTM-B``,
the model Chapter 4's RQ2 x RQ3 result rests on -- had only a closed-form value
in the master table, so it was the single method that could not legally be set
beside ``Quantile-LSTM`` on a proper score.

The comparison is not academic. On this three-point grid the trapezoidal
estimate understates the closed form by roughly 17%, so putting one method's
grid value next to another's closed form invents a difference that is entirely
an artefact of the estimator. The regression that matters most here is
:func:`test_shared_methods_match_run_uq`: the three methods Phase 6 also scores
must come out **bit-identical**, because two published numbers for the same
quantity differing in the 16th digit is its own defect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.rolling import ACTUAL_COL
from src.experiments.run_combined import (
    COMBINED_NAME,
    GAUSS_NAME,
    LOGNORMAL_CRPS,
    MC_NAME,
    Q_NAME,
    assert_crps_distinct,
    crps_table,
    quantile_grid,
)
from src.experiments.run_uq import crps_table as uq_crps_table

TAG = "90"
QUANTILES = [0.05, 0.5, 0.95]
N = 96


def _frames(seed: int = 7):
    """A Phase-6 frame and a Phase-7 frame with independent predictive laws."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=N, freq="B")

    mu = pd.Series(np.log(rng.uniform(2e-5, 4e-4, N)), index=idx)
    sd = pd.Series(rng.uniform(0.45, 0.75, N), index=idx)
    sd_ale = pd.Series(rng.uniform(0.40, 0.60, N), index=idx)
    y = pd.Series(np.exp(mu.to_numpy() + rng.normal(0, 0.5, N)), index=idx)

    uqp = pd.DataFrame({
        ACTUAL_COL: y,
        "mc_mu_log": mu,
        "mc_sigma_log": sd,
        "mc_sd_epistemic": np.sqrt(np.clip(sd**2 - sd_ale**2, 1e-12, None)),
        "mc_sd_aleatoric": sd_ale,
        MC_NAME: np.exp(mu + 0.5 * sd**2),
        # Phase 6 persists LSTM-Gaussian on the variance scale only, so its
        # log-space mean is recovered -- the path lognormal_law must match.
        GAUSS_NAME: np.exp(mu + 0.5 * sd_ale**2),
        Q_NAME: np.exp(mu),
        f"{Q_NAME}_lo{TAG}": np.exp(mu - 1.64 * sd_ale),
        f"{Q_NAME}_hi{TAG}": np.exp(mu + 1.64 * sd_ale),
    }, index=idx)

    # Phase 7's own law is genuinely different, not a copy of Phase 6's.
    mu_c = mu + 0.05
    sd_c = sd * 1.10
    preds = pd.DataFrame({
        ACTUAL_COL: y,
        COMBINED_NAME: np.exp(mu_c + 0.5 * sd_c**2),
        "mc_mu_log": mu_c,
        "mc_sigma_log": sd_c,
        "mc_sd_aleatoric": sd_ale,
        "mc_mu_log_det": mu_c,
    }, index=idx)
    return preds, uqp


# --------------------------------------------------------------- quantile_grid
def test_quantile_grid_recovers_the_taus_from_the_persisted_band():
    _, uqp = _frames()
    grid = quantile_grid(uqp)
    assert sorted(grid) == [0.05, 0.5, 0.95]
    pd.testing.assert_series_equal(grid[0.05], uqp[f"{Q_NAME}_lo{TAG}"],
                                   check_names=False)
    pd.testing.assert_series_equal(grid[0.95], uqp[f"{Q_NAME}_hi{TAG}"],
                                   check_names=False)


@pytest.mark.parametrize("tag, lo, hi", [
    ("90", 0.05, 0.95),
    ("95", 0.025, 0.975),
    ("98", 0.01, 0.99),
    ("80", 0.10, 0.90),
    ("50", 0.25, 0.75),
])
def test_quantile_grid_keys_are_the_exact_trained_literals(tag, lo, hi):
    """Exact equality, because callers index the grid by the literal.

    ``(1 - 90 / 100) / 2`` is ``0.04999999999999999``: a value
    ``pytest.approx`` accepts and ``grid[0.05]`` does not. A tolerant
    assertion here would pass while every caller naming a tau raised
    ``KeyError``, so the assertion is exact and the lookup is exercised.
    """
    _, uqp = _frames()
    uqp = uqp.rename(columns={f"{Q_NAME}_lo{TAG}": f"{Q_NAME}_lo{tag}",
                              f"{Q_NAME}_hi{TAG}": f"{Q_NAME}_hi{tag}"})
    grid = quantile_grid(uqp)
    assert set(grid) == {lo, 0.5, hi}
    pd.testing.assert_series_equal(grid[lo], uqp[f"{Q_NAME}_lo{tag}"],
                                   check_names=False)
    pd.testing.assert_series_equal(grid[hi], uqp[f"{Q_NAME}_hi{tag}"],
                                   check_names=False)


def test_quantile_grid_drops_the_median_when_the_head_had_none():
    _, uqp = _frames()
    grid = quantile_grid(uqp.drop(columns=[Q_NAME]))
    assert sorted(grid) == [0.05, 0.95]


def test_quantile_grid_refuses_an_ambiguous_band():
    _, uqp = _frames()
    uqp = uqp.copy()
    uqp[f"{Q_NAME}_lo95"] = uqp[f"{Q_NAME}_lo{TAG}"]
    uqp[f"{Q_NAME}_hi95"] = uqp[f"{Q_NAME}_hi{TAG}"]
    with pytest.raises(ValueError, match="exactly one persisted"):
        quantile_grid(uqp)


# ----------------------------------------------------------------- crps_table
def test_every_method_appears_once_per_estimator_it_supports():
    preds, uqp = _frames()
    t = crps_table(preds, uqp)
    assert list(t.columns) == ["method", "crps", "estimator", "n"]
    closed = set(t.loc[t["estimator"] == LOGNORMAL_CRPS, "method"])
    assert closed == {COMBINED_NAME, MC_NAME, GAUSS_NAME}
    grid = set(t.loc[t["estimator"] != LOGNORMAL_CRPS, "method"])
    assert grid == {Q_NAME, f"{COMBINED_NAME} (same grid)",
                    f"{MC_NAME} (same grid)", f"{GAUSS_NAME} (same grid)"}
    assert t["n"].eq(N).all()
    assert not t["crps"].isna().any()


def test_the_quantile_head_gets_no_closed_form():
    """A pinball head asserts no predictive law, so it has no closed form at all."""
    preds, uqp = _frames()
    t = crps_table(preds, uqp)
    assert LOGNORMAL_CRPS not in set(t.loc[t["method"] == Q_NAME, "estimator"])
    assert f"{Q_NAME} (same grid)" not in set(t["method"])


def test_the_combined_model_is_scored_both_ways_and_they_differ():
    """The point of the table: the headline model can now be read against the
    quantile head on the grid, and its closed form is still reported."""
    preds, uqp = _frames()
    t = crps_table(preds, uqp).set_index("method")["crps"]
    closed = t[COMBINED_NAME]
    grid = t[f"{COMBINED_NAME} (same grid)"]
    assert np.isfinite(closed) and np.isfinite(grid)
    # a 3-point trapezoidal rule is a lower bound on the true CRPS
    assert grid < closed


def test_shared_methods_match_run_uq(): 
    """The three methods Phase 6 also scores must come out bit-identical.

    Two committed artefacts publishing different values for the same quantity is
    the defect this project has already had once; agreement to float64 is the
    only acceptable standard, not agreement to five significant figures.
    """
    preds, uqp = _frames()
    mine = crps_table(preds, uqp).set_index("method")["crps"]
    theirs = uq_crps_table(uqp, QUANTILES).set_index("method")["crps"]
    for m in (MC_NAME, GAUSS_NAME, Q_NAME,
              f"{MC_NAME} (same grid)", f"{GAUSS_NAME} (same grid)"):
        assert mine[m] == theirs[m], m


def test_guard_rejects_two_methods_sharing_a_crps():
    dup = pd.DataFrame({"method": ["a", "b"], "crps": [1.0, 1.0]})
    with pytest.raises(ValueError, match="identical CRPS"):
        assert_crps_distinct(dup, model_col="method")
    assert_crps_distinct(pd.DataFrame({"method": ["a", "b"], "crps": [1.0, 2.0]}),
                         model_col="method")


def test_guard_still_defends_the_master_table_shape():
    """The widened signature must not break its original caller."""
    dup = pd.DataFrame({"model": ["a", "b"], "crps": [1.0, 1.0]})
    with pytest.raises(ValueError, match="identical CRPS"):
        assert_crps_distinct(dup)
