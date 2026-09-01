"""Tests for the Phase-7 master table's CRPS column (``run_combined``).

Why this file exists
--------------------
The Phase-7 master table published **one model's CRPS on three models' rows**.
``_interval_bits`` read ``mc_mu_log`` / ``mc_sigma_log`` off whatever frame it
was handed, regardless of which method the interval belonged to, and
``MC-Dropout-LSTM``, ``LSTM-Gaussian`` and ``Quantile-LSTM`` are all passed the
same Phase-6 frame -- so all three reported 4.3058928982516726e-05 instead of
4.3058929e-05 / 4.3014955e-05 / (no closed form). Phase 6's own ``crps_table``
was correct throughout, so the two committed artefacts disagreed with each
other, and the wrong one is the table Chapter 4's headline is built from.

This is the sixth instance of the project's recurring failure family: a value
assembled from one input while a second input bears on it, published without
anything cross-checking the two. The fix is executable, not merely intended --
:func:`~src.experiments.run_combined.assert_crps_distinct` runs inside
``master_table`` on every build, and the round-trip below pins the behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import lognormal_interval
from src.evaluation.metrics import crps_lognormal
from src.evaluation.rolling import ACTUAL_COL
from src.experiments.run_combined import (
    COMBINED_NAME,
    GAUSS_NAME,
    LOGNORMAL_CRPS,
    MC_NAME,
    Q_NAME,
    REGIME_STATE_COL,
    assert_crps_distinct,
    lognormal_law,
    master_table,
)

LEVEL = 0.90
TAG = "90"
N = 80


@pytest.fixture()
def frames():
    """A minimal Phase-6 + Phase-7 pair whose three methods have *different* laws.

    The laws are deliberately distinct -- MC-Dropout's spread includes an
    epistemic term the Gaussian ablation does not, and the combined model is a
    different fit again -- so any row that borrows another row's law shows up as
    an equal CRPS.
    """
    # one business day of run-up so the t-1 regime label is reachable for every
    # evaluated date -- the label-source rule, and it keeps the fixture from
    # emitting a RegimeLabelReachWarning that would mask a real one later
    full_idx = pd.date_range("2019-01-01", periods=N + 1, freq="B", name="date")
    idx = full_idx[1:]
    rng = np.random.default_rng(17)

    mu_log = -9.2 + 0.35 * rng.standard_normal(N)          # log realised variance
    y = np.exp(mu_log + 0.4 * rng.standard_normal(N))      # the realisation
    sd_ale = np.full(N, 0.52)                              # aleatoric only
    sd_mc = np.sqrt(sd_ale ** 2 + 0.21 ** 2)               # + epistemic
    mu_det = mu_log - 0.03                                 # dropout-off mean differs

    reg_state = pd.Series(rng.integers(0, 3, N + 1), index=full_idx, dtype="Int64")

    uqp = pd.DataFrame(index=idx)
    uqp[ACTUAL_COL] = y
    uqp[REGIME_STATE_COL] = reg_state.reindex(idx)
    uqp[MC_NAME] = np.exp(mu_log + 0.5 * sd_mc ** 2)
    uqp[GAUSS_NAME] = np.exp(mu_det + 0.5 * sd_ale ** 2)
    uqp[Q_NAME] = np.exp(mu_log)                           # the pinball median
    uqp["mc_mu_log"] = mu_log
    uqp["mc_sigma_log"] = sd_mc
    uqp["mc_sd_aleatoric"] = sd_ale
    lo, hi = lognormal_interval(mu_log, sd_mc, level=LEVEL)
    uqp[f"{MC_NAME}_lo{TAG}"], uqp[f"{MC_NAME}_hi{TAG}"] = lo, hi
    glo, ghi = lognormal_interval(mu_det, sd_ale, level=LEVEL)
    uqp[f"{GAUSS_NAME}_lo{TAG}"], uqp[f"{GAUSS_NAME}_hi{TAG}"] = glo, ghi
    uqp[f"{Q_NAME}_lo{TAG}"] = np.exp(mu_log - 1.5 * sd_ale)   # no log-normal law
    uqp[f"{Q_NAME}_hi{TAG}"] = np.exp(mu_log + 1.5 * sd_ale)

    cmu, csd = mu_log + 0.05, sd_mc * 1.04                 # a different fit again
    preds = pd.DataFrame(index=idx)
    preds[ACTUAL_COL] = y
    preds[REGIME_STATE_COL] = reg_state.reindex(idx)
    preds[COMBINED_NAME] = np.exp(cmu + 0.5 * csd ** 2)
    preds["mc_mu_log"] = cmu
    preds["mc_sigma_log"] = csd
    preds["mc_sd_aleatoric"] = sd_ale
    preds["mc_mu_log_det"] = cmu - 0.02
    clo, chi = lognormal_interval(cmu, csd, level=LEVEL)
    preds[f"{COMBINED_NAME}_lo{TAG}"], preds[f"{COMBINED_NAME}_hi{TAG}"] = clo, chi

    base = pd.DataFrame({ACTUAL_COL: y, "HAR-RV": np.exp(mu_log) * 1.02}, index=idx)

    return preds, base, uqp, reg_state.astype(float)


# --------------------------------------------------------------------------- #
# lognormal_law                                                                #
# --------------------------------------------------------------------------- #
def test_law_is_the_methods_own_moments(frames):
    preds, _, uqp, _ = frames
    mu, sd = lognormal_law(uqp, MC_NAME)
    assert np.array_equal(mu.to_numpy(), uqp["mc_mu_log"].to_numpy())
    assert np.array_equal(sd.to_numpy(), uqp["mc_sigma_log"].to_numpy())

    cmu, csd = lognormal_law(preds, COMBINED_NAME)
    assert np.array_equal(cmu.to_numpy(), preds["mc_mu_log"].to_numpy())
    assert np.array_equal(csd.to_numpy(), preds["mc_sigma_log"].to_numpy())


def test_gaussian_law_is_aleatoric_only_and_differs_from_mc(frames):
    _, _, uqp, _ = frames
    gmu, gsd = lognormal_law(uqp, GAUSS_NAME)
    # spread is the aleatoric term, not MC's total
    assert np.array_equal(gsd.to_numpy(), uqp["mc_sd_aleatoric"].to_numpy())
    assert not np.allclose(gsd.to_numpy(), uqp["mc_sigma_log"].to_numpy())
    # Phase 6 stores no mu_log_det, so the log-space mean is recovered from the
    # point; it must invert exp(mu + sigma^2/2) exactly
    expected = np.log(uqp[GAUSS_NAME].to_numpy()) - 0.5 * uqp["mc_sd_aleatoric"].to_numpy() ** 2
    assert np.array_equal(gmu.to_numpy(), expected)


def test_gaussian_law_prefers_a_persisted_mu_log_det(frames):
    """Phase 7 persists the log-space mean; use it rather than re-deriving it."""
    preds, _, _, _ = frames
    frame = preds.rename(columns={COMBINED_NAME: GAUSS_NAME})
    mu, _ = lognormal_law(frame, GAUSS_NAME)
    assert np.array_equal(mu.to_numpy(), preds["mc_mu_log_det"].to_numpy())


def test_quantile_head_has_no_closed_form_law(frames):
    """A pinball head asserts no distribution, so it gets no closed-form CRPS."""
    _, _, uqp, _ = frames
    assert lognormal_law(uqp, Q_NAME) is None


def test_law_is_none_when_the_moments_are_absent(frames):
    _, _, uqp, _ = frames
    assert lognormal_law(uqp.drop(columns=["mc_mu_log"]), MC_NAME) is None
    assert lognormal_law(uqp.drop(columns=["mc_sd_aleatoric"]), GAUSS_NAME) is None


# --------------------------------------------------------------------------- #
# the regression itself                                                        #
# --------------------------------------------------------------------------- #
def test_master_table_crps_is_per_method(frames):
    """The defect: three rows sharing one law's CRPS."""
    preds, base, uqp, reg_state = frames
    tbl = master_table(preds, base, uqp, reg_state, ["calm", "transitional", "crisis"],
                       level=LEVEL).set_index("model")

    mc, gauss, comb = tbl.at[MC_NAME, "crps"], tbl.at[GAUSS_NAME, "crps"], tbl.at[COMBINED_NAME, "crps"]
    assert np.isfinite([mc, gauss, comb]).all()
    assert mc != gauss and mc != comb and gauss != comb

    # and each equals the score of that method's own law
    y = preds[ACTUAL_COL].to_numpy(float)
    for name, frame in ((MC_NAME, uqp), (GAUSS_NAME, uqp), (COMBINED_NAME, preds)):
        mu, sd = lognormal_law(frame, name)
        assert tbl.at[name, "crps"] == crps_lognormal(y, mu.to_numpy(float), sd.to_numpy(float))


def test_master_table_leaves_the_quantile_crps_blank(frames):
    """Better an empty cell than a number from someone else's estimator."""
    preds, base, uqp, reg_state = frames
    tbl = master_table(preds, base, uqp, reg_state, ["calm", "transitional", "crisis"],
                       level=LEVEL).set_index("model")
    assert pd.isna(tbl.at[Q_NAME, "crps"])
    assert tbl.at[Q_NAME, "crps_estimator"] == ""
    # the interval metrics are still reported -- only the CRPS is withheld
    assert np.isfinite(tbl.at[Q_NAME, "picp"])
    assert np.isfinite(tbl.at[Q_NAME, "winkler"])


def test_master_table_labels_the_estimator(frames):
    preds, base, uqp, reg_state = frames
    tbl = master_table(preds, base, uqp, reg_state, ["calm", "transitional", "crisis"],
                       level=LEVEL).set_index("model")
    for name in (MC_NAME, GAUSS_NAME, COMBINED_NAME):
        assert tbl.at[name, "crps_estimator"] == LOGNORMAL_CRPS
    assert tbl.at["HAR-RV", "crps_estimator"] == ""     # no interval, no estimator


# --------------------------------------------------------------------------- #
# the guard                                                                    #
# --------------------------------------------------------------------------- #
def test_guard_rejects_a_shared_crps():
    """The exact shape of the committed defect."""
    shared = 4.3058928982516726e-05
    tbl = pd.DataFrame({"model": [MC_NAME, GAUSS_NAME, Q_NAME],
                        "crps": [shared, shared, shared]})
    with pytest.raises(ValueError, match="identical CRPS"):
        assert_crps_distinct(tbl)


def test_guard_allows_distinct_and_missing_values():
    tbl = pd.DataFrame({"model": [MC_NAME, GAUSS_NAME, Q_NAME, "HAR-RV"],
                        "crps": [4.3058929e-05, 4.3014955e-05, np.nan, np.nan]})
    assert_crps_distinct(tbl)          # several NaNs are not a collision
    assert_crps_distinct(pd.DataFrame({"model": ["a"], "qlike": [1.0]}))  # no column at all
