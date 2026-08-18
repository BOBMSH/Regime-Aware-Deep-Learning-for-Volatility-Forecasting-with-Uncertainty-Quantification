"""Tests for the formal DM test, the Giacomini-White conditional predictive
ability test, and the Model Confidence Set (Ch2 §2.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import qlike as qlike_mean
from src.evaluation.significance import (
    diebold_mariano,
    dm_lag_sensitivity,
    dm_pairs_table,
    giacomini_white,
    ljung_box,
    loss_differential,
    model_confidence_set,
    qlike_loss,
    regime_test_function,
    sample_acf,
)


def _synthetic(n=400, seed=0):
    """A positive RV-like target and three forecasts of decreasing quality."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    y = np.exp(rng.normal(-9.0, 0.5, size=n))            # ~ realized variance scale
    good = y * np.exp(rng.normal(0.0, 0.05, size=n))     # near-perfect
    mid = y * np.exp(rng.normal(0.0, 0.25, size=n))      # noisier
    bad = np.full(n, float(np.median(y) * 6.0))          # badly biased constant
    to_s = lambda a: pd.Series(a, index=idx)
    return to_s(y), to_s(good), to_s(mid), to_s(bad)


# --------------------------------------------------------------------------- #
# Per-observation loss                                                         #
# --------------------------------------------------------------------------- #
def test_qlike_loss_mean_matches_metrics_qlike():
    y, good, _, _ = _synthetic()
    assert qlike_loss(y, good).mean() == pytest.approx(qlike_mean(y, good), rel=1e-9)


def test_qlike_loss_zero_at_perfect_forecast():
    y, *_ = _synthetic()
    assert qlike_loss(y, y) == pytest.approx(np.zeros(len(y)), abs=1e-12)


def _regime_synthetic(n=600, seed=3, calm_penalty=0.6, crisis_penalty=0.9):
    """Two forecasts whose ranking *flips across regimes* but nets out overall.

    ``a`` is worse in the calm state and much better in the crisis state, tuned so
    the pooled DM test is **not** significant while the regime-conditional
    difference is large. That is precisely the Phase-5 situation — an edge
    concentrated in ~70 crisis days that a pooled average washes out — and the
    reason Ch2 §2.7 names Giacomini–White alongside DM.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    # Persistent 3-state regime path (calm / transitional / crisis).
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.97 else rng.integers(0, 3)
    y = np.exp(rng.normal(-9.0 + states, 0.4))
    err_a = rng.normal(0.0, 0.25, n)
    err_b = rng.normal(0.0, 0.25, n)
    err_b[states == 2] += crisis_penalty   # b badly biased in crisis
    err_a[states == 0] += calm_penalty     # a badly biased in calm -> nets out
    a = y * np.exp(err_a)
    b = y * np.exp(err_b)
    to_s = lambda v: pd.Series(v, index=idx)
    return to_s(y), to_s(a), to_s(b), pd.Series(states, index=idx)


# --------------------------------------------------------------------------- #
# Diebold–Mariano (formal)                                                     #
# --------------------------------------------------------------------------- #
def test_dm_sign_favours_better_model():
    y, good, _, bad = _synthetic()
    r = diebold_mariano(y, good, bad)          # a=good has lower loss -> negative
    assert r["dm_stat"] < 0
    assert r["p_value"] < 0.05
    # backward-compatible keys the run scripts rely on
    assert {"dm_stat", "p_value", "mean_loss_diff", "n", "lag"} <= set(r)


def test_dm_antisymmetric():
    y, good, mid, _ = _synthetic()
    ab = diebold_mariano(y, good, mid)
    ba = diebold_mariano(y, mid, good)
    assert ab["dm_stat"] == pytest.approx(-ba["dm_stat"], rel=1e-9)
    assert ab["p_value"] == pytest.approx(ba["p_value"], rel=1e-9)


def test_dm_identical_forecasts_not_significant():
    y, good, _, _ = _synthetic()
    r = diebold_mariano(y, good, good)
    assert r["dm_stat"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_hln_correction_shrinks_statistic():
    y, good, mid, _ = _synthetic()
    corrected = diebold_mariano(y, good, mid, hln=True)
    raw = diebold_mariano(y, good, mid, hln=False)
    assert abs(corrected["dm_stat"]) <= abs(raw["dm_stat"])
    assert corrected["dm_stat_uncorrected"] == pytest.approx(raw["dm_stat"], rel=1e-9)
    # a longer horizon applies a larger finite-sample shrinkage
    h5 = diebold_mariano(y, good, mid, horizon=5)
    assert abs(h5["dm_stat"]) < abs(raw["dm_stat"])


def test_dm_pairs_table_shape_and_direction():
    y, good, mid, bad = _synthetic()
    preds = pd.DataFrame({"y_true": y, "good": good, "mid": mid, "bad": bad})
    tbl = dm_pairs_table(preds["y_true"], preds, [("good", "bad"), ("mid", "bad")])
    assert list(tbl["better"]) == ["good", "mid"]
    assert (tbl["p_value"] <= 1.0).all() and (tbl["p_value"] >= 0.0).all()


# --------------------------------------------------------------------------- #
# DM regularity diagnostics (Ch2 §2.7)                                         #
# --------------------------------------------------------------------------- #
def _ar1(n=800, rho=0.7, seed=11):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_sample_acf_recovers_ar1_structure():
    """ACF of an AR(1) decays as rho^k and is flagged significant at lag 1."""
    x = _ar1(rho=0.7)
    acf = sample_acf(x, nlags=5)
    assert acf["acf"].iloc[0] == pytest.approx(0.7, abs=0.06)
    assert acf["acf"].iloc[1] == pytest.approx(0.49, abs=0.08)
    assert acf["significant_bartlett"].iloc[0]
    # Bartlett bands widen with lag once earlier lags are correlated.
    assert acf["se_bartlett"].is_monotonic_increasing


def test_sample_acf_white_noise_mostly_insignificant():
    rng = np.random.default_rng(3)
    acf = sample_acf(rng.normal(size=2000), nlags=20)
    assert acf["acf"].abs().max() < 0.1
    assert acf["significant_bartlett"].sum() <= 2      # ~5% false positives of 20


def test_ljung_box_rejects_ar1_decisively():
    lb = ljung_box(_ar1(rho=0.7), lags=(5, 10, 20))
    assert (lb["p_value"] < 1e-6).all()
    assert list(lb["df"]) == [5, 10, 20]


def test_ljung_box_is_calibrated_under_white_noise():
    """Checks the *size* of the test rather than one lucky draw.

    A single white-noise series is a weak test: under the null the p-value is
    uniform, so ~10-15% of seeds will show a sub-5% p-value across three lags
    (seed 5 does). Rejecting in a handful of draws is correct behaviour; rejecting
    in most of them would mean the statistic or its degrees of freedom are wrong.
    """
    min_ps = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        min_ps.append(float(ljung_box(rng.normal(size=1500))["p_value"].min()))
    rejects = sum(p < 0.05 for p in min_ps)
    assert rejects <= 3, f"rejected {rejects}/10 white-noise draws — test is over-sized"
    assert float(np.median(min_ps)) > 0.2


def test_loss_differential_matches_dm_mean():
    y, good, mid, _ = _synthetic()
    d = loss_differential(y, good, mid)
    r = diebold_mariano(y, good, mid)
    assert d.mean() == pytest.approx(r["mean_loss_diff"], rel=1e-12)
    assert d.size == r["n"]


def test_dm_hac_lag_override_reproduces_default():
    """Passing the automatic lag explicitly must not change the statistic."""
    y, good, mid, _ = _synthetic()
    auto = diebold_mariano(y, good, mid)
    forced = diebold_mariano(y, good, mid, hac_lag=auto["lag"])
    assert forced["dm_stat"] == pytest.approx(auto["dm_stat"], rel=1e-12)
    assert forced["lag"] == auto["lag"]


def test_dm_hac_lag_zero_gives_iid_variance():
    """At lag 0 the long-run variance collapses to the sample variance."""
    y, good, mid, _ = _synthetic()
    r = diebold_mariano(y, good, mid, hac_lag=0)
    assert r["lrv_over_gamma0"] == pytest.approx(1.0, rel=1e-12)


def test_dm_lag_sensitivity_grid_and_default_flag():
    y, good, mid, _ = _synthetic()
    tbl = dm_lag_sensitivity(y, good, mid)
    assert tbl["is_default"].sum() == 1
    default_row = tbl[tbl["is_default"]].iloc[0]
    auto = diebold_mariano(y, good, mid)
    assert default_row["p_value"] == pytest.approx(auto["p_value"], rel=1e-12)
    assert (tbl["hac_lag"].diff().dropna() > 0).all()      # strictly increasing grid
    assert tbl["lrv_over_gamma0"].iloc[0] == pytest.approx(1.0, rel=1e-12)


def test_lag_sensitivity_flags_a_persistent_differential():
    """With a strongly autocorrelated differential the HAC lag must matter.

    Constructed so the two forecasts differ by a persistent AR(1) wedge: the
    long-run variance then grows with the lag, so the DM statistic shrinks. This
    is precisely the failure mode Ch2 §2.7 says must be inspected.
    """
    n = 800
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(2)
    y = pd.Series(np.exp(rng.normal(-9.0, 0.4, n)), index=idx)
    wedge = _ar1(n=n, rho=0.95, seed=7) * 0.05
    a = pd.Series(y.to_numpy() * np.exp(wedge + 0.02), index=idx)
    b = pd.Series(y.to_numpy() * np.exp(-wedge), index=idx)
    tbl = dm_lag_sensitivity(y, a, b)
    assert tbl["lrv_over_gamma0"].max() > 2.0              # HAC inflates the variance
    assert tbl["lrv_over_gamma0"].is_monotonic_increasing
    assert abs(tbl["dm_stat"].iloc[-1]) < abs(tbl["dm_stat"].iloc[0])


# --------------------------------------------------------------------------- #
# Giacomini–White (conditional predictive ability)                             #
# --------------------------------------------------------------------------- #
def test_gw_constant_test_function_equals_squared_dm():
    """The q=1 GW statistic IS the squared uncorrected DM statistic.

    Guards the whole construction: same loss differential, same HAC kernel, same
    lag, same normalisation (``hac_divisor='n-k'`` matches what DM uses
    internally) -- so any drift between the two implementations breaks this.
    """
    y, good, mid, _ = _synthetic()
    dm = diebold_mariano(y, good, mid, hln=False)
    gw = giacomini_white(
        y, good, mid,
        np.ones(len(y)),
        hac_lag=dm["lag"], hac_divisor="n-k",
    )
    assert gw["df"] == 1
    assert gw["gw_stat"] == pytest.approx(dm["dm_stat"] ** 2, rel=1e-10)


def test_gw_default_divisor_close_to_dm():
    """The PSD-safe default divisor differs from DM's only by O(lag/n)."""
    y, good, mid, _ = _synthetic()
    dm = diebold_mariano(y, good, mid, hln=False)
    gw = giacomini_white(y, good, mid, np.ones(len(y)), hac_lag=dm["lag"])
    assert gw["gw_stat"] == pytest.approx(dm["dm_stat"] ** 2, rel=0.02)


def test_gw_identical_forecasts_not_significant():
    y, good, _, _ = _synthetic()
    gw = giacomini_white(y, good, good, np.ones(len(y)))
    assert gw["gw_stat"] == pytest.approx(0.0, abs=1e-12)
    assert gw["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_gw_is_direction_free_but_moments_are_signed():
    """The chi-square statistic is symmetric in (a, b); the moments carry the sign."""
    y, good, mid, _ = _synthetic()
    h = np.ones(len(y))
    ab = giacomini_white(y, good, mid, h)
    ba = giacomini_white(y, mid, good, h)
    assert ab["gw_stat"] == pytest.approx(ba["gw_stat"], rel=1e-9)
    assert ab["moments"][0] == pytest.approx(-ba["moments"][0], rel=1e-9)
    assert ab["moments"][0] < 0            # 'good' (model a) has the lower loss


def test_gw_detects_regime_dependent_edge_that_pooled_dm_misses():
    """The motivating case: an edge concentrated in one state.

    ``a`` and ``b`` are built so the pooled loss difference is statistically
    indistinguishable from zero while the *conditional* difference is enormous.
    A DM test therefore reports "no difference" and GW reports "the difference is
    regime-dependent" -- which is exactly the RQ2 claim the Phase-5 per-regime
    table makes on 70 crisis days without testing it.
    """
    y, a, b, states = _regime_synthetic()
    h = regime_test_function(states, n_states=3, labels=["calm", "trans", "crisis"])
    cond = giacomini_white(y, a, b, h)
    uncond = giacomini_white(y, a, b, np.ones(len(y)))
    pooled = diebold_mariano(y, a, b)

    assert pooled["p_value"] > 0.05        # the pooled test sees nothing
    assert cond["df"] == 3
    assert cond["p_value"] < 0.01          # the conditional test sees it clearly
    assert cond["gw_stat"] > uncond["gw_stat"]
    # Signs localise the effect: 'a' wins in crisis (most negative moment) and
    # loses in calm (most positive) -- the flip a pooled mean hides.
    assert np.argmin(cond["moments"]) == 2
    assert np.argmax(cond["moments"]) == 0


def test_gw_rejects_collinear_test_functions():
    """Constant + a complete one-hot basis is rank-deficient -> explicit error."""
    y, a, b, states = _regime_synthetic()
    h = regime_test_function(states, n_states=3)
    h.insert(0, "const", 1.0)
    with pytest.raises(ValueError, match="collinear"):
        giacomini_white(y, a, b, h)


def test_gw_default_test_function_is_constant_plus_lagged_differential():
    y, good, mid, _ = _synthetic()
    gw = giacomini_white(y, good, mid)
    assert gw["df"] == 2
    assert gw["n"] == len(y) - 1          # one observation consumed by the lag


def test_regime_test_function_lags_by_one_day():
    """h must be measurable at t-1; the first row is therefore unusable."""
    states = pd.Series([0, 1, 1, 2], index=pd.date_range("2020-01-01", periods=4))
    h = regime_test_function(states, n_states=3)
    assert list(h.columns) == ["state0", "state1", "state2"]
    assert h.iloc[0].isna().all()                       # nothing known before t=0
    assert h.iloc[1].tolist() == [1.0, 0.0, 0.0]        # day 2 conditions on day 1
    assert h.iloc[3].tolist() == [0.0, 1.0, 0.0]
    # Each usable row is a valid one-hot.
    assert h.dropna().sum(axis=1).eq(1.0).all()


def test_regime_test_function_rows_drop_when_state_missing():
    y, a, b, states = _regime_synthetic()
    states = states.astype(float)
    states.iloc[:5] = np.nan
    h = regime_test_function(states, n_states=3)
    gw = giacomini_white(y, a, b, h)
    assert gw["n"] == len(y) - 6           # 5 NaN states + 1 lost to the lag


def test_gw_rejects_test_function_column_that_never_fires():
    """A regime with no days in the window is a degenerate moment, named as such."""
    y, a, b, states = _regime_synthetic()
    h = regime_test_function(states, n_states=3)
    h["never"] = 0.0
    with pytest.raises(ValueError, match="identically zero"):
        giacomini_white(y, a, b, h)


# --------------------------------------------------------------------------- #
# Model Confidence Set                                                         #
# --------------------------------------------------------------------------- #
def test_mcs_keeps_best_drops_dominated():
    y, good, mid, bad = _synthetic()
    out = model_confidence_set(
        y, {"good": good, "mid": mid, "bad": bad}, alpha=0.10, reps=500, seed=17
    )
    # tidy frame contract
    assert list(out.columns) == ["avg_loss", "mcs_pvalue", "in_mcs", "rank"]
    assert (out["mcs_pvalue"] >= 0).all() and (out["mcs_pvalue"] <= 1).all()
    assert set(out.attrs["included"]) | set(out.attrs["excluded"]) == {"good", "mid", "bad"}
    assert not (set(out.attrs["included"]) & set(out.attrs["excluded"]))
    # the clearly-best model survives; the badly-biased constant is eliminated
    assert "good" in out.attrs["included"]
    assert "bad" in out.attrs["excluded"]
    # lowest average loss is ranked first and is in the set
    assert out.iloc[0]["rank"] == 1
    assert bool(out.iloc[0]["in_mcs"]) is True


def test_mcs_best_model_pvalue_is_one():
    y, good, mid, bad = _synthetic()
    out = model_confidence_set(y, {"good": good, "mid": mid, "bad": bad}, reps=500, seed=17)
    # the never-eliminated (lowest-loss) model has MCS p-value 1 by construction
    assert out.iloc[0]["mcs_pvalue"] == pytest.approx(1.0, abs=1e-9)


def test_mcs_requires_two_models():
    y, good, _, _ = _synthetic()
    with pytest.raises(ValueError):
        model_confidence_set(y, {"good": good})
