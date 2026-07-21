"""Tests for the formal DM test and the Model Confidence Set (Ch2 §2.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import qlike as qlike_mean
from src.evaluation.significance import (
    diebold_mariano,
    dm_pairs_table,
    model_confidence_set,
    qlike_loss,
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
