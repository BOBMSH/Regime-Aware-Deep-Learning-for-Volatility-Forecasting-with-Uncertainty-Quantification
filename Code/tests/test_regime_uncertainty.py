"""Unit tests for the Phase-7 combined regime x MC-Dropout forecaster.

The combined model's whole evidential value rests on one claim: it is the
Phase-5 mixture-of-experts with uncertainty added, *not* a different point
model. If training drifted, then any calibration difference between
``MC-Dropout-LSTM`` and ``MC-Dropout-Regime-LSTM-B`` would be attributable to
the retrained network rather than to regime conditioning, and the Phase-7
comparison would measure nothing. The first test class pins that down; the rest
cover the predictive law, the gate's role in it, reproducibility and leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from src.data.splits import Fold  # noqa: E402
from src.models.deep import (  # noqa: E402
    MCDropoutRegimeExpertForecaster,
    RegimeExpertForecaster,
)


# --------------------------------------------------------------------------- #
# Fixtures (mirroring tests/test_regime_lstm.py so the two are comparable)     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def regime_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 420
    idx = pd.bdate_range("2015-01-01", periods=n, name="date")
    rv = np.exp(rng.normal(-9.0, 0.6, n))
    oc = rng.normal(0, 0.01, n)
    logits = rng.normal(0, 1.0, size=(n, 3))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {"rv": rv, "log_return": oc, "oc_log_return": oc,
         "reg_p0": p[:, 0], "reg_p1": p[:, 1], "reg_p2": p[:, 2],
         "reg_state": p.argmax(axis=1)},
        index=idx,
    )


def _fold(idx: pd.DatetimeIndex, train_end_pos: int) -> Fold:
    return Fold(fold_idx=0, refit_date=idx[train_end_pos + 1], train_start=idx[0],
                train_end=idx[train_end_pos], predict_start=idx[train_end_pos + 1],
                predict_end=idx[-1])


def _small(**kw):
    base = dict(lookback=5, hidden_size=8, max_epochs=3, patience=5, batch_size=32, seed=0)
    base.update(kw)
    return base


def _combined(**kw):
    return MCDropoutRegimeExpertForecaster(
        base_features=("log_rv", "oc_log_return"),
        gate_cols=("reg_p0", "reg_p1", "reg_p2"),
        mc_samples=kw.pop("mc_samples", 12),
        **_small(**kw),
    )


# --------------------------------------------------------------------------- #
# The combined model must BE the Phase-5 model, plus uncertainty               #
# --------------------------------------------------------------------------- #
class TestSamePointModelAsPhase5:
    def test_deterministic_mixture_matches_the_phase5_forecaster(self, regime_frame):
        """Same seed, same hyperparameters -> same trained experts, so the
        dropout-off mixture must reproduce the Phase-5 log-space point exactly.

        This is the test that makes the Phase-6-vs-Phase-7 contrast
        interpretable: it certifies that the only thing Phase 7 changed is
        inference."""
        fold = _fold(regime_frame.index, 320)
        base = RegimeExpertForecaster(base_features=("log_rv", "oc_log_return"),
                                      gate_cols=("reg_p0", "reg_p1", "reg_p2"),
                                      **_small())
        comb = _combined()
        base_pred = base.forecast_fold(regime_frame, fold)
        frame_uq = comb.forecast_fold_uq(regime_frame, fold)

        # Phase-5 point = exp(log_rv + smear/2); the combined model's dropout-off
        # column is that same log_rv, so rebuild and compare on the variance scale.
        rebuilt = np.exp(frame_uq["mu_log_det"].to_numpy() + 0.5 * comb._smear_var)
        np.testing.assert_allclose(rebuilt, base_pred.to_numpy(), rtol=1e-10)

    def test_smearing_variance_is_inherited_not_recomputed(self, regime_frame):
        fold = _fold(regime_frame.index, 320)
        base = RegimeExpertForecaster(base_features=("log_rv", "oc_log_return"),
                                      gate_cols=("reg_p0", "reg_p1", "reg_p2"),
                                      **_small())
        comb = _combined()
        base.forecast_fold(regime_frame, fold)
        comb.forecast_fold_uq(regime_frame, fold)
        assert comb._smear_var == pytest.approx(base._smear_var, rel=1e-12)


# --------------------------------------------------------------------------- #
# The predictive law                                                           #
# --------------------------------------------------------------------------- #
class TestPredictiveLaw:
    def test_columns_match_the_phase6_contract(self, regime_frame):
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        for c in ("mean", "mu_log", "sigma_log", "sd_epistemic", "sd_aleatoric",
                  "mu_log_det", "gate_entropy"):
            assert c in out.columns, c
        assert np.isfinite(out.to_numpy()).all()

    def test_total_variance_is_epistemic_plus_aleatoric(self, regime_frame):
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        np.testing.assert_allclose(
            out["sigma_log"].to_numpy() ** 2,
            out["sd_epistemic"].to_numpy() ** 2 + out["sd_aleatoric"].to_numpy() ** 2,
            rtol=1e-10,
        )

    def test_point_forecast_is_the_lognormal_mean(self, regime_frame):
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        np.testing.assert_allclose(
            out["mean"].to_numpy(),
            np.exp(out["mu_log"].to_numpy() + 0.5 * out["sigma_log"].to_numpy() ** 2),
            rtol=1e-10,
        )

    def test_epistemic_term_is_strictly_positive(self, regime_frame):
        """Dropout is on, so the T mixtures must actually differ. A zero here
        would mean the masks never fired and the 'uncertainty' is a constant."""
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        assert (out["sd_epistemic"].to_numpy() > 0).all()

    def test_aleatoric_term_is_one_constant(self, regime_frame):
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        assert out["sd_aleatoric"].nunique() == 1

    def test_forecast_fold_returns_the_predictive_mean(self, regime_frame):
        f = _combined()
        fold = _fold(regime_frame.index, 320)
        uq = f.forecast_fold_uq(regime_frame, fold)
        pt = f.forecast_fold(regime_frame, fold)
        np.testing.assert_allclose(pt.to_numpy(), uq["mean"].to_numpy(), rtol=1e-12)
        assert pt.name == f.name


# --------------------------------------------------------------------------- #
# The gate                                                                     #
# --------------------------------------------------------------------------- #
class TestGate:
    def test_draws_are_mixed_inside_each_pass(self):
        """The epistemic variance must be the variance of the *mixture*, not the
        mixture of per-expert variances. With deterministic experts of differing
        constant output and a split gate, a per-pass mixture is the gate-weighted
        mean; anything else would show up as a different value here."""
        class _Dummy:
            def __init__(self, v): self.v = float(v)
            def eval(self): return self
            def modules(self): return iter(())
            def __call__(self, x): return torch.full((x.shape[0],), self.v)

        f = _combined(mc_samples=5)
        f._experts = [_Dummy(1.0), _Dummy(2.0), _Dummy(3.0)]
        X = np.zeros((4, 5, 2), dtype=np.float32)
        G = np.tile([0.25, 0.25, 0.5], (4, 1))
        draws = f._mixture_draws(X, G)
        assert draws.shape == (5, 4)
        np.testing.assert_allclose(draws, 0.25 * 1 + 0.25 * 2 + 0.5 * 3)

    def test_gate_entropy_is_bounded_by_log_k(self, regime_frame):
        out = _combined().forecast_fold_uq(regime_frame, _fold(regime_frame.index, 320))
        ent = out["gate_entropy"].to_numpy()
        assert (ent >= -1e-12).all()
        assert (ent <= np.log(3) + 1e-9).all()

    def test_gate_is_lagged_so_day_t_posterior_cannot_matter(self, regime_frame):
        """Perturbing the regime posterior *on* a forecast date must not change
        that date's forecast -- the gate for day t is the posterior at t-1."""
        fold = _fold(regime_frame.index, 320)
        f = _combined()
        base = f.forecast_fold_uq(regime_frame, fold)

        target = base.index[5]
        tampered = regime_frame.copy()
        tampered.loc[target, ["reg_p0", "reg_p1", "reg_p2"]] = [1.0, 0.0, 0.0]
        after = f.forecast_fold_uq(tampered, fold)      # already trained; no refit
        assert after.loc[target, "mean"] == pytest.approx(base.loc[target, "mean"], rel=1e-12)
        # ... while the NEXT day, whose gate is that posterior, does move
        nxt = base.index[6]
        assert after.loc[nxt, "mean"] != pytest.approx(base.loc[nxt, "mean"], rel=1e-12)


# --------------------------------------------------------------------------- #
# Reproducibility, leakage and guards                                          #
# --------------------------------------------------------------------------- #
class TestDiscipline:
    def test_intervals_are_reproducible_across_calls(self, regime_frame):
        fold = _fold(regime_frame.index, 320)
        f = _combined()
        a = f.forecast_fold_uq(regime_frame, fold)
        b = f.forecast_fold_uq(regime_frame, fold)     # no refit; same mc_seed
        pd.testing.assert_frame_equal(a, b)

    def test_no_forecast_on_or_before_train_end(self, regime_frame):
        fold = _fold(regime_frame.index, 320)
        out = _combined().forecast_fold_uq(regime_frame, fold)
        assert (out.index > fold.train_end).all()

    def test_zero_dropout_rejected(self):
        with pytest.raises(ValueError, match="dropout > 0"):
            MCDropoutRegimeExpertForecaster(dropout=0.0, **_small())

    def test_single_mc_sample_rejected(self):
        with pytest.raises(ValueError, match="mc_samples must be >= 2"):
            MCDropoutRegimeExpertForecaster(mc_samples=1, **_small())

    def test_config_reports_the_mc_settings(self):
        c = _combined(mc_samples=7).config()
        assert c["mc_samples"] == 7 and "mc_seed" in c
        assert c["name"] == "MC-Dropout-Regime-LSTM-B"
        assert c["K"] == 3
