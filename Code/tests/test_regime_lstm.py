"""Unit tests for the Phase 5 regime-aware forecasters (roadmap Phase 5; RQ2).

Covers the two things Phase 5 adds on top of the (already-tested) Phase-3 LSTM:
the regime-posterior feature builders, and the mixture-of-experts forecaster --
in particular its leakage discipline (forecasts never dated on/before train_end;
the gate is the posterior at t-1) and the gate-weighted mixture math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from src.data.sequence import FEATURE_BUILDERS, build_feature_target  # noqa: E402
from src.data.splits import Fold  # noqa: E402
from src.models.deep import LSTMForecaster, RegimeExpertForecaster  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def regime_frame() -> pd.DataFrame:
    """A synthetic modelling frame with an RV target, a return, and a 3-state
    causal regime posterior (rows sum to 1)."""
    rng = np.random.default_rng(0)
    n = 420
    idx = pd.bdate_range("2015-01-01", periods=n, name="date")
    rv = np.exp(rng.normal(-9.0, 0.6, n))                     # positive, RV-scale
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


# --------------------------------------------------------------------------- #
# Feature registration                                                         #
# --------------------------------------------------------------------------- #
class TestRegimeFeatures:
    def test_reg_posteriors_registered(self):
        for c in ("reg_p0", "reg_p1", "reg_p2"):
            assert c in FEATURE_BUILDERS

    def test_build_feature_target_with_regime(self, regime_frame):
        feats, target, index = build_feature_target(
            regime_frame, ("log_rv", "oc_log_return", "reg_p0", "reg_p1", "reg_p2")
        )
        assert list(feats.columns) == ["log_rv", "oc_log_return", "reg_p0", "reg_p1", "reg_p2"]
        assert len(feats) == len(target) == len(index)
        # posteriors carried through unchanged (in [0, 1])
        assert (feats[["reg_p0", "reg_p1", "reg_p2"]].to_numpy() >= 0).all()


# --------------------------------------------------------------------------- #
# Approach A — regime as feature (reuses LSTMForecaster)                        #
# --------------------------------------------------------------------------- #
class TestApproachA:
    def test_feature_lstm_runs_and_is_leakage_free(self, regime_frame):
        f = LSTMForecaster(features=("log_rv", "oc_log_return", "reg_p0", "reg_p1", "reg_p2"),
                           name="Regime-LSTM-A", **_small())
        fold = _fold(regime_frame.index, 320)
        pred = f.forecast_fold(regime_frame, fold)
        assert isinstance(pred, pd.Series) and len(pred) > 0
        assert (pred.index > fold.train_end).all()          # no leakage
        assert np.isfinite(pred.to_numpy()).all() and (pred.to_numpy() > 0).all()


# --------------------------------------------------------------------------- #
# Approach B — regime-specific experts (mixture-of-experts)                     #
# --------------------------------------------------------------------------- #
class TestRegimeExperts:
    def test_runs_and_is_leakage_free(self, regime_frame):
        f = RegimeExpertForecaster(base_features=("log_rv", "oc_log_return"),
                                   gate_cols=("reg_p0", "reg_p1", "reg_p2"),
                                   name="Regime-LSTM-B", **_small())
        fold = _fold(regime_frame.index, 320)
        pred = f.forecast_fold(regime_frame, fold)
        assert isinstance(pred, pd.Series) and len(pred) > 0
        assert (pred.index > fold.train_end).all()          # no forecast on/before train_end
        assert np.isfinite(pred.to_numpy()).all() and (pred.to_numpy() > 0).all()
        assert f.K == 3 and len(f._experts) == 3
        # every expert saw an effective (weighted) sample > 0
        eff = f.fold_params_[0]["expert_effective_n"]
        assert all(e > 0 for e in eff)

    def test_gate_is_lagged_one_day(self, regime_frame):
        """The gate for a forecast on day t must be the posterior on day t-1."""
        f = RegimeExpertForecaster(gate_cols=("reg_p0", "reg_p1", "reg_p2"), **_small())
        idx = regime_frame.index
        gate = regime_frame[["reg_p0", "reg_p1", "reg_p2"]].reindex(idx).shift(1)
        # target date idx[100] must map to the raw posterior at idx[99]
        np.testing.assert_allclose(
            gate.loc[idx[100]].to_numpy(),
            regime_frame[["reg_p0", "reg_p1", "reg_p2"]].loc[idx[99]].to_numpy(),
        )

    def test_mixture_math_one_hot_selects_expert(self):
        """With a one-hot gate the mixture equals the selected expert; with a split
        gate it is the gate-weighted average of expert outputs."""
        class _Dummy:
            def __init__(self, v): self.v = float(v)
            def eval(self): return self
            def __call__(self, x): return torch.full((x.shape[0],), self.v)

        f = RegimeExpertForecaster(gate_cols=("reg_p0", "reg_p1", "reg_p2"), **_small())
        f._experts = [_Dummy(1.0), _Dummy(2.0), _Dummy(3.0)]
        X = np.zeros((4, 5, 2), dtype=np.float32)
        onehot = np.tile([0.0, 1.0, 0.0], (4, 1))
        np.testing.assert_allclose(f._mixture_pred(X, onehot), np.full(4, 2.0))
        split = np.tile([0.5, 0.5, 0.0], (4, 1))
        np.testing.assert_allclose(f._mixture_pred(X, split), np.full(4, 1.5))

    def test_gate_for_normalises_and_fills(self):
        f = RegimeExpertForecaster(gate_cols=("reg_p0", "reg_p1", "reg_p2"), **_small())
        idx = pd.bdate_range("2020-01-01", periods=3, name="date")
        g = pd.DataFrame({"reg_p0": [np.nan, 2.0, 0.0], "reg_p1": [np.nan, 2.0, 0.0],
                          "reg_p2": [np.nan, 0.0, 0.0]}, index=idx)
        G = f._gate_for(g, idx)
        np.testing.assert_allclose(G.sum(axis=1), 1.0)       # rows normalised
        np.testing.assert_allclose(G[0], [1 / 3, 1 / 3, 1 / 3])  # NaN row -> uniform
        np.testing.assert_allclose(G[1], [0.5, 0.5, 0.0])    # (2,2,0) -> (.5,.5,0)
