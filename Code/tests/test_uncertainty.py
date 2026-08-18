"""Unit tests for the Phase 6 uncertainty-aware forecasters (roadmap Phase 6; RQ3).

Behavioural tests for the two things Phase 6 adds on top of the (already-tested)
Phase-3 LSTM: MC-Dropout predictive sampling and the monotone quantile head. Not
a re-test of PyTorch; epoch budgets are tiny and a synthetic AR(1)-in-log-variance
series makes the target learnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.splits import Fold  # noqa: E402
from src.evaluation.metrics import pinball_loss, qlike  # noqa: E402
from src.models.deep import (  # noqa: E402
    MCDropoutLSTMForecaster,
    QuantileLSTMForecaster,
    QuantileLSTMRegressor,
)


@pytest.fixture
def rv_frame() -> pd.DataFrame:
    """AR(1) in log-variance -> persistent, learnable realized variance."""
    rng = np.random.default_rng(0)
    n = 700
    logrv = np.empty(n)
    logrv[0] = -9.0
    for t in range(1, n):
        logrv[t] = 0.94 * logrv[t - 1] + 0.06 * (-9.0) + rng.normal(0.0, 0.22)
    rv = np.exp(logrv)
    ret = rng.normal(0.0, np.sqrt(rv))
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"log_return": ret, "oc_log_return": ret, "rv": rv}, index=idx)


def _fold(frame, n_predict=60) -> Fold:
    return Fold(
        fold_idx=0, refit_date=frame.index[-n_predict],
        train_start=frame.index[0], train_end=frame.index[-n_predict - 1],
        predict_start=frame.index[-n_predict], predict_end=frame.index[-1],
    )


def _mc(**kw) -> MCDropoutLSTMForecaster:
    base = dict(features=("log_rv", "oc_log_return"), lookback=10, hidden_size=16,
                num_layers=1, dropout=0.2, max_epochs=8, patience=4, batch_size=64,
                seed=17, device="cpu", mc_samples=30)
    base.update(kw)
    return MCDropoutLSTMForecaster(**base)


def _q(**kw) -> QuantileLSTMForecaster:
    base = dict(features=("log_rv", "oc_log_return"), quantiles=(0.05, 0.5, 0.95),
                lookback=10, hidden_size=16, num_layers=1, dropout=0.1, max_epochs=10,
                patience=5, batch_size=64, seed=17, device="cpu")
    base.update(kw)
    return QuantileLSTMForecaster(**base)


# --------------------------------------------------------------------------- #
# MC-Dropout                                                                   #
# --------------------------------------------------------------------------- #
class TestMCDropout:
    def test_zero_dropout_rejected(self):
        with pytest.raises(ValueError):
            _mc(dropout=0.0)

    def test_uq_frame_shape_and_leakage(self, rv_frame):
        m = _mc()
        fold = _fold(rv_frame, 60)
        u = m.forecast_fold_uq(rv_frame, fold)
        assert list(u.columns) == ["mean", "mu_log", "sigma_log", "sd_epistemic",
                                   "sd_aleatoric", "mu_log_det"]
        assert len(u) == 60
        assert (u.index > fold.train_end).all()          # no leakage
        assert (u["mean"] > 0).all()
        assert np.isfinite(u.to_numpy()).all()

    def test_variance_decomposition(self, rv_frame):
        """Total predictive variance = epistemic (MC spread) + aleatoric (residual)."""
        m = _mc()
        u = m.forecast_fold_uq(rv_frame, _fold(rv_frame, 60))
        recon = np.sqrt(u["sd_epistemic"] ** 2 + u["sd_aleatoric"] ** 2)
        np.testing.assert_allclose(u["sigma_log"].to_numpy(), recon.to_numpy(), rtol=1e-6)
        assert (u["sd_epistemic"] > 0).any()             # dropout>0 => some epistemic spread
        assert (u["sd_aleatoric"] > 0).all()             # smearing on => aleatoric floor

    def test_forecast_fold_returns_predictive_mean(self, rv_frame):
        m = _mc()
        fold = _fold(rv_frame, 50)
        pt = m.forecast_fold(rv_frame, fold)
        u = m.forecast_fold_uq(rv_frame, fold)
        assert pt.name == m.name
        np.testing.assert_allclose(pt.to_numpy(), u["mean"].to_numpy(), rtol=1e-6)

    def test_seeded_mc_is_deterministic(self, rv_frame):
        fold = _fold(rv_frame, 50)
        a = _mc().forecast_fold_uq(rv_frame, fold)
        b = _mc().forecast_fold_uq(rv_frame, fold)
        np.testing.assert_allclose(a["mean"].to_numpy(), b["mean"].to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(a["sigma_log"].to_numpy(), b["sigma_log"].to_numpy(), rtol=1e-6)

    def test_more_dropout_widens_epistemic(self, rv_frame):
        """A higher dropout rate should not collapse the epistemic term to zero."""
        fold = _fold(rv_frame, 50)
        lo = _mc(dropout=0.1).forecast_fold_uq(rv_frame, fold)["sd_epistemic"].mean()
        hi = _mc(dropout=0.4).forecast_fold_uq(rv_frame, fold)["sd_epistemic"].mean()
        assert hi > lo > 0


# --------------------------------------------------------------------------- #
# Quantile regression                                                          #
# --------------------------------------------------------------------------- #
class TestQuantileRegressor:
    def test_head_is_monotone_by_construction(self):
        """Even at random init the quantile outputs must not cross."""
        torch.manual_seed(0)
        net = QuantileLSTMRegressor(input_size=2, hidden_size=8, n_quantiles=3)
        out = net(torch.randn(20, 10, 2)).detach().numpy()
        assert out.shape == (20, 3)
        assert (np.diff(out, axis=1) >= -1e-6).all()     # q0 <= q1 <= q2 for every row


class TestQuantileForecaster:
    def test_quantiles_ordered_and_leakage_free(self, rv_frame):
        m = _q()
        fold = _fold(rv_frame, 60)
        q = m.forecast_fold_uq(rv_frame, fold)
        assert list(q.columns) == ["q0.05", "q0.5", "q0.95"]
        assert (q.index > fold.train_end).all()
        assert (q["q0.05"] <= q["q0.5"] + 1e-12).all()
        assert (q["q0.5"] <= q["q0.95"] + 1e-12).all()
        assert (q["q0.05"] > 0).all()

    def test_forecast_fold_returns_median(self, rv_frame):
        m = _q()
        fold = _fold(rv_frame, 50)
        pt = m.forecast_fold(rv_frame, fold)
        q = m.forecast_fold_uq(rv_frame, fold)
        np.testing.assert_allclose(pt.to_numpy(), q["q0.5"].to_numpy(), rtol=1e-6)

    def test_pinball_improves_over_training(self, rv_frame):
        """The trained median must beat a constant-mean median on pinball@0.5."""
        m = _q(max_epochs=30, patience=10, hidden_size=24)
        fold = _fold(rv_frame, 80)
        q = m.forecast_fold_uq(rv_frame, fold)
        y = rv_frame["rv"].loc[q.index]
        train_med = rv_frame["rv"].loc[: fold.train_end].median()
        const = pd.Series(train_med, index=q.index)
        assert pinball_loss(y, q["q0.5"], tau=0.5) < pinball_loss(y, const, tau=0.5)

    def test_rejects_unordered_quantiles(self):
        with pytest.raises(ValueError):
            _q(quantiles=(0.5, 0.05, 0.95))
        with pytest.raises(ValueError):
            _q(quantiles=(0.0, 0.5, 1.0))

    def test_runs_through_walk_forward_engine(self, rv_frame):
        """The quantile median is a VolForecaster: it runs through the shared engine."""
        from src.data.splits import SplitConfig
        from src.evaluation.rolling import run_walk_forward
        cfg = SplitConfig(
            train_end=rv_frame.index[450], val_start=rv_frame.index[451],
            val_end=rv_frame.index[550], test_start=rv_frame.index[551],
            test_end=rv_frame.index[-1], refit_frequency_days=60,
        )
        preds = run_walk_forward([_q()], rv_frame, cfg, eval_segment="test", strict=True)
        assert "Quantile-LSTM" in preds.columns
        assert (preds["Quantile-LSTM"].dropna() > 0).all()
