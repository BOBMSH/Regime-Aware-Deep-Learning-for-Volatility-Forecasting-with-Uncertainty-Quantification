"""Unit tests for the Phase 3 LSTM forecaster.

Behavioural tests (shape, alignment, leakage-safety, determinism, ability to
learn a persistent signal) -- not a re-test of PyTorch. Epoch budgets are tiny so
the suite stays fast; a synthetic AR(1)-in-log-variance series makes the target
genuinely learnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.splits import Fold, SplitConfig, walk_forward_folds
from src.evaluation.metrics import qlike
from src.models.deep.lstm import LSTMForecaster, LSTMRegressor


@pytest.fixture
def rv_frame() -> pd.DataFrame:
    """AR(1) in log-variance -> persistent, learnable realized variance."""
    rng = np.random.default_rng(0)
    n = 900
    logrv = np.empty(n)
    logrv[0] = -9.0
    for t in range(1, n):
        logrv[t] = 0.94 * logrv[t - 1] + 0.06 * (-9.0) + rng.normal(0.0, 0.22)
    rv = np.exp(logrv)
    ret = rng.normal(0.0, np.sqrt(rv))
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame({"log_return": ret, "oc_log_return": ret, "rv": rv}, index=idx)


def _fast_model(**kw) -> LSTMForecaster:
    base = dict(features=("log_rv", "oc_log_return"), lookback=10, hidden_size=16,
                num_layers=1, dropout=0.0, max_epochs=8, patience=4, batch_size=64,
                seed=17, device="cpu")
    base.update(kw)
    return LSTMForecaster(**base)


def _one_fold(frame, n_predict=60) -> Fold:
    return Fold(
        fold_idx=0, refit_date=frame.index[-n_predict],
        train_start=frame.index[0], train_end=frame.index[-n_predict - 1],
        predict_start=frame.index[-n_predict], predict_end=frame.index[-1],
    )


class TestRegressorModule:
    def test_forward_shape(self):
        net = LSTMRegressor(input_size=2, hidden_size=8, num_layers=1, dropout=0.0)
        out = net(torch.randn(5, 10, 2))
        assert out.shape == (5,)


class TestForecastFold:
    def test_positive_aligned_and_no_leakage(self, rv_frame):
        m = _fast_model()
        fold = _one_fold(rv_frame, 60)
        pred = m.forecast_fold(rv_frame, fold)
        assert (pred > 0).all()                          # QLIKE needs positivity
        assert (pred.index > fold.train_end).all()       # no leakage onto training dates
        assert pred.index.min() == fold.predict_start
        assert pred.index.max() == fold.predict_end
        assert len(pred) == 60

    def test_scalers_fit_on_train_only(self, rv_frame):
        m = _fast_model()
        m.forecast_fold(rv_frame, _one_fold(rv_frame, 60))
        # feature scaler mean must match the *training* rows, not the full sample.
        from src.data.sequence import build_feature_target
        feats, _, index = build_feature_target(rv_frame, m.features)
        train_end = rv_frame.index[-61]
        tr = feats.to_numpy()[np.asarray(index <= train_end)]
        np.testing.assert_allclose(m._feat_scaler.mean_, tr.mean(0), rtol=1e-6)

    def test_single_feature_rv_only(self, rv_frame):
        """The RV-only robustness variant (one input feature) must build and run."""
        m = _fast_model(features=("log_rv",))
        pred = m.forecast_fold(rv_frame, _one_fold(rv_frame, 40))
        assert (pred > 0).all() and len(pred) == 40
        assert m._model.lstm.input_size == 1

    def test_deterministic(self, rv_frame):
        fold = _one_fold(rv_frame, 40)
        a = _fast_model().forecast_fold(rv_frame, fold)
        b = _fast_model().forecast_fold(rv_frame, fold)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5, atol=1e-12)

    def test_beats_constant_mean_on_persistent_series(self, rv_frame):
        """A trained LSTM should beat a static 'forecast the training mean' rule
        on QLIKE for a genuinely persistent target (sanity that it learns)."""
        m = _fast_model(max_epochs=25, patience=8, hidden_size=24)
        fold = _one_fold(rv_frame, 80)
        pred = m.forecast_fold(rv_frame, fold)
        y = rv_frame["rv"].loc[pred.index]
        train_mean = rv_frame["rv"].loc[: fold.train_end].mean()
        const = pd.Series(train_mean, index=pred.index)
        assert qlike(y, pred) < qlike(y, const)

    def test_runs_through_walk_forward_engine(self, rv_frame):
        """End-to-end through the shared engine on a small val-style segment."""
        from src.evaluation.rolling import run_walk_forward
        cfg = SplitConfig(
            train_end=rv_frame.index[500], val_start=rv_frame.index[501],
            val_end=rv_frame.index[700], test_start=rv_frame.index[701],
            test_end=rv_frame.index[-1], refit_frequency_days=60,
        )
        m = _fast_model()
        preds = run_walk_forward([m], rv_frame, cfg, eval_segment="test", strict=True)
        assert m.name in preds.columns and "y_true" in preds.columns
        assert preds[m.name].notna().mean() > 0.99      # ~full OOS coverage
        assert (preds[m.name].dropna() > 0).all()


class TestRefitCadence:
    def test_train_once_reuses_model(self, rv_frame):
        m = _fast_model(refit_every_folds=0)
        cfg = SplitConfig(
            train_end=rv_frame.index[500], val_start=rv_frame.index[501],
            val_end=rv_frame.index[600], test_start=rv_frame.index[601],
            test_end=rv_frame.index[-1], refit_frequency_days=40,
        )
        from src.evaluation.rolling import run_walk_forward
        run_walk_forward([m], rv_frame, cfg, eval_segment="test", strict=True)
        # trained on the first fold only.
        assert m._trained_at_fold == 0
        assert len(m.fold_params_) == 1
