"""Unit tests for the Phase 4 regime models (HMM + Nystrup jump model).

Behavioural tests, not a re-test of hmmlearn: two-regime recovery, volatility
state-ordering, BIC parameter counting, the jump-DP limiting cases, and — the one
that actually protects the dissertation — that the *causal* posterior/filter at
time t is invariant to the future (x_{t+1:T}), i.e. leakage-free. A synthetic
series with a calm (low-variance) and a crisis (high-variance) regime makes the
target genuinely recoverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from src.models.regime.features import (
    RegimeFeatureSpec,
    TrainStandardizer,
    build_regime_features,
)
from src.models.regime.hmm import GaussianHMMRegime
from src.models.regime.jump import JumpModel, _jump_dp, select_jump_penalty


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_regime() -> tuple[np.ndarray, np.ndarray]:
    """12 alternating 250-day segments: calm sigma=0.5, crisis sigma=2.5, ~0 mean.

    Standardised to unit variance overall, matching the pre-registered emission
    (a Gaussian over *standardised* daily log returns)."""
    rng = np.random.default_rng(0)
    seg = 250
    xs, tr, st = [], [], 0
    for _ in range(12):
        xs.append(rng.normal(0.0, 0.5 if st == 0 else 2.5, seg))
        tr.append(np.full(seg, st))
        st ^= 1
    x = np.concatenate(xs)
    x = (x - x.mean()) / x.std()
    return x[:, None], np.concatenate(tr)


def _align_acc(pred: np.ndarray, truth: np.ndarray) -> float:
    """Best label-permutation accuracy (states are only identified up to order)."""
    return max((pred == truth).mean(), (pred == (1 - truth)).mean())


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #
class TestFeatures:
    def test_named_sets_and_columns(self):
        frame = pd.DataFrame(
            {"log_return": [0.01, -0.02, 0.0], "rv": [1e-4, 2e-4, 1.5e-4]},
            index=pd.bdate_range("2020-01-01", periods=3),
        )
        f1 = build_regime_features(frame, "returns")
        f2 = build_regime_features(frame, "returns_logrv")
        assert list(f1.columns) == ["ret"]
        assert list(f2.columns) == ["ret", "log_rv"]

    def test_train_only_standardizer(self):
        feats = pd.DataFrame({"ret": np.arange(100.0)})
        mask = np.zeros(100, bool)
        mask[:50] = True
        z = TrainStandardizer().fit_transform(feats, mask)
        # mean/std computed on the first 50 rows only.
        np.testing.assert_allclose(z.to_numpy()[:50].mean(), 0.0, atol=1e-9)
        assert z.to_numpy()[50:].mean() > 0  # later rows shifted by train stats


# --------------------------------------------------------------------------- #
# Gaussian HMM                                                                 #
# --------------------------------------------------------------------------- #
class TestGaussianHMM:
    def test_recovers_two_regimes(self, two_regime):
        X, truth = two_regime
        m = GaussianHMMRegime(2, n_init=8, seed=17).fit(X)
        assert _align_acc(m.decode(X), truth) > 0.95

    def test_states_ordered_by_volatility(self, two_regime):
        X, _ = two_regime
        m = GaussianHMMRegime(3, n_init=8, seed=17).fit(X)
        traces = np.trace(m.covars_, axis1=1, axis2=2)
        assert np.all(np.diff(traces) >= 0)  # ascending volatility by construction

    def test_posteriors_are_distributions(self, two_regime):
        X, _ = two_regime
        m = GaussianHMMRegime(2, n_init=6, seed=17).fit(X)
        for P in (m.smoothed_proba(X), m.filtered_proba(X)):
            assert P.shape == (len(X), 2)
            np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
            assert (P >= 0).all()

    def test_bic_param_count(self, two_regime):
        X, _ = two_regime
        m = GaussianHMMRegime(3, covariance_type="full", n_init=4, seed=17).fit(X)
        # K=3, D=1, full cov: (K-1) + K(K-1) + K*D + K*D(D+1)/2 = 2 + 6 + 3 + 3 = 14
        assert m.n_params() == 14
        assert np.isfinite(m.bic(X)) and np.isfinite(m.aic(X))

    def test_filtered_is_causal(self, two_regime):
        """The filtered posterior at t must not depend on observations after t."""
        X, _ = two_regime
        m = GaussianHMMRegime(2, n_init=6, seed=17).fit(X)
        t = 1500
        X2 = X.copy()
        X2[t + 1:] += 5.0  # arbitrarily corrupt the future
        f1 = m.filtered_proba(X)
        f2 = m.filtered_proba(X2)
        np.testing.assert_allclose(f1[: t + 1], f2[: t + 1], atol=1e-10)

    def test_smoothed_uses_future(self, two_regime):
        """Contrast: the smoothed posterior *does* move when the future changes —
        which is exactly why it cannot be a forecasting feature."""
        X, _ = two_regime
        m = GaussianHMMRegime(2, n_init=6, seed=17).fit(X)
        t = 1500
        X2 = X.copy()
        X2[t + 1:] += 5.0
        s1 = m.smoothed_proba(X)
        s2 = m.smoothed_proba(X2)
        assert not np.allclose(s1[: t + 1], s2[: t + 1], atol=1e-6)

    def test_persistence_durations(self, two_regime):
        X, _ = two_regime
        m = GaussianHMMRegime(2, n_init=8, seed=17).fit(X)
        pers = m.persistence(m.decode(X))
        assert (pers.expected_duration > 1).all()
        np.testing.assert_allclose(pers.stationary.sum(), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Jump model                                                                   #
# --------------------------------------------------------------------------- #
class TestJumpDP:
    def test_zero_penalty_is_argmin(self):
        rng = np.random.default_rng(0)
        loss = rng.random((200, 3))
        path, _ = _jump_dp(loss, 0.0)
        np.testing.assert_array_equal(path, loss.argmin(axis=1))

    def test_huge_penalty_is_single_state(self):
        rng = np.random.default_rng(0)
        loss = rng.random((200, 3))
        path, _ = _jump_dp(loss, 1e9)
        assert len(set(path.tolist())) == 1

    def test_objective_includes_penalty(self):
        loss = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        _, obj0 = _jump_dp(loss, 0.0)
        _, obj_big = _jump_dp(loss, 10.0)
        assert obj_big > obj0  # switching now costs, so the optimum is higher


class TestJumpModel:
    def test_recovers_two_regimes(self, two_regime):
        X, truth = two_regime
        m = JumpModel(2, jump_penalty=20.0, n_init=6, seed=17).fit(X)
        assert _align_acc(m.decode(X), truth) > 0.95

    def test_states_ordered_by_volatility(self, two_regime):
        X, _ = two_regime
        m = JumpModel(3, jump_penalty=20.0, n_init=6, seed=17).fit(X)
        tot = m.variances_.sum(axis=1)
        assert np.all(np.diff(tot) >= 0)

    def test_penalty_reduces_jumps(self, two_regime):
        X, _ = two_regime
        lo = JumpModel(2, jump_penalty=1.0, n_init=4, seed=17).fit(X)
        hi = JumpModel(2, jump_penalty=50.0, n_init=4, seed=17).fit(X)
        assert hi.n_jumps() <= lo.n_jumps()  # more penalty -> fewer switches

    def test_filtered_is_causal(self, two_regime):
        X, _ = two_regime
        m = JumpModel(2, jump_penalty=20.0, n_init=4, seed=17).fit(X)
        t = 1500
        X2 = X.copy()
        X2[t + 1:] += 5.0
        np.testing.assert_array_equal(
            m.filtered_states(X)[: t + 1], m.filtered_states(X2)[: t + 1]
        )

    def test_proba_is_distribution(self, two_regime):
        X, _ = two_regime
        m = JumpModel(2, jump_penalty=20.0, n_init=4, seed=17).fit(X)
        P = m.predict_proba(X)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)

    def test_penalty_selection_table(self, two_regime):
        X, _ = two_regime
        tbl = select_jump_penalty(X, 2, [1.0, 20.0, 100.0], n_init=3, seed=17)
        assert {"jump_penalty", "n_jumps", "mean_duration", "bic"} <= set(tbl.columns)
        # more penalty -> (weakly) fewer jumps, longer mean duration
        assert tbl.sort_values("jump_penalty")["n_jumps"].is_monotonic_decreasing
