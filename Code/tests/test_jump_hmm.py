"""Unit tests for the jump-penalised HMM (Ch2 §2.4/§2.8; Nystrup et al. 2020).

This is the estimator that produces the regime signal Phases 5–7 condition on, so
the tests target the three properties the dissertation actually leans on:

1. **It is an HMM.** It must satisfy the whole ``GaussianHMMRegime`` contract —
   volatility-ordered states, a proper stochastic transition matrix, a usable
   posterior — because every downstream consumer (Phase-5 features, the MoE gate,
   Phase-6 per-regime calibration) is written against that contract.
2. **The penalty does something.** A larger λ must yield a more persistent chain;
   otherwise conditioning on it is indistinguishable from conditioning on the
   Baum-Welch HMM and Chapter 2's argument for adopting it is empty.
3. **The posterior is causal.** The filtered probability at t must be invariant to
   everything after t. This is the leakage guarantee the whole Phase-5 result
   rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from src.models.regime.hmm import GaussianHMMRegime
from src.models.regime.jump_hmm import JumpPenalisedHMM, select_jump_penalty_causal


@pytest.fixture
def three_regime() -> tuple[np.ndarray, np.ndarray]:
    """Persistent calm / mid / crisis series: 18 segments of 200 days each.

    Standardised overall, matching the pre-registered emission (a Gaussian over
    *standardised* daily log returns).
    """
    rng = np.random.default_rng(11)
    sigmas = [0.4, 1.0, 2.6]
    xs, truth = [], []
    for i in range(18):
        state = i % 3
        xs.append(rng.normal(0.0, sigmas[state], 200))
        truth.append(np.full(200, state))
    x = np.concatenate(xs)
    x = (x - x.mean()) / x.std()
    return x[:, None], np.concatenate(truth)


# --------------------------------------------------------------------------- #
# The GaussianHMMRegime contract                                               #
# --------------------------------------------------------------------------- #
def test_is_a_gaussian_hmm_regime(three_regime):
    """Downstream code is written against the parent type -- keep it that way."""
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=3, seed=17).fit(X)
    assert isinstance(m, GaussianHMMRegime)
    for attr in ("means_", "covars_", "transmat_", "startprob_"):
        assert hasattr(m, attr)
    assert m.means_.shape == (3, 1)
    assert m.covars_.shape == (3, 1, 1)


def test_states_are_volatility_ordered(three_regime):
    """State 0 = calmest ... state K-1 = most turbulent, the invariant every
    downstream table, colour and probability column assumes."""
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=3, seed=17).fit(X)
    traces = np.trace(m.covars_, axis1=1, axis2=2)
    assert np.all(np.diff(traces) > 0)


def test_transition_matrix_is_stochastic_and_irreducible(three_regime):
    """Rows sum to 1 and no off-diagonal is exactly zero.

    A zero off-diagonal would freeze the forward filter in whatever state it
    entered -- the Dirichlet pseudo-count exists to prevent exactly that.
    """
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=30.0, n_init=3, seed=17).fit(X)
    np.testing.assert_allclose(m.transmat_.sum(axis=1), 1.0, atol=1e-12)
    assert (m.transmat_ > 0).all()
    np.testing.assert_allclose(m.startprob_.sum(), 1.0, atol=1e-12)


def test_recovers_three_persistent_states(three_regime):
    """The decoded path should track the true regime and stay in it."""
    X, truth = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=5, seed=17).fit(X)
    assert (m.jump_path_ == truth).mean() > 0.8
    assert np.diag(m.transmat_).min() > 0.9


# --------------------------------------------------------------------------- #
# The penalty must actually buy persistence                                    #
# --------------------------------------------------------------------------- #
def test_larger_penalty_gives_a_more_persistent_chain(three_regime):
    """λ = 0 is k-means (switches freely); a large λ must switch less.

    This is the mechanism Ch2 §2.4 adopts the estimator *for*, so if the ordering
    ever broke, the chapter's argument would no longer describe the code.
    """
    X, _ = three_regime
    loose = JumpPenalisedHMM(3, jump_penalty=0.0, n_init=3, seed=17).fit(X)
    tight = JumpPenalisedHMM(3, jump_penalty=50.0, n_init=3, seed=17).fit(X)
    assert tight.n_jumps_ <= loose.n_jumps_
    assert np.diag(tight.transmat_).mean() >= np.diag(loose.transmat_).mean()


def test_degenerate_penalty_does_not_produce_invalid_parameters(three_regime):
    """A λ so large that states go unused must still yield a usable model.

    The pooled-moment fallback keeps the emission covariances positive-definite
    instead of raising deep inside the forward filter.
    """
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=1e6, n_init=2, seed=17).fit(X)
    assert np.isfinite(m.means_).all()
    assert np.isfinite(m.covars_).all()
    for k in range(3):
        assert np.linalg.eigvalsh(m.covars_[k]).min() > 0
    p = m.filtered_proba(X)
    assert np.isfinite(p).all()


# --------------------------------------------------------------------------- #
# Leakage: the posterior must be causal                                        #
# --------------------------------------------------------------------------- #
def test_filtered_posterior_is_causal(three_regime):
    """The filtered posterior at t must not depend on observations after t.

    The single most important test in this file: the Phase-5 conditioning feature
    is this posterior, so if it saw the future the entire regime result would be
    contaminated.
    """
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=3, seed=17).fit(X)
    t = 1500
    X2 = X.copy()
    X2[t + 1:] += 5.0  # arbitrarily corrupt the future
    np.testing.assert_allclose(
        m.filtered_proba(X)[: t + 1], m.filtered_proba(X2)[: t + 1], atol=1e-10
    )


def test_filtered_posterior_is_a_probability(three_regime):
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=3, seed=17).fit(X)
    p = m.filtered_proba(X)
    assert p.shape == (len(X), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
    assert ((p >= 0) & (p <= 1)).all()


def test_smoothed_and_filtered_differ_but_agree_mostly(three_regime):
    """The smoother uses the future; the filter does not. They should agree on
    most days and diverge around turning points -- the Phase-4 milestone claim."""
    X, _ = three_regime
    m = JumpPenalisedHMM(3, jump_penalty=10.0, n_init=3, seed=17).fit(X)
    filt = m.filtered_proba(X).argmax(axis=1)
    smooth = m.smoothed_proba(X).argmax(axis=1)
    agree = (filt == smooth).mean()
    assert 0.8 < agree < 1.0


# --------------------------------------------------------------------------- #
# Penalty selection on training rows only                                      #
# --------------------------------------------------------------------------- #
def test_select_jump_penalty_causal_takes_smallest_lambda_meeting_the_floor(three_regime):
    X, _ = three_regime
    grid = [0.0, 1.0, 10.0, 100.0]
    lam, table = select_jump_penalty_causal(
        X, 3, grid, min_mean_duration_days=15.0, n_init=2, seed=17
    )
    assert lam in grid
    ok = table[table["mean_duration"] >= 15.0]
    if len(ok):
        assert lam == float(ok.iloc[0]["jump_penalty"])
        # nothing smaller on the grid clears the floor
        assert (table[table["jump_penalty"] < lam]["mean_duration"] < 15.0).all()
    assert list(table["jump_penalty"]) == grid


def test_rejects_nonpositive_transition_prior():
    with pytest.raises(ValueError, match="transition_prior"):
        JumpPenalisedHMM(3, transition_prior=0.0)


def test_rejects_negative_penalty():
    with pytest.raises(ValueError, match="jump_penalty"):
        JumpPenalisedHMM(3, jump_penalty=-1.0)
