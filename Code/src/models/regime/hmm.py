"""Gaussian-emission Hidden Markov regime model (roadmap Phase 4, Ch2 §2.4).

A first-order HMM with Gaussian emissions is the workhorse regime model of the
volatility literature (Hamilton 1989; Rydén, Teräsvirta & Åsbrink 1998; Nystrup
et al. 2020). Latent state ``s_t ∈ {0..K-1}`` evolves as a Markov chain with
transition matrix ``A``; conditional on the state, the day's feature vector is
drawn from a state-specific Gaussian ``N(μ_k, Σ_k)``. Fit by Baum–Welch (EM), the
model recovers persistent low- and high-variance regimes without any labels.

What this wrapper adds on top of ``hmmlearn``
--------------------------------------------
1. **Volatility-ordered states.** EM labels states arbitrarily and the labelling
   changes across refits/seeds. We permute the fitted parameters so state 0 is
   always the *calmest* (smallest total emission variance) and state ``K-1`` the
   most turbulent. Every downstream table, colour and probability column is then
   comparable across K, across seeds and across refits — essential once these
   posteriors become a Phase-5 input.
2. **A causal (filtered) posterior, not just the smoothed one.** ``hmmlearn`` only
   exposes the *smoothed* posterior ``P(s_t | x_1..x_T)`` (``predict_proba``),
   which conditions on the **whole** sequence including the future — using it as a
   forecasting feature would leak. We add :meth:`filtered_proba`, the forward
   recursion ``P(s_t | x_1..x_t)`` that conditions only on the past, so Phase 5
   can attach a regime signal to an out-of-sample day without look-ahead. Both are
   provided; the milestone shows they diverge exactly where look-ahead would
   matter (around turning points).
3. **Transparent BIC/AIC and persistence statistics** for model selection over K
   and for reporting expected regime durations.

Restarts: EM is only locally optimal and sensitive to initialisation, so
:meth:`fit` runs ``n_init`` random starts and keeps the highest-likelihood fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("regime.hmm")

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class RegimePersistence:
    """Per-state persistence summary derived from the transition matrix + path."""

    diag: np.ndarray            # P(stay) = A_kk
    expected_duration: np.ndarray  # 1 / (1 - A_kk), in trading days
    stationary: np.ndarray      # stationary distribution of A
    empirical_freq: np.ndarray  # fraction of decoded days spent in each state
    mean_run_length: np.ndarray  # average consecutive-day run length per state


def _gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(x | mean, cov) for a full covariance, evaluated row-wise over X.

    Cholesky-based: ``Σ = L Lᵀ`` gives the Mahalanobis term as ``||L⁻¹(x-μ)||²``
    and ``log|Σ| = 2 Σ log diag(L)`` without ever forming ``Σ⁻¹`` — numerically
    stable and O(D²) per row. A tiny jitter rescues a non-PD covariance.
    """
    from scipy.linalg import solve_triangular

    d = mean.shape[0]
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(d))
    diff = np.asarray(X, dtype=float) - mean
    z = solve_triangular(L, diff.T, lower=True)
    maha = np.sum(z ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * _LOG_2PI + logdet + maha)


def _stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Left eigenvector of A for eigenvalue 1, normalised to a probability."""
    vals, vecs = np.linalg.eig(A.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, k])
    pi = np.abs(pi)
    s = pi.sum()
    return pi / s if s > 0 else np.full(A.shape[0], 1.0 / A.shape[0])


def _mean_run_length(path: np.ndarray, n_states: int) -> np.ndarray:
    """Average length of consecutive same-state runs, per state."""
    out = np.zeros(n_states)
    counts = np.zeros(n_states)
    if len(path) == 0:
        return out
    run_state = path[0]
    run_len = 1
    for s in path[1:]:
        if s == run_state:
            run_len += 1
        else:
            out[run_state] += run_len
            counts[run_state] += 1
            run_state, run_len = s, 1
    out[run_state] += run_len
    counts[run_state] += 1
    return np.divide(out, counts, out=np.zeros_like(out), where=counts > 0)


class GaussianHMMRegime:
    """Volatility-ordered Gaussian HMM with smoothed *and* causal posteriors.

    Parameters
    ----------
    n_states : number of latent regimes K.
    covariance_type : ``"full"`` (default) or ``"diag"`` — passed to hmmlearn and
        matched in the manual causal filter and the BIC parameter count.
    n_init : random restarts of EM; the highest-likelihood fit is kept.
    n_iter : max EM iterations per restart.
    seed : base RNG seed (restart ``i`` uses ``seed + i``).
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        covariance_type: str = "full",
        n_init: int = 10,
        n_iter: int = 200,
        tol: float = 1e-4,
        seed: int = 17,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self._fitted = False

    # -- fitting ----------------------------------------------------------- #
    def fit(self, X: np.ndarray | pd.DataFrame) -> "GaussianHMMRegime":
        from hmmlearn.hmm import GaussianHMM

        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        self.n_features_ = Xa.shape[1]

        best = None
        best_ll = -np.inf
        for i in range(self.n_init):
            m = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.seed + i,
                init_params="stmc",
            )
            try:
                m.fit(Xa)
                ll = m.score(Xa)
            except Exception as exc:  # noqa: BLE001 -- a bad restart shouldn't kill the fit
                log.debug("HMM restart %d failed: %s", i, exc)
                continue
            if np.isfinite(ll) and ll > best_ll:
                best_ll, best = ll, m
        if best is None:
            raise RuntimeError("all HMM restarts failed to fit")

        # Pull parameters out, order by volatility, store as plain numpy.
        means = np.asarray(best.means_, dtype=float)
        covars = np.asarray(best.covars_, dtype=float)  # (K, D, D) for any cov type
        transmat = np.asarray(best.transmat_, dtype=float)
        startprob = np.asarray(best.startprob_, dtype=float)

        order = self._volatility_order(covars)
        self.means_ = means[order]
        self.covars_ = covars[order]
        self.transmat_ = transmat[np.ix_(order, order)]
        self.startprob_ = startprob[order]
        self.startprob_ /= self.startprob_.sum()
        self._loglik_ = float(best_ll)
        self._n_obs_ = Xa.shape[0]
        self._fitted = True
        return self

    def _volatility_order(self, covars: np.ndarray) -> np.ndarray:
        """State order (ascending) by total emission variance = trace(Σ_k).

        Trace of the covariance is the summed variance across feature dimensions —
        a scalar 'how turbulent is this state' score that works for both the
        returns-only (1-D) and returns+log-RV (2-D) emissions. Ties are broken by
        the first diagonal entry (the return variance).
        """
        traces = np.trace(covars, axis1=1, axis2=2)
        first_diag = covars[:, 0, 0]
        return np.lexsort((first_diag, traces))

    # -- emission / posteriors -------------------------------------------- #
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """(T, K) matrix of per-state Gaussian log-densities."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        logB = np.empty((T, self.n_states))
        for k in range(self.n_states):
            logB[:, k] = _gaussian_logpdf(X, self.means_[k], self.covars_[k])
        return logB

    def filtered_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Causal forward posterior ``P(s_t | x_1..x_t)`` — leakage-free.

        This is the online filter: the belief on day ``t`` uses only observations
        up to and including ``t`` (never the future), so it is the version safe to
        turn into a Phase-5 forecasting feature. Computed in log-space:

            logα_1(k) = log π_k + log b_k(x_1)
            logα_t(k) = log b_k(x_t) + logΣ_j exp(logα_{t-1}(j) + log A_{jk})

        and the filtered posterior is ``softmax_k logα_t(k)`` at each t.
        """
        self._check_fitted()
        logB = self._log_emission(X)
        T = logB.shape[0]
        log_A = np.log(np.clip(self.transmat_, 1e-300, None))
        log_pi = np.log(np.clip(self.startprob_, 1e-300, None))

        filt = np.empty((T, self.n_states))
        log_alpha = log_pi + logB[0]
        filt[0] = _softmax(log_alpha)
        for t in range(1, T):
            # predict: logsumexp over previous state of (alpha_{t-1} + log A[:,k])
            m = log_alpha.max()
            pred = m + np.log(np.exp(log_alpha - m) @ self.transmat_)
            log_alpha = logB[t] + pred
            filt[t] = _softmax(log_alpha)
        return filt

    def smoothed_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Smoothed posterior ``P(s_t | x_1..x_T)`` (forward–backward).

        Conditions on the **entire** sequence, so it is the right tool to *describe*
        the historical regime path but must NOT be used as a forecasting feature.
        Delegates to hmmlearn via a parameter-locked model in the volatility order.
        """
        self._check_fitted()
        model = self._as_hmmlearn()
        Xa = self._as2d(X)
        _, post = model.score_samples(Xa)
        return post

    def decode(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Viterbi most-likely state path (smoothed / full-sequence)."""
        self._check_fitted()
        model = self._as_hmmlearn()
        _, path = model.decode(self._as2d(X), algorithm="viterbi")
        return path

    def filtered_states(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Causal hard state path: argmax of the filtered posterior each day."""
        return self.filtered_proba(X).argmax(axis=1)

    # -- model selection / reporting -------------------------------------- #
    def n_params(self) -> int:
        """Free-parameter count for BIC/AIC (full or diagonal covariance)."""
        K, D = self.n_states, self.n_features_
        start = K - 1
        trans = K * (K - 1)
        means = K * D
        if self.covariance_type == "full":
            cov = K * D * (D + 1) // 2
        else:  # diag
            cov = K * D
        return start + trans + means + cov

    def bic(self, X: np.ndarray | pd.DataFrame | None = None) -> float:
        ll = self._loglik_ if X is None else self._score(X)
        n = self._n_obs_ if X is None else self._as2d(X).shape[0]
        return -2.0 * ll + self.n_params() * np.log(n)

    def aic(self, X: np.ndarray | pd.DataFrame | None = None) -> float:
        ll = self._loglik_ if X is None else self._score(X)
        return -2.0 * ll + 2.0 * self.n_params()

    def loglik(self) -> float:
        self._check_fitted()
        return self._loglik_

    def persistence(self, path: np.ndarray | None = None) -> RegimePersistence:
        """Persistence summary; ``path`` (a decoded state sequence) enables the
        empirical frequency and mean-run-length fields."""
        self._check_fitted()
        diag = np.diag(self.transmat_).copy()
        expected = np.where(diag < 1.0, 1.0 / np.clip(1.0 - diag, 1e-12, None), np.inf)
        stat = _stationary_distribution(self.transmat_)
        if path is not None:
            path = np.asarray(path)
            freq = np.array([(path == k).mean() for k in range(self.n_states)])
            runs = _mean_run_length(path, self.n_states)
        else:
            freq = np.full(self.n_states, np.nan)
            runs = np.full(self.n_states, np.nan)
        return RegimePersistence(diag, expected, stat, freq, runs)

    # -- helpers ----------------------------------------------------------- #
    def _as_hmmlearn(self):
        from hmmlearn.hmm import GaussianHMM

        m = GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type)
        m.startprob_ = self.startprob_
        m.transmat_ = self.transmat_
        m.means_ = self.means_
        m.covars_ = self.covars_
        m.n_features = self.n_features_
        return m

    def _score(self, X) -> float:
        return float(self._as_hmmlearn().score(self._as2d(X)))

    @staticmethod
    def _as2d(X) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        return Xa[:, None] if Xa.ndim == 1 else Xa

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")


def _softmax(logv: np.ndarray) -> np.ndarray:
    m = logv.max()
    e = np.exp(logv - m)
    return e / e.sum()
