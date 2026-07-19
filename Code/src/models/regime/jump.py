"""Statistical jump model for regime detection (Nystrup, Lindström & Madsen 2020).

The jump model is the second regime detector Chapter 2 §2.4 commits to, run
alongside the Gaussian HMM as a robustness cross-check. Where the HMM encodes
persistence *softly* through a transition matrix estimated by EM, the jump model
encodes it *directly*: it clusters the daily feature vectors into K states while
charging a fixed penalty ``λ`` every time the state sequence switches. It is the
temporal generalisation of k-means — recovering k-means at ``λ = 0`` and a single
constant state as ``λ → ∞`` — and it fits by coordinate descent over

    minimise   Σ_t  ℓ(x_t, θ_{s_t})  +  λ · Σ_t 1{s_t ≠ s_{t-1}}

alternating two steps to convergence:

* **Fit step (θ | s).** With the assignment fixed, each state's parameters are the
  MLE on its assigned days — here a Gaussian mean and (diagonal) variance, so the
  loss ``ℓ`` is the Gaussian negative log-likelihood. This is the "Gaussian jump
  model"; it reduces to k-means centroids when ``ℓ`` is squared distance.
* **Assignment step (s | θ).** With θ fixed, the penalised objective is a shortest
  path over a T×K trellis and is solved *exactly* by dynamic programming. The
  naive DP is O(TK²); we use the min / second-min trick to make each step O(TK),
  because the only way to enter state ``k`` at ``t`` is either to stay (cost of the
  same state at ``t−1``) or to jump from the cheapest *other* state (cost ``λ`` plus
  that state's value) — so we never need the full K×K comparison.

Restarts (``n_init``) guard against the local optima of the non-convex objective;
the lowest-objective fit is kept and its states are ordered by volatility so the
labels line up with the HMM's.

Causality. The DP assignment is a *smoother* — deciding a jump at ``t`` can use
``t+1`` to see whether the move persists. For an out-of-sample forecasting feature
we therefore also expose :meth:`filtered_states`, a greedy online pass that assigns
each new day using only the previous state, never the future (leakage-free, the
Phase-5-safe counterpart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("regime.jump")

_LOG_2PI = float(np.log(2.0 * np.pi))
_VAR_FLOOR = 1e-6


def _jump_dp(loss: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Exact min-cost state path for the jump-penalised objective, O(T·K).

    ``loss`` is the (T, K) per-day per-state loss; ``lam`` the switch penalty.
    Returns ``(path, objective)`` where ``objective`` includes the jump penalties.

    Recursion: ``V_t(k) = loss_t(k) + min( V_{t-1}(k),  λ + min_{j} V_{t-1}(j) )``
    where the second branch's argmin excludes ``k`` — handled by carrying both the
    best and second-best previous value so the per-t work is O(K) not O(K²).
    """
    T, K = loss.shape
    if T == 0:
        return np.empty(0, dtype=int), 0.0
    V = np.empty((T, K))
    back = np.zeros((T, K), dtype=int)
    V[0] = loss[0]
    for t in range(1, T):
        prev = V[t - 1]
        order = np.argsort(prev)
        j1 = order[0]
        j2 = order[1] if K > 1 else order[0]
        m1, m2 = prev[j1], prev[j2]
        # For each k, the cheapest "other" previous state (value + argmin):
        best_other_val = np.full(K, m1)
        best_other_arg = np.full(K, j1, dtype=int)
        best_other_val[j1] = m2
        best_other_arg[j1] = j2
        stay_val = prev
        jump_val = best_other_val + lam
        use_stay = stay_val <= jump_val
        V[t] = loss[t] + np.where(use_stay, stay_val, jump_val)
        back[t] = np.where(use_stay, np.arange(K), best_other_arg)
    # Backtrack from the cheapest terminal state.
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmin(V[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(V[-1].min())


class JumpModel:
    """Gaussian statistical jump model with volatility-ordered states.

    Parameters
    ----------
    n_states : number of regimes K.
    jump_penalty : λ, the fixed cost charged per state switch (larger ⇒ fewer,
        longer regimes). Selected by :func:`select_jump_penalty` or swept in the
        Phase-4 driver.
    n_init : random restarts of coordinate descent; lowest-objective kept.
    max_iter : max coordinate-descent sweeps per restart.
    seed : base RNG seed (restart ``i`` uses ``seed + i``).
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        jump_penalty: float = 50.0,
        n_init: int = 10,
        max_iter: int = 30,
        seed: int = 17,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if jump_penalty < 0:
            raise ValueError("jump_penalty must be >= 0")
        self.n_states = int(n_states)
        self.jump_penalty = float(jump_penalty)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self._fitted = False

    # -- fitting ----------------------------------------------------------- #
    def fit(self, X: np.ndarray | pd.DataFrame) -> "JumpModel":
        Xa = self._as2d(X)
        T, D = Xa.shape
        self.n_features_ = D
        if T < self.n_states:
            raise ValueError("fewer observations than states")

        best = None  # (objective, means, variances, path)
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            # Init 0 is a deterministic volatility-quantile seeding so at least one
            # restart separates states by *variance* — essential for the
            # pre-registered 1-D standardised-return emission, where the regimes
            # share a ~zero mean and a plain k-means++ (distance-based) seed would
            # place both centroids near zero and never split them. Remaining
            # restarts use randomised k-means++ for diversity.
            path = (self._volatility_init(Xa) if i == 0
                    else self._kmeanspp_init(Xa, rng))
            prev_obj = np.inf
            for _ in range(self.max_iter):
                means, variances = self._fit_step(Xa, path)
                loss = self._loss_matrix(Xa, means, variances)
                path, obj = _jump_dp(loss, self.jump_penalty)
                if np.isclose(obj, prev_obj, rtol=1e-9, atol=1e-12):
                    break
                prev_obj = obj
            if best is None or obj < best[0]:
                best = (obj, means, variances, path)

        obj, means, variances, path = best
        order = self._volatility_order(variances)
        remap = np.argsort(order)  # old-state -> new-state index
        self.means_ = means[order]
        self.variances_ = variances[order]
        self.objective_ = float(obj)
        self.labels_ = remap[path]
        self._n_obs_ = T
        self._fitted = True
        return self

    def _volatility_init(self, X: np.ndarray) -> np.ndarray:
        """Seed the state path by quantile bins of a local-volatility proxy.

        The proxy is a short rolling mean of the row-wise squared deviation from
        the sample mean — a scale signal that is large in turbulent stretches and
        small in calm ones regardless of return sign. Binning it into K quantiles
        gives an initial assignment whose states differ in *variance*, which is the
        structure the Gaussian-NLL coordinate descent then sharpens. This is what
        lets the jump model recover volatility regimes from signed returns alone.
        """
        T = X.shape[0]
        dev2 = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
        w = min(10, max(1, T // 20))
        proxy = pd.Series(dev2).rolling(w, min_periods=1).mean().to_numpy()
        # rank -> K quantile bins (ascending volatility -> ascending state index)
        ranks = proxy.argsort().argsort()
        return np.minimum((ranks * self.n_states) // T, self.n_states - 1)

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ seeding, then assign each point to its nearest seed."""
        T = X.shape[0]
        centers = [X[rng.integers(T)]]
        for _ in range(1, self.n_states):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(T, 1.0 / T)
            centers.append(X[rng.choice(T, p=probs)])
        C = np.vstack(centers)
        return np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)

    def _fit_step(self, X: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """MLE Gaussian mean + diagonal variance per state (empty states reseeded)."""
        K, D = self.n_states, X.shape[1]
        means = np.zeros((K, D))
        variances = np.zeros((K, D))
        global_mean = X.mean(axis=0)
        global_var = X.var(axis=0)
        for k in range(K):
            rows = X[path == k]
            if rows.shape[0] == 0:
                means[k] = global_mean
                variances[k] = np.maximum(global_var, _VAR_FLOOR)
            else:
                means[k] = rows.mean(axis=0)
                variances[k] = np.maximum(rows.var(axis=0), _VAR_FLOOR)
        return means, variances

    def _loss_matrix(self, X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
        """(T, K) Gaussian negative log-likelihood loss ℓ(x_t, θ_k)."""
        T = X.shape[0]
        loss = np.empty((T, self.n_states))
        for k in range(self.n_states):
            var = variances[k]
            diff2 = (X - means[k]) ** 2
            loss[:, k] = 0.5 * np.sum(diff2 / var + np.log(2.0 * np.pi * var), axis=1)
        return loss

    def _volatility_order(self, variances: np.ndarray) -> np.ndarray:
        """Ascending state order by total variance (Σ_d var_kd)."""
        total = variances.sum(axis=1)
        return np.lexsort((variances[:, 0], total))

    # -- posteriors / decoding -------------------------------------------- #
    def decode(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Smoothed hard path: re-run the jump DP with the fitted parameters."""
        self._check_fitted()
        loss = self._loss_matrix(self._as2d(X), self.means_, self.variances_)
        path, _ = _jump_dp(loss, self.jump_penalty)
        return path

    def filtered_states(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Causal (online) state path — leakage-free, the Phase-5-safe counterpart
        to :meth:`decode`.

        This is the **forward pass** of the jump DP with the argmin read off at each
        step and *no* backward pass: the running cost
        ``V_t(k) = ℓ(x_t,θ_k) + min_j[ V_{t-1}(j) + λ·1{j≠k} ]`` accumulates all past
        evidence (so, unlike a one-step greedy rule, a state only flips once the
        penalised evidence up to ``t`` favours it), yet never consults ``t+1``. The
        smoother (:meth:`decode`) differs only by back-tracking from the terminal
        state, which is what lets it use the future — and is why it must not feed a
        forecast.
        """
        self._check_fitted()
        loss = self._loss_matrix(self._as2d(X), self.means_, self.variances_)
        T, K = loss.shape
        lam = self.jump_penalty
        path = np.empty(T, dtype=int)
        V = loss[0].copy()
        path[0] = int(np.argmin(V))
        for t in range(1, T):
            order = np.argsort(V)
            j1 = order[0]
            j2 = order[1] if K > 1 else order[0]
            best_other = np.full(K, V[j1])
            best_other[j1] = V[j2]
            V = loss[t] + np.minimum(V, lam + best_other)  # stay vs jump-from-best-other
            path[t] = int(np.argmin(V))
        return path

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Soft state membership ``softmax_k(−ℓ(x_t, θ_k))``.

        A continuous relaxation of the hard assignment (Nystrup's soft jump model
        limit): useful as a smooth Phase-5 feature. It reflects only the emission
        fit, not the jump penalty, so it is a membership weight, not a filtered
        HMM posterior — documented as such.
        """
        self._check_fitted()
        loss = self._loss_matrix(self._as2d(X), self.means_, self.variances_)
        z = -loss
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # -- model selection / reporting -------------------------------------- #
    def n_params(self) -> int:
        """Emission free parameters: K means + K diagonal variances (λ is a
        tuning hyperparameter, not counted; the number of jumps is data-driven)."""
        return 2 * self.n_states * self.n_features_

    def loglik(self, X: np.ndarray | pd.DataFrame) -> float:
        """Gaussian log-likelihood at the MAP (decoded) assignment; the jump
        penalty is excluded so this is a comparable emission likelihood for BIC."""
        self._check_fitted()
        Xa = self._as2d(X)
        loss = self._loss_matrix(Xa, self.means_, self.variances_)
        path = self.decode(Xa)
        return float(-loss[np.arange(len(path)), path].sum())

    def bic(self, X: np.ndarray | pd.DataFrame) -> float:
        Xa = self._as2d(X)
        ll = self.loglik(Xa)
        n = Xa.shape[0]
        return -2.0 * ll + self.n_params() * np.log(n)

    def aic(self, X: np.ndarray | pd.DataFrame) -> float:
        ll = self.loglik(self._as2d(X))
        return -2.0 * ll + 2.0 * self.n_params()

    def transition_counts(self, path: np.ndarray | None = None) -> np.ndarray:
        """Empirical K×K transition-count matrix from a decoded path."""
        self._check_fitted()
        p = self.labels_ if path is None else np.asarray(path)
        A = np.zeros((self.n_states, self.n_states))
        for a, b in zip(p[:-1], p[1:]):
            A[a, b] += 1
        return A

    def n_jumps(self, path: np.ndarray | None = None) -> int:
        p = self.labels_ if path is None else np.asarray(path)
        return int((np.diff(p) != 0).sum())

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _as2d(X) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)
        return Xa[:, None] if Xa.ndim == 1 else Xa

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() first")


def select_jump_penalty(
    X: np.ndarray | pd.DataFrame,
    n_states: int,
    penalties: list[float],
    *,
    n_init: int = 10,
    seed: int = 17,
) -> pd.DataFrame:
    """Sweep λ and report jumps / mean-duration / BIC for each — the persistence
    knob is chosen from this curve (Nystrup et al. 2020 pick λ for a plausible
    number of regimes rather than by likelihood, which always prefers λ→0)."""
    Xa = JumpModel._as2d(X)
    T = Xa.shape[0]
    rows = []
    for lam in penalties:
        m = JumpModel(n_states, jump_penalty=lam, n_init=n_init, seed=seed).fit(Xa)
        nj = m.n_jumps()
        rows.append(
            {
                "jump_penalty": lam,
                "n_jumps": nj,
                "mean_duration": T / (nj + 1),
                "objective": m.objective_,
                "bic": m.bic(Xa),
            }
        )
    return pd.DataFrame(rows)
