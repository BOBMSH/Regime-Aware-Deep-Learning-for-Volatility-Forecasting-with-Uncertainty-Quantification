"""Jump-penalised Gaussian HMM — Nystrup, Lindström & Madsen (2020) (Ch2 §2.4/§2.8).

Chapter 2 commits the dissertation to conditioning the deep models on *"a
Gaussian-emission HMM estimated with the Nystrup et al. (2020) jump penalty"*
(§2.8), having argued in §2.4 that maximum-likelihood EM "switches states more
rapidly than the data warrant, contaminating any forecast that conditions on
them". This module is that estimator.

Why a separate class rather than the jump model itself
------------------------------------------------------
:class:`~src.models.regime.jump.JumpModel` solves the penalised objective

    min_{s, θ}  Σ_t ℓ(x_t, θ_{s_t})  +  λ Σ_t 1{s_t ≠ s_{t-1}}

and returns a *hard* state path. Two of its properties make it unsuitable as a
direct drop-in for the Phase-5 conditioning signal:

* its ``predict_proba`` is a softmax over the **emission loss only** — it ignores
  the jump penalty and the state history entirely, so it is a membership weight,
  not a persistence-aware posterior; and
* a hard path carries no notion of *confidence*, which the mixture-of-experts
  gate (Approach B) needs to weight its experts smoothly.

Nystrup et al. resolve this by treating the jump model as a *route to learning an
HMM*: the penalised path identifies persistent states, and the HMM parameters are
then read off that path. That is exactly what :class:`JumpPenalisedHMM` does —

1. fit the jump model on the (training) sample, obtaining a penalised state path;
2. estimate the Gaussian emissions ``(μ_k, Σ_k)`` as the per-state moments of the
   days that path assigns to each state;
3. estimate the transition matrix ``A`` from the path's transition counts (with a
   small Dirichlet pseudo-count so the chain stays irreducible and the filter can
   still react to strong evidence);
4. order the states by volatility and hand the parameters to the *inherited*
   forward filter, which yields a genuine causal posterior ``P(s_t | x_1..x_t)``.

The result is a first-order Gaussian HMM in every downstream respect — it is a
:class:`~src.models.regime.hmm.GaussianHMMRegime`, so ``filtered_proba``,
``smoothed_proba``, ``decode``, ``persistence``, ``bic`` and the volatility-
ordering invariant all work unchanged — but its transition matrix inherits the
jump penalty's persistence rather than EM's excessive switching. Phase 5 and 6
therefore consume the *same* column contract (``*_filt_p0..p{K-1}``) with no
change to any model code.

Leakage discipline (roadmap §1.3)
---------------------------------
Nothing here is causal on its own: :meth:`fit` is an in-sample estimator and must
be handed **training rows only**. The step-2 path comes from the jump *smoother*
(``decode``), which is legitimate because every row it sees is training data; the
out-of-sample causality comes from applying the inherited forward filter to the
full series afterwards. λ must likewise be selected on training rows only — use
:func:`select_jump_penalty_causal`, which is the training-window counterpart of
the full-sample sweep the Phase-4 descriptive map uses (roadmap Phase 4: "the
regularisation parameter is also CV-selected on training-only data").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.regime.hmm import GaussianHMMRegime
from src.models.regime.jump import JumpModel, select_jump_penalty
from src.utils.logging import get_logger

log = get_logger("regime.jump_hmm")

# Diagonal variance floor on standardised features; mirrors ``jump._VAR_FLOOR``
# so a degenerate state cannot produce a singular emission covariance.
_VAR_FLOOR = 1e-6


class JumpPenalisedHMM(GaussianHMMRegime):
    """Gaussian HMM whose parameters are estimated from a jump-penalised path.

    Parameters
    ----------
    n_states : number of latent regimes K.
    jump_penalty : λ, the per-switch cost of the underlying jump model. Larger ⇒
        fewer, longer regimes ⇒ a more persistent transition matrix. Select it on
        training data with :func:`select_jump_penalty_causal`.
    covariance_type : ``"full"`` (default) or ``"diag"``; matched in the inherited
        BIC parameter count and in the emission moments estimated below.
    n_init : restarts of the jump model's coordinate descent (lowest objective
        kept).
    max_iter : max coordinate-descent sweeps per restart.
    seed : base RNG seed, forwarded to the jump model.
    transition_prior : Dirichlet pseudo-count added to **every** transition-count
        cell before normalising. Must be > 0. With a strongly penalised path the
        raw off-diagonal counts can be single digits over two decades of data;
        the pseudo-count keeps the chain irreducible, keeps ``log A`` finite, and
        leaves a floor of roughly ``prior / n_k`` on each off-diagonal so the
        filter can still switch when the emission evidence is decisive. Default
        ``1.0`` (Laplace) ⇒ an off-diagonal floor of ~4e-4 at n_k ≈ 2500.
    cov_jitter : ridge added to each emission covariance for numerical safety.

    Attributes (set by :meth:`fit`, in addition to the inherited HMM parameters)
    --------------------------------------------------------------------------
    jump_model_ : the fitted :class:`~src.models.regime.jump.JumpModel`.
    jump_path_ : its decoded path, relabelled into the volatility order.
    n_jumps_ : number of switches in that path (the persistence diagnostic).
    state_counts_ : days assigned to each (volatility-ordered) state.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        jump_penalty: float = 3.0,
        covariance_type: str = "full",
        n_init: int = 10,
        max_iter: int = 40,
        seed: int = 17,
        transition_prior: float = 1.0,
        cov_jitter: float = 1e-8,
    ) -> None:
        super().__init__(
            n_states,
            covariance_type=covariance_type,
            n_init=n_init,
            n_iter=max_iter,
            seed=seed,
        )
        if jump_penalty < 0:
            raise ValueError("jump_penalty must be >= 0")
        if transition_prior <= 0:
            raise ValueError(
                "transition_prior must be > 0 so the transition matrix stays "
                "irreducible (a zero off-diagonal would freeze the filter)"
            )
        self.jump_penalty = float(jump_penalty)
        self.max_iter = int(max_iter)
        self.transition_prior = float(transition_prior)
        self.cov_jitter = float(cov_jitter)

    # -- fitting ----------------------------------------------------------- #
    def fit(self, X: np.ndarray | pd.DataFrame) -> "JumpPenalisedHMM":
        """Estimate HMM parameters from a jump-penalised state path.

        ``X`` must be **training rows only** — this is an in-sample estimator (see
        the module docstring). Causality is delivered downstream by the inherited
        :meth:`~src.models.regime.hmm.GaussianHMMRegime.filtered_proba`.
        """
        Xa = self._as2d(X)
        T, D = Xa.shape
        if T < self.n_states:
            raise ValueError("fewer observations than states")
        self.n_features_ = D

        jm = JumpModel(
            self.n_states,
            jump_penalty=self.jump_penalty,
            n_init=self.n_init,
            max_iter=self.max_iter,
            seed=self.seed,
        ).fit(Xa)
        path = np.asarray(jm.decode(Xa), dtype=int)

        means, covars = self._emission_moments(Xa, path)
        transmat = self._transition_matrix(path)
        startprob = self._start_distribution(path)

        # Volatility ordering (state 0 = calmest), the invariant every downstream
        # table, colour and probability column relies on. The jump model already
        # orders its own states, so this is usually the identity — applying it
        # unconditionally makes the guarantee independent of that.
        order = self._volatility_order(covars)
        inverse = np.argsort(order)

        self.means_ = means[order]
        self.covars_ = covars[order]
        self.transmat_ = transmat[np.ix_(order, order)]
        sp = startprob[order]
        self.startprob_ = sp / sp.sum()

        self.jump_model_ = jm
        self.jump_path_ = inverse[path]
        self.n_jumps_ = int(np.count_nonzero(np.diff(self.jump_path_)))
        self.state_counts_ = np.bincount(self.jump_path_, minlength=self.n_states)

        self._n_obs_ = T
        self._fitted = True
        # True HMM log-likelihood under the estimated parameters, so ``bic``/``aic``
        # inherited from the parent stay comparable with the Baum-Welch HMM's.
        self._loglik_ = self._score(Xa)

        log.info(
            "jump-penalised HMM: K=%d lambda=%.1f jumps=%d counts=%s diag(A)=%s",
            self.n_states,
            self.jump_penalty,
            self.n_jumps_,
            self.state_counts_.tolist(),
            np.round(np.diag(self.transmat_), 4).tolist(),
        )
        return self

    # -- parameter estimation from the penalised path ---------------------- #
    def _emission_moments(
        self, X: np.ndarray, path: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-state Gaussian moments of the days the penalised path assigns.

        A state that the path leaves (nearly) empty cannot support a covariance,
        so it falls back to the pooled moments and is logged — silently emitting a
        singular state would corrupt every downstream posterior.
        """
        K, D = self.n_states, X.shape[1]
        pooled_mean = X.mean(axis=0)
        pooled_cov = np.atleast_2d(np.cov(X, rowvar=False)).reshape(D, D)

        means = np.empty((K, D))
        covars = np.empty((K, D, D))
        for k in range(K):
            rows = X[path == k]
            if rows.shape[0] > D:  # need > D rows for a non-degenerate covariance
                mu = rows.mean(axis=0)
                cov = np.atleast_2d(np.cov(rows, rowvar=False)).reshape(D, D)
            else:
                log.warning(
                    "state %d has %d assigned day(s) at lambda=%.1f; falling back "
                    "to pooled emission moments",
                    k,
                    rows.shape[0],
                    self.jump_penalty,
                )
                mu, cov = pooled_mean, pooled_cov
            if self.covariance_type == "diag":
                cov = np.diag(np.diag(cov))
            cov = cov + self.cov_jitter * np.eye(D)
            np.fill_diagonal(cov, np.maximum(np.diag(cov), _VAR_FLOOR))
            means[k] = mu
            covars[k] = cov
        return means, covars

    def _transition_matrix(self, path: np.ndarray) -> np.ndarray:
        """Row-normalised transition counts of the penalised path + Dirichlet prior."""
        K = self.n_states
        counts = np.zeros((K, K), dtype=float)
        if path.size > 1:
            np.add.at(counts, (path[:-1], path[1:]), 1.0)
        counts += self.transition_prior
        return counts / counts.sum(axis=1, keepdims=True)

    def _start_distribution(self, path: np.ndarray) -> np.ndarray:
        """Empirical state frequency (smoothed) as the initial distribution.

        The frequency is preferred to a one-hot on ``path[0]``: over a multi-decade
        filter the initial condition washes out within a few days, and a one-hot
        would make the first days' posterior an artefact of one observation.
        """
        counts = np.bincount(path, minlength=self.n_states).astype(float)
        counts += self.transition_prior
        return counts / counts.sum()


# --------------------------------------------------------------------------- #
# Training-window penalty selection                                            #
# --------------------------------------------------------------------------- #
def select_jump_penalty_causal(
    X: np.ndarray | pd.DataFrame,
    n_states: int,
    penalties: list[float],
    *,
    min_mean_duration_days: float = 15.0,
    n_init: int = 10,
    seed: int = 17,
) -> tuple[float, pd.DataFrame]:
    """Choose λ on **training rows only** — the leakage-free counterpart of the
    Phase-4 descriptive sweep.

    Applies the same rule the descriptive map uses: sweep the grid and take the
    *smallest* λ whose mean regime duration clears ``min_mean_duration_days``
    (falling back to the largest λ on the grid if none does). Selecting on
    likelihood is not an option — it always prefers λ → 0 (Nystrup et al. 2020),
    which is precisely the over-switching pathology the penalty exists to fix.

    Returns ``(lambda, sweep_table)``; the table is reported in the milestone so
    the choice is auditable rather than asserted.
    """
    table = select_jump_penalty(X, n_states, list(penalties), n_init=n_init, seed=seed)
    ok = table[table["mean_duration"] >= float(min_mean_duration_days)]
    lam = float(ok.iloc[0]["jump_penalty"]) if len(ok) else float(
        table.iloc[-1]["jump_penalty"]
    )
    log.info(
        "causal lambda selected on training rows: %.1f (persistence floor %.0f days)",
        lam,
        float(min_mean_duration_days),
    )
    return lam, table
