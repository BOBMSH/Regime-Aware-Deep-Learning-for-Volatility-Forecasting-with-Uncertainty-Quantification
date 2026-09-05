"""The regime configs must state what ``run_regimes`` actually does.

Three comments in ``configs/hmm.yaml`` had drifted from the code (audit xi D8,
the third instance of that family), and the reason they could drift is that
nothing executed them:

* "BIC selects K" -- it does not. ``run_regimes`` fits at ``headline_k`` and only
  *logs* the BIC argmin. On the published sample the two detectors disagree
  anyway (HMM prefers K=3, the jump model K=4), so there is no single BIC answer
  to select.
* ``jump_penalty: 30.0`` was labelled "headline lambda". It never is:
  ``select_by_persistence`` is enabled, so the persistence rule always overrides
  it. Its real job is to fix the penalty at which the **K-selection BIC sweep**
  fits the jump model, which makes it load-bearing for ``m04_*_bic.csv``.

These tests pin both statements so the next edit has to keep them true.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.run_regimes import HEADLINE_K, bic_over_k
from src.utils.config import load_config

REGIME_CONFIGS = ["hmm", "hmm_rq4"]


# --------------------------------------------------------------------------- #
# The declarations reach the caller                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", REGIME_CONFIGS)
def test_headline_k_is_declared_and_inside_the_swept_grid(name):
    cfg = load_config(name)
    assert "headline_k" in cfg, f"{name}.yaml must declare headline_k"
    assert int(cfg.headline_k) in [int(k) for k in cfg.k_grid], (
        "the headline K must be one of the K values the BIC sweep reports, "
        "or the sweep says nothing about the model that was actually fitted"
    )


@pytest.mark.parametrize("name", REGIME_CONFIGS)
def test_declared_headline_k_matches_the_module_default(name):
    """``run_regimes`` falls back to HEADLINE_K when the key is absent.

    If the two ever differ, the same run means two things depending on which
    config it is given, and neither the code nor the config would say so.
    """
    assert int(load_config(name).headline_k) == HEADLINE_K


@pytest.mark.parametrize("name", REGIME_CONFIGS)
def test_jump_penalty_is_never_the_headline_lambda(name):
    """The comment says the persistence rule always overrides ``jump_penalty``.

    That is only true while the rule is switched on. Disabling it would silently
    promote 30.0 -- a penalty chosen for the BIC sweep -- to the headline jump
    model, which is a different study.
    """
    cfg = load_config(name)
    assert bool(cfg.select_by_persistence.enabled), (
        "select_by_persistence is off, so cfg.jump_penalty now IS the headline "
        "lambda -- update the config comments and Chapter 3 before enabling this"
    )
    assert float(cfg.select_by_persistence.min_mean_duration_days) > 0


@pytest.mark.parametrize("name", REGIME_CONFIGS)
def test_bic_sweep_penalty_is_a_point_on_the_reported_lambda_grid(name):
    """``m04_*_bic.csv``'s jump column is fitted at ``cfg.jump_penalty``.

    Keeping that value on the swept grid is what lets a reader find the same fit
    in ``m04_*_jump_lambda.csv`` and see for themselves which penalty the K
    comparison was made at.
    """
    cfg = load_config(name)
    grid = [float(x) for x in cfg.jump_penalty_grid]
    assert float(cfg.jump_penalty) in grid
    assert set(grid) <= {float(x) for x in cfg.jump_penalty_sensitivity_grid}, (
        "the selection grid must be a subset of the sensitivity grid, or the "
        "sensitivity sweep does not contain the point selection was made on"
    )


# --------------------------------------------------------------------------- #
# The penalty is wired, and reaches only the jump column                       #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def two_regime_series() -> np.ndarray:
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0, 0.5, 120)
    crisis = rng.normal(0.0, 3.0, 60)
    x = np.concatenate([calm, crisis, calm[:60]])
    return ((x - x.mean()) / x.std()).reshape(-1, 1)


def test_bic_over_k_uses_the_jump_penalty_it_is_given(two_regime_series):
    """`bic_over_k(..., float(cfg.jump_penalty), ...)` -- prove the argument is
    wired, that it moves the jump column, and that it moves nothing else."""
    pytest.importorskip("hmmlearn")
    from src.models.regime.jump import JumpModel

    Xz = two_regime_series
    mcfg = SimpleNamespace(
        hmm=SimpleNamespace(covariance_type="full", n_init=2, n_iter=50, tol=1e-3),
        jump=SimpleNamespace(n_init=2, max_iter=10),
    )
    lo = bic_over_k(Xz, [2], mcfg, 1.0, seed=17).iloc[0]
    hi = bic_over_k(Xz, [2], mcfg, 300.0, seed=17).iloc[0]

    # a larger penalty buys fewer switches, so the jump column must move
    assert lo["jump_n_jumps"] > hi["jump_n_jumps"]
    assert not np.isclose(lo["jump_bic"], hi["jump_bic"])

    # and it must be the SAME fit a reader gets from the model directly
    direct = JumpModel(2, jump_penalty=300.0, n_init=2, max_iter=10, seed=17).fit(Xz)
    assert np.isclose(hi["jump_bic"], direct.bic(Xz))

    # the HMM column is estimated without any penalty and must not move with it,
    # which is what makes "K is fixed, BIC is reported" readable per detector
    assert np.isclose(lo["hmm_bic"], hi["hmm_bic"])
    assert np.isclose(lo["hmm_loglik"], hi["hmm_loglik"])
