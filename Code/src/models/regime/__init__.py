"""Regime-detection models (roadmap Phase 4, Ch2 §2.4).

Two unsupervised detectors of latent volatility regimes, run side by side so the
regime map's robustness to the modelling choice is visible rather than assumed:

* :class:`~src.models.regime.hmm.GaussianHMMRegime` — a first-order Gaussian HMM
  (Hamilton 1989; Rydén et al. 1998), the workhorse regime model. Persistence is
  encoded through an estimated transition matrix; states are ordered by volatility
  and both smoothed (descriptive) and causal-filtered (Phase-5-safe) posteriors
  are exposed.
* :class:`~src.models.regime.jump.JumpModel` — the statistical jump model of
  Nystrup, Lindström & Madsen (2020): temporal k-means with an explicit switch
  penalty ``λ``, fit by coordinate descent (Gaussian MLE + jump-penalised DP).

Both consume the same feature contract from
:mod:`src.models.regime.features` (standardised daily returns; a returns+log-RV
ablation). The Phase-5 network conditions on the *causal* posterior of the chosen
detector, which is why the leakage-free filter is built and tested here.
"""

from __future__ import annotations

from src.models.regime.features import (
    FEATURE_SETS,
    RegimeFeatureSpec,
    TrainStandardizer,
    build_regime_features,
)
from src.models.regime.hmm import GaussianHMMRegime, RegimePersistence
from src.models.regime.jump import JumpModel, select_jump_penalty

__all__ = [
    "GaussianHMMRegime",
    "RegimePersistence",
    "JumpModel",
    "select_jump_penalty",
    "build_regime_features",
    "RegimeFeatureSpec",
    "TrainStandardizer",
    "FEATURE_SETS",
]
