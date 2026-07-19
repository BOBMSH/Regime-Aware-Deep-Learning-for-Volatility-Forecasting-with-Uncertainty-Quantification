"""Regime-detection features + train-only standardisation (roadmap Phase 4, §2.4).

Phase 4 detects *volatility regimes* — latent market states (calm / normal /
crisis) that the Phase 5 network conditions on. The regime models (a Gaussian HMM
and a Nystrup jump model) are unsupervised: they read a small feature vector per
trading day and cluster the days into persistent states. This module builds that
feature vector.

Design choices
--------------
* **Daily return is the primary observable.** A volatility regime is, at heart, a
  shift in the second moment of returns, so the classic 2–3-state regime model
  (Hamilton 1989; Nystrup et al. 2020) emits on the daily return: calm states are
  a tight zero-mean Gaussian, crisis states a wide, often negatively-skewed one.
  We default to the **close-to-close** log return because that is the return the
  regime literature and the VIX (our external validation series) are defined on.
* **Ablation adds contemporaneous log-RV.** Realised variance measures volatility
  directly, so appending ``log(RV_t)`` to the emission sharpens the separation
  between states (the crisis state's higher RV is now an explicit dimension, not
  an inferred variance). We report both the returns-only model and this richer
  ``returns+logrv`` model so the regime map's robustness to the feature set is
  visible rather than assumed.
* **Standardisation is fit on the training window only.** Both models are scale
  sensitive — the jump model's loss is a squared distance, and a multivariate
  Gaussian emission mixing a ~1e-2 return with a ~-9 log-RV would be dominated by
  the larger-variance dimension. We z-score every feature, but the mean/std are
  estimated on *training* rows only and then applied everywhere (roadmap §1.3).
  This keeps the causal, leakage-free pipeline that Phase 5 consumes honest: an
  out-of-sample day is never standardised with statistics that peeked at it.

Pure numpy/pandas — no hmmlearn, no torch — so the feature construction and the
train-only scaling are unit-testable on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EPS = 1e-16


# --------------------------------------------------------------------------- #
# Named feature builders                                                       #
# --------------------------------------------------------------------------- #
# Each maps the aligned modelling frame (``build_econometric_frame`` output) to
# one date-indexed column. Returns are used raw (already ~zero-mean, O(1e-2));
# RV-derived features are logged (RV spans ~4 orders of magnitude, so the log is
# what makes a Gaussian emission remotely appropriate). Extend here, not in the
# model, so both the HMM and the jump model see an identical feature contract.
FEATURE_BUILDERS: dict[str, callable] = {
    "ret": lambda f: f["log_return"].astype(float),
    "oc_ret": lambda f: f["oc_log_return"].astype(float),
    "abs_ret": lambda f: f["log_return"].astype(float).abs(),
    "log_rv": lambda f: np.log(f["rv"].astype(float).clip(lower=_EPS)),
    "log_rv_d": lambda f: np.log(f["rv_d"].astype(float).clip(lower=_EPS)),
}


# Canonical feature sets referenced by the Phase 4 config and milestone.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "returns": ("ret",),
    "returns_logrv": ("ret", "log_rv"),
}


@dataclass(frozen=True)
class RegimeFeatureSpec:
    """Which columns form the regime emission, and whether to standardise them."""

    name: str
    columns: tuple[str, ...]
    standardize: bool = True

    @classmethod
    def from_set(cls, name: str, *, standardize: bool = True) -> "RegimeFeatureSpec":
        if name not in FEATURE_SETS:
            raise KeyError(f"unknown feature set {name!r}; valid: {sorted(FEATURE_SETS)}")
        return cls(name=name, columns=FEATURE_SETS[name], standardize=standardize)


def build_regime_features(
    frame: pd.DataFrame, spec: RegimeFeatureSpec | str
) -> pd.DataFrame:
    """Build the *raw* (unstandardised) regime feature frame.

    Parameters
    ----------
    frame : aligned modelling frame indexed by trading date (must carry the base
        columns the requested features read, e.g. ``log_return`` and ``rv``).
    spec : a :class:`RegimeFeatureSpec` or the name of a canonical feature set.

    Returns
    -------
    A date-indexed frame with one column per requested feature, NaN rows dropped.
    Standardisation is applied *separately* by :class:`TrainStandardizer` so the
    fit-on-train contract stays explicit at the call site.
    """
    if isinstance(spec, str):
        spec = RegimeFeatureSpec.from_set(spec)
    unknown = [c for c in spec.columns if c not in FEATURE_BUILDERS]
    if unknown:
        raise KeyError(f"unknown feature(s) {unknown}; valid: {sorted(FEATURE_BUILDERS)}")
    cols = {c: FEATURE_BUILDERS[c](frame) for c in spec.columns}
    feats = pd.DataFrame(cols, index=frame.index)[list(spec.columns)]
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    feats.index.name = frame.index.name or "date"
    return feats


class TrainStandardizer:
    """Column-wise z-score standardiser fit on a chosen date window only.

    Mirrors :class:`src.data.sequence.Standardizer` but keyed by a boolean/date
    mask so the leakage contract ("fit on train rows only") is visible in the
    regime pipeline exactly as it is in the deep pipeline. ``fit_transform`` fits
    on the rows selected by ``train_mask`` and transforms the whole frame.
    """

    mean_: np.ndarray
    std_: np.ndarray
    columns_: list[str]

    def fit(self, feats: pd.DataFrame, train_mask: np.ndarray | pd.Series) -> "TrainStandardizer":
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != bool:
            raise TypeError("train_mask must be boolean")
        if train_mask.sum() < 2:
            raise ValueError("need >= 2 training rows to estimate a std")
        a = feats.to_numpy(dtype=float)[train_mask]
        self.mean_ = a.mean(axis=0)
        std = a.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)  # guard constant columns
        self.columns_ = list(feats.columns)
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        if list(feats.columns) != self.columns_:
            raise ValueError("feature columns differ from those seen at fit time")
        z = (feats.to_numpy(dtype=float) - self.mean_) / self.std_
        return pd.DataFrame(z, index=feats.index, columns=feats.columns)

    def fit_transform(self, feats: pd.DataFrame, train_mask: np.ndarray | pd.Series) -> pd.DataFrame:
        return self.fit(feats, train_mask).transform(feats)


def standardize_full_sample(feats: pd.DataFrame) -> pd.DataFrame:
    """Standardise using the whole frame — for the *descriptive* in-sample regime
    map only (never for a forecast).

    The Phase-4 regime characterisation (BIC over K, persistence, VIX confusion)
    is an unsupervised description of the historical record and is fit in-sample,
    exactly as the regime literature does (Nystrup et al. 2020 fit on the full
    series). The leakage-free counterpart — fit on train, filter causally — is
    what feeds Phase 5, and uses :class:`TrainStandardizer` instead.
    """
    mask = np.ones(len(feats), dtype=bool)
    return TrainStandardizer().fit_transform(feats, mask)
