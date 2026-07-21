"""Sliding-window supervised framing + train-only standardisation for the deep
models (roadmap Phase 3, the "sliding windows" stage of the engine, §3/§5).

Design notes
------------
* **Target is log realized-variance.** The deep models regress ``log(RV_t)``.
  Logging tames the extreme right-skew of RV (variance spans ~4 orders of
  magnitude on equity data) and makes the target roughly homoskedastic, which
  is what an MSE-trained network needs to behave. The forecast is mapped back to
  the *variance* scale (with an optional log-normal smearing correction) before
  it ever touches QLIKE/MSE, so scoring stays on the variance scale that
  ``src.evaluation.metrics`` and the econometric baselines use (Patton 2011).
* **One-step-ahead, leakage-safe windows.** To forecast ``RV_t`` the network sees
  the ``lookback`` feature rows dated ``t-L … t-1`` only -- never row ``t`` -- so a
  forecast for day ``t`` uses information known at the close of ``t-1``. This is
  the same leakage rule the HAR/GARCH baselines obey.
* **Standardisation is fit on the training window only.** Both the feature matrix
  and the log-RV target are standardised with means/stds estimated on training
  rows, then applied everywhere (roadmap §1.3: all scaling parameters fit on
  train only). The forecaster owns the train mask; this module only provides the
  fit/transform primitive and the windowing.

Pure numpy/pandas -- no torch -- so the windowing and scaling are unit-testable
on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

_EPS = 1e-16


def _log_rv(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame["rv"].astype(float).clip(lower=_EPS))


# Named feature builders. Each maps the modelling frame to one aligned column.
# RV-derived features are logged (same reasoning as the target); returns are
# used raw; VIX is carried for later phases. Extend here, not in the model.
FEATURE_BUILDERS: dict[str, callable] = {
    "log_rv": _log_rv,
    "rv": lambda f: f["rv"].astype(float),
    "log_return": lambda f: f["log_return"].astype(float),
    "oc_log_return": lambda f: f["oc_log_return"].astype(float),
    "abs_log_return": lambda f: f["log_return"].astype(float).abs(),
    "abs_oc_return": lambda f: f["oc_log_return"].astype(float).abs(),
    "log_rv_d": lambda f: np.log(f["rv_d"].astype(float).clip(lower=_EPS)),
    "log_rv_w": lambda f: np.log(f["rv_w"].astype(float).clip(lower=_EPS)),
    "log_rv_m": lambda f: np.log(f["rv_m"].astype(float).clip(lower=_EPS)),
    "vix_close": lambda f: f["vix_close"].astype(float),
    # Phase 5 (regime-aware): the Phase-4 causal *filtered* HMM state posteriors,
    # P(s_t | x_1..x_t), joined onto the modelling frame as columns reg_p0..reg_p2
    # (calm / transitional / crisis). Used raw (probabilities in [0,1]); the
    # sliding window (ending at t-1) makes their use as a forecast feature causal
    # by construction, so no manual lag is needed here (roadmap Phase 5; Ch2 §2.5).
    "reg_p0": lambda f: f["reg_p0"].astype(float),
    "reg_p1": lambda f: f["reg_p1"].astype(float),
    "reg_p2": lambda f: f["reg_p2"].astype(float),
}

# Internal modelling target: log realized *variance* of day t.
TARGET_NAME = "log_rv"


def build_feature_target(
    frame: pd.DataFrame, feature_names: list[str] | tuple[str, ...]
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build the (features, target, index) triple for one modelling frame.

    Parameters
    ----------
    frame : the aligned modelling frame (``build_econometric_frame`` output) --
        must carry ``rv`` and whatever base columns the requested features need.
    feature_names : ordered feature keys drawn from :data:`FEATURE_BUILDERS`.

    Returns
    -------
    (features, target, index)
        ``features`` is a date-indexed frame with one column per requested
        feature; ``target`` is ``log(RV)`` per date; ``index`` is their shared,
        NaN-free ``DatetimeIndex``.
    """
    unknown = [n for n in feature_names if n not in FEATURE_BUILDERS]
    if unknown:
        raise KeyError(
            f"unknown feature(s) {unknown}; valid: {sorted(FEATURE_BUILDERS)}"
        )
    cols = {n: FEATURE_BUILDERS[n](frame) for n in feature_names}
    feats = pd.DataFrame(cols, index=frame.index)
    target = _log_rv(frame).rename(TARGET_NAME)
    both = pd.concat([feats, target.rename("__y__")], axis=1)
    both = both.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return both[list(feature_names)], both["__y__"], both.index


class Standardizer:
    """Column-wise (z-score) standardiser fit on a chosen slice only.

    A deliberately tiny reimplementation (rather than sklearn) so the leakage
    contract is visible in-repo: ``fit`` is only ever called on training rows.
    """

    mean_: np.ndarray
    std_: np.ndarray

    def fit(self, a: np.ndarray) -> "Standardizer":
        a = np.asarray(a, dtype=float)
        self.mean_ = a.mean(axis=0)
        std = a.std(axis=0)
        # Guard constant columns so transform never divides by zero.
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float) * self.std_ + self.mean_


def make_windows(
    feats: np.ndarray, target: np.ndarray, index: pd.DatetimeIndex, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Frame a (possibly standardised) feature/target series as supervised windows.

    Returns ``(X, y, y_dates)`` where, for each ``i``:

    * ``X[i]`` has shape ``(lookback, n_features)`` and holds feature rows dated
      ``t-L … t-1``;
    * ``y[i]`` is the target on day ``t`` (``= target[t]``);
    * ``y_dates[i]`` is ``t``.

    The window ends strictly before its target date, so no window can see its own
    label (leakage-safe by construction). ``X`` and ``y`` are ``float32``.
    """
    feats = np.ascontiguousarray(np.asarray(feats, dtype=np.float32))
    target = np.asarray(target, dtype=np.float32)
    n = feats.shape[0]
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if n <= lookback:
        # Not enough history to form a single window.
        d = feats.shape[1] if feats.ndim == 2 else 1
        return (
            np.empty((0, lookback, d), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DatetimeIndex([]),
        )
    if feats.ndim == 1:
        feats = feats[:, None]
    # sliding_window_view over the time axis -> (n-L+1, n_features, L); move the
    # time axis last-but-one so each window reads (L, n_features).
    win = sliding_window_view(feats, window_shape=lookback, axis=0)  # (n-L+1, d, L)
    win = np.ascontiguousarray(win.transpose(0, 2, 1))               # (n-L+1, L, d)
    X = win[: n - lookback]              # windows ending at t-1 for t = L … n-1
    y = target[lookback:]                # target on day t
    y_dates = index[lookback:]
    return X.astype(np.float32, copy=False), y.astype(np.float32, copy=False), y_dates
