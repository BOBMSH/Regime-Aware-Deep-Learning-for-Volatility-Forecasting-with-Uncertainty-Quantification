"""Prediction-interval construction and calibration diagnostics (roadmap Phase 6;
Ch2 §2.6; RQ3).

Phase 6 turns the Phase-3 LSTM into an *uncertainty-aware* forecaster in two
ways (see :mod:`src.models.deep.uncertainty`):

* **MC-Dropout** yields, per forecast day, a Gaussian predictive law on **log**
  realized variance -- a mean ``mu`` and a std ``sigma`` that combines the model
  (epistemic) spread across stochastic passes with the irreducible (aleatoric)
  residual noise (Gal & Ghahramani 2016). Because the target is log-RV, the
  induced law on the *variance* scale is **log-normal**, whose central intervals
  and quantiles are closed-form (``exp`` of the Gaussian ones).
* **Quantile regression** yields the variance quantiles directly (distribution
  free), so it needs no parametric interval at all.

This module is the bridge from a predictive law to the calibration numbers the
gate reports -- PICP / MPIW / Winkler (in :mod:`src.evaluation.metrics`), the
reliability curve, and the per-regime breakdown that is the phase's distinctive
contribution. It is pure numpy/pandas/scipy so it is unit-testable without torch
and reused unchanged by both UQ methods.

Convention: ``level`` is the *nominal central coverage* of an interval (e.g.
``0.90``); ``tau`` is a *quantile level* in (0, 1) (e.g. ``0.05``). A central
``level`` interval is the ``[(1-level)/2, (1+level)/2]`` quantile pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import coverage_error, mpiw, picp, winkler_score

# Variance positivity floor, matched to the LSTM's forecast floor so interval
# endpoints and point forecasts live on the same scale.
_VAR_FLOOR = 1e-14


def _z(p: float) -> float:
    """Standard-normal quantile ``Phi^{-1}(p)`` (scipy imported lazily)."""
    from scipy import stats as _st

    return float(_st.norm.ppf(p))


def lognormal_quantile(mu: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    """The ``tau``-quantile on the **variance** scale of a log-normal predictive law.

    ``mu``/``sigma`` are the mean/std of the Gaussian law on log-variance; the
    variance-scale quantile is ``exp(mu + Phi^{-1}(tau) * sigma)`` (``exp`` is
    monotone, so quantile ordering is preserved).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.clip(np.exp(mu + _z(tau) * sigma), _VAR_FLOOR, None)


def lognormal_interval(
    mu: np.ndarray, sigma: np.ndarray, *, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Central ``level`` prediction interval on the variance scale (log-normal).

    Returns ``(lower, upper)`` = the ``[(1-level)/2, (1+level)/2]`` quantiles.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    a = (1.0 - level) / 2.0
    return lognormal_quantile(mu, sigma, a), lognormal_quantile(mu, sigma, 1.0 - a)


def lognormal_mean(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Mean of the log-normal predictive law on the variance scale: ``exp(mu + sigma^2/2)``.

    This is the mean-unbiased point forecast for the conditional *variance* (the
    QLIKE-appropriate target, Patton 2011) and generalises the Phase-3 Duan
    smearing correction to a predictive ``sigma`` that also carries the epistemic
    term.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.clip(np.exp(mu + 0.5 * sigma ** 2), _VAR_FLOOR, None)


def interval_metrics(
    y_true, lower, upper, *, level: float
) -> dict[str, float]:
    """Bundle the calibration numbers for one central ``level`` interval.

    Returns ``n``, ``picp``, ``mpiw``, ``winkler`` (interval score at
    ``alpha = 1 - level``) and ``coverage_error`` (``picp - level``; signed).
    Inputs are coerced to aligned float arrays (Series aligned on their index,
    NaNs dropped) so a model that covered a subset of dates scores cleanly.
    """
    yt, lo, hi = _align_triplet(y_true, lower, upper)
    alpha = 1.0 - level
    return {
        "n": int(yt.size),
        "picp": picp(yt, lo, hi),
        "mpiw": mpiw(lo, hi),
        "winkler": winkler_score(yt, lo, hi, alpha=alpha),
        "coverage_error": coverage_error(yt, lo, hi, level=level),
    }


def reliability_curve(
    y_true, quantile_fn, taus
) -> pd.DataFrame:
    """Empirical vs nominal quantile coverage -- the reliability diagram data.

    For each nominal quantile level ``tau`` in ``taus``, ``quantile_fn(tau)``
    returns the predicted ``tau``-quantile of the variance for every observation;
    the empirical coverage is the fraction of realized values at or below it,
    ``mean(y <= q_tau)``, which a perfectly calibrated model matches to ``tau``.

    Returns a tidy frame with ``tau``, ``empirical`` and ``gap = empirical - tau``,
    sorted by ``tau``. (For a *central* interval of level L, the two relevant
    taus are ``(1±L)/2``; feeding a dense tau grid traces the whole diagram.)
    """
    y = _as_array(y_true)
    rows = []
    for tau in taus:
        q = _as_array(quantile_fn(tau))
        if q.shape != y.shape:
            raise ValueError(f"quantile_fn({tau}) shape {q.shape} != y shape {y.shape}")
        emp = float(np.mean(y <= q))
        rows.append({"tau": float(tau), "empirical": emp, "gap": emp - float(tau)})
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


def per_regime_interval_metrics(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    reg_state: pd.Series,
    labels: list[str],
    *,
    level: float,
) -> pd.DataFrame:
    """Calibration numbers within each regime (RQ3's core: does coverage degrade?).

    All inputs are date-indexed and aligned on their shared index. ``reg_state``
    holds the causal filtered hard regime label (0..K-1, matched to ``labels``).
    Returns one row per regime plus an ``all`` row, with ``n``, ``picp``,
    ``mpiw``, ``winkler`` and ``coverage_error`` at the given ``level``.
    """
    j = pd.concat(
        [y_true.rename("y"), lower.rename("lo"), upper.rename("hi"),
         reg_state.rename("state")],
        axis=1, join="inner",
    ).dropna(subset=["y", "lo", "hi"])
    rows = []
    for r, lab in enumerate(labels):
        sub = j[j["state"] == r]
        rec = {"regime": lab, "n": int(len(sub))}
        if len(sub):
            rec.update({k: v for k, v in
                        interval_metrics(sub["y"], sub["lo"], sub["hi"], level=level).items()
                        if k != "n"})
        else:  # empty regime in this window -> NaNs, not a crash
            rec.update({"picp": np.nan, "mpiw": np.nan, "winkler": np.nan,
                        "coverage_error": np.nan})
        rows.append(rec)
    rec = {"regime": "all", "n": int(len(j))}
    rec.update({k: v for k, v in
                interval_metrics(j["y"], j["lo"], j["hi"], level=level).items() if k != "n"})
    rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# small array helpers                                                         #
# --------------------------------------------------------------------------- #
def _as_array(x) -> np.ndarray:
    return x.to_numpy(dtype=float) if isinstance(x, pd.Series) else np.asarray(x, dtype=float)


def _align_triplet(y_true, lower, upper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align (y, lower, upper) on the shared index when all are Series; drop NaNs."""
    if all(isinstance(v, pd.Series) for v in (y_true, lower, upper)):
        j = pd.concat(
            [y_true.rename("y"), lower.rename("lo"), upper.rename("hi")],
            axis=1, join="inner",
        ).replace([np.inf, -np.inf], np.nan).dropna()
        return j["y"].to_numpy(float), j["lo"].to_numpy(float), j["hi"].to_numpy(float)
    yt, lo, hi = _as_array(y_true), _as_array(lower), _as_array(upper)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: {yt.shape}, {lo.shape}, {hi.shape}")
    m = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
    return yt[m], lo[m], hi[m]
