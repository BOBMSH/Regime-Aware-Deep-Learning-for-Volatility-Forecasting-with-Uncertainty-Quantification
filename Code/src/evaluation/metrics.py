"""Forecast-evaluation losses (roadmap Phase 2 gate, §4/§5).

All point losses are defined on the **variance** scale, i.e. both arguments are
realized *variance* (the RV proxy) and the model's *variance* forecast. This is a
deliberate design decision:

* Patton (2011) derives the proxy-robustness of MSE and QLIKE for the conditional
  *variance*, so evaluating on variance is what makes the "noisy RV proxy does not
  bias the ranking" guarantee actually hold. Chapter 2 §2.3 commits to QLIKE on
  this basis.
* GARCH/EGARCH forecast a conditional variance natively; HAR-RV is fit on the RV
  series. Scoring every model on the same variance target keeps the comparison
  apples-to-apples (Hansen & Lunde 2005).

QLIKE is the **primary** metric (Patton 2011); MSE and MAE are reported as
robustness checks. QLIKE is written in its non-negative Bregman form

    QLIKE(sigma2, h) = sigma2 / h - log(sigma2 / h) - 1,

which is >= 0 and minimised uniquely at ``h == sigma2``. This differs from the
raw ``log h + sigma2 / h`` form only by terms independent of the forecast ``h``,
so it induces an identical ranking while being easier to read (0 = perfect).

The interval metrics (``picp``, ``mpiw``, ``winkler_score``) are not used by the
Phase 2 baselines -- they are the calibration losses the Phase 6 uncertainty
models will need -- but they live here so the evaluation engine is built once
(roadmap §5/§6) and unit-tested from the start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numerical floor. Realized variance for daily equity data is O(1e-4); 1e-16 is
# far below any real value and only guards against log/÷ of an exact zero.
_EPS = 1e-16

ArrayLike = np.ndarray | pd.Series | list[float]


def _align(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays, align on index when both are Series, drop NaNs.

    Aligning on the index (rather than by position) is the safe default: it makes
    a metric call robust to a model that returned forecasts for a subset of dates,
    and it surfaces misalignment as dropped rows rather than a silent off-by-one.
    """
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        joined = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1, join="inner")
        yt = joined["y"].to_numpy(dtype=float)
        yp = joined["yhat"].to_numpy(dtype=float)
    else:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if yt.shape != yp.shape:
            raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        raise ValueError("no finite (y_true, y_pred) pairs to score")
    return yt[mask], yp[mask]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error on the variance scale."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error (same units as the variance target)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error on the variance scale."""
    yt, yp = _align(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def qlike(y_true: ArrayLike, y_pred: ArrayLike, *, eps: float = _EPS) -> float:
    """QLIKE loss (Patton 2011), non-negative Bregman form.

        L = mean( sigma2 / h - log(sigma2 / h) - 1 )

    Parameters
    ----------
    y_true : realized variance (the proxy), strictly positive.
    y_pred : forecast variance, strictly positive.
    eps : positivity floor applied to both arguments before the ratio.

    Notes
    -----
    QLIKE penalises *under*-prediction more heavily than over-prediction, which
    aligns the objective with the asymmetric cost of understating risk -- the
    property that makes it the literature's default for volatility (Patton 2011,
    Chapter 2 §2.3). Forecasts must be positive; the caller (e.g. HAR-RV) is
    responsible for flooring negative forecasts before scoring, but we clip here
    too as a last line of defence.
    """
    yt, yp = _align(y_true, y_pred)
    yt = np.clip(yt, eps, None)
    yp = np.clip(yp, eps, None)
    ratio = yt / yp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def point_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Bundle the Phase 2 point-forecast losses into one record.

    Returns ``n``, ``mse``, ``rmse``, ``mae`` and ``qlike``. QLIKE is listed last
    but is the primary metric for model ranking.
    """
    yt, yp = _align(y_true, y_pred)
    return {
        "n": int(yt.size),
        "mse": mse(yt, yp),
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "qlike": qlike(yt, yp),
    }


# --------------------------------------------------------------------------- #
# Interval / calibration metrics -- used from Phase 6 (roadmap §4 lists them   #
# in metrics.py; built now so the engine is complete and tested).             #
# --------------------------------------------------------------------------- #


def picp(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Prediction Interval Coverage Probability: fraction of truths inside [l, u]."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("picp requires y_true, lower, upper of identical shape")
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mpiw(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean Prediction Interval Width (sharpness)."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("mpiw requires lower and upper of identical shape")
    return float(np.mean(hi - lo))


def winkler_score(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, *, alpha: float = 0.05
) -> float:
    """Winkler / interval score for a central (1 - alpha) interval.

    Gneiting & Raftery (2007): a strictly proper interval score that rewards
    narrow intervals but penalises misses by ``2/alpha`` times the shortfall. For
    a 95% interval pass ``alpha=0.05``. Lower is better.
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("winkler_score requires identical shapes")
    width = hi - lo
    below = yt < lo
    above = yt > hi
    penalty = np.zeros_like(yt, dtype=float)
    penalty[below] = (2.0 / alpha) * (lo[below] - yt[below])
    penalty[above] = (2.0 / alpha) * (yt[above] - hi[above])
    return float(np.mean(width + penalty))
