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


def _finite_interval(
    *arrays: ArrayLike, what: str
) -> tuple[np.ndarray, ...]:
    """Coerce equal-shaped interval arrays to float and drop non-finite rows.

    The point losses above drop non-finite pairs in :func:`_align`; before
    2026-08-30 the interval metrics did not, and the asymmetry was a trap rather
    than a saving. ``picp`` in particular compared ``NaN >= lower``, which is
    ``False``, so a missing endpoint or a missing realisation was silently
    scored as an interval **miss** — understating coverage by a plausible-looking
    amount with no warning. That never fired on the ``.SPX`` panel, where every
    series is complete over the 784 evaluation days, but RQ3 is entirely a
    coverage question and Phase 8 introduces assets on other trading calendars,
    which is exactly where a silent gap would appear.

    All interval metrics now share one contract with the point losses: score the
    rows where every input is finite, and raise if none are.
    """
    out = [np.asarray(a, dtype=float) for a in arrays]
    shapes = {a.shape for a in out}
    if len(shapes) != 1:
        raise ValueError(f"{what} requires inputs of identical shape, got {shapes}")
    mask = np.ones(out[0].shape, dtype=bool)
    for a in out:
        mask &= np.isfinite(a)
    if not mask.any():
        raise ValueError(f"no finite observations to score in {what}")
    return tuple(a[mask] for a in out)


def picp(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Prediction Interval Coverage Probability: fraction of truths inside [l, u].

    Scored on the rows where ``y_true``, ``lower`` and ``upper`` are all finite
    (see :func:`_finite_interval`), so a gap is excluded from the denominator
    rather than counted as a miss.
    """
    yt, lo, hi = _finite_interval(y_true, lower, upper, what="picp")
    return float(np.mean((yt >= lo) & (yt <= hi)))


def mpiw(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean Prediction Interval Width (sharpness); non-finite rows dropped."""
    lo, hi = _finite_interval(lower, upper, what="mpiw")
    return float(np.mean(hi - lo))


def winkler_score(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, *, alpha: float = 0.05
) -> float:
    """Winkler / interval score for a central (1 - alpha) interval.

    Gneiting & Raftery (2007): a strictly proper interval score that rewards
    narrow intervals but penalises misses by ``2/alpha`` times the shortfall. For
    a 95% interval pass ``alpha=0.05``. Lower is better. Non-finite rows are
    dropped (see :func:`_finite_interval`).
    """
    yt, lo, hi = _finite_interval(y_true, lower, upper, what="winkler_score")
    width = hi - lo
    below = yt < lo
    above = yt > hi
    penalty = np.zeros_like(yt, dtype=float)
    penalty[below] = (2.0 / alpha) * (lo[below] - yt[below])
    penalty[above] = (2.0 / alpha) * (yt[above] - hi[above])
    return float(np.mean(width + penalty))


def coverage_error(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, *, level: float) -> float:
    """Signed calibration gap for a central ``level`` interval: ``PICP - level``.

    Positive → the interval over-covers (too wide / conservative); negative →
    it under-covers (too narrow / over-confident). ``level`` is the *nominal*
    coverage (e.g. ``0.90`` for a 90% interval); the well-calibrated target is 0.
    Used from Phase 6 (RQ3) to read miscalibration directionally, per regime.
    """
    return float(picp(y_true, lower, upper) - float(level))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, *, tau: float) -> float:
    """Mean pinball (quantile / check) loss at quantile level ``tau`` ∈ (0, 1).

        L_tau(y, q) = mean( max( tau (y - q), (tau - 1)(y - q) ) )

    The loss whose minimiser is the conditional ``tau``-quantile (Koenker &
    Bassett 1978); the objective the Phase-6 quantile-regression LSTM is trained
    on and the proper score its quantile forecasts are evaluated with. At
    ``tau = 0.5`` it equals ``0.5 * MAE``. Lower is better.

    ``y_true``/``y_pred`` are aligned on their index when both are Series (NaNs
    dropped), matching the point-loss convention above.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    yt, yp = _align(y_true, y_pred)
    e = yt - yp
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def mean_pinball_loss(
    y_true: ArrayLike, quantile_preds: dict[float, ArrayLike]
) -> float:
    """Average pinball loss over a set of ``{tau: forecast}`` quantiles.

    A compact summary of quantile-forecast quality that rewards both calibration
    and sharpness across the interval, not just at its centre. It is *not* the
    CRPS: see :func:`crps_from_quantiles`, which applies the correct quadrature
    weighting, and :func:`crps_lognormal` for the closed form.
    """
    if not quantile_preds:
        raise ValueError("quantile_preds is empty")
    return float(np.mean([pinball_loss(y_true, q, tau=t) for t, q in quantile_preds.items()]))


def crps_from_quantiles(
    y_true: ArrayLike, quantile_preds: dict[float, ArrayLike]
) -> float:
    """CRPS approximated from a set of predictive quantiles (Gneiting & Raftery 2007).

    The continuous ranked probability score has the quantile representation

        CRPS(F, y) = 2 * integral_0^1 PB_tau(y, F^{-1}(tau)) d tau

    where ``PB_tau`` is the pinball loss at level ``tau``. This function evaluates
    that integral by the trapezoidal rule on whatever ``tau`` grid is supplied,
    which is why it is **not** the same object as :func:`mean_pinball_loss`: the
    latter is an unweighted average over the grid points, with no ``2 x`` factor
    and no spacing weights, so it neither converges to the CRPS nor shares its
    units. On a dense, regular grid the two differ by roughly a factor of two;
    on the Phase-6 grid ``{0.05, 0.5, 0.95}`` they differ by more, because the
    trapezoidal weights are far from uniform.

    Accuracy caveat, stated because a three-point grid is what this project
    actually has: the integrand is only sampled where quantiles were estimated,
    and the tails outside ``[min(tau), max(tau)]`` are not sampled at all, so the
    result is a *lower bound* on the true CRPS that tightens as the grid
    densifies. Report it as "CRPS (trapezoidal, K-point grid)" rather than as
    the CRPS, or use :func:`crps_lognormal` where a parametric predictive law is
    available. Lower is better; units are those of ``y``.
    """
    if len(quantile_preds) < 2:
        raise ValueError("crps_from_quantiles needs at least two quantile levels")
    taus = sorted(float(t) for t in quantile_preds)
    if not all(0.0 < t < 1.0 for t in taus):
        raise ValueError("every tau must lie in (0, 1)")
    pb = np.array([pinball_loss(y_true, quantile_preds[t], tau=t) for t in taus], dtype=float)
    return float(2.0 * np.trapezoid(pb, np.asarray(taus, dtype=float)))


def crps_lognormal(
    y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike
) -> float:
    """Closed-form CRPS for a log-normal predictive law (Gneiting & Raftery 2007).

    The Phase-6 MC-Dropout forecaster produces a Gaussian law on **log** variance
    with mean ``mu`` and std ``sigma``, so the induced law on the variance scale
    is log-normal and its CRPS has an exact expression -- no grid, no truncation,
    no quadrature error:

        CRPS(y) = y (2 Phi(w) - 1)
                  - 2 exp(mu + sigma^2/2) [ Phi(w - sigma) + Phi(sigma/sqrt(2)) - 1 ]

    with ``w = (log y - mu) / sigma``. This is the strictly proper score Chapter 2
    §2.6/§2.7 invokes when it says a probabilistic forecast "cannot be evaluated
    by its point estimate alone": unlike coverage, it rewards the whole predictive
    distribution, and unlike the Winkler score it is not tied to one nominal
    level. Lower is better; units are those of ``y`` (variance).

    ``mu``/``sigma`` are on the log scale; ``y_true`` is on the variance scale and
    must be positive.
    """
    yt = np.asarray(y_true, dtype=float)
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if not (yt.shape == m.shape == s.shape):
        raise ValueError(f"shape mismatch: {yt.shape}, {m.shape}, {s.shape}")
    if np.any(s <= 0):
        raise ValueError("sigma must be strictly positive")
    mask = np.isfinite(yt) & np.isfinite(m) & np.isfinite(s) & (yt > 0)
    if not mask.any():
        raise ValueError("no finite, positive observations to score")
    yt, m, s = yt[mask], m[mask], s[mask]

    from scipy import stats as _st

    w = (np.log(yt) - m) / s
    term_obs = yt * (2.0 * _st.norm.cdf(w) - 1.0)
    term_pred = 2.0 * np.exp(m + 0.5 * s**2) * (
        _st.norm.cdf(w - s) + _st.norm.cdf(s / np.sqrt(2.0)) - 1.0
    )
    return float(np.mean(term_obs - term_pred))
