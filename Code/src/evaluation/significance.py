"""Formal forecast-comparison significance tests (Ch2 §2.7).

Three procedures, all operating on the proxy-robust QLIKE loss (Patton 2011) that
Chapter 2 §2.3/§2.7 commits to, so every pairwise, conditional and joint
comparison in the dissertation uses one consistent, leakage-free loss:

1. **Diebold–Mariano (1995), formalised.** The pairwise test of equal predictive
   accuracy, with a Newey–West HAC long-run variance, the **Harvey, Leybourne &
   Newbold (1997) small-sample correction**, and a Student-``t`` reference
   distribution (``df = n-1``). This is the "formal DM" the earlier milestones
   deferred: it upgrades the indicative normal-approximation readout used during
   Phases 3–5 without changing the loss or the HAC variance, so the statistics are
   backward-compatible (at the dissertation's ``n≈784``, ``h=1`` the HLN factor and
   the ``t``-vs-normal difference are numerically tiny, but the test is now the one
   the literature actually reports).

2. **Giacomini–White (2006) conditional predictive ability.** DM asks whether one
   model is better *on average*; GW asks whether the loss difference is
   *predictable* from information available when the forecast was made. Passing
   the lagged regime indicators as the test function turns the Phase-5 per-regime
   table — which reports point estimates on as few as 70 crisis days — into a
   properly sized joint test on the whole sample, which is what RQ2 needs. Ch2
   §2.7 names it alongside DM as the pairwise-comparison apparatus.

3. **Model Confidence Set (Hansen, Lunde & Nason 2011).** A single pairwise test
   cannot adjudicate nine models without inflating the family-wise error rate; the
   MCS returns the subset of models that are statistically indistinguishable from
   the best at a chosen confidence level, controlling for multiple comparison.
   It is computed with the stationary-bootstrap MCS of the ``arch`` package
   (Sheppard) — already a project dependency (used for GARCH/EGARCH), and the
   canonical implementation — over the same QLIKE loss.

Design note: Chapter 2 §2.7 names *both* the Diebold–Mariano test and the Model
Confidence Set as the comparison apparatus. Keeping them here — beside the losses
in :mod:`src.evaluation.metrics`, and behind one QLIKE definition — is what lets
the run scripts report a formal, multiple-comparison-controlled verdict as part of
the model comparison rather than as a promise deferred to the writeup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-16

ArrayLike = np.ndarray | pd.Series | list


# --------------------------------------------------------------------------- #
# Per-observation losses (variance scale; match src.evaluation.metrics)       #
# --------------------------------------------------------------------------- #
def qlike_loss(y: ArrayLike, h: ArrayLike, *, eps: float = _EPS) -> np.ndarray:
    """Per-observation QLIKE (Patton 2011), non-negative Bregman form.

    ``L_t = sigma2_t / h_t - log(sigma2_t / h_t) - 1``. Its mean equals
    :func:`src.evaluation.metrics.qlike`, so the significance tests and the point
    rankings use one identical loss.
    """
    y = np.clip(np.asarray(y, dtype=float), eps, None)
    h = np.clip(np.asarray(h, dtype=float), eps, None)
    r = y / h
    return r - np.log(r) - 1.0


def se_loss(y: ArrayLike, h: ArrayLike) -> np.ndarray:
    """Per-observation squared error on the variance scale (MSE robustness loss)."""
    y = np.asarray(y, dtype=float)
    h = np.asarray(h, dtype=float)
    return (y - h) ** 2


def _loss_obs(y: np.ndarray, h: np.ndarray, loss: str) -> np.ndarray:
    if loss == "qlike":
        return qlike_loss(y, h)
    if loss in ("mse", "se"):
        return se_loss(y, h)
    raise ValueError(f"unknown loss {loss!r}; use 'qlike' or 'mse'")


def _join3(y: pd.Series, h_a: pd.Series, h_b: pd.Series) -> pd.DataFrame:
    """Inner-join (y, h_a, h_b) on their shared index, dropping non-finite rows.

    Factored out so :func:`_align3` and :func:`giacomini_white` cannot drift: the
    latter needs the surviving *index* to align its conditioning variables to the
    same rows the loss differential was computed on.
    """
    return (
        pd.concat([y.rename("y"), h_a.rename("a"), h_b.rename("b")], axis=1, join="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _align3(
    y: ArrayLike, h_a: ArrayLike, h_b: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align (y, h_a, h_b) on their shared index when all are Series; drop NaNs."""
    if isinstance(y, pd.Series) and isinstance(h_a, pd.Series) and isinstance(h_b, pd.Series):
        j = _join3(y, h_a, h_b)
        return j["y"].to_numpy(float), j["a"].to_numpy(float), j["b"].to_numpy(float)
    y = np.asarray(y, dtype=float)
    a = np.asarray(h_a, dtype=float)
    b = np.asarray(h_b, dtype=float)
    if not (y.shape == a.shape == b.shape):
        raise ValueError(f"shape mismatch: {y.shape}, {a.shape}, {b.shape}")
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    return y[m], a[m], b[m]


# --------------------------------------------------------------------------- #
# Diebold–Mariano (formal: HAC + HLN correction + Student-t)                   #
# --------------------------------------------------------------------------- #
def diebold_mariano(
    y: ArrayLike,
    h_a: ArrayLike,
    h_b: ArrayLike,
    *,
    loss: str = "qlike",
    horizon: int = 1,
    hln: bool = True,
    hac_lag: int | None = None,
) -> dict:
    """Formal Diebold–Mariano test on the loss differential ``d = L(a) - L(b)``.

    A **negative** statistic means model *a* has the lower loss (is better). The
    long-run variance of ``d`` is estimated by a Bartlett-kernel Newey–West HAC
    with lag ``max(horizon-1, floor(n**(1/3)))``; the statistic is then rescaled by
    the Harvey–Leybourne–Newbold (1997) finite-sample factor and referred to a
    Student-``t`` distribution with ``n-1`` degrees of freedom (two-sided p-value).

    ``hac_lag`` overrides the automatic Bartlett lag; it exists so
    :func:`dm_lag_sensitivity` can sweep it and report how much the verdict depends
    on that choice (Ch2 §2.7's regularity-condition robustness exercise). ``None``
    (the default) preserves the automatic rule exactly.

    Returns a dict with ``dm_stat`` (HLN-corrected), ``p_value``, ``mean_loss_diff``,
    ``n``, ``lag`` (backward-compatible keys) plus ``horizon``, ``hln``,
    ``lrv_over_gamma0`` (how far the HAC inflates the variance over the iid case)
    and ``dm_stat_uncorrected`` for transparency.
    """
    yv, a, b = _align3(y, h_a, h_b)
    d = _loss_obs(yv, a, loss) - _loss_obs(yv, b, loss)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 3:
        raise ValueError("Diebold–Mariano needs at least 3 aligned observations")
    dbar = float(d.mean())
    h = max(int(horizon), 1)
    lag = (int(hac_lag) if hac_lag is not None
           else max(h - 1, int(np.floor(n ** (1.0 / 3.0)))))
    if lag < 0:
        raise ValueError("hac_lag must be >= 0")

    # Newey–West HAC long-run variance (Bartlett weights).
    gamma0 = float(np.mean((d - dbar) ** 2))
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        cov = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        lrv += 2.0 * w * cov
    lrv = max(lrv, 1e-24)
    stat = dbar / np.sqrt(lrv / n)

    # Harvey–Leybourne–Newbold (1997) small-sample correction.
    if hln:
        factor = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        stat_corr = stat * factor
    else:
        stat_corr = stat

    from scipy import stats as _st

    p = float(2.0 * _st.t.sf(abs(stat_corr), df=n - 1))
    return {
        "dm_stat": float(stat_corr),
        "p_value": p,
        "mean_loss_diff": dbar,
        "n": n,
        "lag": int(lag),
        "horizon": h,
        "dm_stat_uncorrected": float(stat),
        "hln": bool(hln),
        # >1 means serial dependence in the loss differential inflates the standard
        # error relative to the iid case; ~1 means the DM regularity conditions are
        # comfortably met on this pair (Ch2 §2.7).
        "lrv_over_gamma0": float(lrv / max(gamma0, 1e-300)),
    }


def dm_pairs_table(
    y: ArrayLike, forecasts: pd.DataFrame, pairs, *, loss: str = "qlike", horizon: int = 1
) -> pd.DataFrame:
    """Formal DM for a list of ``(a, b)`` model-column pairs → tidy DataFrame."""
    rows = []
    for a, b in pairs:
        if a not in forecasts.columns or b not in forecasts.columns:
            continue
        r = diebold_mariano(y, forecasts[a], forecasts[b], loss=loss, horizon=horizon)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "dm_stat": r["dm_stat"],
                "p_value": r["p_value"],
                "mean_loss_diff": r["mean_loss_diff"],
                "better": a if r["dm_stat"] < 0 else b,
                "n": r["n"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# DM regularity diagnostics (Ch2 §2.7 robustness exercise)                     #
# --------------------------------------------------------------------------- #
def loss_differential(y: ArrayLike, h_a: ArrayLike, h_b: ArrayLike, *, loss: str = "qlike"):
    """The per-observation loss differential ``d_t = L(a)_t − L(b)_t``."""
    yv, a, b = _align3(y, h_a, h_b)
    d = _loss_obs(yv, a, loss) - _loss_obs(yv, b, loss)
    return d[np.isfinite(d)]


def sample_acf(x: ArrayLike, nlags: int = 20) -> pd.DataFrame:
    """Sample autocorrelations with white-noise and Bartlett standard errors.

    Chapter 2 §2.7 promises that "the test's regularity conditions, in particular
    weak dependence in the loss differential, are inspected and reported as a
    robustness exercise". This is the inspection: Diebold–Mariano assumes the loss
    differential is covariance-stationary with *summable* autocovariances, and a
    HAC variance only repairs dependence it is given enough lags to see. Strong,
    slowly-decaying autocorrelation in ``d_t`` is what "degrades the test's power
    and inflates its size".

    Two standard errors are reported because they answer different questions:
    ``se_white`` = ``1/√n`` tests each lag against strict white noise, while
    ``se_bartlett`` = ``√((1 + 2 Σ_{j<k} r_j²)/n)`` is the correct band once the
    series is already known to be autocorrelated at shorter lags (Bartlett's
    formula) and is the more conservative — and therefore the reported — test.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        raise ValueError("sample_acf needs at least 3 observations")
    nlags = int(min(nlags, n - 1))
    dev = a - a.mean()
    denom = float(np.sum(dev ** 2))
    rows = []
    r_prev: list[float] = []
    for k in range(1, nlags + 1):
        r = float(np.sum(dev[k:] * dev[:-k]) / denom) if denom > 0 else 0.0
        se_w = 1.0 / np.sqrt(n)
        se_b = np.sqrt((1.0 + 2.0 * sum(v ** 2 for v in r_prev)) / n)
        rows.append({
            "lag": k,
            "acf": r,
            "se_white": se_w,
            "se_bartlett": se_b,
            "significant_white": bool(abs(r) > 1.96 * se_w),
            "significant_bartlett": bool(abs(r) > 1.96 * se_b),
        })
        r_prev.append(r)
    out = pd.DataFrame(rows)
    out.attrs["n"] = n
    return out


def ljung_box(x: ArrayLike, lags: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Ljung–Box portmanteau test for autocorrelation up to each lag in ``lags``.

    ``Q(m) = n(n+2) Σ_{k=1..m} r_k² / (n−k)`` ~ ``χ²_m`` under the null of no
    autocorrelation. Applied to the loss differential this is the formal version
    of the ACF inspection: a small p-value says the DM regularity conditions are
    strained and the HAC lag choice matters (see :func:`dm_lag_sensitivity`).
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    max_lag = max(lags)
    acf = sample_acf(a, nlags=max_lag)
    r = acf["acf"].to_numpy()

    from scipy import stats as _st

    rows = []
    for m in lags:
        m = int(m)
        if m >= n:
            continue
        q = n * (n + 2.0) * float(np.sum([r[k - 1] ** 2 / (n - k) for k in range(1, m + 1)]))
        rows.append({
            "lag": m,
            "lb_stat": q,
            "df": m,
            "p_value": float(_st.chi2.sf(q, df=m)),
            "n": n,
        })
    return pd.DataFrame(rows)


def dm_lag_sensitivity(
    y: ArrayLike,
    h_a: ArrayLike,
    h_b: ArrayLike,
    *,
    loss: str = "qlike",
    horizon: int = 1,
    lags: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Re-run the DM test across a grid of HAC lags; report the verdict's stability.

    The automatic rule (``⌊n^{1/3}⌋``) is one defensible choice among many. If the
    p-value crosses a decision threshold as the lag varies, the conclusion is an
    artefact of that choice and must be reported as such; if it does not, the
    inference is robust and can be stated plainly. The row matching the automatic
    rule is flagged ``is_default``.
    """
    d = loss_differential(y, h_a, h_b, loss=loss)
    n = d.size
    default_lag = max(max(int(horizon), 1) - 1, int(np.floor(n ** (1.0 / 3.0))))
    if lags is None:
        lags = tuple(sorted({0, 1, 2, 5, default_lag, 10, 15, 20, 30}))
    rows = []
    for lag in lags:
        if lag >= n:
            continue
        r = diebold_mariano(y, h_a, h_b, loss=loss, horizon=horizon, hac_lag=int(lag))
        rows.append({
            "hac_lag": int(lag),
            "dm_stat": r["dm_stat"],
            "p_value": r["p_value"],
            "lrv_over_gamma0": r["lrv_over_gamma0"],
            "is_default": bool(int(lag) == default_lag),
            "n": r["n"],
        })
    out = pd.DataFrame(rows)
    out.attrs["default_lag"] = default_lag
    return out


# --------------------------------------------------------------------------- #
# Giacomini–White (2006) conditional predictive ability                        #
# --------------------------------------------------------------------------- #
def _hac_matrix(Z: np.ndarray, lag: int, *, divisor: str = "n") -> np.ndarray:
    """Newey–West (Bartlett) long-run covariance of the rows of ``Z``.

    ``divisor`` selects the autocovariance normalisation:

    * ``"n"`` (default) — ``γ̂_k = (1/n) Σ z_t z_{t-k}'``, the textbook Newey–West
      form, which is guaranteed positive semi-definite. A Wald statistic built on
      a non-PSD covariance can come out negative, so this is the safe default for
      the multivariate (q > 1) case that matters here.
    * ``"n-k"`` — divides by the ``n − k`` products actually summed. This is the
      convention :func:`diebold_mariano` uses internally, so passing it makes the
      ``q = 1`` Giacomini–White statistic *exactly* the squared uncorrected DM
      statistic (asserted in the unit tests). The two divisors differ by a factor
      ``(n−k)/n`` per lag — under 0.2% of the statistic at the dissertation's
      ``n ≈ 784``, ``lag = 9``.
    """
    if divisor not in ("n", "n-k"):
        raise ValueError(f"divisor must be 'n' or 'n-k', got {divisor!r}")
    n = Z.shape[0]
    Zc = Z - Z.mean(axis=0)
    omega = (Zc.T @ Zc) / n
    for k in range(1, int(lag) + 1):
        w = 1.0 - k / (lag + 1)
        den = n if divisor == "n" else (n - k)
        gamma = (Zc[k:].T @ Zc[:-k]) / den
        omega += w * (gamma + gamma.T)
    return omega


def giacomini_white(
    y: ArrayLike,
    h_a: ArrayLike,
    h_b: ArrayLike,
    test_fn: ArrayLike | pd.DataFrame | None = None,
    *,
    loss: str = "qlike",
    horizon: int = 1,
    hac_lag: int | None = None,
    hac_divisor: str = "n",
) -> dict:
    """Giacomini–White (2006) test of **conditional** predictive ability.

    Diebold–Mariano asks whether model *a* is better *on average* over the test
    window. Giacomini–White asks the sharper question this dissertation actually
    needs for RQ2: **given what was knowable yesterday, can we predict which model
    will be better tomorrow?** Formally, with loss differential
    ``d_t = L(a)_t − L(b)_t`` and a vector of test functions ``h_{t-1}`` that is
    measurable with respect to the information set at ``t−1``, it tests

        H0 :  E[ h_{t-1} · d_t ] = 0        (equal conditional predictive ability)

    via the Wald statistic ``GW = n · z̄' Ω̂⁻¹ z̄`` where ``z_t = h_{t-1} d_t``,
    ``z̄`` is its sample mean and ``Ω̂`` its (long-run) covariance. Under H0,
    ``GW ~ χ²_q`` with ``q = dim(h)``.

    Why it matters here: the Phase-5 per-regime table shows the regime-aware
    models gaining almost all of their edge in the crisis state, on ~70 days. A
    pairwise DM on the pooled window cannot test that, and a DM on the 70-day
    subsample throws away the rest of the data. Passing the (lagged) regime
    indicators as ``h`` tests exactly the RQ2 claim — *is the loss difference
    regime-dependent?* — while using the whole sample and controlling the size of
    the test. Chapter 2 §2.7 names Giacomini & White (2006) alongside
    Diebold–Mariano as the inferential apparatus; this is that commitment.

    Parameters
    ----------
    y, h_a, h_b : realized variance and the two competing variance forecasts. When
        all three are Series they are aligned on their shared index.
    test_fn : the ``(n, q)`` matrix of conditioning variables, **already lagged**
        into the ``t−1`` information set (see :func:`regime_test_function`, which
        does the lagging for you). A Series/DataFrame is aligned on the index; an
        array must match the aligned sample length. ``None`` (default) uses the
        canonical GW choice ``h_{t-1} = (1, d_{t-1})`` — a constant plus the
        lagged loss differential — which tests whether *past* relative
        performance predicts *future* relative performance.
    loss : ``"qlike"`` (default, Patton 2011) or ``"mse"``.
    horizon : forecast horizon τ. At ``τ = 1`` the score is a martingale
        difference under H0 and no HAC is needed, so ``hac_lag`` defaults to 0;
        for ``τ > 1`` it defaults to ``τ − 1``.
    hac_lag : override the Bartlett lag of the long-run covariance.
    hac_divisor : autocovariance normalisation, ``"n"`` (default, PSD-safe) or
        ``"n-k"`` (matches :func:`diebold_mariano` exactly); see :func:`_hac_matrix`.

    Returns
    -------
    dict with ``gw_stat``, ``p_value``, ``df``, ``n``, ``hac_lag``, ``loss``,
    ``mean_loss_diff``, and ``moments`` (the per-column mean of ``h_{t-1} d_t``,
    whose *signs* say which model wins in which state — the statistic itself is
    two-sided and directionless).

    Notes
    -----
    * **Estimation-window caveat — state this in Chapter 3.** Giacomini & White's
      asymptotics assume the forecasts come from a *finite* (rolling / fixed)
      estimation window, so that parameter-estimation error does not vanish and
      the test is about the *forecasting method* rather than the population
      model. This project's econometric baselines are refit on an **anchored
      (expanding)** window, which formally violates that condition; the deep
      models, trained once and frozen (``refit_every_folds: 0``), satisfy it. In
      practice the anchored window is standard in applied volatility work and the
      test is used here as it is in that literature, but the assumption is
      violated on one side of every deep-vs-econometric pair and should be
      declared rather than discovered at the viva. It affects the GW test's
      formal justification, not the descriptive per-regime moments.
    * A **negative** entry in ``moments`` means model *a* has the lower loss in
      that state (same sign convention as :func:`diebold_mariano`).
    * With a constant test function, a matched ``hac_lag`` and
      ``hac_divisor="n-k"``, ``gw_stat`` equals the squared uncorrected DM
      statistic — the unconditional test is the ``q = 1`` special case, and the
      unit tests assert exactly that.
    * The columns of ``h`` must be linearly independent: passing a constant
      *together with* a full set of one-hot indicators is rank-deficient (they sum
      to the constant) and raises. Drop one or the other.
    """
    yv, a, b = _align3(y, h_a, h_b)
    d_all = _loss_obs(yv, a, loss) - _loss_obs(yv, b, loss)

    # Build the test-function matrix, aligned to the loss differential.
    if test_fn is None:
        # Canonical GW: h_{t-1} = (1, d_{t-1}) -> the first observation is lost.
        d = d_all[1:]
        H = np.column_stack([np.ones(d.size), d_all[:-1]])
    else:
        all_series = (
            isinstance(y, pd.Series)
            and isinstance(h_a, pd.Series)
            and isinstance(h_b, pd.Series)
        )
        if isinstance(test_fn, (pd.Series, pd.DataFrame)) and all_series:
            # Re-align the conditioning variables onto the surviving rows.
            idx = _join3(y, h_a, h_b).index
            H = np.asarray(
                pd.DataFrame(test_fn).reindex(idx).to_numpy(dtype=float), dtype=float
            )
        else:
            H = np.asarray(test_fn, dtype=float)
        if H.ndim == 1:
            H = H[:, None]
        if H.shape[0] != d_all.size:
            raise ValueError(
                f"test_fn has {H.shape[0]} rows but the aligned sample has "
                f"{d_all.size}; pass a Series/DataFrame indexed like the forecasts"
            )
        d = d_all
        keep = np.isfinite(H).all(axis=1) & np.isfinite(d)
        H, d = H[keep], d[keep]

    n, q = H.shape[0], H.shape[1]
    if n <= q + 2:
        raise ValueError(f"Giacomini–White needs more observations ({n}) than test functions ({q})")

    # A test function that never fires (e.g. a regime with no days in the
    # evaluation window) contributes an all-zero column and a singular Omega.
    # Catch it by name -- the generic "collinear" message below would send the
    # reader looking for the wrong problem.
    dead = np.flatnonzero(~np.any(np.abs(H) > 0, axis=0))
    if dead.size:
        raise ValueError(
            f"test_fn column(s) {dead.tolist()} are identically zero over the "
            "evaluation window (a regime with no days?), so the conditional "
            "moment is degenerate; drop those columns or widen the window"
        )

    Z = H * d[:, None]
    zbar = Z.mean(axis=0)

    tau = max(int(horizon), 1)
    lag = int(hac_lag) if hac_lag is not None else max(tau - 1, 0)

    if not np.any(np.abs(Z) > 0):
        # Identical forecasts: the loss differential is exactly zero everywhere,
        # so Omega is degenerate. There is no evidence of a difference, which is
        # the same answer diebold_mariano gives via its variance floor.
        return {
            "gw_stat": 0.0,
            "p_value": 1.0,
            "df": int(q),
            "n": int(n),
            "hac_lag": int(lag),
            "horizon": tau,
            "loss": loss,
            "mean_loss_diff": 0.0,
            "moments": zbar,
        }

    omega = _hac_matrix(Z, lag, divisor=hac_divisor)

    # Rank check: a singular Omega means collinear test functions (e.g. a constant
    # alongside a complete one-hot basis), which makes the Wald statistic
    # undefined rather than merely large.
    scale = np.sqrt(np.clip(np.diag(omega), 1e-300, None))
    corr = omega / np.outer(scale, scale)
    if np.linalg.matrix_rank(corr, tol=1e-8) < q:
        raise ValueError(
            "test_fn columns are collinear (rank-deficient long-run covariance): "
            "drop the constant when passing a full set of one-hot indicators, or "
            "drop one indicator column"
        )

    stat = float(n * zbar @ np.linalg.solve(omega, zbar))

    from scipy import stats as _st

    p = float(_st.chi2.sf(stat, df=q))
    return {
        "gw_stat": stat,
        "p_value": p,
        "df": int(q),
        "n": int(n),
        "hac_lag": int(lag),
        "horizon": tau,
        "loss": loss,
        "mean_loss_diff": float(d.mean()),
        "moments": zbar,
    }


def regime_test_function(
    states: pd.Series | np.ndarray,
    *,
    n_states: int | None = None,
    shift: int = 1,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot regime indicators, lagged into the ``t−1`` information set.

    ``states`` is the causal hard regime label per date (Phase-4 ``*_filt_state``).
    The default ``shift=1`` is not cosmetic: Giacomini–White requires ``h`` to be
    measurable at ``t−1``, and it is also *exactly* the information the regime-
    aware models condition on (their input window ends at ``t−1``). Using the
    contemporaneous state would test a different, non-implementable rule.

    Returns a DataFrame of ``K`` indicator columns (no constant — the indicators
    already span it, and adding one would make the Wald statistic singular).
    Rows whose lagged state is missing are left as NaN for the caller to drop.
    """
    s = pd.Series(states).astype("float")
    s = s.shift(int(shift))
    K = int(n_states) if n_states is not None else int(np.nanmax(s.to_numpy())) + 1
    names = labels if labels is not None else [f"state{k}" for k in range(K)]
    if len(names) != K:
        raise ValueError(f"labels has {len(names)} entries but K={K}")
    out = pd.DataFrame(index=s.index, columns=names, dtype=float)
    for k, nm in enumerate(names):
        col = (s == float(k)).astype(float)
        out[nm] = col.where(s.notna(), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Model Confidence Set (Hansen, Lunde & Nason 2011)                            #
# --------------------------------------------------------------------------- #
def model_confidence_set(
    y: ArrayLike,
    forecasts: dict | pd.DataFrame,
    *,
    loss: str = "qlike",
    alpha: float = 0.10,
    method: str = "R",
    reps: int = 2000,
    block_size: int | None = 10,
    bootstrap: str = "stationary",
    seed: int = 17,
) -> pd.DataFrame:
    """Model Confidence Set over ``forecasts`` at confidence ``1 - alpha``.

    Returns the surviving set of models that are statistically indistinguishable
    from the best under the QLIKE loss, controlling the family-wise error across
    all models at once (Hansen, Lunde & Nason 2011). Computed with the
    stationary-bootstrap MCS of the ``arch`` package.

    Parameters
    ----------
    y : realized variance (the proxy target), one value per date.
    forecasts : ``{model_name: variance_forecast}`` mapping or a DataFrame whose
        columns are model forecasts (all aligned to ``y``).
    loss : ``"qlike"`` (default, primary) or ``"mse"``.
    alpha : MCS test size; the returned set has confidence ``1 - alpha`` (0.10 →
        the 90% MCS).
    method : elimination statistic — ``"R"`` (range, default) or ``"max"``.
    reps, block_size, bootstrap, seed : bootstrap controls (fixed ``seed`` for
        reproducibility, per roadmap §9).

    Returns
    -------
    DataFrame indexed by model (sorted by mean loss) with columns ``avg_loss``,
    ``mcs_pvalue``, ``in_mcs`` and ``rank``. The surviving/eliminated model lists,
    ``alpha``, ``method`` and the bootstrap settings are attached on ``.attrs``.
    """
    from arch.bootstrap import MCS

    ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    if isinstance(forecasts, pd.DataFrame):
        items = [(c, forecasts[c]) for c in forecasts.columns]
    else:
        items = list(forecasts.items())
    cols = {}
    for name, h in items:
        hs = h.reindex(ys.index) if isinstance(h, pd.Series) else pd.Series(
            np.asarray(h, dtype=float), index=ys.index
        )
        cols[name] = _loss_obs(ys.to_numpy(float), hs.to_numpy(float), loss)
    losses = pd.DataFrame(cols, index=ys.index).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if losses.shape[1] < 2:
        raise ValueError("MCS needs at least two models")

    mcs = MCS(
        losses,
        size=float(alpha),
        reps=int(reps),
        block_size=(None if block_size is None else int(block_size)),
        method=method,
        bootstrap=bootstrap,
        seed=int(seed),
    )
    mcs.compute()
    pvals = mcs.pvalues["Pvalue"]
    included = set(mcs.included)

    avg = losses.mean()
    out = pd.DataFrame(
        {
            "avg_loss": avg,
            "mcs_pvalue": pvals.reindex(avg.index),
            "in_mcs": [name in included for name in avg.index],
        }
    ).sort_values("avg_loss")
    out["rank"] = np.arange(1, len(out) + 1)
    out.attrs.update(
        {
            "included": sorted(included),
            "excluded": sorted(set(mcs.excluded)),
            "alpha": float(alpha),
            "method": method,
            "reps": int(reps),
            "block_size": block_size,
            "bootstrap": bootstrap,
            "seed": int(seed),
            "loss": loss,
            "n": int(len(losses)),
        }
    )
    return out
