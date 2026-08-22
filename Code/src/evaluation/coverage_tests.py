"""Formal tests of interval coverage (roadmap Phase 7; Ch2 §2.6/§2.7; RQ3).

What this closes
----------------
Phase 6 answered RQ3 *descriptively*: it reported PICP, MPIW, the Winkler score
and CRPS as point estimates, per regime and per transition split, with no
hypothesis test anywhere. That is precisely the gap Giacomini–White closed for
RQ2 — a point estimate on 76 crisis days is not evidence, and "0.844 vs 0.842"
cannot be called a null without a test that could have rejected. This module is
the inferential half of RQ3.

Three tests, all standard in the risk-management literature and all already
cited by Chapter 2:

1. **Kupiec (1995) proportion-of-failures / unconditional coverage.** Are
   violations occurring at the nominal rate? ``LR_uc ~ χ²(1)``. This is a test
   of the interval's *level*.
2. **Christoffersen (1998) independence and conditional coverage.** Are
   violations *clustered*? ``LR_ind ~ χ²(1)`` tests first-order Markov
   dependence in the hit sequence; ``LR_cc = LR_uc + LR_ind ~ χ²(2)`` tests
   level and clustering jointly. This is a test of the interval's *timing*.
3. **Engle & Manganelli (2004) dynamic quantile (DQ).** Is the demeaned hit
   predictable from anything in the ``t−1`` information set — lagged hits, the
   interval width itself, the lagged regime? ``DQ ~ χ²(k)``. Christoffersen's
   independence test only looks one lag back and only at the hit's own history;
   DQ is the general version, and it is the one that can accept the regime
   indicators as regressors. Chapter 2 §2.6 already cites the test; this is that
   commitment.

Plus one test the literature does *not* supply and this project needs:

4. :func:`two_sample_coverage_test` — a **difference-in-coverage** test between
   two subsamples. See "Why a difference test is not optional" below.

The hit indicator
-----------------
All four tests operate on a Bernoulli *hit* (violation) sequence. The literature
states them for a one-sided VaR exceedance ``1{y_t < VaR_t}`` at level ``α``;
here the object is a **central** ``(1−α)`` prediction interval, so the natural
generalisation is

    hit_t = 1{ y_t < lower_t  or  y_t > upper_t },     E[hit_t] = α = 1 − level

under correct calibration. Nothing in Kupiec, Christoffersen or DQ depends on
the hit being one-sided — they need an iid Bernoulli(α) null and nothing more.
The one thing that *is* weakened is interpretation: pooling the two tails means
a model that is too narrow above and too wide below can pass on net. Both
one-sided views are therefore available through ``side=`` and are reported
alongside the pooled test, because for a volatility forecast the upper tail
(realized variance above the band — the under-prediction of risk) is the
economically consequential one.

Why a difference test is not optional (the 2026-08-19/(iv) caution)
-------------------------------------------------------------------
The Quantile-LSTM under-covers **globally**: PICP 0.821 against a nominal 0.90.
A Kupiec test against the nominal level therefore rejects on *any* subsample
large enough to have power — measured p = 0.046 on the ex-ante transition
subsample despite a flagged-vs-rest gap of only −1.5 pp. Reading that rejection
as "coverage fails at transitions" would attribute a **level** failure to
**timing**, which is the exact confusion Phase 7 exists to resolve.

So: whenever a coverage claim is about a subsample, the quantity of interest is
``PICP(subsample) − PICP(rest)``, not ``PICP(subsample) − nominal``. Use
:func:`two_sample_coverage_test` for that. It reports both an iid two-proportion
statistic and a Newey–West HAC version from the regression ``hit_t = a + b·g_t``,
because hits in a volatility model are not serially independent even under the
null of correct *unconditional* coverage.

And the selector defining any such subsample must be measurable at ``t−1``
(:func:`src.evaluation.regime_timing.selector_is_implementable`) — a subsample
chosen with the day-``t`` outcome has no nominal size for any of these tests.

Pure numpy/scipy: no torch, no project model code, so these run against any
persisted predictions file and are unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-300

ArrayLike = np.ndarray | pd.Series | list


# --------------------------------------------------------------------------- #
# Hit construction                                                             #
# --------------------------------------------------------------------------- #
def interval_hits(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    side: str = "both",
) -> np.ndarray:
    """Binary violation indicator for a central prediction interval.

    Parameters
    ----------
    y_true, lower, upper : aligned realized values and interval endpoints. When
        all three are Series they are inner-joined on the index and non-finite
        rows dropped, so a method covering a subset of dates scores cleanly.
    side : which violations count.

        * ``"both"`` (default) — ``y < lower`` or ``y > upper``; expected rate
          ``α = 1 − level``.
        * ``"upper"`` — only ``y > upper``, i.e. realized variance above the
          band. This is the **risk-relevant** tail for a volatility forecast:
          it is the case where the model understated how volatile the day would
          be. Expected rate ``α/2`` for a symmetric-in-probability interval.
        * ``"lower"`` — only ``y < lower``; expected rate ``α/2``.

    Returns
    -------
    ``np.ndarray`` of 0/1 floats, in date order.
    """
    if side not in ("both", "upper", "lower"):
        raise ValueError(f"side must be 'both', 'upper' or 'lower', got {side!r}")
    if all(isinstance(v, pd.Series) for v in (y_true, lower, upper)):
        j = (
            pd.concat(
                [y_true.rename("y"), lower.rename("lo"), upper.rename("hi")],
                axis=1,
                join="inner",
            )
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        y, lo, hi = (j["y"].to_numpy(float), j["lo"].to_numpy(float),
                     j["hi"].to_numpy(float))
    else:
        y = np.asarray(y_true, dtype=float)
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if not (y.shape == lo.shape == hi.shape):
            raise ValueError(f"shape mismatch: {y.shape}, {lo.shape}, {hi.shape}")
        m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        y, lo, hi = y[m], lo[m], hi[m]
    if y.size == 0:
        raise ValueError("no finite (y, lower, upper) triples to score")
    if side == "upper":
        return (y > hi).astype(float)
    if side == "lower":
        return (y < lo).astype(float)
    return ((y < lo) | (y > hi)).astype(float)


def expected_rate(level: float, side: str = "both") -> float:
    """Violation rate implied by a nominal central ``level`` for a given side."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    a = 1.0 - float(level)
    return a if side == "both" else a / 2.0


# --------------------------------------------------------------------------- #
# 1. Kupiec (1995) unconditional coverage / proportion of failures             #
# --------------------------------------------------------------------------- #
def kupiec_pof(hits: ArrayLike, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    Under ``H0`` the hits are Bernoulli(``alpha``). With ``x`` violations in
    ``n`` days and ``π̂ = x/n``,

        LR_uc = −2 [ x log α + (n−x) log(1−α)
                     − x log π̂ − (n−x) log(1−π̂) ]   ~   χ²(1).

    A small p-value says the interval's **level** is wrong — it is too narrow
    (``π̂ > α``) or too wide (``π̂ < α``). It says nothing about *when* the
    violations happen; that is :func:`christoffersen_independence`.

    Returns ``{'test','stat','df','p_value','n','n_violations','rate',
    'expected_rate','direction'}`` where ``direction`` is ``'under-covers'``,
    ``'over-covers'`` or ``'exact'``.
    """
    h = _as_hits(hits)
    n = int(h.size)
    if n < 2:
        raise ValueError("kupiec_pof needs at least 2 observations")
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    x = int(h.sum())
    pi = x / n

    ll_null = x * np.log(a) + (n - x) * np.log(1.0 - a)
    if x == 0 or x == n:
        # π̂ is 0 or 1, so the unrestricted likelihood is exactly 1 and its
        # log-likelihood is 0 (the usual 0·log 0 = 0 convention).
        ll_alt = 0.0
    else:
        ll_alt = x * np.log(pi) + (n - x) * np.log(1.0 - pi)
    stat = float(max(-2.0 * (ll_null - ll_alt), 0.0))

    from scipy import stats as _st

    return {
        "test": "Kupiec LR_uc",
        "stat": stat,
        "df": 1,
        "p_value": float(_st.chi2.sf(stat, df=1)),
        "n": n,
        "n_violations": x,
        "rate": float(pi),
        "expected_rate": a,
        "direction": ("exact" if np.isclose(pi, a)
                      else "under-covers" if pi > a else "over-covers"),
    }


# --------------------------------------------------------------------------- #
# 2. Christoffersen (1998) independence + conditional coverage                 #
# --------------------------------------------------------------------------- #
def christoffersen_independence(hits: ArrayLike) -> dict:
    """Christoffersen (1998) first-order Markov independence test.

    Compares a first-order Markov chain on the hit sequence against an iid
    Bernoulli one:

        LR_ind = −2 [ log L(π̂) − log L(π̂01, π̂11) ]   ~   χ²(1)

    with ``π̂ij`` the estimated probability of a hit at ``t`` given state ``i`` at
    ``t−1``. A small p-value means violations **cluster** — one breach makes the
    next more (or less) likely — which is the signature of an interval whose
    width fails to react to conditions.

    Note the null here is independence, *not* correct level: the test is
    deliberately silent about whether the overall rate is right, which is what
    makes ``LR_uc`` and ``LR_ind`` separable pieces of evidence. Degenerate
    sequences (no transitions of one type) return ``stat = 0``, ``p = 1`` and
    ``degenerate = True`` rather than a spurious rejection.
    """
    h = _as_hits(hits)
    n = int(h.size)
    if n < 3:
        raise ValueError("christoffersen_independence needs at least 3 observations")
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    from scipy import stats as _st

    base = {
        "test": "Christoffersen LR_ind",
        "df": 1,
        "n": n,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }
    # No day ever follows a violation (or none is ever followed by a non-
    # violation): the transition probabilities are not both identified, so the
    # test has nothing to say. Report that rather than a number.
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {**base, "stat": 0.0, "p_value": 1.0, "pi": float(h.mean()),
                "pi01": float("nan"), "pi11": float("nan"), "degenerate": True}

    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)

    ll_ind = (_xlog(n00 + n10, 1.0 - pi) + _xlog(n01 + n11, pi))
    ll_markov = (_xlog(n00, 1.0 - pi01) + _xlog(n01, pi01)
                 + _xlog(n10, 1.0 - pi11) + _xlog(n11, pi11))
    stat = float(max(-2.0 * (ll_ind - ll_markov), 0.0))
    return {
        **base,
        "stat": stat,
        "p_value": float(_st.chi2.sf(stat, df=1)),
        "pi": float(pi), "pi01": float(pi01), "pi11": float(pi11),
        "degenerate": False,
    }


def christoffersen_cc(hits: ArrayLike, alpha: float) -> dict:
    """Christoffersen (1998) joint conditional-coverage test.

    ``LR_cc = LR_uc + LR_ind ~ χ²(2)``: correct level *and* no clustering. The
    decomposition is the point — a rejection of ``LR_cc`` is only interesting
    once you know which component drove it, so this returns both alongside the
    joint statistic.
    """
    uc = kupiec_pof(hits, alpha)
    ind = christoffersen_independence(hits)
    stat = float(uc["stat"] + ind["stat"])

    from scipy import stats as _st

    return {
        "test": "Christoffersen LR_cc",
        "stat": stat,
        "df": 2,
        "p_value": float(_st.chi2.sf(stat, df=2)),
        "n": uc["n"],
        "lr_uc": uc["stat"], "p_uc": uc["p_value"],
        "lr_ind": ind["stat"], "p_ind": ind["p_value"],
        "ind_degenerate": bool(ind.get("degenerate", False)),
    }


# --------------------------------------------------------------------------- #
# 3. Engle & Manganelli (2004) dynamic quantile test                           #
# --------------------------------------------------------------------------- #
def dynamic_quantile(
    hits: ArrayLike,
    alpha: float,
    *,
    lags: int = 4,
    exog: ArrayLike | pd.DataFrame | None = None,
    exog_names: list[str] | None = None,
) -> dict:
    """Engle & Manganelli (2004) out-of-sample dynamic quantile (DQ) test.

    With the **demeaned** hit ``Hit_t = 1{violation} − α`` and a regressor matrix
    ``X`` whose columns are all measurable at ``t−1``,

        DQ = (Hit' X (X'X)^{-1} X' Hit) / (α (1 − α))   ~   χ²(k),  k = ncol(X).

    Under correct conditional coverage ``E[Hit_t | F_(t−1)] = 0``, so ``Hit`` is
    orthogonal to anything knowable yesterday and the quadratic form is
    asymptotically ``χ²``. A small p-value says the misses are **predictable** —
    the sharpest possible statement that an interval's width is mis-timed rather
    than merely mis-levelled.

    ``X`` always contains a constant and ``lags`` lagged hits (the first ``lags``
    rows are dropped, as in the original). ``exog`` appends further columns —
    this is how the lagged regime indicators enter, which is what the roadmap's
    Phase-7 entry means by *"run on the FULL sample with regressors lagged into
    t−1"*. Do **not** pass a contemporaneous regime label or an ex-post
    transition dummy: both read the day-``t`` outcome and the test loses its
    nominal size (see :mod:`src.evaluation.regime_timing`).

    A rank-deficient ``X`` (e.g. a constant plus a complete one-hot basis, or an
    indicator that never fires) raises rather than silently returning a
    meaningless statistic.
    """
    h = _as_hits(hits)
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    L = int(lags)
    if L < 0:
        raise ValueError("lags must be >= 0")
    n_all = int(h.size)
    if n_all <= L + 2:
        raise ValueError(f"dynamic_quantile needs more than {L + 2} observations, got {n_all}")

    hit = h - a                                   # demeaned hit
    y = hit[L:] if L else hit
    cols = [np.ones(y.size)]
    names = ["const"]
    for k in range(1, L + 1):
        cols.append(hit[L - k: n_all - k])
        names.append(f"hit_lag{k}")

    if exog is not None:
        E = pd.DataFrame(exog).to_numpy(dtype=float) if isinstance(
            exog, (pd.Series, pd.DataFrame)) else np.asarray(exog, dtype=float)
        if E.ndim == 1:
            E = E[:, None]
        if E.shape[0] != n_all:
            raise ValueError(
                f"exog has {E.shape[0]} rows but the hit series has {n_all}; "
                "pass regressors aligned to the same dates as the hits"
            )
        E = E[L:] if L else E
        cols.extend(E[:, j] for j in range(E.shape[1]))
        if exog_names is not None:
            if len(exog_names) != E.shape[1]:
                raise ValueError("exog_names length does not match exog columns")
            names.extend(exog_names)
        else:
            names.extend(f"exog{j}" for j in range(E.shape[1]))

    X = np.column_stack(cols)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[keep], y[keep]
    n, k = X.shape
    if n <= k + 2:
        raise ValueError(f"dynamic_quantile: {n} usable rows for {k} regressors")

    dead = np.flatnonzero(np.ptp(X, axis=0) == 0)
    dead = [int(j) for j in dead if names[int(j)] != "const"]
    if dead:
        raise ValueError(
            f"DQ regressor(s) {[names[j] for j in dead]} are constant over the "
            "sample (an indicator that never fires?), so the design is singular; "
            "drop them or widen the window"
        )
    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx, tol=1e-10) < k:
        raise ValueError(
            "DQ design matrix is rank-deficient: a constant alongside a complete "
            "set of one-hot indicators is collinear — drop the constant or one "
            "indicator column"
        )

    xty = X.T @ y
    stat = float((xty @ np.linalg.solve(xtx, xty)) / (a * (1.0 - a)))
    stat = max(stat, 0.0)

    from scipy import stats as _st

    return {
        "test": "Engle-Manganelli DQ",
        "stat": stat,
        "df": int(k),
        "p_value": float(_st.chi2.sf(stat, df=k)),
        "n": int(n),
        "lags": L,
        "regressors": names,
        "expected_rate": a,
    }


def dq_incremental(
    hits: ArrayLike,
    alpha: float,
    exog: ArrayLike | pd.DataFrame,
    *,
    lags: int = 4,
    exog_names: list[str] | None = None,
) -> dict:
    """Does ``exog`` predict a miss **beyond** what the level and hit dynamics do?

    Why this exists — the level/timing trap, one level up
    -----------------------------------------------------
    The DQ design always contains a constant, so ``DQ`` tests
    ``E[Hit_t] = 0`` jointly with everything else. A model whose *level* is
    wrong therefore rejects DQ through the constant alone, whatever the other
    regressors do. On this project every interval is too narrow globally
    (PICP 0.821–0.865 against 0.90), so the full DQ rejects for every method and
    every conditioner — and reading that as "the regime predicts a miss" would
    be the same level-for-timing substitution that
    :func:`two_sample_coverage_test` exists to prevent, restated in χ² form.

    The fix is the standard nested comparison. ``DQ`` is a quadratic form in the
    projection of ``Hit`` onto the design,
    ``Hit' P_X Hit / (α(1−α))``. For nested designs ``X₀ ⊂ X``, ``P_X − P_{X₀}``
    is itself a projection onto the orthogonal complement, so

        DQ_incremental = DQ_full − DQ_restricted   ~   χ²(k − k₀)

    under the null that the added columns carry no predictive content. That is
    the statistic this function returns: the restricted model is the ordinary
    DQ (constant + ``lags`` lagged hits), the full model adds ``exog``, and the
    difference isolates **exactly** the question RQ3 asks — *given that the band
    is the wrong width on average, is there anything knowable at t−1 that says
    where it will break?*

    Returns the incremental statistic plus both parent statistics, so a table
    can show the decomposition rather than a bare number.
    """
    restricted = dynamic_quantile(hits, alpha, lags=lags)
    full = dynamic_quantile(hits, alpha, lags=lags, exog=exog, exog_names=exog_names)
    if full["n"] != restricted["n"]:
        raise ValueError(
            f"nested DQ designs scored different samples ({full['n']} vs "
            f"{restricted['n']}); align exog to the hit series before calling"
        )
    df = int(full["df"] - restricted["df"])
    if df <= 0:
        raise ValueError("exog added no columns, so there is no incremental test")
    stat = float(max(full["stat"] - restricted["stat"], 0.0))

    from scipy import stats as _st

    added = [r for r in full["regressors"] if r not in restricted["regressors"]]
    return {
        "test": "DQ incremental",
        "stat": stat,
        "df": df,
        "p_value": float(_st.chi2.sf(stat, df=df)),
        "n": int(full["n"]),
        "dq_full": full["stat"],
        "df_full": full["df"],
        "p_full": full["p_value"],
        "dq_restricted": restricted["stat"],
        "df_restricted": restricted["df"],
        "p_restricted": restricted["p_value"],
        "added_regressors": added,
    }


def holm_adjust(p_values: ArrayLike) -> np.ndarray:
    """Holm–Bonferroni step-down adjustment of a family of p-values.

    Phase 7 runs a lot of subsample coverage tests — three regime buckets and a
    transition split, two sides, several methods — and at that count something
    crosses 0.05 by chance alone. Holm controls the family-wise error rate
    without assuming independence (the tests here share a hit series, so they
    are emphatically not independent), and is uniformly more powerful than plain
    Bonferroni. Adjusted values are clipped to 1 and made monotone in the
    original ordering, as the procedure requires.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# 4. Difference in coverage between two subsamples                             #
# --------------------------------------------------------------------------- #
def two_sample_coverage_test(
    hits: ArrayLike,
    group: ArrayLike,
    *,
    hac_lag: int | None = None,
) -> dict:
    """Test whether the violation rate differs between a flagged subsample and the rest.

    **This is the test that must accompany any subsample coverage claim**, and
    the reason is in the module docstring: a model that under-covers globally
    fails a Kupiec test against nominal on every subsample, so a rejection there
    is evidence about the model's *level*, not about the subsample. The
    difference in rates is the quantity that isolates the subsample.

    Two statistics are returned because they answer the same question under
    different assumptions:

    * ``z_iid`` — the classical two-proportion statistic with a pooled variance.
      Valid if hits are serially independent, which is exactly what
      :func:`christoffersen_independence` is there to check.
    * ``z_hac`` — the ``t``-statistic on ``b`` in ``hit_t = a + b·g_t`` with a
      Newey–West (Bartlett) HAC variance. This is the one to quote when the
      independence test rejects, or when the group indicator is itself
      persistent — a regime dummy is, by construction.

    ``group`` is a boolean/0-1 indicator aligned to ``hits``. It must be
    measurable at ``t−1``.
    """
    h = _as_hits(hits)
    g = np.asarray(group).astype(float)
    if g.shape != h.shape:
        raise ValueError(f"group shape {g.shape} != hits shape {h.shape}")
    m = np.isfinite(h) & np.isfinite(g)
    h, g = h[m], (g[m] > 0.5).astype(float)
    n1, n0 = int(g.sum()), int((1 - g).sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"need >= 2 observations on each side, got {n1} flagged / {n0} rest")

    p1 = float(h[g == 1].mean())
    p0 = float(h[g == 0].mean())
    diff = p1 - p0

    # -- iid two-proportion z (pooled variance under H0 of equal rates) --
    n = n0 + n1
    p_pool = float(h.mean())
    se_iid = float(np.sqrt(max(p_pool * (1.0 - p_pool), 0.0) * (1.0 / n1 + 1.0 / n0)))
    z_iid = diff / se_iid if se_iid > 0 else 0.0

    # -- HAC regression hit = a + b*g --
    X = np.column_stack([np.ones(n), g])
    beta, *_ = np.linalg.lstsq(X, h, rcond=None)
    resid = h - X @ beta
    lag = int(hac_lag) if hac_lag is not None else int(np.floor(n ** (1.0 / 3.0)))
    U = X * resid[:, None]
    S = (U.T @ U) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gk = (U[k:].T @ U[:-k]) / n
        S += w * (gk + gk.T)
    xtx_inv = np.linalg.pinv((X.T @ X) / n)
    V = xtx_inv @ S @ xtx_inv / n
    se_hac = float(np.sqrt(max(V[1, 1], 0.0)))
    z_hac = float(beta[1] / se_hac) if se_hac > 0 else 0.0

    from scipy import stats as _st

    return {
        "test": "difference in coverage",
        "n_flagged": n1,
        "n_rest": n0,
        "violation_rate_flagged": p1,
        "violation_rate_rest": p0,
        "picp_flagged": 1.0 - p1,
        "picp_rest": 1.0 - p0,
        "diff_violation_rate": float(diff),
        "z_iid": float(z_iid),
        "p_iid": float(2.0 * _st.norm.sf(abs(z_iid))),
        "se_iid": se_iid,
        "z_hac": z_hac,
        "p_hac": float(2.0 * _st.norm.sf(abs(z_hac))),
        "se_hac": se_hac,
        "hac_lag": lag,
    }


# --------------------------------------------------------------------------- #
# Convenience: the whole suite for one interval                                #
# --------------------------------------------------------------------------- #
def coverage_test_suite(
    y_true: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    *,
    level: float,
    method: str,
    sides: tuple[str, ...] = ("both", "upper"),
    dq_lags: int = 4,
    exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Kupiec, Christoffersen (ind + cc) and DQ for one method's interval.

    Returns one tidy row per (side, test). ``exog`` — if given — is appended to
    the DQ design and must be aligned to ``y_true``'s index and measurable at
    ``t−1``; its columns are named in the ``regressors`` field so a reader can
    see exactly what the test conditioned on.

    ``sides`` defaults to the pooled two-sided view plus the upper tail, which
    for a variance forecast is the tail that matters: a realized variance above
    the band is the day the model said "calm" and the market disagreed.
    """
    # Resolve the DQ sample once, up front: with exogenous regressors the test
    # can only use days where every regressor is available, so the hits fed to
    # DQ are rebuilt on that same index rather than trimmed afterwards. Doing it
    # here (not per-side) keeps every side's DQ on one identical sample.
    dq_index = y_true.index
    dq_exog = None
    if exog is not None:
        ex = pd.DataFrame(exog).reindex(y_true.index)
        dq_index = ex.dropna(how="any").index
        dq_exog = ex.loc[dq_index]

    rows = []
    for side in sides:
        h = interval_hits(y_true, lower, upper, side=side)
        a = expected_rate(level, side)
        uc = kupiec_pof(h, a)
        ind = christoffersen_independence(h)
        cc = christoffersen_cc(h, a)
        if dq_exog is not None:
            h_dq = interval_hits(
                y_true.reindex(dq_index), lower.reindex(dq_index),
                upper.reindex(dq_index), side=side)
            dq = dynamic_quantile(h_dq, a, lags=dq_lags,
                                  exog=dq_exog.to_numpy(),
                                  exog_names=list(dq_exog.columns))
        else:
            dq = dynamic_quantile(h, a, lags=dq_lags)

        common = {"method": method, "side": side, "nominal_level": float(level),
                  "expected_violation_rate": a, "n": uc["n"]}
        rows.append({**common, "test": "Kupiec LR_uc", "stat": uc["stat"], "df": uc["df"],
                     "p_value": uc["p_value"], "n_violations": uc["n_violations"],
                     "violation_rate": uc["rate"], "direction": uc["direction"],
                     "regressors": ""})
        rows.append({**common, "test": "Christoffersen LR_ind", "stat": ind["stat"],
                     "df": ind["df"], "p_value": ind["p_value"],
                     "n_violations": uc["n_violations"], "violation_rate": uc["rate"],
                     "direction": "clustered" if ind["p_value"] < 0.05 else "unclustered",
                     "regressors": "hit_(t-1)"})
        rows.append({**common, "test": "Christoffersen LR_cc", "stat": cc["stat"],
                     "df": cc["df"], "p_value": cc["p_value"],
                     "n_violations": uc["n_violations"], "violation_rate": uc["rate"],
                     "direction": "", "regressors": "level + hit_(t-1)"})
        rows.append({**common, "test": "Engle-Manganelli DQ", "stat": dq["stat"],
                     "df": dq["df"], "p_value": dq["p_value"],
                     "n_violations": uc["n_violations"], "violation_rate": uc["rate"],
                     "direction": "", "regressors": ", ".join(dq["regressors"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _as_hits(hits: ArrayLike) -> np.ndarray:
    h = (hits.to_numpy(dtype=float) if isinstance(hits, pd.Series)
         else np.asarray(hits, dtype=float))
    h = h[np.isfinite(h)]
    uniq = np.unique(h)
    if uniq.size and not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError(f"hits must be 0/1, got values {uniq[:5]}")
    return h


def _xlog(count: int, p: float) -> float:
    """``count * log(p)``, treating ``0 * log(0)`` as 0 (the usual convention)."""
    if count == 0:
        return 0.0
    return float(count) * float(np.log(max(p, _EPS)))
