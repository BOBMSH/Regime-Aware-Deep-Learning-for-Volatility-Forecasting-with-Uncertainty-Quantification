"""Formal forecast-comparison significance tests (Ch2 §2.7).

Two procedures, both operating on the proxy-robust QLIKE loss (Patton 2011) that
Chapter 2 §2.3/§2.7 commits to, so every pairwise and joint comparison in the
dissertation uses one consistent, leakage-free loss:

1. **Diebold–Mariano (1995), formalised.** The pairwise test of equal predictive
   accuracy, with a Newey–West HAC long-run variance, the **Harvey, Leybourne &
   Newbold (1997) small-sample correction**, and a Student-``t`` reference
   distribution (``df = n-1``). This is the "formal DM" the earlier milestones
   deferred: it upgrades the indicative normal-approximation readout used during
   Phases 3–5 without changing the loss or the HAC variance, so the statistics are
   backward-compatible (at the dissertation's ``n≈784``, ``h=1`` the HLN factor and
   the ``t``-vs-normal difference are numerically tiny, but the test is now the one
   the literature actually reports).

2. **Model Confidence Set (Hansen, Lunde & Nason 2011).** A single pairwise test
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


def _align3(
    y: ArrayLike, h_a: ArrayLike, h_b: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align (y, h_a, h_b) on their shared index when all are Series; drop NaNs."""
    if isinstance(y, pd.Series) and isinstance(h_a, pd.Series) and isinstance(h_b, pd.Series):
        j = pd.concat(
            [y.rename("y"), h_a.rename("a"), h_b.rename("b")], axis=1, join="inner"
        ).replace([np.inf, -np.inf], np.nan).dropna()
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
) -> dict:
    """Formal Diebold–Mariano test on the loss differential ``d = L(a) - L(b)``.

    A **negative** statistic means model *a* has the lower loss (is better). The
    long-run variance of ``d`` is estimated by a Bartlett-kernel Newey–West HAC
    with lag ``max(horizon-1, floor(n**(1/3)))``; the statistic is then rescaled by
    the Harvey–Leybourne–Newbold (1997) finite-sample factor and referred to a
    Student-``t`` distribution with ``n-1`` degrees of freedom (two-sided p-value).

    Returns a dict with ``dm_stat`` (HLN-corrected), ``p_value``, ``mean_loss_diff``,
    ``n``, ``lag`` (backward-compatible keys) plus ``horizon``, ``hln`` and
    ``dm_stat_uncorrected`` for transparency.
    """
    yv, a, b = _align3(y, h_a, h_b)
    d = _loss_obs(yv, a, loss) - _loss_obs(yv, b, loss)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 3:
        raise ValueError("Diebold–Mariano needs at least 3 aligned observations")
    dbar = float(d.mean())
    h = max(int(horizon), 1)
    lag = max(h - 1, int(np.floor(n ** (1.0 / 3.0))))

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
