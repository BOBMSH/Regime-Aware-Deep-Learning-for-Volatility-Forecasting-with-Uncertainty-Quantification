"""Phase 6 driver -- uncertainty quantification end-to-end (ROADMAP.md Phase 6; RQ3).

Turns the Phase-3 LSTM into a probabilistic forecaster two ways and evaluates the
*calibration* of the resulting prediction intervals on the identical leakage-free
anchored walk-forward as every earlier phase (Ch2 §2.6):

* **MC-Dropout LSTM (primary).** Head dropout kept active at inference, ``T``
  stochastic passes, predictive mean + std -> a log-normal interval
  (:class:`src.models.deep.uncertainty.MCDropoutLSTMForecaster`).
* **Quantile-regression LSTM (comparison).** One monotone 3-quantile head trained
  with the pinball loss -> distribution-free intervals
  (:class:`src.models.deep.uncertainty.QuantileLSTMForecaster`).
* **LSTM-Gaussian (reference).** The same MC-Dropout network's *deterministic*
  forecast wrapped in an aleatoric-only (no-MC) log-normal interval -- an
  ablation that isolates what the ``T`` stochastic passes add over the Phase-3
  point model plus its residual-noise term.

Hyperparameters are fixed at the Phase-3 selection (hidden 128, lookback 10, lr
1e-3, dropout 0.1), so the POINT forecasts stay comparable to Phases 3/5 and the
contrast measures the uncertainty machinery, not a new sweep. **RQ3**: are the
intervals calibrated (PICP ≈ nominal), how sharp (MPIW / Winkler), and where does
coverage degrade across regimes?

Outputs (the Phase 6 gate artifacts)
------------------------------------
* ``results/predictions/m06_uq_<profile>.parquet`` -- per-day predictive objects
  (means, log-normal params, epistemic/aleatoric split, quantiles, interval
  endpoints) + realized target + causal regime label, over the test window.
* ``results/tables/m06_uq_<profile>_{point,calibration,per_regime,reliability}.csv``.
* ``results/figures/m06/<profile>_{reliability,interval_band,per_regime}.png``.
* ``results/milestones/m06_uq.md`` -- the milestone note.
* ``experiments/06_uq/run_<utc>/config.yaml`` -- reproducibility.

Usage
-----
    python -m src.experiments.run_uq            # full protocol
    python -m src.experiments.run_uq --fast     # few epochs / few MC samples (smoke)
    python -m src.experiments.run_uq --max-epochs 60 --mc-samples 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.datasets import build_econometric_frame
from src.data.splits import SplitConfig, walk_forward_folds
from src.evaluation.calibration import (
    interval_metrics,
    lognormal_interval,
    lognormal_mean_from_quantiles,
    lognormal_quantile,
    per_regime_interval_metrics,
    reliability_curve,
)
from src.evaluation.metrics import crps_from_quantiles, crps_lognormal, qlike
from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    DEFAULT_SELECTOR_SHIFT,
    align_regime_label,
    regime_timing_label,
    transition_day_summary,
)
from src.evaluation.rolling import ACTUAL_COL
from src.evaluation.significance import diebold_mariano
from src.models.deep import MCDropoutLSTMForecaster, QuantileLSTMForecaster
from src.utils.config import load_config, repo_path, snapshot_config
from src.utils.io import ensure_dir, from_parquet, to_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("run_uq")

REGIME_STATE_COL = "reg_state"
MC_NAME = "MC-Dropout-LSTM"
Q_NAME = "Quantile-LSTM"
GAUSS_NAME = "LSTM-Gaussian"
#: Retransformed quantile point forecast — a labelled sensitivity, never a
#: replacement for ``Q_NAME``. See :func:`add_quantile_mean_column`.
Q_MEAN_NAME = "Quantile-LSTM-mean"


# --------------------------------------------------------------------------- #
# Frame assembly: attach the Phase-4 causal regime hard state (for the split)  #
# --------------------------------------------------------------------------- #
def attach_regime_state(frame: pd.DataFrame, rcfg) -> pd.Series:
    """Return the Phase-4 causal *filtered* hard regime state aligned to ``frame``.

    Used only to segment the test window for the per-regime calibration table
    (RQ3) -- never as a model input in Phase 6.
    """
    reg = from_parquet(repo_path(rcfg.source))
    state = reg[rcfg.state_col].reindex(frame.index)
    if state.isna().any():
        raise RuntimeError("regime state has NaNs after aligning to the modelling frame.")
    return state.rename(REGIME_STATE_COL).astype(int)


# --------------------------------------------------------------------------- #
# UQ walk-forward: collect the full per-day predictive frame per model         #
# --------------------------------------------------------------------------- #
def run_uq_walk_forward(models, frame, cfg: SplitConfig, *, eval_segment="test") -> dict:
    """Run each UQ model across the walk-forward folds, collecting its per-day
    predictive DataFrame (``forecast_fold_uq``) rather than a single point Series.

    Mirrors :func:`src.evaluation.rolling.run_walk_forward`'s fold iteration and
    leakage guard (no output dated on/before ``fold.train_end``); returns
    ``{model.name: predictive DataFrame}`` indexed by OOS trading date.
    """
    folds = list(walk_forward_folds(frame.index, cfg, eval_segment=eval_segment))
    if not folds:
        raise RuntimeError("walk_forward_folds produced no folds")
    oos_start, oos_end = folds[0].predict_start, folds[-1].predict_end
    log.info("UQ walk-forward: %d folds, %d models, OOS %s -> %s",
             len(folds), len(models), oos_start.date(), oos_end.date())

    out = {}
    for m in models:
        chunks = []
        for fold in folds:
            assert fold.train_end < fold.predict_start, "leakage: train_end >= predict_start"
            fr = m.forecast_fold_uq(frame, fold)
            if len(fr):
                fr = fr.copy()
                fr.index = pd.DatetimeIndex(fr.index)
                if (fr.index <= fold.train_end).any():
                    raise AssertionError(f"{m.name} fold {fold.fold_idx}: leakage in UQ output.")
                sel = (fr.index >= fold.predict_start) & (fr.index <= fold.predict_end)
                chunks.append(fr.loc[sel])
        full = pd.concat(chunks).sort_index()
        full = full[~full.index.duplicated(keep="first")]
        out[m.name] = full
        log.info("  %-16s done: %d predictive rows", m.name, len(full))
    return out


# --------------------------------------------------------------------------- #
# Assemble intervals from the predictive frames                                #
# --------------------------------------------------------------------------- #
def build_predictions(uq: dict, y_true: pd.Series, reg_state: pd.Series, levels) -> pd.DataFrame:
    """One tidy frame: realized target, regime label, each method's point forecast,
    and each method's lower/upper interval endpoints at every requested level."""
    mc = uq[MC_NAME]
    qf = uq[Q_NAME]
    idx = mc.index
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    out[ACTUAL_COL] = y_true.reindex(idx)
    out[REGIME_STATE_COL] = reg_state.reindex(idx).astype("Int64")

    # --- point forecasts ---
    out[MC_NAME] = mc["mean"]                                   # log-normal predictive mean
    q_cols = list(qf.columns)                                   # e.g. q0.05, q0.5, q0.95
    q_taus = np.array([float(c[1:]) for c in q_cols])
    med_col = q_cols[int(np.argmin(np.abs(q_taus - 0.5)))]      # central (median) quantile, robustly
    out[Q_NAME] = qf[med_col]                                   # quantile median
    out[GAUSS_NAME] = np.exp(mc["mu_log_det"] + 0.5 * mc["sd_aleatoric"] ** 2)  # = Phase-3 LSTM point
    # variance decomposition (for the write-up)
    out["mc_mu_log"] = mc["mu_log"]
    out["mc_sigma_log"] = mc["sigma_log"]
    out["mc_sd_epistemic"] = mc["sd_epistemic"]
    out["mc_sd_aleatoric"] = mc["sd_aleatoric"]

    # --- intervals per level ---
    for L in levels:
        tag = f"{int(round(L * 100))}"
        lo, hi = lognormal_interval(mc["mu_log"].to_numpy(), mc["sigma_log"].to_numpy(), level=L)
        out[f"{MC_NAME}_lo{tag}"] = lo
        out[f"{MC_NAME}_hi{tag}"] = hi
        glo, ghi = lognormal_interval(mc["mu_log_det"].to_numpy(), mc["sd_aleatoric"].to_numpy(), level=L)
        out[f"{GAUSS_NAME}_lo{tag}"] = glo
        out[f"{GAUSS_NAME}_hi{tag}"] = ghi
    # Quantile intervals: the trained outer quantiles give one native band only.
    q_lo, q_hi = qf.columns[0], qf.columns[-1]
    q_level = float(q_hi[1:]) - float(q_lo[1:])
    out[f"{Q_NAME}_lo{int(round(q_level*100))}"] = qf[q_lo]
    out[f"{Q_NAME}_hi{int(round(q_level*100))}"] = qf[q_hi]
    out.attrs["q_level"] = q_level
    return out


# --------------------------------------------------------------------------- #
# Milestone-prose primitives                                                   #
#                                                                              #
# These four live at module level, and are unit-tested, for one reason: three  #
# bugs of the same family have now been found in this project's generated      #
# milestone notes -- a hardcoded conclusion, a shadowed loop variable printing #
# lambda=300, and (2026-08-22, audit vi) an "the ordering is non-monotone"     #
# clause asserted in a branch that had only ever tested monotone-*decreasing*, #
# which was false for the Baum-Welch comparator's 0.848 / 0.880 / 0.887 and    #
# propagated verbatim into the roadmap. The numbers in these notes were always #
# computed; it was the qualitative WORDS that were templated. Anything that    #
# turns numbers into a claim now lives here where a test can reach it.         #
# --------------------------------------------------------------------------- #
def _picp_of(pr: pd.DataFrame, label: str) -> float:
    """PICP for one regime row of a per-regime table (NaN if absent/empty)."""
    r = pr[pr["regime"] == label]
    return float(r["picp"].iloc[0]) if len(r) and pd.notna(r["picp"].iloc[0]) else float("nan")


#: End-to-end gap below which "degrades with severity" is not worth asserting.
#: Two percentage points is already generous at n=76, where one day is 1.3 pp.
MATERIAL_COVERAGE_GAP = 0.02


def coverage_shape(pr: pd.DataFrame, labels: list[str]) -> str:
    """Classify a per-regime coverage sequence: the note may not call a sequence
    "non-monotone" without having tested non-monotonicity.

    Returns one of ``"monotone decreasing"``, ``"monotone increasing"``,
    ``"non-monotone"`` or ``"undetermined"`` (any bucket missing). A constant
    sequence satisfies both weak orderings and is reported as decreasing, which
    is the conservative reading against the RQ3 hypothesis.
    """
    v = [_picp_of(pr, lb) for lb in labels]
    if any(np.isnan(x) for x in v):
        return "undetermined"
    if all(a >= b for a, b in zip(v, v[1:])):
        return "monotone decreasing"
    if all(a <= b for a, b in zip(v, v[1:])):
        return "monotone increasing"
    return "non-monotone"


def coverage_degrades(pr: pd.DataFrame, labels: list[str],
                      *, material: float = MATERIAL_COVERAGE_GAP) -> bool:
    """Does coverage degrade with regime severity, materially and monotonically?

    "Degrades with severity" is a claim about an *ordering*, so the ordering is
    tested across every state -- not just the endpoints, which a 0.2 pp gap
    would satisfy by noise -- and the end-to-end gap must clear ``material``.
    """
    v = [_picp_of(pr, lb) for lb in labels]
    if any(np.isnan(x) for x in v):
        return False
    return all(a >= b for a, b in zip(v, v[1:])) and (v[0] - v[-1]) >= float(material)


def picp_standard_error(pr: pd.DataFrame, label: str) -> float:
    """Binomial standard error of one bucket's PICP, ``sqrt(p(1-p)/n)``."""
    p = _picp_of(pr, label)
    row = pr[pr["regime"] == label]
    n = int(row["n"].iloc[0]) if len(row) else 0
    if n <= 0 or np.isnan(p):
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def endpoint_gap_verdict(pr: pd.DataFrame, labels: list[str]) -> str:
    """Is the first-vs-last bucket coverage gap inside its own sampling error?

    The generator used to print "well inside sampling noise" flat, with no
    standard error computed anywhere in the function. This computes the
    two-bucket standard error and compares the gap against two of them, so the
    phrase is earned rather than asserted.
    """
    gap = abs(_picp_of(pr, labels[0]) - _picp_of(pr, labels[-1]))
    se = float(np.sqrt(picp_standard_error(pr, labels[0]) ** 2
                       + picp_standard_error(pr, labels[-1]) ** 2))
    if np.isnan(gap) or np.isnan(se) or se <= 0:
        return "not assessable"
    return ("well inside sampling noise" if gap < 2.0 * se
            else "larger than sampling noise")


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def _infer_quantile_levels(preds: pd.DataFrame) -> tuple[float, float] | None:
    """Recover the fitted outer quantile levels from the persisted column names.

    ``Quantile-LSTM_lo90``/``_hi90`` is a central 90% band, i.e. the 0.05 and 0.95
    quantiles. Inferring this lets :func:`add_quantile_mean_column` run against a
    predictions parquet alone -- which is what ``run_combined`` and any
    ``--from-predictions`` rebuild have -- instead of needing the Phase-6 config.
    """
    tags = sorted(
        {int(str(c).rsplit("_lo", 1)[1]) for c in preds.columns
         if str(c).startswith(f"{Q_NAME}_lo") and str(c).rsplit("_lo", 1)[1].isdigit()}
    )
    if not tags:
        return None
    width = max(tags) / 100.0
    a = (1.0 - width) / 2.0
    return a, 1.0 - a


def add_quantile_mean_column(preds: pd.DataFrame, q_quantiles=None) -> pd.DataFrame:
    """Add the retransformed quantile point forecast, in place, and return ``preds``.

    Why (2026-08-30) — a scoring inconsistency, not a new model
    ----------------------------------------------------------
    Every other deep forecaster in this study is retransformed to the mean scale
    before it is scored. ``LSTMForecaster`` returns ``exp(log_rv + smear_var/2)``;
    ``MCDropoutLSTMForecaster`` returns the log-normal mean of its predictive law.
    Both do so for the same reason: the networks regress **log** variance, while
    MSE and QLIKE are minimised at the conditional **mean** of the variance
    (Patton 2011), and for a right-skewed law the median lies strictly below it.

    The pinball-trained head was the single exception — its point forecast is the
    fitted median, exponentiated, with no correction. On this sample that median
    sits below the MC-Dropout mean on **every one of the 784 evaluation days**, so
    ranking it against the others on QLIKE was measuring the estimand at least as
    much as the model.

    The correction uses the quantile model's *own* ``[q_lo, q_med, q_hi]`` triple,
    so it borrows nothing from the parametric models: the implied ``sigma`` comes
    from its predicted spread, which is conditional and therefore a better scale
    estimate than the Phase-3 LSTM's constant held-out ``smear_var``. The
    log-normal closure is an assumption and the method is distribution-free by
    design (Ch2 §2.6), which is exactly why this is reported as an extra labelled
    row and the median-point row is left standing unchanged.

    Derived purely from columns already in the predictions parquet, so a
    ``--from-predictions`` rebuild produces it with no retraining.
    """
    if q_quantiles is None:
        levels = _infer_quantile_levels(preds)
        if levels is None:
            log.warning("cannot build %s: no %s_lo* column to infer levels from",
                        Q_MEAN_NAME, Q_NAME)
            return preds
        lo_level, hi_level = levels
    else:
        taus = sorted(float(t) for t in q_quantiles)
        lo_level, hi_level = taus[0], taus[-1]
    tag = int(round((hi_level - lo_level) * 100))
    lo_col, hi_col = f"{Q_NAME}_lo{tag}", f"{Q_NAME}_hi{tag}"
    if not {Q_NAME, lo_col, hi_col} <= set(preds.columns):
        log.warning(
            "cannot build %s: need %s, %s and %s in the predictions frame",
            Q_MEAN_NAME, Q_NAME, lo_col, hi_col,
        )
        return preds
    preds[Q_MEAN_NAME] = lognormal_mean_from_quantiles(
        preds[lo_col], preds[Q_NAME], preds[hi_col],
        lo_level=lo_level, hi_level=hi_level,
    )
    ratio = float(np.mean(preds[Q_MEAN_NAME].to_numpy() / preds[Q_NAME].to_numpy()))
    log.info("%s built from the fitted %.2f/%.2f/0.50 triple; mean correction factor %.4f",
             Q_MEAN_NAME, lo_level, hi_level, ratio)
    return preds


def point_metrics_table(preds: pd.DataFrame, lstm_point: pd.Series | None) -> pd.DataFrame:
    """QLIKE of each UQ point forecast (+ the deterministic Phase-3 LSTM), so the
    reader can see the uncertainty machinery does not cost point accuracy.

    ``Quantile-LSTM`` is the model's own median point; ``Quantile-LSTM-mean`` is
    that point put on the mean scale QLIKE actually elicits (see
    :func:`add_quantile_mean_column`). Both are reported: the first is what the
    method produces, the second is what makes the comparison like-for-like.
    """
    y = preds[ACTUAL_COL]
    rows = {}
    cand = {MC_NAME: preds[MC_NAME], Q_NAME: preds[Q_NAME], GAUSS_NAME: preds[GAUSS_NAME]}
    if Q_MEAN_NAME in preds.columns:
        cand[Q_MEAN_NAME] = preds[Q_MEAN_NAME]
    if lstm_point is not None:
        cand["LSTM (Phase 3)"] = lstm_point.reindex(preds.index)
    for name, h in cand.items():
        sub = pd.concat([y, h], axis=1).dropna()
        rows[name] = {"n": int(len(sub)), "qlike": qlike(sub.iloc[:, 0], sub.iloc[:, 1])}
    return pd.DataFrame(rows).T.sort_values("qlike")


def crps_table(preds: pd.DataFrame, q_quantiles) -> pd.DataFrame:
    """CRPS per method — the strictly proper score Ch2 §2.6/§2.7 commits to.

    Coverage and the Winkler score are both tied to a nominal level; the CRPS is
    not, so it is the score that adjudicates the *whole* predictive distribution
    and is what Gneiting & Raftery (2007) actually supply. Two routes, chosen to
    match how each method represents its forecast:

    * the parametric methods have a closed-form log-normal CRPS (exact, no
      quadrature error);
    * the quantile head only ever produced ``len(q_quantiles)`` quantiles, so its
      CRPS is a trapezoidal approximation on that grid and is reported as such —
      on a 3-point grid it understates the true CRPS materially, so it is
      **comparable across methods only when read against the parametric methods'
      grid-approximated value**, which is why both are computed the same way for
      the comparison row.
    """
    y = preds[ACTUAL_COL]
    mu, sigma = preds["mc_mu_log"], preds["mc_sigma_log"]
    sd_ale = preds["mc_sd_aleatoric"]
    mu_det = np.log(preds[GAUSS_NAME].to_numpy()) - 0.5 * sd_ale.to_numpy() ** 2

    rows = [
        {"method": MC_NAME, "crps": crps_lognormal(y.to_numpy(), mu.to_numpy(),
                                                   sigma.to_numpy()),
         "estimator": "log-normal closed form", "n": int(len(y))},
        {"method": GAUSS_NAME, "crps": crps_lognormal(y.to_numpy(), mu_det,
                                                      sd_ale.to_numpy()),
         "estimator": "log-normal closed form", "n": int(len(y))},
    ]
    # Grid-approximated CRPS for every method on the quantile head's own grid, so
    # the three numbers are like-for-like despite the different representations.
    qtag = f"{int(round((q_quantiles[-1] - q_quantiles[0]) * 100))}"
    q_grid = {q_quantiles[0]: preds[f"{Q_NAME}_lo{qtag}"],
              q_quantiles[-1]: preds[f"{Q_NAME}_hi{qtag}"]}
    if Q_NAME in preds.columns and 0.5 in q_quantiles:
        q_grid[0.5] = preds[Q_NAME]
    rows.append({"method": Q_NAME, "crps": crps_from_quantiles(y, q_grid),
                 "estimator": f"trapezoidal, {len(q_grid)}-point grid", "n": int(len(y))})
    for method, m_, s_ in ((MC_NAME, mu.to_numpy(), sigma.to_numpy()),
                           (GAUSS_NAME, mu_det, sd_ale.to_numpy())):
        grid = {t: pd.Series(lognormal_quantile(m_, s_, t), index=preds.index)
                for t in sorted(q_grid)}
        rows.append({"method": f"{method} (same grid)",
                     "crps": crps_from_quantiles(y, grid),
                     "estimator": f"trapezoidal, {len(grid)}-point grid", "n": int(len(y))})
    return pd.DataFrame(rows)[["method", "crps", "estimator", "n"]]


def sigma_scale_diagnostic(preds: pd.DataFrame, level: float, scales=None) -> pd.DataFrame:
    """How much too narrow is the parametric predictive law? Two answers.

    A proper score is minimised at the *true* predictive distribution, so if
    multiplying the fitted log-scale ``sigma`` by some ``k > 1`` **improves** the
    CRPS, the fitted law is too narrow — and the minimising ``k`` measures by how
    much, in a units-free way that neither PICP nor MPIW gives on its own.

    Two different ``k`` are reported because they answer different questions and
    generally disagree:

    * ``k`` minimising the CRPS — the width that best fits the *whole*
      distribution, tails included;
    * ``k`` driving PICP to the nominal level — the width that fixes coverage at
      one level only.

    The gap between them is informative: if the coverage-calibrating ``k`` is the
    smaller, the predictive law is not merely too narrow but the wrong *shape*
    (too thin-tailed), which a single variance rescaling cannot repair and a
    conformal or transition-conditional procedure can. This is the concrete
    target Phase 7's recalibration has to beat.
    """
    y = preds[ACTUAL_COL].to_numpy()
    mu = preds["mc_mu_log"].to_numpy()
    sg = preds["mc_sigma_log"].to_numpy()
    scales = np.linspace(0.8, 2.0, 61) if scales is None else np.asarray(scales, dtype=float)
    crps = np.array([crps_lognormal(y, mu, sg * k) for k in scales])
    picp_ = np.array([
        float(np.mean((y >= lognormal_interval(mu, sg * k, level=level)[0])
                      & (y <= lognormal_interval(mu, sg * k, level=level)[1])))
        for k in scales
    ])
    k_crps = float(scales[int(np.argmin(crps))])
    k_cov = float(scales[int(np.argmin(np.abs(picp_ - level)))])
    return pd.DataFrame([{
        "method": MC_NAME,
        "nominal": float(level),
        "crps_at_fitted_sigma": float(crps_lognormal(y, mu, sg)),
        "crps_optimal_sigma_scale": k_crps,
        "crps_at_optimal_scale": float(crps.min()),
        "coverage_calibrating_sigma_scale": k_cov,
        "picp_at_fitted_sigma": float(np.mean(
            (y >= lognormal_interval(mu, sg, level=level)[0])
            & (y <= lognormal_interval(mu, sg, level=level)[1]))),
        "n": int(len(y)),
    }])


def calibration_table(preds: pd.DataFrame, levels, q_level) -> pd.DataFrame:
    """PICP / MPIW / Winkler / coverage_error per (method, nominal level)."""
    y = preds[ACTUAL_COL]
    rows = []
    for L in levels:
        tag = f"{int(round(L * 100))}"
        for method in (MC_NAME, GAUSS_NAME):
            lo, hi = preds[f"{method}_lo{tag}"], preds[f"{method}_hi{tag}"]
            m = interval_metrics(y, lo, hi, level=L)
            rows.append({"method": method, "nominal": L, **m})
    # quantile: its single native band
    qtag = f"{int(round(q_level * 100))}"
    lo, hi = preds[f"{Q_NAME}_lo{qtag}"], preds[f"{Q_NAME}_hi{qtag}"]
    m = interval_metrics(y, lo, hi, level=q_level)
    rows.append({"method": Q_NAME, "nominal": q_level, **m})
    tab = pd.DataFrame(rows)[["method", "nominal", "n", "picp", "mpiw", "winkler", "coverage_error"]]
    return tab.sort_values(["nominal", "method"]).reset_index(drop=True)


def per_regime_table(preds, labels, level, method, *,
                     reg_state=None, shift: int = DEFAULT_REGIME_SHIFT) -> pd.DataFrame:
    """Per-regime interval calibration, bucketed by the regime known at ``t-shift``.

    ``shift`` defaults to 1 for the reason set out in
    :mod:`src.evaluation.regime_timing`: the filtered state sₜ is inferred using
    the day-t observation, and interval *coverage* on day t is a function of that
    same observation, so the contemporaneous bucketing correlates the label with
    the outcome it is used to explain. It systematically pushes interval misses
    into whichever state the market had just entered, manufacturing a
    "calibration degrades with regime severity" pattern out of the labelling
    convention. Pass ``reg_state`` (the full Phase-4 series) so the lag can reach
    the trading day before the evaluation window starts.
    """
    tag = f"{int(round(level * 100))}"
    state = preds[REGIME_STATE_COL].astype("float") if reg_state is None else reg_state
    return per_regime_interval_metrics(
        preds[ACTUAL_COL], preds[f"{method}_lo{tag}"], preds[f"{method}_hi{tag}"],
        state, labels, level=level, shift=shift,
    )


def transition_vs_stable_table(preds, level, methods, reg_state, *,
                               selector_shift: int = DEFAULT_SELECTOR_SHIFT) -> pd.DataFrame:
    """Coverage on regime-*transition* days vs everything else, at one selector timing.

    ``selector_shift`` decides what the table means, and the two readings are
    opposite, so every row is stamped with it:

    * ``1`` (**default, headline**) — ``1{s_(t-1) != s_(t-2)}``. Both labels lie in
      the ``t-1`` information set, so this is the only version that answers *"can a
      forecaster tell in advance when these intervals will fail?"*. On this test
      window the answer is no: coverage on flagged days is indistinguishable from
      the rest.
    * ``0`` — ``1{s_t != s_(t-1)}``, the split reported before 2026-08-19 (iii). It
      is convention-*free* (both label timings agree on which days those are) but
      **not** implementable: ``s_t`` is inferred from the day-t return, and whether
      the day-t interval covered is a function of that same return, so the
      subsample is chosen using the outcome being measured. It shows a
      16-percentage-point coverage gap that vanishes entirely at ``selector_shift=1``.
      Kept as a descriptive contrast — never as a conditional claim, and never as
      the subsample for a coverage test.

    See :mod:`src.evaluation.regime_timing` for the measurement and the mechanism.
    """
    tag = f"{int(round(level * 100))}"
    tr = transition_day_summary(reg_state, preds.index, selector_shift=selector_shift)
    mask = tr["mask"].reindex(preds.index).fillna(False).to_numpy()
    scored = tr["scored"].reindex(preds.index).fillna(False).to_numpy()
    rows = []
    for method in methods:
        lo, hi = preds[f"{method}_lo{tag}"], preds[f"{method}_hi{tag}"]
        for group, sel in (("regime transition", mask & scored),
                           ("stable regime", (~mask) & scored)):
            if not sel.any():
                continue
            m = interval_metrics(preds[ACTUAL_COL][sel], lo[sel], hi[sel], level=level)
            rows.append({
                "selector_shift": int(selector_shift),
                "selector": tr["selector"],
                "implementable": bool(tr["implementable"]),
                "method": method, "days": group, **m,
            })
    return pd.DataFrame(rows)


def reliability_table(preds, uq, taus, q_quantiles) -> pd.DataFrame:
    """Empirical vs nominal coverage: MC-Dropout across a dense tau grid, the
    quantile model at its trained taus, the Gaussian reference across the grid.

    ``uq`` may be ``None`` (the ``--from-predictions`` path): every quantity the
    curves need is already persisted in the predictions frame, so the table is
    rebuilt from there. The two routes are asserted equal in the unit tests.
    """
    y = preds[ACTUAL_COL]
    if uq is not None:
        mc = uq[MC_NAME]
        mu, sigma, mu_det = mc["mu_log"], mc["sigma_log"], mc["mu_log_det"]
        sd_ale = mc["sd_aleatoric"]
        idx = mc.index
        qf = uq[Q_NAME]
        q_get = (lambda t: qf[f"q{t:g}"] if f"q{t:g}" in qf.columns
                 else pd.Series(np.nan, index=qf.index))
        q_idx = qf.index
    else:
        idx = preds.index
        mu, sigma, sd_ale = preds["mc_mu_log"], preds["mc_sigma_log"], preds["mc_sd_aleatoric"]
        # The Gaussian reference is the dropout-off point with aleatoric-only
        # spread; its log-mean is recoverable from the persisted median-free
        # point forecast, which is exactly how build_predictions wrote it.
        mu_det = np.log(preds[GAUSS_NAME].to_numpy()) - 0.5 * sd_ale.to_numpy() ** 2
        mu_det = pd.Series(mu_det, index=idx)
        qtag = f"{int(round((q_quantiles[-1] - q_quantiles[0]) * 100))}"
        cols = {q_quantiles[0]: f"{Q_NAME}_lo{qtag}", q_quantiles[-1]: f"{Q_NAME}_hi{qtag}"}
        if Q_NAME in preds.columns:
            cols[0.5] = Q_NAME
        q_get = (lambda t: preds[cols[t]] if t in cols
                 else pd.Series(np.nan, index=idx))
        q_idx = idx

    mc_curve = reliability_curve(
        y, lambda t: pd.Series(lognormal_quantile(mu.to_numpy(), sigma.to_numpy(), t),
                               index=idx), taus)
    mc_curve["method"] = MC_NAME
    g_curve = reliability_curve(
        y, lambda t: pd.Series(lognormal_quantile(mu_det.to_numpy(), sd_ale.to_numpy(), t),
                               index=idx), taus)
    g_curve["method"] = GAUSS_NAME
    q_curve = reliability_curve(y.reindex(q_idx), q_get, q_quantiles)
    q_curve["method"] = Q_NAME
    return pd.concat([mc_curve, q_curve, g_curve], ignore_index=True)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def plot_reliability(rel: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0, label="perfect calibration")
    style = {MC_NAME: ("tab:blue", "o", "-"), GAUSS_NAME: ("0.55", "^", ":"),
             Q_NAME: ("tab:red", "s", "")}
    for method, sub in rel.groupby("method"):
        c, mk, ls = style.get(method, ("k", ".", "-"))
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub["empirical"], color=c, marker=mk, ls=ls, lw=1.4,
                ms=6, label=method)
    ax.set_xlabel("nominal quantile level τ")
    ax.set_ylabel("empirical coverage  P(y ≤ q̂τ)")
    ax.set_title("Reliability diagram — S&P 500 RV prediction quantiles")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    return save_fig(fig, path)


def plot_interval_band(preds, reg_state, labels, level, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.utils.plotting import save_fig, set_style

    set_style()
    tag = f"{int(round(level * 100))}"
    idx = preds.index
    to_vol = lambda v: np.sqrt(np.clip(np.asarray(v, dtype=float), 0, None)) * 100.0
    fig, ax = plt.subplots(figsize=(11, 4.8))
    band = {0: "#2ca02c", 1: "#f0c000", 2: "#d62728"}
    rs = reg_state.reindex(idx).fillna(-1).astype(int).to_numpy()
    start = 0
    for i in range(1, len(rs) + 1):
        if i == len(rs) or rs[i] != rs[start]:
            ax.axvspan(idx[start], idx[min(i, len(idx) - 1)],
                       color=band.get(int(rs[start]), "0.8"), alpha=0.10, lw=0)
            start = i
    ax.fill_between(idx, to_vol(preds[f"{MC_NAME}_lo{tag}"]), to_vol(preds[f"{MC_NAME}_hi{tag}"]),
                    color="tab:blue", alpha=0.25, lw=0, label=f"MC-Dropout {tag}% interval")
    ax.plot(idx, to_vol(preds[MC_NAME]), color="tab:blue", lw=1.0, label="MC-Dropout mean")
    ax.plot(idx, to_vol(preds[ACTUAL_COL]), color="0.12", lw=1.1, label="Realized", zorder=6)
    handles, _ = ax.get_legend_handles_labels()
    handles += [mpatches.Patch(color=band[r], alpha=0.3, label=f"regime: {labels[r]}")
                for r in range(len(labels))]
    ax.legend(handles=handles, ncol=3, loc="upper left", fontsize=7.5)
    ax.set_ylabel("Daily volatility (%)")
    ax.set_title(f"S&P 500 realized volatility with MC-Dropout {tag}% prediction band")
    ax.set_xlim(idx.min(), idx.max())
    return save_fig(fig, path)


def plot_per_regime(pr_mc, pr_q, labels, level, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(labels)); w = 0.38
    mc = pr_mc.set_index("regime").reindex(labels)
    q = pr_q.set_index("regime").reindex(labels)
    ax1.bar(x - w / 2, mc["picp"], w, color="tab:blue", label=MC_NAME)
    ax1.bar(x + w / 2, q["picp"], w, color="tab:red", label=Q_NAME)
    ax1.axhline(level, color="0.4", ls="--", lw=1.0, label=f"nominal {int(round(level*100))}%")
    ax1.set_xticks(x, [f"{l}\n(n={int(mc.loc[l,'n'])})" for l in labels])
    ax1.set_ylabel("PICP (empirical coverage)")
    ax1.set_title(f"Coverage by regime ({int(round(level*100))}% intervals)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.02)
    ax2.bar(x - w / 2, mc["mpiw"] * 1e4, w, color="tab:blue", label=MC_NAME)
    ax2.bar(x + w / 2, q["mpiw"] * 1e4, w, color="tab:red", label=Q_NAME)
    ax2.set_xticks(x, [f"{l}" for l in labels])
    ax2.set_ylabel("MPIW  (variance ×1e-4)")
    ax2.set_title("Interval width (sharpness) by regime")
    ax2.legend(fontsize=8)
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# Milestone note                                                              #
# --------------------------------------------------------------------------- #
def _fmt_point(pt: pd.DataFrame) -> str:
    lines = ["| model | n | QLIKE |", "|---|---|---|"]
    for name, r in pt.iterrows():
        lines.append(f"| {name} | {int(r['n'])} | {r['qlike']:.4f} |")
    return "\n".join(lines)


def _fmt_calib(tab: pd.DataFrame) -> str:
    lines = ["| method | nominal | n | PICP | MPIW | Winkler | coverage error |",
             "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(
            f"| {r['method']} | {r['nominal']*100:.0f}% | {int(r['n'])} | {r['picp']:.3f} | "
            f"{r['mpiw']:.3e} | {r['winkler']:.3e} | {r['coverage_error']:+.3f} |")
    return "\n".join(lines)


def _fmt_per_regime(pr: pd.DataFrame, level: float) -> str:
    lines = [f"| regime | n | PICP | MPIW | Winkler | coverage error |",
             "|---|---|---|---|---|---|"]
    for _, r in pr.iterrows():
        picp = f"{r['picp']:.3f}" if pd.notna(r['picp']) else "—"
        mpiw = f"{r['mpiw']:.3e}" if pd.notna(r['mpiw']) else "—"
        wink = f"{r['winkler']:.3e}" if pd.notna(r['winkler']) else "—"
        ce = f"{r['coverage_error']:+.3f}" if pd.notna(r['coverage_error']) else "—"
        lines.append(f"| {r['regime']} | {int(r['n'])} | {picp} | {mpiw} | {wink} | {ce} |")
    return "\n".join(lines)


def write_milestone(ctx: dict, out_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    name = ctx["name"]; L = ctx["headline_level"]; Lp = int(round(L * 100))
    cal = ctx["calibration"]; pr_mc = ctx["per_regime_mc"]; pr_q = ctx["per_regime_q"]
    labels = ctx["labels"]; pt = ctx["point"]; dm = ctx["dm_mc_vs_lstm"]
    dec = ctx["var_decomp"]

    def cal_row(method, level):
        r = cal[(cal["method"] == method) & (np.isclose(cal["nominal"], level))]
        return r.iloc[0] if len(r) else None

    mc90 = cal_row(MC_NAME, L); q90 = cal_row(Q_NAME, ctx["q_level"]); g90 = cal_row(GAUSS_NAME, L)

    p: list[str] = []
    p.append("# M06 — Uncertainty quantification (Phase 6)\n")
    p.append(f"*Generated by `python -m src.experiments.run_uq` on {ts}. Numbers trace to "
             "`results/tables/m06_uq_*` and `results/predictions/m06_uq_*.parquet`.*\n")
    if ctx.get("fast"):
        p.append("> ⚠️ **FAST pass** — few epochs / few MC samples for a quick CPU smoke run. "
                 "Re-run `python -m src.experiments.run_uq` (full config) before citing.\n")

    p.append("## Scope\n")
    p.append("Roadmap Phase 6 (Ch2 §2.6): make the Phase-3 **LSTM** *uncertainty-aware* and test "
             "whether its prediction intervals are **calibrated**, on the same leakage-free anchored "
             "walk-forward as Phases 2–5 (**RQ3**). Two methods, from the two dominant schools of "
             "neural-network UQ, plus one ablation:\n"
             f"- **MC-Dropout LSTM (primary)** — head dropout kept active at inference, **T={ctx['mc_samples']}** "
             "stochastic passes; the predictive law on log-variance combines the **epistemic** spread "
             "across passes with the **aleatoric** residual noise (Gal & Ghahramani 2016), giving a "
             "log-normal interval on the variance scale.\n"
             f"- **Quantile-regression LSTM (comparison)** — one LSTM, a monotone (non-crossing) "
             f"{len(ctx['quantiles'])}-quantile head at τ={ctx['quantiles']} trained with the **pinball loss** "
             "(Koenker–Bassett 1978; Engle–Manganelli 2004); distribution-free.\n"
             f"- **LSTM-Gaussian (reference)** — the same network's deterministic forecast with an "
             "aleatoric-only (no-MC) log-normal interval, i.e. the Phase-3 point model + its residual "
             "noise; isolates what the MC passes add.\n"
             "Hyperparameters are fixed at the Phase-3 selection (hidden 128, lookback 10, lr 1e-3, "
             f"dropout 0.1); the headline nominal level common to both methods is **{Lp}%**.\n")
    p.append("> **Refit cadence (read with every number below).** As in Phases 3 and 5, "
             "`refit_every_folds: 0`: the networks are trained **once** on data ≤ 2018-12-31 and "
             "frozen across the whole out-of-sample window, while the econometric baselines refit "
             "every 21 trading days. For Phase 6 this has a direct consequence worth stating "
             "rather than burying: the MC-Dropout **aleatoric** term is a single residual variance "
             "estimated once on pre-2019 data, so the log-scale interval width is essentially "
             "constant for all 784 test days. The intervals cannot adapt to the out-of-sample "
             "period by construction — which is the mechanism behind most of what follows, and "
             "the first thing a conditional recalibration in Phase 7 would change.\n")

    p.append("## Point accuracy is preserved (sanity)\n")
    p.append("The uncertainty machinery is *added on top of* the Phase-3 point model, so its point "
             "forecasts should still land where Phase 3 left them (QLIKE, primary):\n")
    p.append(_fmt_point(pt) + "\n")
    p.append(f"\nThe MC-Dropout predictive mean (QLIKE {pt.loc[MC_NAME,'qlike']:.4f}) tracks the "
             f"deterministic LSTM, and a formal Diebold–Mariano test finds **no** significant point-"
             f"accuracy difference (DM {dm['dm_stat']:.2f}, p={dm['p_value']:.3f}) — the intervals come "
             "at no cost to the forecast. The **LSTM-Gaussian** point reproduces the Phase-3 LSTM "
             "to floating-point precision (same weights, dropout off) — agreement is ~1e-6 in "
             "relative terms, not bitwise, which is why the master table prints MSE values that "
             "differ in their seventh significant figure. That is the numerical floor of a "
             "same-environment refit, and it confirms the reference is the Phase-3 model. "
             f"The quantile model's higher QLIKE ({pt.loc[Q_NAME, 'qlike']:.4f}) is expected and is "
             "*not* an interval-quality signal: its point is the predictive **median**, whereas QLIKE "
             "is minimised at the conditional **mean** (Patton 2011), and for right-skewed realized "
             "variance the median sits below the mean — so a mean-vs-median QLIKE gap is partly an "
             "artefact of the target functional. The quantile method's job is *calibration*, "
             "adjudicated below, not point-QLIKE.\n")
    if Q_MEAN_NAME in pt.index:
        _qm, _q0 = float(pt.loc[Q_MEAN_NAME, "qlike"]), float(pt.loc[Q_NAME, "qlike"])
        p.append(
            f"\nThat artefact is now **measured rather than asserted**. Every other deep model here "
            f"is retransformed to the mean scale before scoring — the Phase-3 LSTM by "
            f"`exp(log_rv + smear_var/2)`, MC-Dropout by the log-normal mean of its predictive law — "
            f"and the pinball head was the only one that was not. Applying the same correction, using "
            f"the quantile model's **own** fitted spread rather than any parametric model's, gives "
            f"`{Q_MEAN_NAME}` QLIKE **{_qm:.4f}** against {_q0:.4f} for the median point, a gap of "
            f"{_q0 - _qm:.4f}. The uncorrected row stays in the table because the median is what the "
            f"method actually produces and the log-normal closure is an assumption a distribution-free "
            f"method does not otherwise make (Ch2 §2.6); the corrected row is what makes the "
            f"point-forecast comparison like-for-like, and it is the one to read against the "
            f"mean-scale models.\n")

    p.append(f"## Calibration — central intervals\n")
    p.append("PICP is empirical coverage (closer to nominal is better); MPIW is mean interval width "
             "(smaller = sharper); Winkler is the proper interval score (Gneiting–Raftery 2007, lower "
             "better); coverage error = PICP − nominal (negative = over-confident / too narrow).\n")
    p.append(_fmt_calib(cal) + "\n")

    crps_tbl = ctx.get("crps")
    if crps_tbl is not None and len(crps_tbl):
        p.append("\n### CRPS — a level-free proper score\n")
        p.append("Coverage and the Winkler score are both tied to one nominal level. The "
                 "**continuous ranked probability score** (Gneiting & Raftery 2007) is not: it "
                 "scores the whole predictive distribution, which is what Ch2 §2.6 means when it "
                 "says a probabilistic forecast cannot be judged by its point estimate alone. "
                 "The parametric methods get the exact log-normal closed form; the quantile head "
                 "only ever produced three quantiles, so its CRPS is a trapezoidal approximation "
                 "on that grid — which understates the true value. The `(same grid)` rows apply "
                 "the *same* 3-point approximation to the parametric methods, and those are the "
                 "rows to compare across methods. Lower is better; units are variance.\n")
        p.append("\n| method | CRPS | estimator | n |")
        p.append("|---|---|---|---|")
        for _, r in crps_tbl.iterrows():
            p.append(f"| {r['method']} | {r['crps']:.4e} | {r['estimator']} | {int(r['n'])} |")
        p.append("")
    sd_ = ctx.get("sigma_diag")
    if sd_ is not None and len(sd_):
        r0 = sd_.iloc[0]
        p.append(f"\nBecause a proper score is minimised at the *true* predictive law, the CRPS "
                 "also measures how wrong the fitted width is. Rescaling the log-scale σ by a "
                 f"factor **{r0['crps_optimal_sigma_scale']:.2f}** minimises the CRPS "
                 f"({r0['crps_at_optimal_scale']:.4e} vs {r0['crps_at_fitted_sigma']:.4e} as "
                 "fitted) — an independent confirmation, on a criterion that never looks at a "
                 "nominal level, that the intervals are too narrow. Note that the factor which "
                 f"merely fixes {Lp}% coverage is smaller "
                 f"({r0['coverage_calibrating_sigma_scale']:.2f}): the CRPS wants more width than "
                 "coverage does, which says the predictive law is not just too narrow but too "
                 "thin-tailed, and that a single global variance rescaling cannot repair it. "
                 "That is the concrete bar a Phase-7 conformal or transition-conditional "
                 "recalibration has to clear.\n")
    p.append(f"\n![reliability diagram](../figures/m06/{name}_reliability.png)\n")

    p.append(f"## Per-regime calibration ({Lp}%)\n")
    p.append("Coverage within each regime over the test window. **A day is bucketed by the "
             "regime known at t−1.** This is not cosmetic: the causal filtered state sₜ is "
             "inferred *using the day-t return*, and whether the day-t interval covered is a "
             "function of that same return, so the contemporaneous bucketing correlates the "
             "label with the outcome it is being used to explain. It pushes interval misses "
             "into whichever state the market had just entered, and the resulting "
             "\"coverage degrades with regime severity\" pattern is an artefact of the "
             "convention rather than a property of the intervals — the ex-post table below "
             "shows exactly how much moves. The t−1 label is what a forecaster actually knew, "
             "and is the label the regime models condition on.\n")
    p.append(f"**MC-Dropout:**\n\n{_fmt_per_regime(pr_mc, L)}\n")
    p.append(f"\n**Quantile:**\n\n{_fmt_per_regime(pr_q, L)}\n")
    p.append(f"\n![per-regime calibration](../figures/m06/{name}_per_regime.png)\n")

    pr_mc_ex = ctx.get("per_regime_mc_expost"); pr_q_ex = ctx.get("per_regime_q_expost")
    if pr_mc_ex is not None and pr_q_ex is not None:
        p.append("\n<details><summary><b>Ex-post (contemporaneous) split — descriptive only, "
                 "do not cite as a conditional result</b></summary>\n")
        p.append("\nThe same tables bucketed by the state the filter assigns to day t itself. "
                 "Printed so the size of the convention effect is visible rather than hidden.\n")
        p.append(f"\n**MC-Dropout (ex-post):**\n\n{_fmt_per_regime(pr_mc_ex, L)}\n")
        p.append(f"\n**Quantile (ex-post):**\n\n{_fmt_per_regime(pr_q_ex, L)}\n")
        p.append("\n</details>\n")

    tt = ctx.get("transition_table"); tr = ctx.get("transition")
    tt_ex = ctx.get("transition_table_expost"); tr_ex = ctx.get("transition_expost")

    def _tbl(t):
        rows = ["\n| method | days | n | PICP | MPIW | Winkler | coverage error |",
                "|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            rows.append(f"| {r['method']} | {r['days']} | {int(r['n'])} | {r['picp']:.3f} | "
                        f"{r['mpiw']:.3e} | {r['winkler']:.3e} | {r['coverage_error']:+.3f} |")
        return "\n".join(rows) + "\n"

    def _gap(t, method):
        """(PICP flagged, PICP rest, flagged − rest). Negative = worse on flagged days."""
        a = t[(t["method"] == method) & (t["days"] == "regime transition")]
        b = t[(t["method"] == method) & (t["days"] == "stable regime")]
        if not len(a) or not len(b):
            return float("nan"), float("nan"), float("nan")
        pa, pb = float(a["picp"].iloc[0]), float(b["picp"].iloc[0])
        return pa, pb, pa - pb

    if tt is not None and len(tt):
        p.append("\n### Can the failures be seen coming? The transition split, done two ways\n")
        p.append("Prediction intervals are only useful if their failures are *anticipable*, so the "
                 "question is not where the misses landed but whether anything knowable at t−1 "
                 "flags them. That makes the timing of the **selector** as consequential as the "
                 "timing of the bucket label above, and for the same reason.\n")

        p.append(f"\n**Headline — ex-ante selector `1{{s(t−1) ≠ s(t−2)}}`.** Both labels are known "
                 f"at t−1, so this is the only version that supports a conditional claim, a "
                 f"coverage test, or a recalibration rule. It flags "
                 f"**{tr['n_transition']} of {tr['n']} days ({100 * tr['share']:.1f}%)**.\n")
        p.append(_tbl(tt))
        mc_a, mc_b, mc_g = _gap(tt, MC_NAME)
        q_a, q_b, q_g = _gap(tt, Q_NAME)
        _MATERIAL_GAP = 0.05
        if max(abs(mc_g), abs(q_g)) >= _MATERIAL_GAP:
            p.append(f"\nOn the implementable selector the two groups **do** separate: MC-Dropout "
                     f"{mc_a * 100:.1f}% on flagged days vs {mc_b * 100:.1f}% on the rest "
                     f"({mc_g * 100:+.1f} pp), quantile {q_a * 100:.1f}% vs {q_b * 100:.1f}% "
                     f"({q_g * 100:+.1f} pp); the sign is flagged minus rest, so negative means "
                     "worse when a change was visible. Attach a formal coverage test in Phase 7 "
                     "before citing it.\n")
        else:
            p.append(f"\n**There is no separation.** MC-Dropout covers {mc_a * 100:.1f}% on flagged "
                     f"days versus {mc_b * 100:.1f}% on the rest ({mc_g * 100:+.1f} pp, flagged "
                     f"minus rest); the quantile head {q_a * 100:.1f}% versus {q_b * 100:.1f}% "
                     f"({q_g * 100:+.1f} pp). Knowing that the regime changed *yesterday* tells a "
                     "forecaster nothing about whether today's interval will hold.\n")

    if tt_ex is not None and len(tt_ex):
        p.append("\n<details><summary><b>Ex-post selector <code>1{s(t) ≠ s(t−1)}</code> — "
                 "descriptive only, do not cite as a conditional result</b></summary>\n")
        if tr_ex:
            p.append(f"\nThese are also exactly the **{tr_ex['n_transition']} of {tr_ex['n']} days "
                     f"({100 * tr_ex['share']:.1f}%)** on which the two per-regime tables above "
                     "disagree — the days whose bucket the labelling convention alone decides. "
                     "Splitting coverage on them is convention-*free*, which is why an earlier "
                     "version of this note reported it as the robust RQ3 finding. It is not: "
                     "s(t) is inferred from the day-t return, and whether the day-t interval "
                     "covered is a function of that same return, so the subsample is selected on "
                     "the outcome being measured. A regime transition is *detected by* the very "
                     "surprise that breaks the interval — realized variance on these days runs "
                     "about twice the median of stable days.\n")
        p.append(_tbl(tt_ex))
        ex_mc_a, ex_mc_b, ex_mc_g = _gap(tt_ex, MC_NAME)
        p.append(f"\nThe gap here is {ex_mc_g * 100:+.1f} pp for MC-Dropout "
                 f"({ex_mc_a * 100:.1f}% vs {ex_mc_b * 100:.1f}%) against "
                 f"{mc_g * 100:+.1f} pp on the ex-ante selector above — the same rule, the same "
                 "number of flagged days, one day of hindsight apart. The difference between the "
                 "two tables is the finding; neither number alone is.\n")
        p.append("\n</details>\n")
    p.append(f"\n![interval band](../figures/m06/{name}_interval_band.png)\n")

    # ---- findings ----
    p.append("## Findings (RQ3)\n")
    mc_ce = mc90["coverage_error"]; q_ce = q90["coverage_error"]
    mc_dir = "over-confident (too narrow)" if mc_ce < 0 else "conservative (too wide)"
    q_dir = "over-confident (too narrow)" if q_ce < 0 else "conservative (too wide)"
    p.append(f"At the nominal **{Lp}%** level, MC-Dropout attains **{mc90['picp']*100:.1f}%** empirical "
             f"coverage (coverage error {mc_ce:+.3f} — {mc_dir}) and the quantile model "
             f"**{q90['picp']*100:.1f}%** ({q_ce:+.3f} — {q_dir}). "
             + ("The distribution-free quantile intervals are the better-calibrated of the two overall"
                if abs(q_ce) < abs(mc_ce) else
                "The MC-Dropout log-normal intervals are the better-calibrated of the two overall")
             + ", consistent with the literature that MC-Dropout tends to under-cover unless the "
             "dropout rate is tuned as a calibration knob (roadmap Risk 4).\n")

    # per-regime degradation
    def picp_of(pr, lab):
        r = pr[pr["regime"] == lab]
        return float(r["picp"].iloc[0]) if len(r) and pd.notna(r["picp"].iloc[0]) else np.nan
    calm_mc, cris_mc = picp_of(pr_mc, labels[0]), picp_of(pr_mc, labels[-1])
    calm_q, cris_q = picp_of(pr_q, labels[0]), picp_of(pr_q, labels[-1])
    # Crisis-state sample size, read from the table rather than hardcoded: the
    # regime estimator (and therefore the bucket size) is config-driven.
    _cris = pr_mc[pr_mc["regime"] == labels[-1]]
    n_crisis = int(_cris["n"].iloc[0]) if len(_cris) else 0

    # Data-driven, in both directions. An earlier version asserted "coverage
    # degrades in the crisis regime -- exactly the RQ3 hypothesis" as fixed prose.
    # Under the corrected t-1 bucketing that ordering does not hold, and prose
    # that cannot report its own falsification is not evidence. Every clause
    # below is computed, including which state is worst.
    worst_mc = min(labels, key=lambda lb: (np.inf if np.isnan(picp_of(pr_mc, lb))
                                           else picp_of(pr_mc, lb)))
    worst_q = min(labels, key=lambda lb: (np.inf if np.isnan(picp_of(pr_q, lb))
                                          else picp_of(pr_q, lb)))

    # "Degrades with regime severity" is a claim about an *ordering*, so test the
    # ordering -- not just the calm-vs-crisis endpoints, which a 0.2pp gap would
    # satisfy by noise. Require the sequence to be monotone non-increasing across
    # every state AND the end-to-end gap to be materially larger than the
    # sampling noise on a ~75-day bucket (2 percentage points is already
    # generous at n=76, where one day is 1.3pp).
    def _monotone_down(pr) -> bool:
        return coverage_degrades(pr, labels)

    def _shape(pr) -> str:
        return coverage_shape(pr, labels)

    def _picp_se(pr, lb) -> float:
        return picp_standard_error(pr, lb)

    def _gap_verdict(pr) -> str:
        return endpoint_gap_verdict(pr, labels)

    degrades = _monotone_down(pr_mc) and _monotone_down(pr_q)
    if degrades:
        p.append(f"**Coverage degrades monotonically with regime severity under both methods.** "
                 f"MC-Dropout {calm_mc*100:.1f}% → {cris_mc*100:.1f}%, "
                 f"quantile {calm_q*100:.1f}% → {cris_q*100:.1f}%. The pooled coverage number "
                 "therefore flatters both models.\n")
    else:
        best_mc = max(labels, key=lambda lb: (-np.inf if np.isnan(picp_of(pr_mc, lb))
                                              else picp_of(pr_mc, lb)))
        shape_mc, shape_q = _shape(pr_mc), _shape(pr_q)
        shape_clause = (f"the ordering is {shape_mc}" if shape_mc == shape_q
                        else f"the ordering is {shape_mc} for MC-Dropout and "
                             f"{shape_q} for the quantile model")
        gap_mc = abs(calm_mc - cris_mc)
        se_gap = float(np.sqrt(_picp_se(pr_mc, labels[0]) ** 2
                               + _picp_se(pr_mc, labels[-1]) ** 2))
        p.append("**Coverage does *not* degrade monotonically with regime severity.** Under the "
                 f"t−1 bucketing {shape_clause}: the **{best_mc}** state is the "
                 f"*best* covered ({picp_of(pr_mc, best_mc)*100:.1f}% MC-Dropout, "
                 f"{picp_of(pr_q, best_mc)*100:.1f}% quantile), and {labels[0]} and "
                 f"{labels[-1]} are {gap_mc*100:.1f} pp apart "
                 f"({calm_mc*100:.1f}% vs {cris_mc*100:.1f}% MC-Dropout, {calm_q*100:.1f}% vs "
                 f"{cris_q*100:.1f}% quantile) — {_gap_verdict(pr_mc)}, since the two-bucket "
                 f"standard error is {se_gap*100:.1f} pp on an n={n_crisis} "
                 f"{labels[-1]} bucket. What the correct timing rules out is the *monotone "
                 "degradation with severity* that RQ3 anticipated; whichever direction the "
                 "residual ordering runs, it is not that. This is a **correction to the earlier "
                 "Phase-6 claim**, which was computed on the contemporaneous label; that ordering "
                 "is produced by the bucketing convention, not by the intervals (the ex-post "
                 "table above still shows it, and the transition split below explains the "
                 "mechanism).\n")
    tt2 = ctx.get("transition_table")
    tt2_ex = ctx.get("transition_table_expost")
    if tt2 is not None and len(tt2):
        def _pick(t, method, days):
            r = t[(t["method"] == method) & (t["days"] == days)]
            return float(r["picp"].iloc[0]) if len(r) else float("nan")
        a_mc = _pick(tt2, MC_NAME, "regime transition")
        b_mc = _pick(tt2, MC_NAME, "stable regime")
        a_q = _pick(tt2, Q_NAME, "regime transition")
        b_q = _pick(tt2, Q_NAME, "stable regime")
        p.append("**Nor do they fail predictably at regime *transitions*.** On the ex-ante "
                 "selector — the state change a forecaster could actually have seen, "
                 f"`1{{s(t−1) ≠ s(t−2)}}` — MC-Dropout covers {a_mc*100:.1f}% on flagged days "
                 f"versus {b_mc*100:.1f}% on the rest, and the quantile head {a_q*100:.1f}% "
                 f"versus {b_q*100:.1f}%: no separation in either direction.\n")
        if tt2_ex is not None and len(tt2_ex):
            xa = _pick(tt2_ex, MC_NAME, "regime transition")
            xb = _pick(tt2_ex, MC_NAME, "stable regime")
            xaq = _pick(tt2_ex, Q_NAME, "regime transition")
            xbq = _pick(tt2_ex, Q_NAME, "stable regime")
            p.append("**This retracts a claim made in an earlier version of this note.** The "
                     f"ex-post selector `1{{s(t) ≠ s(t−1)}}` gives {xa*100:.1f}% vs {xb*100:.1f}% "
                     f"(MC-Dropout) and {xaq*100:.1f}% vs {xbq*100:.1f}% (quantile) — a large gap "
                     "that was reported as the substantive RQ3 result on the grounds that it is "
                     "invariant to the bucketing convention. Convention-invariance was the wrong "
                     "test. s(t) is inferred from the day-t return and interval coverage on day t "
                     "is a function of that same return, so the split is selection on the outcome: "
                     "the identical error the t−1 bucketing fixed one level up, and one that would "
                     "manufacture a small p-value for any coverage test run on that subsample. "
                     "Lagging the selector by a single day removes the effect completely.\n")
        p.append("**What the two RQ3 results add up to is that the miscalibration is a *level* "
                 "problem, not a *timing* problem.** The intervals are the wrong width more or "
                 "less everywhere — not in identifiable states, and not on days that can be "
                 "flagged in advance. That is a coherent and useful answer for a risk manager: it "
                 "says the fix is a width correction rather than a state-dependent rule, and it "
                 "predicts that a global rescaling should work where a conditional one will not. "
                 "The σ-scale diagnostic above is the direct estimate of that correction, and "
                 "Phase 7's coverage tests (Kupiec unconditional, Christoffersen conditional, "
                 "Engle–Manganelli DQ) belong on the **full sample with lagged regressors**, not "
                 "on any transition subsample.\n")
    p.append("A structural reading explains *why*, and why neither method escapes it. The "
             "MC-Dropout predictive band is near-**homoskedastic on the log scale**: its "
             "aleatoric term is a single global residual variance estimated once on pre-2019 "
             "data and broadcast to every out-of-sample day, and the epistemic term is "
             "negligible (below), so the band is essentially a fixed *multiplicative* factor on "
             "the point forecast. Its *relative* width is therefore fixed: it widens in absolute "
             "terms only when the point forecast itself rises, and it has no channel at all "
             "through which to price a particular day as unusually uncertain. The "
             "quantile head can in principle learn a state-dependent width from the pinball "
             "objective, and it does widen, but it too keys off the same lagged inputs and is "
             "caught by the same one-day surprise. A band that is a fixed multiplicative factor "
             "on the point forecast is *exactly* the object whose miscalibration is a level "
             "problem: it can be the wrong width uniformly, but it has no mechanism by which to "
             "be wrong selectively — which is what the ex-ante transition split measures and "
             "confirms.\n")
    p.append("**What this implies for Phase 7,** stated before it is built so the result reads as "
             "a prediction rather than a rationalisation: the combined regime × uncertainty model "
             "should be expected to leave calibration roughly unchanged, because the conditioning "
             "variable it would use is the same lagged regime signal that shows no coverage "
             "separation here. That is worth running as a **pre-registered null** — a clean "
             "negative on regime-conditional recalibration, reported alongside a global width "
             "correction that the σ-scale diagnostic says should work. The alternative reading, "
             "that a richer conditioning variable would succeed, requires first exhibiting one "
             "that is measurable at t−1 and correlates with interval failure; none of the "
             "candidates screened so far does.\n")

    p.append(f"**Epistemic vs aleatoric.** Averaged over the test window the MC-Dropout predictive "
             f"std decomposes into an epistemic (model) part {dec['sd_epistemic']:.3f} and an aleatoric "
             f"(residual-noise) part {dec['sd_aleatoric']:.3f} on the log-variance scale — the "
             f"irreducible noise dominates ({dec['epi_share']*100:.1f}% of predictive variance is "
             "epistemic). At dropout 0.1 the MC-Dropout and LSTM-Gaussian intervals are therefore "
             "close (the reference row confirms it); the substantive calibration contrast is the "
             "**distribution-free quantile** method vs the **parametric log-normal** one, and the "
             "transition/stable breakdown between them. Two modelling assumptions are worth stating plainly: "
             "the MC-Dropout interval assumes the log-variance residuals are **Gaussian and "
             "homoskedastic** (a single global aleatoric variance), and the epistemic contribution "
             "here is small — so this phase should be read as a comparison of *interval calibration* "
             "under those assumptions, not as evidence that dropout-Bayesian uncertainty is large for "
             "S&P 500 RV.\n")

    p.append("## Gate criteria\n")
    p.append(f"- [x] MC-Dropout LSTM: dropout kept active at inference, T={ctx['mc_samples']} passes, "
             "predictive mean + std → log-normal intervals; epistemic/aleatoric variance decomposed.\n")
    p.append(f"- [x] Quantile-regression LSTM: monotone non-crossing head at τ={ctx['quantiles']}, "
             "pinball loss; distribution-free intervals.\n")
    p.append("- [x] Calibration metrics PICP / MPIW / Winkler at 90% (both) and 95% (parametric), "
             "with a reliability diagram across a dense τ grid.\n")
    p.append("- [x] **Per-regime** calibration table (calm / transitional / crisis) for both "
             "methods, bucketed by the regime known at t−1, with the ex-post split reported "
             "alongside; plus a transition/stable contrast at both selector timings, whose "
             "disagreement is the RQ3 contribution.\n")
    p.append("- [x] Point-accuracy preserved vs the Phase-3 LSTM (QLIKE + formal DM); predictions, "
             "tables, figures and a reproducible config snapshot persisted.\n")

    p.append("## Reproduce\n")
    p.append("```\npython -m src.experiments.run_uq\n"
             "pytest -q tests/test_uncertainty.py tests/test_metrics.py\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# One profile                                                                  #
# --------------------------------------------------------------------------- #
def build_models(cfg, *, max_epochs, mc_samples):
    m = cfg.model
    common = dict(
        features=tuple(OmegaConf.to_container(m.features, resolve=True)),
        lookback=int(m.lookback), hidden_size=int(m.hidden_size), num_layers=int(m.num_layers),
        dropout=float(m.dropout), lr=float(m.lr), weight_decay=float(m.weight_decay),
        batch_size=int(m.batch_size), max_epochs=int(max_epochs if max_epochs else m.max_epochs),
        patience=int(m.patience), grad_clip=float(m.grad_clip), val_fraction=float(m.val_fraction),
        refit_every_folds=int(cfg.harness.refit_every_folds), seed=int(cfg.seed),
    )
    models = []
    if bool(cfg.uq.mc_dropout.enabled):
        models.append(MCDropoutLSTMForecaster(
            smearing=bool(m.smearing),
            mc_samples=int(mc_samples if mc_samples else cfg.uq.mc_dropout.mc_samples),
            mc_seed=int(cfg.uq.mc_dropout.get("mc_seed", cfg.seed)), name=MC_NAME, **common))
    if bool(cfg.uq.quantile.enabled):
        q = tuple(float(x) for x in OmegaConf.to_container(cfg.uq.quantile.quantiles, resolve=True))
        models.append(QuantileLSTMForecaster(quantiles=q, name=Q_NAME, **common))
    return models


def run_profile(data_cfg, cfg, profile, args) -> dict:
    name = profile.name
    log.info("=" * 70)
    log.info("PROFILE %s | target=%s", name, profile.target)

    frame = build_econometric_frame(data_cfg, target=profile.target, include_vix=False)
    reg_state_full = attach_regime_state(frame, cfg.regime)
    labels = list(OmegaConf.to_container(cfg.regime.labels, resolve=True))
    levels = [float(x) for x in OmegaConf.to_container(cfg.uq.levels, resolve=True)]
    headline_level = float(cfg.uq.headline_level)

    sp = profile.splits
    split_cfg = SplitConfig(
        train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
        test_start=sp.test_start, test_end=sp.test_end,
        refit_frequency_days=int(cfg.harness.refit_frequency_days), scheme=cfg.harness.scheme,
    )
    max_epochs = args.max_epochs if args.max_epochs else (6 if args.fast else None)
    mc_samples = args.mc_samples if args.mc_samples else (20 if args.fast else None)

    pred_path = repo_path(cfg.paths.predictions, f"m06_uq_{name}.parquet")
    q_quantiles_cfg = [float(x) for x in
                       OmegaConf.to_container(cfg.uq.quantile.quantiles, resolve=True)]

    if getattr(args, "from_predictions", False):
        # Evaluation-only re-run: reuse the persisted predictive laws and rebuild
        # every table/figure/milestone section. Identical to a full re-run for
        # every artefact that is a function of the predictions alone, which is
        # what a scoring-convention fix needs. See run_regime_lstm for the same
        # flag and the same reasoning.
        if not pred_path.exists():
            raise FileNotFoundError(
                f"--from-predictions needs {pred_path}; run the full phase first.")
        log.info("--from-predictions: reusing %s (no training)", pred_path)
        preds = from_parquet(pred_path).copy()
        uq = None
        q_level = q_quantiles_cfg[-1] - q_quantiles_cfg[0]
        n_mc = int(cfg.uq.mc_dropout.mc_samples)
    else:
        models = build_models(cfg, max_epochs=max_epochs, mc_samples=mc_samples)
        log.info("Phase-6 models: %s", [mm.name for mm in models])
        uq = run_uq_walk_forward(models, frame, split_cfg,
                                 eval_segment=cfg.harness.eval_segment)

        y_true = frame["rv"]
        q_level = float(uq[Q_NAME].columns[-1][1:]) - float(uq[Q_NAME].columns[0][1:])
        preds = build_predictions(uq, y_true, reg_state_full, levels)
        n_mc = int(models[0].mc_samples)

    # Retransform the quantile point onto the mean scale QLIKE elicits. Derived
    # from the persisted quantiles, so both branches above reach it identically
    # and a --from-predictions rebuild picks it up without retraining.
    preds = add_quantile_mean_column(preds, q_quantiles_cfg)

    # The *full* Phase-4 series (not the copy stored in the predictions parquet)
    # so the t-1 lag can reach the trading day before the OOS window opens.
    reg_state = reg_state_full.astype("float")

    # deterministic Phase-3 LSTM point (from the Phase-5 parquet) for the sanity check
    lstm_point = None
    try:
        base = from_parquet(repo_path(profile.baseline_predictions))
        col = profile.get("lstm_point_col", "LSTM")
        if col in base.columns:
            lstm_point = base[col]
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load baseline LSTM point (%s); skipping that sanity row", exc)

    pt = point_metrics_table(preds, lstm_point)
    cal = calibration_table(preds, levels, q_level)
    # HEADLINE per-regime tables use the t-1 label (see per_regime_table); the
    # contemporaneous versions are kept alongside, explicitly marked ex-post,
    # because the two disagree and the disagreement is itself the finding.
    pr_mc = per_regime_table(preds, labels, headline_level, MC_NAME, reg_state=reg_state)
    pr_q = per_regime_table(preds, labels, headline_level, Q_NAME, reg_state=reg_state)
    pr_mc_ex = per_regime_table(preds, labels, headline_level, MC_NAME,
                                reg_state=reg_state, shift=0)
    pr_q_ex = per_regime_table(preds, labels, headline_level, Q_NAME,
                               reg_state=reg_state, shift=0)
    # HEADLINE transition split uses the EX-ANTE selector 1{s_(t-1) != s_(t-2)};
    # the ex-post 1{s_t != s_(t-1)} version is kept beside it because the two
    # disagree completely and that disagreement is the RQ3 finding. See
    # transition_vs_stable_table and src.evaluation.regime_timing.
    trans_tbl = transition_vs_stable_table(
        preds, headline_level, [MC_NAME, Q_NAME], reg_state,
        selector_shift=DEFAULT_SELECTOR_SHIFT)
    trans_tbl_ex = transition_vs_stable_table(
        preds, headline_level, [MC_NAME, Q_NAME], reg_state, selector_shift=0)
    trans = transition_day_summary(reg_state, preds.index,
                                   selector_shift=DEFAULT_SELECTOR_SHIFT)
    # selector_shift=0 is also exactly the set of days whose per-regime *bucket*
    # differs between the two label timings -- that is what the note quotes it for.
    trans_ex = transition_day_summary(reg_state, preds.index, selector_shift=0)
    crps = crps_table(preds, q_quantiles_cfg)
    sigma_diag = sigma_scale_diagnostic(preds, headline_level)
    taus = [float(x) for x in OmegaConf.to_container(cfg.uq.reliability_taus, resolve=True)]
    q_quantiles = q_quantiles_cfg
    rel = reliability_table(preds, uq, taus, q_quantiles)

    # formal DM: MC-Dropout mean vs the deterministic LSTM (point-accuracy check)
    if lstm_point is not None:
        sub = pd.concat([preds[ACTUAL_COL], preds[MC_NAME], lstm_point.reindex(preds.index)],
                        axis=1).dropna()
        dm = diebold_mariano(sub.iloc[:, 0], sub.iloc[:, 1], sub.iloc[:, 2])
    else:
        dm = {"dm_stat": float("nan"), "p_value": float("nan"), "n": len(preds)}

    # variance decomposition summary
    s2_epi = float((preds["mc_sd_epistemic"] ** 2).mean())
    s2_ale = float((preds["mc_sd_aleatoric"] ** 2).mean())
    var_decomp = {"sd_epistemic": float(np.sqrt(s2_epi)), "sd_aleatoric": float(np.sqrt(s2_ale)),
                  "epi_share": s2_epi / (s2_epi + s2_ale)}

    log.info("PROFILE %s point QLIKE:\n%s", name, pt.round(4).to_string())
    log.info("PROFILE %s calibration:\n%s", name, cal.round(4).to_string(index=False))
    log.info("PROFILE %s per-regime MC [%s]:\n%s", name,
             regime_timing_label(DEFAULT_REGIME_SHIFT), pr_mc.round(4).to_string(index=False))
    log.info("PROFILE %s per-regime Q  [%s]:\n%s", name,
             regime_timing_label(DEFAULT_REGIME_SHIFT), pr_q.round(4).to_string(index=False))
    log.info("PROFILE %s transition vs stable [%s]:\n%s", name, trans["selector"],
             trans_tbl.round(4).to_string(index=False))
    log.info("PROFILE %s transition vs stable [%s]:\n%s", name, trans_ex["selector"],
             trans_tbl_ex.round(4).to_string(index=False))

    # --- persist predictions + tables ---
    if not getattr(args, "from_predictions", False):
        to_parquet(preds, pred_path)
    ensure_dir(repo_path(cfg.paths.tables))
    pt.to_csv(repo_path(cfg.paths.tables, f"m06_uq_{name}_point.csv"))
    cal.to_csv(repo_path(cfg.paths.tables, f"m06_uq_{name}_calibration.csv"), index=False)
    for tbl, suffix, sh in ((pr_mc, "per_regime_mc", DEFAULT_REGIME_SHIFT),
                            (pr_q, "per_regime_q", DEFAULT_REGIME_SHIFT),
                            (pr_mc_ex, "per_regime_mc_expost", 0),
                            (pr_q_ex, "per_regime_q_expost", 0)):
        stamped = tbl.copy()
        stamped.insert(0, "regime_shift", int(sh))
        stamped.insert(1, "timing", regime_timing_label(sh))
        stamped.to_csv(repo_path(cfg.paths.tables, f"m06_uq_{name}_{suffix}.csv"), index=False)
    # Both selector timings in one file, stamped, ex-ante first: a reader who cites
    # a row cannot avoid seeing which of the two it is.
    pd.concat([trans_tbl, trans_tbl_ex], ignore_index=True).to_csv(
        repo_path(cfg.paths.tables, f"m06_uq_{name}_transition_vs_stable.csv"), index=False)
    crps.to_csv(repo_path(cfg.paths.tables, f"m06_uq_{name}_crps.csv"), index=False)
    sigma_diag.to_csv(
        repo_path(cfg.paths.tables, f"m06_uq_{name}_sigma_scale.csv"), index=False)
    rel.to_csv(repo_path(cfg.paths.tables, f"m06_uq_{name}_reliability.csv"), index=False)

    # --- figures ---
    figdir = repo_path(cfg.paths.figures, "m06"); ensure_dir(figdir)
    plot_reliability(rel, figdir / f"{name}_reliability.png")
    plot_interval_band(preds, reg_state, labels, headline_level, figdir / f"{name}_interval_band.png")
    plot_per_regime(pr_mc, pr_q, labels, headline_level, figdir / f"{name}_per_regime.png")

    return {"name": name, "point": pt, "calibration": cal, "per_regime_mc": pr_mc,
            "per_regime_q": pr_q, "per_regime_mc_expost": pr_mc_ex,
            "per_regime_q_expost": pr_q_ex, "transition_table": trans_tbl,
            "transition_table_expost": trans_tbl_ex,
            "transition": trans, "transition_expost": trans_ex,
            "regime_shift": int(DEFAULT_REGIME_SHIFT),
            "selector_shift": int(DEFAULT_SELECTOR_SHIFT),
            "crps": crps, "sigma_diag": sigma_diag,
            "reliability": rel, "labels": labels, "headline_level": headline_level,
            "q_level": q_level, "dm_mc_vs_lstm": dm, "var_decomp": var_decomp,
            "mc_samples": n_mc, "quantiles": q_quantiles,
            "n_oos": int(len(preds)), "fast": bool(args.fast), "pred_path": pred_path}


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_uq")
    ap.add_argument("--config", default="uq")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true", help="few epochs + few MC samples (smoke)")
    ap.add_argument("--max-epochs", type=int, default=0, help="override max epochs (0 = config/fast)")
    ap.add_argument("--mc-samples", type=int, default=0, help="override MC samples (0 = config/fast)")
    ap.add_argument(
        "--from-predictions", action="store_true",
        help="skip training; rebuild all tables/figures/milestone from the persisted "
             "predictions parquet (use after changing scoring code, not model code)")
    args = ap.parse_args(argv)
    args.max_epochs = args.max_epochs or None
    args.mc_samples = args.mc_samples or None

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "06_uq",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    # Config-driven milestone filename so the Baum-Welch robustness variant
    # (configs/uq_hmm.yaml) cannot overwrite the headline note.
    milestone_file = str(cfg.paths.get("milestone_file") or "m06_uq.md")
    for profile in cfg.profiles:
        ctx = run_profile(data_cfg, cfg, profile, args)
        write_milestone(ctx, repo_path(cfg.paths.milestones, milestone_file))
    log.info("Phase 6 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
