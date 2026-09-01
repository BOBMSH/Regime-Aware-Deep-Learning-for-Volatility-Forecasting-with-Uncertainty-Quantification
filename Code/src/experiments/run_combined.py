"""Phase 7 driver -- combined regime x uncertainty model + inferential RQ3
(ROADMAP.md Phase 7; Ch2 §2.6/§2.7; RQ3, and the RQ2 x RQ3 interaction).

Phase 7 does two things, and they are deliberately separable because only one of
them needs a GPU-hour:

1. **The combined model.** ``MC-Dropout-Regime-LSTM-B`` -- "best of Phase 5"
   (:class:`~src.models.deep.regime_uncertainty.MCDropoutRegimeExpertForecaster`)
   with the Phase-6 MC-Dropout inference bolted on and *training inherited
   unchanged*. Selection rule, fixed before this ran: the Phase-5 winner is
   Regime-LSTM-B, chosen on the **transitional** result -- the only bucket where
   the regime effect is both large and formally tested (per-regime DM
   p = 4.3e-09 against HAR-RV; the state carrying the Giacomini-White moment
   behind chi2(3) = 40.1). Calm and crisis are point estimates on 379 and 76
   days with no significant test behind them, and crisis reverses sign between
   regime estimators; selecting on either would be choosing an architecture on
   noise.

2. **Making RQ3 inferential.** Phase 6 reported coverage as point estimates with
   no hypothesis test. The Kupiec / Christoffersen / DQ machinery lives in
   :mod:`src.evaluation.coverage_tests` and is driven from
   :mod:`src.experiments.report_coverage`, which reads parquets only; this
   script calls the same functions so the combined model is tested identically
   to the Phase-6 methods, and writes the master table that is the Phase-7 gate.

The pre-registered expectation (roadmap Phase 7, recorded before the run)
-------------------------------------------------------------------------
Kupiec rejects decisively; Christoffersen's independence component does not; no
``t-1`` conditioning variable predicts a miss. If that holds, miscalibration is
a **level** problem rather than a **timing** problem, the combined model is a
pre-registered null, and the deliverable is a global width correction.

The milestone this script writes computes that verdict from the tables in both
directions -- it can report its own falsification, which is the only way a
pre-registration is worth anything. Two guards make that real:

* every conditional DQ is **incremental** over level + hit dynamics
  (:func:`~src.evaluation.coverage_tests.dq_incremental`), because the raw DQ
  design contains a constant and a globally mis-levelled model rejects it
  through that constant alone; and
* every subsample coverage statement is a **difference** between the flagged
  days and the rest, never a comparison against nominal -- the 2026-08-22 (iv)
  caution -- with Holm control across the family.

Outputs (the Phase 7 gate artifacts)
------------------------------------
* ``results/predictions/m07_combined_<profile>.parquet``
* ``results/tables/m07_combined_<profile>_{master,point,calibration,per_regime,
  dm,gw,sigma_scale,coverage_tests,coverage_conditional}.csv``
* ``results/figures/m07/<profile>_{calibration,width_by_regime}.png``
* ``results/milestones/m07_combined.md``
* ``experiments/07_combined/run_<utc>/config.yaml``

Usage
-----
    python -m src.experiments.run_combined                  # full protocol
    python -m src.experiments.run_combined --fast           # smoke test
    python -m src.experiments.run_combined --from-predictions
    python -m src.experiments.run_combined --config combined_hmm
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
from src.data.splits import SplitConfig
from src.evaluation.calibration import (
    lognormal_interval,
    per_regime_interval_metrics,
)
from src.evaluation.coverage_tests import expected_rate, interval_hits
from src.evaluation.metrics import crps_lognormal, point_metrics, qlike
from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    DEFAULT_SELECTOR_SHIFT,
    align_regime_label,
    regime_timing_label,
    selector_timing_label,
)
from src.evaluation.rolling import ACTUAL_COL
from src.evaluation.significance import (
    diebold_mariano,
    giacomini_white,
    regime_test_function,
)
from src.experiments.report_coverage import conditional_table, unconditional_table
from src.experiments.run_uq import add_quantile_mean_column, run_uq_walk_forward
from src.models.deep import MCDropoutRegimeExpertForecaster
from src.utils.config import load_config, repo_path, snapshot_config
from src.utils.io import ensure_dir, from_parquet, to_parquet
from src.utils.logging import get_logger
from src.utils.plotting import save_fig
from src.utils.seeding import set_seed

log = get_logger("run_combined")

REGIME_STATE_COL = "reg_state"
COMBINED_NAME = "MC-Dropout-Regime-LSTM-B"
MC_NAME = "MC-Dropout-LSTM"
Q_NAME = "Quantile-LSTM"
GAUSS_NAME = "LSTM-Gaussian"
#: Mean-scale retransformation of Q_NAME. Defined in run_uq (which builds it);
#: imported here so the two modules cannot disagree about the column name.
Q_MEAN_NAME = "Quantile-LSTM-mean"
#: The Phase-5 architecture Phase 7 combines with MC-Dropout, and the rule that
#: picked it. Kept as a constant so the milestone cannot describe a different
#: choice from the one the code made.
BEST_OF_PHASE5 = "Regime-LSTM-B"
BEST_OF_PHASE5_RULE = (
    "selected on the transitional result -- the only regime bucket where the "
    "effect is both large and formally tested (per-regime DM p=4.3e-09 vs "
    "HAR-RV, and the state carrying the Giacomini-White moment behind "
    "chi2(3)=40.1). Calm and crisis are point estimates on 379 and 76 days with "
    "no significant test behind them, and crisis reverses sign between the two "
    "regime estimators."
)


# --------------------------------------------------------------------------- #
# Frame assembly                                                               #
# --------------------------------------------------------------------------- #
def attach_regime(frame: pd.DataFrame, rcfg) -> pd.DataFrame:
    """Join the Phase-4 causal filtered posteriors on as ``reg_p0..reg_p{K-1}``.

    Identical to :func:`src.experiments.run_regime_lstm.attach_regime` -- the
    combined model consumes exactly the Phase-5 conditioning signal, and any
    divergence here would silently make the Phase-5 -> Phase-7 comparison
    measure two different inputs.
    """
    reg = from_parquet(repo_path(rcfg.source))
    post = list(rcfg.posterior_cols)
    rename = {post[k]: f"reg_p{k}" for k in range(len(post))}
    rename[rcfg.state_col] = REGIME_STATE_COL
    regf = reg[post + [rcfg.state_col]].rename(columns=rename)

    joined = frame.join(regf, how="left")
    reg_cols = [f"reg_p{k}" for k in range(len(post))]
    missing = int(joined[reg_cols].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(
            f"{missing} frame rows have no regime posterior after the join -- the "
            "Phase-4 parquet and the modelling frame are misaligned."
        )
    log.info("regime signal attached: %s + %s (%d rows, %s -> %s)", reg_cols,
             REGIME_STATE_COL, len(joined), joined.index.min().date(),
             joined.index.max().date())
    return joined


def build_predictions(uq_frame: pd.DataFrame, y_true: pd.Series,
                      reg_state: pd.Series, levels) -> pd.DataFrame:
    """Tidy predictions frame for the combined model.

    Column contract is deliberately identical to Phase 6's
    (``<method>_lo<NN>`` / ``_hi<NN>``, ``mc_*`` decomposition), so
    :mod:`src.experiments.report_coverage` and every Phase-6 calibration helper
    run against this file unchanged -- no Phase-7-specific scoring code, and
    therefore no chance of the two phases being scored by two conventions.
    """
    idx = uq_frame.index
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    out[ACTUAL_COL] = y_true.reindex(idx)
    out[REGIME_STATE_COL] = reg_state.reindex(idx).astype("Int64")
    out[COMBINED_NAME] = uq_frame["mean"]
    out["mc_mu_log"] = uq_frame["mu_log"]
    out["mc_sigma_log"] = uq_frame["sigma_log"]
    out["mc_sd_epistemic"] = uq_frame["sd_epistemic"]
    out["mc_sd_aleatoric"] = uq_frame["sd_aleatoric"]
    out["mc_mu_log_det"] = uq_frame["mu_log_det"]
    out["gate_entropy"] = uq_frame["gate_entropy"]
    for L in levels:
        tag = f"{int(round(L * 100))}"
        lo, hi = lognormal_interval(uq_frame["mu_log"].to_numpy(),
                                    uq_frame["sigma_log"].to_numpy(), level=L)
        out[f"{COMBINED_NAME}_lo{tag}"] = lo
        out[f"{COMBINED_NAME}_hi{tag}"] = hi
    return out


#: Tolerance for the Phase-5 inheritance check, on the log-variance scale.
#: A same-environment refit of this codebase reproduces to ~1e-6 -- Phase 6
#: re-trains the Phase-3 LSTM from the same config and matches it to 9.6e-7 --
#: so 1e-3 sits three orders above the numerical floor while staying far below
#: the 1.44e-2 sd / 0.18 max drift that a change of interpreter produced.
PHASE5_INHERITANCE_TOL = 1e-3


def assert_inherits_phase5_training(
    preds: pd.DataFrame, base: pd.DataFrame, *, tol: float = PHASE5_INHERITANCE_TOL
) -> float:
    """Verify on the *artefacts* that the combined model is the Phase-5 model.

    Why this is here and not (only) in a unit test -- audit (vii), 2026-08-23
    ------------------------------------------------------------------------
    Phase 7's whole interpretive claim is that training is inherited unchanged,
    so the contrast ``MC-Dropout-LSTM -> MC-Dropout-Regime-LSTM-B`` isolates
    regime conditioning and nothing else. ``tests/test_regime_uncertainty.py``
    certifies it -- and structurally cannot certify what matters, because it
    fits both objects inside a single process. It catches a coding error; it
    cannot catch the two artefacts having been produced by two different
    interpreters, which is exactly what happened: Phase 5 was fitted under
    CPython 3.10 and Phase 7 under 3.12, and the dropout-off mixture missed the
    committed Phase-5 forecast by up to 18% on 2020-03-03.

    This function checks the claim where both artefacts are already in memory.
    The dropout-off column ``mc_mu_log_det`` is the Phase-5 log-space point, so

        exp(mu_log_det + s2_aleatoric / 2)

    must reproduce the Phase-5 ``Regime-LSTM-B`` variance forecast, both being
    the same log-normal retransformation of the same mixture. Returns the worst
    absolute log deviation and raises above ``tol``.

    A missing ``Regime-LSTM-B`` column (an unusual baseline file) downgrades to
    a warning: the check is a guard on a claim, not a hard dependency of the
    phase.
    """
    if BEST_OF_PHASE5 not in base.columns:
        log.warning(
            "Phase-5 inheritance check SKIPPED: no '%s' column in the baseline "
            "predictions; the 'training inherited unchanged' claim is unverified "
            "for this run.", BEST_OF_PHASE5)
        return float("nan")

    det = np.exp(preds["mc_mu_log_det"].astype(float)
                 + 0.5 * preds["mc_sd_aleatoric"].astype(float) ** 2)
    j = pd.concat([det.rename("det"), base[BEST_OF_PHASE5].astype(float).rename("ref")],
                  axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if j.empty:
        raise RuntimeError(
            "Phase-5 inheritance check: the combined predictions and the Phase-5 "
            "baseline share no dates -- the two profiles are misaligned.")

    dev = np.abs(np.log(j["det"].to_numpy() / j["ref"].to_numpy()))
    worst, spread, n = float(dev.max()), float(dev.std()), int(len(j))
    log.info("Phase-5 inheritance check: max |log dev| = %.3g, sd = %.3g over "
             "%d days (tolerance %.0e)", worst, spread, n, tol)
    if worst > tol:
        raise RuntimeError(
            f"Phase-5 inheritance check FAILED: the dropout-off mixture does not "
            f"reproduce '{BEST_OF_PHASE5}' in {profile_hint(base)}.\n"
            f"  max |log deviation| = {worst:.4g} (sd {spread:.4g}) over {n} days, "
            f"tolerance {tol:.0e}; worst day {j.index[int(np.argmax(dev))].date()}.\n"
            f"A same-environment refit of this codebase reproduces to ~1e-6, so a "
            f"deviation this large means the two artefacts were NOT produced by the "
            f"same fit. The usual cause is that the Phase-5 predictions on disk were "
            f"trained under a different interpreter or library set from the one "
            f"running now (audit vii, 2026-08-23), or that the Phase-4 regime parquet "
            f"changed after Phase 5 was trained.\n"
            f"Fix: re-run Phases 2 -> 7 in one session under a single environment, "
            f"then re-tag. Compare the '_environment' block of the Phase-5 and "
            f"Phase-7 snapshots in experiments/*/run_*/config.yaml.")
    return worst


def profile_hint(base: pd.DataFrame) -> str:
    """Short human-readable identifier for a baseline frame, for error text."""
    try:
        return (f"the baseline predictions covering "
                f"{base.index.min().date()} -> {base.index.max().date()}")
    except Exception:  # pragma: no cover - non-datetime index
        return "the baseline predictions"


LOGNORMAL_CRPS = "log-normal closed form"


def lognormal_law(frame: pd.DataFrame, method: str):
    """The ``(mu, sigma)`` of *method*'s predictive law on log variance, or ``None``.

    Every interval-bearing method carries its **own** predictive law, and a CRPS
    is only that method's CRPS if it is computed from that law. Until 2026-09-01
    this was not enforced: :func:`master_table` read ``mc_mu_log`` /
    ``mc_sigma_log`` off whatever frame it happened to be handed, regardless of
    which method the interval belonged to -- and ``MC-Dropout-LSTM``,
    ``LSTM-Gaussian`` and ``Quantile-LSTM`` are all passed the *same* Phase-6
    frame. All three therefore published MC-Dropout's CRPS
    (4.3058928982516726e-05) rather than their own 4.3058929e-05 / 4.3014955e-05
    / 3.5498179e-05. Phase 6's own :func:`~src.experiments.run_uq.crps_table`
    had it right throughout; only the Phase-7 master table was wrong, which is
    exactly the artefact Chapter 4's headline table is built from.

    The laws:

    * ``MC-Dropout-LSTM`` and ``MC-Dropout-Regime-LSTM-B`` -- the MC predictive
      moments, persisted as ``mc_mu_log`` / ``mc_sigma_log``.
    * ``LSTM-Gaussian`` -- the dropout-off mean with an aleatoric-only spread.
      Phase 7 persists the log-space mean as ``mc_mu_log_det``; Phase 6 persists
      only the point on the variance scale, so the log-space mean is recovered
      as ``log(point) - sigma^2 / 2``, which is what ``run_uq.crps_table`` does.
    * ``Quantile-LSTM`` -- **None**. A pinball head asserts no distribution, so
      it has no closed-form CRPS at all. Its grid-approximated value, and the
      ``(same grid)`` comparators that make it comparable to the parametric
      methods, live in ``m06_*_crps.csv``; putting a trapezoidal number in the
      same column as three closed-form ones would recreate the defect in a
      subtler form.
    """
    cols = set(frame.columns)
    if method in (MC_NAME, COMBINED_NAME):
        if {"mc_mu_log", "mc_sigma_log"} <= cols:
            return frame["mc_mu_log"].astype(float), frame["mc_sigma_log"].astype(float)
        return None
    if method == GAUSS_NAME:
        if "mc_sd_aleatoric" not in cols:
            return None
        sd = frame["mc_sd_aleatoric"].astype(float)
        if "mc_mu_log_det" in cols:
            return frame["mc_mu_log_det"].astype(float), sd
        if method in cols:
            # Recover the log-space mean on ndarrays, in exactly the order
            # run_uq.crps_table uses, so the two tables agree bit-for-bit rather
            # than in the 16th digit -- a gratuitous discrepancy between two
            # published numbers for the same quantity is its own defect.
            mu = (np.log(frame[method].to_numpy(dtype=float))
                  - 0.5 * sd.to_numpy(dtype=float) ** 2)
            return pd.Series(mu, index=frame.index), sd
        return None
    return None


def assert_crps_distinct(table: pd.DataFrame) -> None:
    """Refuse a master table in which two models share a bit-identical CRPS.

    The 2026-09-01 defect was invisible in review precisely because a wrong
    number looks like a number. Two distinct predictive laws agreeing to all 17
    significant digits of a float64 does not happen; one frame's law being
    broadcast across several rows does. This makes the fix executable rather
    than merely intended -- the same lesson as the config block nothing read.
    """
    if "crps" not in table.columns:
        return
    sub = table.loc[table["crps"].notna(), ["model", "crps"]]
    dup = sub[sub.duplicated("crps", keep=False)]
    if not dup.empty:
        rows = ", ".join(f"{m}={c!r}" for m, c in zip(dup["model"], dup["crps"]))
        raise ValueError(
            "two models report an identical CRPS, which means one method's "
            f"predictive law was used for another's row: {rows}"
        )


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def master_table(preds: pd.DataFrame, base: pd.DataFrame, uqp: pd.DataFrame,
                 reg_state: pd.Series, labels, *, level: float) -> pd.DataFrame:
    """The Phase-7 master table: every model, one row, point *and* interval.

    This is the table the roadmap names as the Phase-7 gate and the one that
    goes into Chapter 4. It deliberately puts point accuracy and interval
    quality side by side, because the study's whole framing (Ch2 §2.8) is that a
    model which adds calibrated intervals without lowering point error is still
    a useful result -- a claim a reader can only assess if both are on the same
    page.

    Columns: pooled QLIKE / MSE / MAE; transitional-bucket QLIKE on the t-1
    label (the bucket RQ2's effect lives in); and, for models that produce a
    predictive distribution, PICP / MPIW / Winkler / CRPS at the headline level.
    ``crps`` is the closed-form log-normal score computed from each method's
    **own** predictive law (:func:`lognormal_law`), and ``crps_estimator``
    names it; a method asserting no distribution -- the pinball head -- gets a
    blank cell rather than a number produced by a different estimator, and its
    grid-approximated CRPS is read from ``m06_*_crps.csv`` instead.
    Models without intervals get NaN there rather than being dropped -- the
    comparison is the point of the table.

    Ranked rows vs sensitivity rows (2026-08-30)
    --------------------------------------------
    ``rank_qlike`` is assigned to competing *models* only. ``Quantile-LSTM-mean``
    is the same forecast as ``Quantile-LSTM`` under the mean-scale
    retransformation every other deep model already receives (see
    ``run_uq.add_quantile_mean_column``), so ranking it as a rival would both
    imply a model competes with itself and silently shift every published rank
    below it. It carries ``rank_qlike = NaN`` and the family
    ``sensitivity (retransformed point)``: visible in the gate artefact, quoted in
    Chapter 4, counted in no ranking.
    """
    y = preds[ACTUAL_COL].astype(float)
    lagged = align_regime_label(reg_state, y.index, shift=DEFAULT_REGIME_SHIFT,
                                warn_unreachable=True)
    trans_mask = (lagged == 1.0).to_numpy() if len(labels) > 1 else np.zeros(len(y), bool)
    tag = f"{int(round(level * 100))}"

    # point-only models from Phase 5, plus every interval-bearing method
    point_cols = [c for c in base.columns
                  if c not in (ACTUAL_COL, REGIME_STATE_COL)
                  and pd.api.types.is_numeric_dtype(base[c])]
    rows = []

    def _interval_bits(frame: pd.DataFrame, method: str) -> dict:
        lo_c, hi_c = f"{method}_lo{tag}", f"{method}_hi{tag}"
        if lo_c not in frame.columns or hi_c not in frame.columns:
            return {}
        j = pd.concat([y.rename("y"), frame[lo_c].rename("lo"),
                       frame[hi_c].rename("hi")], axis=1, join="inner").dropna()
        if j.empty:
            return {}
        yy, lo, hi = (j["y"].to_numpy(float), j["lo"].to_numpy(float),
                      j["hi"].to_numpy(float))
        inside = (yy >= lo) & (yy <= hi)
        alpha = 1.0 - level
        pen = np.zeros_like(yy)
        below, above = yy < lo, yy > hi
        pen[below] = (2.0 / alpha) * (lo[below] - yy[below])
        pen[above] = (2.0 / alpha) * (yy[above] - hi[above])
        bits = {"picp": float(inside.mean()), "mpiw": float((hi - lo).mean()),
                "winkler": float((hi - lo + pen).mean()),
                "coverage_error": float(inside.mean() - level)}
        # CRPS in closed form from THIS method's own predictive law (see
        # lognormal_law); blank where the method asserts no distribution, so
        # every value in the column is computed the same way and comparable.
        law = lognormal_law(frame, method)
        if law is not None:
            mu = law[0].reindex(j.index).to_numpy(float)
            sd = law[1].reindex(j.index).to_numpy(float)
            if np.isfinite(mu).all() and np.isfinite(sd).all():
                bits["crps"] = crps_lognormal(j["y"].to_numpy(float), mu, sd)
                bits["crps_estimator"] = LOGNORMAL_CRPS
        return bits

    def _row(name: str, point: pd.Series, frame: pd.DataFrame | None,
             method: str | None, family: str, *, ranked: bool = True) -> dict:
        j = pd.concat([y.rename("y"), point.rename("h")], axis=1, join="inner").dropna()
        pm = point_metrics(j["y"], j["h"])
        m = lagged.reindex(j.index) == 1.0
        rec = {"model": name, "family": family, "ranked": bool(ranked),
               "n": int(pm["n"]),
               "qlike": pm["qlike"], "mse": pm["mse"], "mae": pm["mae"],
               "qlike_transitional": (qlike(j["y"][m], j["h"][m])
                                      if int(m.sum()) > 0 else np.nan),
               "n_transitional": int(m.sum())}
        rec.update({"picp": np.nan, "mpiw": np.nan, "winkler": np.nan,
                    "coverage_error": np.nan, "crps": np.nan,
                    "crps_estimator": ""})
        if frame is not None and method is not None:
            rec.update(_interval_bits(frame, method))
        return rec

    for c in point_cols:
        fam = ("econometric" if c in ("GARCH", "EGARCH", "HAR-RV", "RW-RV")
               else "regime-aware deep" if c.startswith("Regime-")
               else "deep")
        rows.append(_row(c, base[c], None, None, fam))
    for c, fr in ((MC_NAME, uqp), (Q_NAME, uqp), (GAUSS_NAME, uqp)):
        if c in fr.columns:
            rows.append(_row(c, fr[c], fr, c, "uncertainty-aware deep"))
    rows.append(_row(COMBINED_NAME, preds[COMBINED_NAME], preds, COMBINED_NAME,
                     "combined (Phase 7)"))
    # Sensitivity row: the same quantile forecast on the mean scale QLIKE
    # elicits. Unranked -- see the docstring.
    if Q_MEAN_NAME in uqp.columns:
        rows.append(_row(Q_MEAN_NAME, uqp[Q_MEAN_NAME], None, None,
                         "sensitivity (retransformed point)", ranked=False))

    out = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
    rank = pd.Series(np.nan, index=out.index, dtype="float")
    is_ranked = out["ranked"].to_numpy(dtype=bool)
    rank[is_ranked] = np.arange(1, int(is_ranked.sum()) + 1)
    out.insert(0, "rank_qlike", rank)
    out.attrs["regime_shift"] = int(DEFAULT_REGIME_SHIFT)
    out.attrs["level"] = float(level)
    out.attrs["n_ranked_models"] = int(is_ranked.sum())
    assert_crps_distinct(out)
    return out


def calibration_table(preds: pd.DataFrame, uqp: pd.DataFrame, levels) -> pd.DataFrame:
    """PICP / MPIW / Winkler for every interval-bearing method at every level."""
    y = preds[ACTUAL_COL].astype(float)
    rows = []
    sources = [(COMBINED_NAME, preds)]
    for m in (MC_NAME, GAUSS_NAME, Q_NAME):
        if any(str(c).startswith(f"{m}_lo") for c in uqp.columns):
            sources.append((m, uqp))
    for method, frame in sources:
        for L in levels:
            tag = f"{int(round(L * 100))}"
            lo_c, hi_c = f"{method}_lo{tag}", f"{method}_hi{tag}"
            if lo_c not in frame.columns:
                continue
            j = pd.concat([y.rename("y"), frame[lo_c].rename("lo"),
                           frame[hi_c].rename("hi")], axis=1, join="inner").dropna()
            yy, lo, hi = (j["y"].to_numpy(float), j["lo"].to_numpy(float),
                          j["hi"].to_numpy(float))
            inside = (yy >= lo) & (yy <= hi)
            alpha = 1.0 - L
            pen = np.zeros_like(yy)
            below, above = yy < lo, yy > hi
            pen[below] = (2.0 / alpha) * (lo[below] - yy[below])
            pen[above] = (2.0 / alpha) * (yy[above] - hi[above])
            rows.append({"method": method, "nominal": float(L), "n": int(len(j)),
                         "picp": float(inside.mean()),
                         "mpiw": float((hi - lo).mean()),
                         "winkler": float((hi - lo + pen).mean()),
                         "coverage_error": float(inside.mean() - L)})
    return pd.DataFrame(rows).sort_values(["nominal", "method"]).reset_index(drop=True)


def sigma_scale_table(preds: pd.DataFrame, reg_state: pd.Series, labels,
                      *, level: float, scales=None) -> pd.DataFrame:
    """The width-correction deliverable: global, and per t-1 regime.

    Two rescalings are reported for each scope because they optimise different
    things and do not coincide:

    * ``coverage_calibrating_scale`` -- the multiplier on ``sigma`` that brings
      PICP to nominal. This is what a risk manager wants: a 90% interval that
      contains 90% of outcomes.
    * ``crps_optimal_scale`` -- the multiplier minimising the CRPS. This is what
      a statistician wants: the best whole predictive distribution. It is
      systematically the larger of the two, because CRPS also rewards the tails.

    Reporting the per-regime scales alongside the global one is what turns the
    Phase-7 verdict into a decision rather than an assertion: if the per-regime
    scales are close to the global one, a single global correction is
    sufficient and the pre-registered null stands; if they diverge materially,
    it does not, and the table says by how much.
    """
    from scipy import stats as _st

    y = preds[ACTUAL_COL].to_numpy(float)
    mu = preds["mc_mu_log"].to_numpy(float)
    sg = preds["mc_sigma_log"].to_numpy(float)
    if scales is None:
        scales = np.round(np.arange(0.50, 3.001, 0.01), 2)
    a = (1.0 - level) / 2.0
    zl, zu = _st.norm.ppf(a), _st.norm.ppf(1.0 - a)

    lagged = align_regime_label(reg_state, preds.index, shift=DEFAULT_REGIME_SHIFT)
    scopes = [("all", np.ones(len(y), dtype=bool))]
    scopes += [(lab, (lagged == float(k)).to_numpy())
               for k, lab in enumerate(labels)]

    rows = []
    for name, m in scopes:
        if int(m.sum()) < 20:
            continue
        yy, mm, ss = y[m], mu[m], sg[m]
        crps = np.array([crps_lognormal(yy, mm, c * ss) for c in scales])
        cov = np.array([np.mean((yy >= np.exp(mm + zl * c * ss))
                                & (yy <= np.exp(mm + zu * c * ss))) for c in scales])
        j_crps = int(np.argmin(crps))
        j_cov = int(np.argmin(np.abs(cov - level)))
        # Flat optima are common here; report the width of the near-optimal
        # plateau so a two-decimal scale is not read as two-decimal precision.
        near = scales[crps <= crps[j_crps] * 1.001]
        rows.append({
            "scope": name, "n": int(m.sum()), "nominal": float(level),
            "picp_at_fitted_sigma": float(np.mean((yy >= np.exp(mm + zl * ss))
                                                  & (yy <= np.exp(mm + zu * ss)))),
            "crps_at_fitted_sigma": float(crps_lognormal(yy, mm, ss)),
            "coverage_calibrating_scale": float(scales[j_cov]),
            "picp_at_calibrating_scale": float(cov[j_cov]),
            "crps_optimal_scale": float(scales[j_crps]),
            "crps_at_optimal_scale": float(crps[j_crps]),
            "crps_plateau_lo": float(near.min()), "crps_plateau_hi": float(near.max()),
        })
    out = pd.DataFrame(rows)
    out.insert(0, "regime_shift", int(DEFAULT_REGIME_SHIFT))
    out.insert(1, "timing", regime_timing_label(DEFAULT_REGIME_SHIFT))
    return out


def uq_significance_tables(preds: pd.DataFrame, base: pd.DataFrame,
                           uqp: pd.DataFrame, reg_state: pd.Series, labels):
    """Extend DM and Giacomini-White to the Phase-7 pairs (roadmap Phase 7).

    The pairs are chosen to isolate one thing each:

    * combined vs ``MC-Dropout-LSTM`` -- what regime conditioning adds *inside*
      the uncertainty-aware family (the Phase-7 question);
    * combined vs ``Regime-LSTM-B`` -- what the MC machinery costs the point
      forecast relative to the Phase-5 model it is built from (should be ~0);
    * combined vs ``HAR-RV`` -- the benchmark any deep alternative must clear;
    * ``MC-Dropout-LSTM`` vs the Phase-3 ``LSTM`` -- the Phase-6 sanity pair,
      carried forward so the master table is self-contained.
    """
    y = preds[ACTUAL_COL].astype(float)
    pool = {COMBINED_NAME: preds[COMBINED_NAME]}
    for c in (MC_NAME, Q_NAME, GAUSS_NAME):
        if c in uqp.columns:
            pool[c] = uqp[c]
    for c in base.columns:
        if c not in (ACTUAL_COL, REGIME_STATE_COL):
            pool.setdefault(c, base[c])

    pairs = [(COMBINED_NAME, MC_NAME), (COMBINED_NAME, BEST_OF_PHASE5),
             (COMBINED_NAME, "HAR-RV"), (COMBINED_NAME, "LSTM"),
             (MC_NAME, "LSTM")]
    h = regime_test_function(reg_state, n_states=len(labels), labels=list(labels),
                             index=y.index)

    dm_rows, gw_rows = [], []
    for a, b in pairs:
        if a not in pool or b not in pool:
            log.warning("skipping pair %s vs %s (column missing)", a, b)
            continue
        j = pd.concat([y.rename("y"), pool[a].rename("a"), pool[b].rename("b")],
                      axis=1, join="inner").dropna()
        d = diebold_mariano(j["y"], j["a"], j["b"])
        dm_rows.append({"model_a": a, "model_b": b, "dm_stat": d["dm_stat"],
                        "p_value": d["p_value"], "mean_loss_diff": d["mean_loss_diff"],
                        "better": a if d["dm_stat"] < 0 else b, "n": d["n"]})
        g = giacomini_white(j["y"], j["a"], j["b"], h.reindex(j.index))
        row = {"model_a": a, "model_b": b, "n": g["n"],
               "gw_regime_stat": g["gw_stat"], "gw_regime_df": g["df"],
               "gw_regime_p": g["p_value"]}
        for lab, mom in zip(labels, g["moments"]):
            row[f"moment_{lab}"] = float(mom)
        moments = np.asarray(g["moments"], dtype=float)
        row["a_relative_edge_in"] = labels[int(np.argmin(moments))]
        wins = [lab for lab, m in zip(labels, moments) if m < 0]
        row["a_wins_in"] = ", ".join(wins) if wins else "none"
        gw_rows.append(row)
    return pd.DataFrame(dm_rows), pd.DataFrame(gw_rows)


def width_by_regime_table(preds: pd.DataFrame, reg_state: pd.Series, labels,
                          *, level: float) -> pd.DataFrame:
    """Does the combined model's interval width and *shape* vary by t-1 regime?

    RQ3's second clause asks whether width "adapts informatively to regime". The
    Phase-6 answer was that it adapts only mechanically -- a fixed multiplicative
    collar on the point forecast, because the aleatoric term is one constant. The
    combined model has one extra channel that could break that: when the regime
    posterior is split, the gate mixes experts that disagree, and the epistemic
    term widens. This table measures whether it does, by reporting mean gate
    entropy and the epistemic share alongside the width in each bucket.
    """
    tag = f"{int(round(level * 100))}"
    y = preds[ACTUAL_COL].astype(float)
    lagged = align_regime_label(reg_state, preds.index, shift=DEFAULT_REGIME_SHIFT)
    lo = preds[f"{COMBINED_NAME}_lo{tag}"].astype(float)
    hi = preds[f"{COMBINED_NAME}_hi{tag}"].astype(float)
    rows = []
    for k, lab in enumerate([*labels, "all"]):
        m = (np.ones(len(preds), dtype=bool) if lab == "all"
             else (lagged == float(k)).to_numpy())
        if int(m.sum()) == 0:
            continue
        s2e = float((preds["mc_sd_epistemic"][m] ** 2).mean())
        s2a = float((preds["mc_sd_aleatoric"][m] ** 2).mean())
        up = interval_hits(y[m], lo[m], hi[m], side="upper")
        dn = interval_hits(y[m], lo[m], hi[m], side="lower")
        rows.append({
            "regime": lab, "n": int(m.sum()),
            "mpiw": float((hi[m] - lo[m]).mean()),
            "mean_log_width": float(np.log(hi[m] / lo[m]).mean()),
            "mean_sigma_log": float(preds["mc_sigma_log"][m].mean()),
            "epistemic_share": s2e / (s2e + s2a) if (s2e + s2a) > 0 else np.nan,
            "mean_gate_entropy": float(preds["gate_entropy"][m].mean()),
            "upper_violation_rate": float(up.mean()),
            "lower_violation_rate": float(dn.mean()),
            "expected_one_sided_rate": expected_rate(level, "upper"),
        })
    out = pd.DataFrame(rows)
    out.insert(0, "regime_shift", int(DEFAULT_REGIME_SHIFT))
    out.insert(1, "timing", regime_timing_label(DEFAULT_REGIME_SHIFT))
    return out


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def plot_calibration(cal: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, grp in cal.groupby("method"):
        g = grp.sort_values("nominal")
        ax.plot(g["nominal"], g["picp"], marker="o", label=method)
    lim = [cal["nominal"].min() - 0.02, cal["nominal"].max() + 0.02]
    ax.plot(lim, lim, ls="--", c="0.4", lw=1, label="perfect calibration")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage (PICP)")
    ax.set_title("Phase 7 — interval calibration, combined vs Phase-6 methods")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return save_fig(fig, path)


def plot_width_by_regime(w: pd.DataFrame, path: Path, *, level: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = w[w["regime"] != "all"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a1.bar(sub["regime"], sub["mpiw"], color="#4C78A8")
    a1.set_yscale("log")
    a1.set_ylabel(f"MPIW at {level:.0%} (log scale)")
    a1.set_title("Interval width by t−1 regime")
    a1.grid(alpha=0.25, axis="y")

    x = np.arange(len(sub))
    a2.bar(x - 0.2, sub["upper_violation_rate"], width=0.4, label="upper", color="#E45756")
    a2.bar(x + 0.2, sub["lower_violation_rate"], width=0.4, label="lower", color="#72B7B2")
    a2.axhline(float(sub["expected_one_sided_rate"].iloc[0]), ls="--", c="0.35", lw=1,
               label="nominal one-sided")
    a2.set_xticks(x, sub["regime"])
    a2.set_ylabel("violation rate")
    a2.set_title("Where the band breaks, by tail")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# Milestone                                                                    #
# --------------------------------------------------------------------------- #
#: Spread in the per-regime coverage-calibrating sigma scale below which one
#: global correction is treated as sufficient.
SIGMA_SPREAD_THRESHOLD = 0.15


def width_correction_verdict(
    sigma: pd.DataFrame,
    *,
    conditional_evidence: bool,
    spread_threshold: float = SIGMA_SPREAD_THRESHOLD,
) -> dict:
    """Global sigma rescale, or per-regime? Decide on **both** available inputs.

    Why this is a function and not two independent sentences
    --------------------------------------------------------
    The first version of this note decided the question twice, in two places,
    from two different inputs, and the two disagreed inside one document: the
    verdict block concluded *"the deliverable is the global width correction"*
    from the coverage tests, while the sigma-scale section seven paragraphs
    later concluded *"the honest deliverable is the per-regime column"* from the
    spread. Both computations were right; neither knew about the other. Same
    failure family as the three generator bugs already logged in this project --
    a claim assembled from one input while a second input bears on it.

    The two inputs answer genuinely different questions and both are needed:

    * **The spread** of the per-regime coverage-calibrating scales is a set of
      *point estimates*. It says how differently each bucket would have to be
      rescaled. It carries no standard error, and the crisis bucket is ~75 days,
      so a wide spread can be noise.
    * **The coverage tests** say whether any t-1 conditioner predicts a miss
      once the level is accounted for and the family-wise rate controlled. That
      is the *tested* question, and it is the one a claim can rest on.

    Four combinations, four different honest conclusions -- in particular a
    material spread with no significant conditioner is **not** licence to
    recommend per-regime rescaling, and a significant conditioner with a narrow
    spread means the interval's *shape* is wrong in a way no rescale fixes.

    Returns ``{'spread', 'material', 'conditional_evidence', 'recommendation',
    'sentence'}``.
    """
    per = sigma[sigma["scope"] != "all"]
    if not len(per):
        return {"spread": float("nan"), "material": False,
                "conditional_evidence": bool(conditional_evidence),
                "recommendation": "global",
                "sentence": "No per-regime scales were estimable, so only the "
                            "global correction is reported."}
    lo = float(per["coverage_calibrating_scale"].min())
    hi = float(per["coverage_calibrating_scale"].max())
    spread = hi - lo
    material = spread > float(spread_threshold)
    ev = bool(conditional_evidence)

    if material and ev:
        rec = "per-regime"
        s = (f"Per-regime coverage-calibrating scales span {lo:.2f}–{hi:.2f} "
             f"(spread {spread:.2f}), and the coverage tests independently find a "
             f"t−1 conditioner that predicts a miss. Point estimates and formal "
             f"tests agree, so **the deliverable is the per-regime column**: a "
             f"single global scale would leave real, testable regime-dependent "
             f"miscalibration on the table.")
    elif material and not ev:
        rec = "global, per-regime indicative"
        s = (f"Per-regime coverage-calibrating scales span {lo:.2f}–{hi:.2f} "
             f"(spread {spread:.2f}), which looks material — but the coverage "
             f"tests find no t−1 conditioner that predicts a miss once the level "
             f"is accounted for and the family-wise rate controlled. These scales "
             f"are point estimates with no standard error attached, on buckets as "
             f"small as the crisis one, so the spread is **not** evidence of "
             f"regime-dependent miscalibration. **The defensible deliverable is "
             f"the global scale**, with the per-regime column reported as "
             f"indicative and explicitly untested.")
    elif ev:
        rec = "global, shape problem noted"
        s = (f"Per-regime coverage-calibrating scales span only {lo:.2f}–{hi:.2f} "
             f"(spread {spread:.2f}), so **one global scale is the right width "
             f"correction** — but the coverage tests do find a t−1 conditioner "
             f"that predicts a miss. A conditioner that matters while the "
             f"required widths do not differ points at the interval's *shape* "
             f"rather than its width: no rescaling of a log-normal band fixes it, "
             f"and that belongs in the Chapter 5 discussion.")
    else:
        rec = "global"
        s = (f"Per-regime coverage-calibrating scales span only {lo:.2f}–{hi:.2f} "
             f"(spread {spread:.2f}) and no t−1 conditioner predicts a miss, so "
             f"**one global scale is sufficient** and the pre-registered "
             f"deliverable stands unamended.")
    return {"spread": spread, "material": material, "conditional_evidence": ev,
            "recommendation": rec, "sentence": s}


def pooled_view_verdict(
    diff_pooled: pd.DataFrame, diff_upper: pd.DataFrame, *, alpha: float = 0.05
) -> dict:
    """Is the pooled two-sided coverage view enough, or does it hide structure?

    Why this is a function and not a sentence -- audits vii / viii
    -------------------------------------------------------------
    The note used to answer this from an **argmax**: if the upper and lower
    tails peaked in the same state it printed *"the pooled view captures the
    structure adequately here"*, and otherwise printed the opposite. Neither
    branch tested anything, and on the headline profile the claim was refuted by
    its own document -- 0 of 16 pooled subsample differences survive Holm
    against 5 of 16 on the upper tail, so the pooled view demonstrably sees
    less. Fifth defect of one family: a hardcoded conclusion, a shadowed loop
    variable printing lambda=300, an untested "non-monotone", a width verdict
    that contradicted itself, and this.

    Adequacy is a claim about the *tests*, so decide it from the tests: how many
    subsample coverage differences survive family-wise control on each side.
    Four cases, four honest readings, with the counts printed alongside so a
    reader can check the arithmetic. The fourth -- neither side finding anything
    -- makes **no** adequacy claim in either direction, which is exactly the
    branch the argmax version could not express.

    Parameters
    ----------
    diff_pooled, diff_upper : the ``difference in coverage`` rows of the
        conditional-tests table for ``side="both"`` and ``side="upper"``, at one
        nominal level. Only ``p_holm`` is read. Empty or ``None`` counts as zero
        tests, which lands in the "no structure either way" branch.

    Returns
    -------
    dict with ``verdict`` (one of ``pooled understates`` / ``pooled is sharper``
    / ``views agree`` / ``no structure either way``), the four counts, and
    ``sentence`` -- the rendered claim, which always carries its own evidence.
    """
    def _counts(df) -> tuple[int, int]:
        if df is None or not len(df):
            return 0, 0
        return int((df["p_holm"] < alpha).sum()), int(len(df))

    up_sig, up_n = _counts(diff_upper)
    po_sig, po_n = _counts(diff_pooled)
    counts = (f"**{po_sig} of {po_n}** pooled and **{up_sig} of {up_n}** "
              f"upper-tail")

    if up_sig > po_sig:
        verdict = "pooled understates"
        sentence = (
            "Whether the pooled two-sided view is enough is a question about the "
            "tests rather than about which state each argmax lands in — and here "
            f"it is not enough: {counts} subsample coverage differences survive "
            "Holm. Pooling nets the two tails against each other and sees less "
            "structure than the one-sided view, which is why both are reported.")
    elif po_sig > up_sig:
        verdict = "pooled is sharper"
        sentence = (
            "On the tests the pooled two-sided view is the sharper of the two "
            f"here: {counts} subsample coverage differences survive Holm, so the "
            "structure is not confined to one tail.")
    elif up_sig > 0:
        verdict = "views agree"
        sentence = (
            f"The two views agree on the tests — {counts} subsample coverage "
            "differences survive Holm — so the pooled table loses nothing the "
            "one-sided tables find.")
    else:
        verdict = "no structure either way"
        sentence = (
            "Neither view finds subsample structure that survives family-wise "
            f"control ({counts} differences survive Holm), so the ordering above "
            "is descriptive only and no adequacy claim is made in either "
            "direction.")

    return {"verdict": verdict, "sentence": sentence, "alpha": float(alpha),
            "n_pooled_significant": po_sig, "n_pooled": po_n,
            "n_upper_significant": up_sig, "n_upper": up_n}


def write_milestone(ctx: dict, out_path: Path) -> Path:
    """Generate ``m07_combined.md``.

    Every qualitative claim below is computed from the tables, in both
    directions, so the note can report the pre-registered expectation being
    *falsified* as readily as confirmed. Three generator bugs of this family
    have already been caught in this project (hardcoded conclusions, a shadowed
    loop variable, and an untested "non-monotone"), so nothing here is a
    template with a verdict baked into the branch.
    """
    name = ctx["name"]
    master: pd.DataFrame = ctx["master"]
    cal: pd.DataFrame = ctx["calibration"]
    cov: pd.DataFrame = ctx["coverage"]
    con: pd.DataFrame = ctx["coverage_conditional"]
    sig: pd.DataFrame = ctx["sigma_scale"]
    width: pd.DataFrame = ctx["width_by_regime"]
    dm: pd.DataFrame = ctx["dm"]
    gw: pd.DataFrame = ctx["gw"]
    labels = ctx["labels"]
    level = float(ctx["headline_level"])
    p: list[str] = []

    p.append(f"# Milestone m07 — Phase 7: combined regime × uncertainty model, "
             f"and RQ3 made inferential\n")
    p.append(f"*Profile:* `{name}` · *test window:* {ctx['oos_start']} → "
             f"{ctx['oos_end']} (n={ctx['n_oos']}) · *headline level:* {level:.0%} · "
             f"*MC samples:* {ctx['mc_samples']}\n")
    if ctx.get("fast"):
        p.append("> ⚠️ **SMOKE RUN** (`--fast`): epochs and MC samples reduced. "
                 "Numbers here are not the reported result.\n")

    # ---- scope ----
    p.append("## Scope and what is being isolated\n")
    p.append(f"The combined model is **{COMBINED_NAME}** — the Phase-5 "
             f"`{BEST_OF_PHASE5}` mixture-of-experts with MC-Dropout inference, "
             f"training inherited unchanged. \"Best of Phase 5\" is "
             f"`{BEST_OF_PHASE5}`, {BEST_OF_PHASE5_RULE} The rule was fixed "
             "before this phase ran.\n")
    p.append("Because training is inherited and the hyperparameters are still "
             "the Phase-3 validation selection, the contrast "
             f"`{MC_NAME}` → `{COMBINED_NAME}` isolates **regime conditioning "
             "inside the uncertainty-aware family** and nothing else: same "
             "network shape, same seed, same folds, same MC sample count.\n")

    # ---- pre-registered expectation, then the verdict ----
    p.append("## The pre-registered expectation\n")
    p.append("Recorded in `ROADMAP.md` **before** this phase was run: *Kupiec "
             "should reject decisively — the intervals are 3.5–7.9 pp narrow — "
             "while Christoffersen's independence component and any t−1 "
             "conditioning variable should not, because the failures are "
             "unclustered and unpredictable.* If that holds, miscalibration is a "
             "**level** problem rather than a **timing** problem, the combined "
             "model is a pre-registered null, and the deliverable is a global "
             "width correction.\n")

    def _cov(method, side, test, lvl=None):
        s = cov[(cov["method"] == method) & (cov["side"] == side)
                & (cov["test"] == test)
                & (cov["nominal_level"] == (level if lvl is None else lvl))]
        return s.iloc[0] if len(s) else None

    methods_tested = [m for m in cov["method"].unique()]
    kup = {m: _cov(m, "both", "Kupiec LR_uc") for m in methods_tested}
    ind = {m: _cov(m, "both", "Christoffersen LR_ind") for m in methods_tested}
    kup_all_reject = all(r is not None and r["p_value"] < 0.05 for r in kup.values())
    ind_none_reject = all(r is not None and r["p_value"] >= 0.05 for r in ind.values())

    inc = con[(con["kind"] == "DQ incremental") & (con["side"] == "both")
              & (con["nominal_level"] == level)]
    inc_any = bool(len(inc) and (inc["p_holm"] < 0.05).any())
    diff_up = con[(con["kind"] == "difference in coverage") & (con["side"] == "upper")
                  & (con["nominal_level"] == level)]
    diff_up_sig = diff_up[diff_up["p_holm"] < 0.05] if len(diff_up) else diff_up
    # The pooled counterpart, needed so the tail section can decide whether the
    # two-sided view is adequate from the TESTS rather than from an argmax.
    diff_both = con[(con["kind"] == "difference in coverage") & (con["side"] == "both")
                    & (con["nominal_level"] == level)]

    p.append("### Verdict, computed from the tables below\n")
    p.append(f"| clause | expected | found |\n|---|---|---|")
    p.append(f"| Kupiec rejects (level is wrong) | yes | "
             f"**{'yes' if kup_all_reject else 'NO'}** — "
             + "; ".join(f"{m} p={kup[m]['p_value']:.3g}" for m in methods_tested
                         if kup[m] is not None) + " |")
    p.append(f"| Christoffersen independence does *not* reject (no clustering) | yes | "
             f"**{'yes' if ind_none_reject else 'NO'}** — "
             + "; ".join(f"{m} p={ind[m]['p_value']:.3g}" for m in methods_tested
                         if ind[m] is not None) + " |")
    p.append(f"| no t−1 conditioner predicts a miss (pooled hit) | yes | "
             f"**{'no conditioner survives Holm' if not inc_any else 'FALSIFIED — at least one does'}** |")
    p.append(f"| no t−1 subsample differs in *upper-tail* coverage | (not pre-registered) | "
             f"**{len(diff_up_sig)} of {len(diff_up)} survive Holm** |\n")

    # One decision, made once, from BOTH inputs -- the tested evidence and the
    # per-regime rescale spread. An earlier version decided it twice, here and
    # again in the deliverable section, from different inputs, and the two
    # disagreed inside one note. See width_correction_verdict.
    conditional_evidence = bool(inc_any or len(diff_up_sig))
    wv = width_correction_verdict(sig, conditional_evidence=conditional_evidence)
    ctx["width_verdict"] = wv

    if kup_all_reject and ind_none_reject and not conditional_evidence:
        p.append("**The pre-registration holds in full.** Miscalibration is a "
                 "level problem, not a timing problem, and the combined model is "
                 "a pre-registered null. "
                 + ("The deliverable is the global width correction in the "
                    "σ-scale table below.\n" if wv["recommendation"] == "global"
                    else "On the width correction itself the σ-scale table needs "
                         "reading with care — see the deliverable section, which "
                         "reconciles the per-regime point estimates against the "
                         "absence of tested evidence.\n"))
    elif kup_all_reject and ind_none_reject:
        p.append("**The pre-registration holds in its main clauses and is "
                 "partially falsified in its last one.** The level is wrong and "
                 "the misses are unclustered, as predicted — but the claim that "
                 "*nothing* knowable at t−1 predicts a miss does not survive "
                 "contact with the one-sided tests. That is the more interesting "
                 "outcome and is reported as such below, not absorbed. On the "
                 f"remedy, the recommendation is **{wv['recommendation']}** — the "
                 "deliverable section gives the reasoning.\n")
    else:
        p.append("**The pre-registration is falsified in a main clause.** Read "
                 "the table above before anything else in this note.\n")

    # ---- master table ----
    p.append(f"\n## Master results table (headline level {level:.0%})\n")
    show = master[["rank_qlike", "model", "family", "qlike", "qlike_transitional",
                   "mse", "mae", "picp", "mpiw", "winkler", "crps", "n"]]
    p.append(_md_table(show, floats={"qlike": 4, "qlike_transitional": 4,
                                     "picp": 4, "coverage_error": 4}))
    p.append(f"\n*`qlike_transitional` is the transitional bucket on the "
             f"**t−1** label ({regime_timing_label(DEFAULT_REGIME_SHIFT)}), "
             f"n={int(master['n_transitional'].max())}. Interval columns are "
             f"blank for models that produce no predictive distribution — that "
             f"contrast is the point of the table.*\n")

    ranked = master[master["ranked"]]
    n_ranked = int(len(ranked))
    if (~master["ranked"]).any():
        _s = master[~master["ranked"]]
        p.append(
            f"\n*Rows with a blank `rank_qlike` are sensitivities, not competitors: "
            f"{', '.join(f'**{r.model}** ({r.qlike:.4f})' for r in _s.itertuples())}. "
            f"`{Q_MEAN_NAME}` is the quantile model's own forecast retransformed onto "
            f"the mean scale QLIKE elicits — the correction every other deep model in "
            f"this table already carries — so it is the row to read when comparing the "
            f"pinball head's point accuracy against the mean-scale models, and it "
            f"competes with nothing. Ranks below count the {n_ranked} models only.*\n")

    best_q = ranked.iloc[0]
    comb = master[master["model"] == COMBINED_NAME].iloc[0]
    mcrow = master[master["model"] == MC_NAME]
    p.append(f"\nPooled, the best model on QLIKE is **{best_q['model']}** "
             f"({best_q['qlike']:.4f}); the combined model ranks "
             f"**{int(comb['rank_qlike'])} of {n_ranked}** at "
             f"{comb['qlike']:.4f}.")
    if len(mcrow):
        mc = mcrow.iloc[0]
        delta = comb["qlike"] - mc["qlike"]
        dpc = 100.0 * delta / mc["qlike"]
        p.append(f" Against the regime-agnostic `{MC_NAME}` "
                 f"({mc['qlike']:.4f}) that is a change of {delta:+.4f} "
                 f"({dpc:+.1f}%), and in the transitional bucket "
                 f"{comb['qlike_transitional']:.4f} vs "
                 f"{mc['qlike_transitional']:.4f}.\n")

    # ---- significance ----
    p.append("\n## Significance on the Phase-7 pairs\n")
    p.append(_md_table(dm, floats={"dm_stat": 4, "p_value": 4, "mean_loss_diff": 6}))
    p.append("\n**Giacomini–White, conditional on the lagged regime indicator.** "
             "A negative moment means the first model has the lower loss in that "
             "state.\n")
    p.append(_md_table(gw, floats={"gw_regime_stat": 4, "gw_regime_p": 6,
                                   **{f"moment_{l}": 6 for l in labels}}))

    # ---- coverage tests ----
    p.append(f"\n## RQ3, made inferential — coverage tests on the full "
             f"{ctx['n_oos']}-day sample\n")
    p.append("Every regressor is lagged into the t−1 information set, and every "
             "test runs on the **full** sample. Two rules make these numbers "
             "mean what they appear to mean, both learned from earlier audits of "
             "this project:\n")
    p.append("1. **A subsample selector must be measurable at t−1.** The ex-post "
             "transition indicator `1{s_t ≠ s_(t−1)}` is detected by the same "
             "one-day surprise that breaks an interval; a test on it has no "
             "nominal size. Only `1{s_(t−1) ≠ s_(t−2)}` appears here.\n")
    p.append("2. **A subsample claim is a *difference*, never a comparison "
             "against nominal.** These intervals are all too narrow globally, so "
             "a Kupiec test against nominal rejects on *any* subsample with "
             "power — that would report a level failure as a timing failure. "
             "Subsample rows below are two-sample differences with Holm control; "
             "conditional DQ rows are **incremental** over level and hit "
             "dynamics, for the same reason one level up.\n")
    p.append(_md_table(
        cov[cov["nominal_level"] == level][
            ["method", "side", "test", "stat", "df", "p_value", "violation_rate",
             "expected_violation_rate", "n"]],
        floats={"stat": 3, "p_value": 5, "violation_rate": 4,
                "expected_violation_rate": 4}))

    p.append("\n### Conditional tests (t−1 conditioners only)\n")
    if len(con):
        c = con[con["nominal_level"] == level][
            ["method", "side", "kind", "conditioner", "stat", "df", "p_value",
             "p_holm", "n", "detail"]]
        p.append(_md_table(c, floats={"stat": 3, "p_value": 5, "p_holm": 5}))
        p.append(f"\n*Holm adjustment is within each (method, level, side) "
                 f"family; family sizes are in the CSV.*\n")
    else:
        p.append("*(no conditional test could be formed)*\n")

    # ---- the upper-tail finding, computed ----
    p.append("\n### Where the band actually breaks\n")
    p.append(_md_table(width, floats={"mpiw": 6, "mean_log_width": 4,
                                      "mean_sigma_log": 4, "epistemic_share": 5,
                                      "mean_gate_entropy": 4,
                                      "upper_violation_rate": 4,
                                      "lower_violation_rate": 4}))
    w_no_all = width[width["regime"] != "all"]
    if len(w_no_all):
        worst_up = w_no_all.loc[w_no_all["upper_violation_rate"].idxmax()]
        worst_dn = w_no_all.loc[w_no_all["lower_violation_rate"].idxmax()]
        nom1 = float(w_no_all["expected_one_sided_rate"].iloc[0])
        p.append(f"\nThe **upper** tail — realized variance above the band, i.e. "
                 f"the day the model understated risk — breaks most often in the "
                 f"**{worst_up['regime']}** state "
                 f"({worst_up['upper_violation_rate']:.4f} against a nominal "
                 f"{nom1:.3f}); the **lower** tail breaks most often in "
                 f"**{worst_dn['regime']}** "
                 f"({worst_dn['lower_violation_rate']:.4f}). ")
        # Descriptive fact first -- which states the two tails peak in -- then
        # the claim, which is decided from the tests. Keeping those two separate
        # is the fix for audit vii's prose defect: an argmax cannot license a
        # statement about whether the pooled view is adequate.
        p.append("The two tails peak in the **same** state. "
                 if worst_up["regime"] == worst_dn["regime"]
                 else "The two tails peak in **different** states. ")
        pv = pooled_view_verdict(diff_both, diff_up)
        ctx["pooled_view_verdict"] = pv
        p.append(pv["sentence"] + "\n")
        if len(diff_up_sig):
            p.append("\nUpper-tail coverage differences surviving Holm:\n")
            for _, r in diff_up_sig.iterrows():
                p.append(f"- `{r['method']}` · **{r['conditioner']}** — "
                         f"{r['detail']}; z_HAC={r['stat']:.3f}, "
                         f"p={r['p_value']:.4g}, Holm={r['p_holm']:.4g}")
            p.append("")

    # ---- gate entropy: the one mechanism the combined model adds ----
    if "mean_gate_entropy" in width.columns and len(w_no_all) > 1:
        ent = w_no_all["mean_gate_entropy"]
        share = w_no_all["epistemic_share"]
        p.append(f"\n**Does the gate widen the band where the regime is "
                 f"ambiguous?** Mean gate entropy ranges "
                 f"{ent.min():.3f}–{ent.max():.3f} nats across states (max "
                 f"possible {np.log(len(labels)):.3f}), and the epistemic share "
                 f"of predictive variance ranges "
                 f"{share.min():.2%}–{share.max():.2%}. "
                 + ("Since the epistemic term is a small fraction of the total "
                    "everywhere, the gate cannot move the interval width much "
                    "even where it is maximally uncertain — this is the "
                    "mechanism behind the combined model being a null on "
                    "calibration.\n" if float(share.max()) < 0.05 else
                    "The epistemic term is large enough in at least one state "
                    "for the gate to move the width materially, so the combined "
                    "model is *not* mechanically constrained to be a null here — "
                    "read the calibration table on its own terms.\n"))

    # ---- the deliverable ----
    p.append(f"\n## The deliverable: a width correction\n")
    p.append(_md_table(sig, floats={"picp_at_fitted_sigma": 4,
                                    "crps_at_fitted_sigma": 8,
                                    "coverage_calibrating_scale": 2,
                                    "picp_at_calibrating_scale": 4,
                                    "crps_optimal_scale": 2,
                                    "crps_at_optimal_scale": 8,
                                    "crps_plateau_lo": 2, "crps_plateau_hi": 2}))
    g = sig[sig["scope"] == "all"]
    if len(g):
        g0 = g.iloc[0]
        per = sig[sig["scope"] != "all"]
        p.append(f"\nGlobally, σ × **{g0['coverage_calibrating_scale']:.2f}** "
                 f"brings coverage to {g0['picp_at_calibrating_scale']:.4f} "
                 f"against a {level:.0%} nominal, and σ × "
                 f"**{g0['crps_optimal_scale']:.2f}** minimises CRPS "
                 f"(plateau {g0['crps_plateau_lo']:.2f}–"
                 f"{g0['crps_plateau_hi']:.2f}, so quote it as approximate).")
        if len(per):
            p.append(" " + ctx["width_verdict"]["sentence"] + "\n")

    p.append("\n## Reproduce\n")
    # Every command here must be runnable as printed. Two things this block used
    # to get wrong: the --from-predictions line dropped --config, so running it
    # from the comparator's note rebuilt the HEADLINE tables; and the
    # report_coverage line passed a bare filename, which `repo_path` resolves
    # against the repo root and so cannot exist.
    cfg_flag = (f" --config {ctx['config_name']}"
                if ctx.get("config_name") not in (None, "combined") else "")
    pred_rel = f"results/predictions/{ctx['pred_path'].name}"
    p.append("```\npython -m src.experiments.run_combined" + cfg_flag
             + "\n# tables only, from the persisted forecasts (seconds, no training):\n"
             f"python -m src.experiments.run_combined{cfg_flag} --from-predictions\n"
             "# the coverage tests alone, against any UQ predictions file:\n"
             f"python -m src.experiments.report_coverage --predictions {pred_rel}\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p), encoding="utf-8")
    log.info("wrote milestone %s", out_path)
    return out_path


def _md_table(df: pd.DataFrame, floats: dict[str, int] | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table with per-column rounding."""
    d = df.copy()
    floats = floats or {}
    for c in d.columns:
        if c in floats:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floats[c]}g}"
                            if abs(v) < 1e-3 and v != 0 else f"{v:.{floats[c]}f}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
        else:
            d[c] = d[c].astype(str)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(r) + " |" for r in d.astype(str).to_numpy()]
    return "\n".join([head, rule, *body])


# --------------------------------------------------------------------------- #
# One profile                                                                  #
# --------------------------------------------------------------------------- #
def build_model(cfg, *, max_epochs, mc_samples) -> MCDropoutRegimeExpertForecaster:
    m = cfg.model
    return MCDropoutRegimeExpertForecaster(
        base_features=tuple(OmegaConf.to_container(m.base_features, resolve=True)),
        gate_cols=tuple(OmegaConf.to_container(m.regime_features, resolve=True)),
        lookback=int(m.lookback), hidden_size=int(m.hidden_size),
        num_layers=int(m.num_layers), dropout=float(m.dropout), lr=float(m.lr),
        weight_decay=float(m.weight_decay), batch_size=int(m.batch_size),
        max_epochs=int(max_epochs if max_epochs else m.max_epochs),
        patience=int(m.patience), grad_clip=float(m.grad_clip),
        val_fraction=float(m.val_fraction), smearing=bool(m.smearing),
        refit_every_folds=int(cfg.harness.refit_every_folds), seed=int(cfg.seed),
        mc_samples=int(mc_samples if mc_samples else cfg.uq.mc_dropout.mc_samples),
        mc_seed=int(cfg.uq.mc_dropout.get("mc_seed", cfg.seed)),
        name=COMBINED_NAME,
    )


def run_profile(data_cfg, cfg, profile, args) -> dict:
    name = profile.name
    log.info("=" * 70)
    log.info("PHASE 7 | PROFILE %s | target=%s", name, profile.target)

    labels = list(OmegaConf.to_container(cfg.regime.labels, resolve=True))
    levels = [float(x) for x in OmegaConf.to_container(cfg.uq.levels, resolve=True)]
    headline_level = float(cfg.uq.headline_level)
    pred_path = repo_path(cfg.paths.predictions, f"m07_combined_{name}.parquet")

    # The FULL Phase-4 regime series (never the copy sliced into a predictions
    # parquet) so every t-1 lag can reach the day before the window opens.
    reg_full = from_parquet(repo_path(cfg.regime.source))[cfg.regime.state_col].astype(float)

    if getattr(args, "from_predictions", False):
        if not pred_path.exists():
            raise FileNotFoundError(
                f"--from-predictions needs {pred_path}; run the full phase first.")
        log.info("--from-predictions: reusing %s (no training)", pred_path)
        preds = from_parquet(pred_path).copy()
        n_mc = int(cfg.uq.mc_dropout.mc_samples)
    else:
        frame = build_econometric_frame(data_cfg, target=profile.target, include_vix=False)
        frame = attach_regime(frame, cfg.regime)
        sp = profile.splits
        split_cfg = SplitConfig(
            train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
            test_start=sp.test_start, test_end=sp.test_end,
            refit_frequency_days=int(cfg.harness.refit_frequency_days),
            scheme=cfg.harness.scheme,
        )
        max_epochs = args.max_epochs if args.max_epochs else (4 if args.fast else None)
        mc_samples = args.mc_samples if args.mc_samples else (10 if args.fast else None)
        model = build_model(cfg, max_epochs=max_epochs, mc_samples=mc_samples)
        log.info("Phase-7 model: %s | config=%s", model.name, model.config())
        uq = run_uq_walk_forward([model], frame, split_cfg,
                                 eval_segment=cfg.harness.eval_segment)
        preds = build_predictions(uq[model.name], frame["rv"],
                                  frame[REGIME_STATE_COL], levels)
        n_mc = int(model.mc_samples)

    base = from_parquet(repo_path(profile.baseline_predictions))
    uqp = from_parquet(repo_path(profile.uq_predictions))
    # Build the retransformed quantile point here rather than requiring the
    # Phase-6 parquet to have been rewritten: it is a pure function of the
    # persisted quantiles, and one implementation (run_uq's) serves both phases.
    uqp = add_quantile_mean_column(uqp.copy())

    # The claim this phase rests on, checked against the artefacts rather than
    # asserted (audit vii). Raises if the combined model is not the Phase-5 one.
    assert_inherits_phase5_training(preds, base)

    master = master_table(preds, base, uqp, reg_full, labels, level=headline_level)
    cal = calibration_table(preds, uqp, levels)
    per_regime = per_regime_interval_metrics(
        preds[ACTUAL_COL].astype(float),
        preds[f"{COMBINED_NAME}_lo{int(round(headline_level*100))}"].astype(float),
        preds[f"{COMBINED_NAME}_hi{int(round(headline_level*100))}"].astype(float),
        reg_full, labels, level=headline_level, shift=DEFAULT_REGIME_SHIFT)
    dm, gw = uq_significance_tables(preds, base, uqp, reg_full, labels)
    sigma = sigma_scale_table(preds, reg_full, labels, level=headline_level)
    width = width_by_regime_table(preds, reg_full, labels, level=headline_level)

    # Coverage tests: the combined model's own intervals PLUS every Phase-6
    # method, scored by the identical code path so the comparison is exact.
    joined = preds.join(
        uqp[[c for c in uqp.columns if c not in preds.columns]], how="left")
    dq_lags = int(cfg.coverage.dq_lags)
    pairs = [(COMBINED_NAME, L) for L in levels]
    for m in (MC_NAME, GAUSS_NAME, Q_NAME):
        for L in levels:
            tag = f"{int(round(L * 100))}"
            if f"{m}_lo{tag}" in joined.columns and f"{m}_hi{tag}" in joined.columns:
                pairs.append((m, L))
    cov = unconditional_table(joined, pairs, dq_lags=dq_lags)
    con = conditional_table(joined, reg_full, pairs, labels, dq_lags=dq_lags)

    log.info("PHASE 7 master:\n%s", master.round(5).to_string(index=False))
    log.info("PHASE 7 calibration:\n%s", cal.round(5).to_string(index=False))
    log.info("PHASE 7 sigma-scale:\n%s", sigma.round(4).to_string(index=False))

    # --- persist ---
    if not getattr(args, "from_predictions", False):
        to_parquet(preds, pred_path)
    tables = repo_path(cfg.paths.tables)
    ensure_dir(tables)
    stem = f"m07_combined_{name}"
    master.to_csv(tables / f"{stem}_master.csv", index=False)
    cal.to_csv(tables / f"{stem}_calibration.csv", index=False)
    pr = per_regime.copy()
    pr.insert(0, "regime_shift", int(DEFAULT_REGIME_SHIFT))
    pr.insert(1, "timing", regime_timing_label(DEFAULT_REGIME_SHIFT))
    pr.to_csv(tables / f"{stem}_per_regime.csv", index=False)
    dm.to_csv(tables / f"{stem}_dm.csv", index=False)
    gw.to_csv(tables / f"{stem}_gw.csv", index=False)
    sigma.to_csv(tables / f"{stem}_sigma_scale.csv", index=False)
    width.to_csv(tables / f"{stem}_width_by_regime.csv", index=False)
    cov.to_csv(tables / f"{stem}_coverage_tests.csv", index=False)
    con.to_csv(tables / f"{stem}_coverage_conditional.csv", index=False)

    figdir = repo_path(cfg.paths.figures, "m07")
    ensure_dir(figdir)
    plot_calibration(cal, figdir / f"{name}_calibration.png")
    plot_width_by_regime(width, figdir / f"{name}_width_by_regime.png",
                         level=headline_level)

    return {
        "name": name, "master": master, "calibration": cal, "per_regime": per_regime,
        "dm": dm, "gw": gw, "sigma_scale": sigma, "width_by_regime": width,
        "coverage": cov, "coverage_conditional": con, "labels": labels,
        "headline_level": headline_level, "mc_samples": n_mc,
        "n_oos": int(len(preds)),
        "oos_start": str(preds.index.min().date()),
        "oos_end": str(preds.index.max().date()),
        "fast": bool(getattr(args, "fast", False)),
        "pred_path": pred_path, "config_name": getattr(args, "config", "combined"),
        "selector": selector_timing_label(DEFAULT_SELECTOR_SHIFT),
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_combined")
    ap.add_argument("--config", default="combined")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true",
                    help="few epochs + few MC samples (smoke test, not a result)")
    ap.add_argument("--max-epochs", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=0)
    ap.add_argument("--from-predictions", action="store_true",
                    help="skip training; rebuild every table/figure/milestone from "
                         "the persisted Phase-7 predictions parquet")
    args = ap.parse_args(argv)
    args.max_epochs = args.max_epochs or None
    args.mc_samples = args.mc_samples or None

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "07_combined",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    milestone_file = str(cfg.paths.get("milestone_file") or "m07_combined.md")
    for profile in cfg.profiles:
        ctx = run_profile(data_cfg, cfg, profile, args)
        write_milestone(ctx, repo_path(cfg.paths.milestones, milestone_file))
    log.info("Phase 7 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
