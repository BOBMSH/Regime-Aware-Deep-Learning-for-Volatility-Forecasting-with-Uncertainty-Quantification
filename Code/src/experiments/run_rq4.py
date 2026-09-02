"""Phase 8 / RQ4 -- does the S&P 500 result transfer to other equity indices?

RQ4 (Ch1 §1.6, pre-registered as *secondary*; method in Ch3 §3.9) asks whether the
answers to RQ1-RQ3 are properties of the **method** or of the **market it was
developed on**. It is answered on a deliberately reduced design -- the econometric
baselines and one deep model, on two further indices whose realized variance is
constructed identically to the S&P's -- because re-running the full board would
multiply comparisons without adding evidence about transfer. That model is
``DEEP_MODEL`` below, fixed by the pre-registered "best of Phase 5" rule (Ch3
§3.9); it is NOT the lowest-QLIKE deep model on the S&P, where it ranks third.

This module does not fit anything. Phases 2, 4 and 5 are re-run per asset from
their own Phase-8 configs; this reads what they wrote and answers the one
question those runs were for::

    python -m src.experiments.run_econometric  --config econometric_rq4
    python -m src.experiments.run_regimes      --config hmm_rq4
    python -m src.experiments.run_regime_lstm  --config regime_lstm_rq4
    python -m src.experiments.run_rq4                                  # <- here

Outputs (``results/tables/``):

* ``m08_rq4_metrics.csv``  -- pooled point metrics, every model on every asset.
* ``m08_rq4_ranks.csv``    -- rank-agreement statistics against the S&P ordering.
* ``m08_rq4_tests.csv``    -- for each asset, the pooled Diebold-Mariano and the
  regime-conditional Giacomini-White test of the deep model against each
  econometric baseline, with the per-state moments.
* ``m08_rq4_per_regime.csv`` -- per-state QLIKE by the ``t-1`` label.
* ``m08_rq4_regimes.csv``  -- each asset's *own* selected penalty, BIC-preferred
  K, state persistence and state shares, beside the S&P's.
* ``m08_rq4_sample.csv``   -- the calendar-join attrition Ch3 §3.9 commits to
  reporting rather than absorbing.
* ``m08_rq4_verdict.csv``  -- the transfer verdict per asset, with the facts it
  was computed from.
* ``results/milestones/m08_rq4.md`` -- the note.

**The reference is read, never hardcoded.** Every S&P quantity this module
compares against is loaded from the committed Phase-5 artefacts at run time. The
project has been bitten six times by a generated claim that asserted what nothing
had tested (see the milestone-prose fixes of 2026-08-19 and 2026-08-24), and a
literal 0.2745 pasted into a "does it replicate?" comparison would be the seventh
and the worst-placed: it would keep agreeing after the number it mirrors moved.

**The verdict is computed by one tested function**, :func:`transfer_verdict`,
whose cases are exhaustive and whose "indeterminate" branch is real. Transfer is
the kind of question where a null is a result, and the note must be able to say
"this does not replicate" as fluently as the alternative.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import attrition_record, frame_for_profile
from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    align_regime_label,
    regime_timing_label,
)
from src.evaluation.rolling import ACTUAL_COL, evaluate_predictions
from src.experiments.report_gw import joint_table, per_regime_table
from src.utils.config import (load_config, milestone_path, repo_path,
                              snapshot_config)
from src.utils.io import ensure_dir, from_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("rq4")

#: The model whose transfer is under test. Fixed by the pre-registered "best of
#: Phase 5" rule (configs/combined.yaml, ROADMAP Phase 7): the architecture whose
#: advantage is both large and carried by a formal test. Re-deriving "best" from
#: this phase's own results would be selection on the outcome.
DEEP_MODEL = "Regime-LSTM-B"

#: The benchmark the deep model must clear. HAR-RV is the de facto benchmark on
#: realized-volatility targets (Corsi 2009; Ch2 §2.3), so it carries the verdict;
#: the others are reported beside it but do not decide replication.
BENCHMARK = "HAR-RV"

#: Baselines the deep model is tested against on every asset, benchmark first.
TEST_BASELINES = (BENCHMARK, "GARCH", "EGARCH", "RW-RV")

#: The primary-asset profile whose result is the reference for "does it transfer?".
SPX_PROFILE = "intraday_2019_2022"

#: Significance level for every replication decision. One level, fixed here, used
#: for both the reference and the robustness assets -- comparing a 5% verdict
#: against a 10% one would manufacture disagreement.
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# The verdict                                                                  #
# --------------------------------------------------------------------------- #
def transfer_verdict(reference: dict, candidate: dict, *, alpha: float = ALPHA) -> dict:
    """Does one asset's deep-vs-benchmark result reproduce the S&P's?

    Both arguments describe the *same* comparison on different assets and carry
    four facts each: ``qlike_deep`` and ``qlike_benchmark`` (which give the
    sign), ``dm_p`` (pooled significance) and ``gw_p`` (regime-conditional
    significance).

    The facts are graded separately because the S&P result is itself a
    conjunction -- the deep model is *ahead in level, not significantly so
    pooled, and significantly so conditional on the lagged state*. Collapsing
    them into one boolean would report "it replicates" for an asset that agrees
    on the easy part and differs on the part the finding rests on.

    ``sign`` is deliberately not tested for significance: it is the direction of
    the point estimate, and ``dm_p`` is where its significance is graded. An
    asset whose sign agrees while the conditional test disagrees is reported as
    agreeing on direction only, which is a weaker claim than "replicates" and is
    labelled as such.

    An input whose ``qlike`` or ``p`` is missing or non-finite yields
    ``indeterminate`` rather than a silently coerced comparison -- the case that
    arises when a model column is absent from a board, which the reduced design
    makes possible.
    """
    def _facts(d: dict):
        try:
            qd, qb = float(d["qlike_deep"]), float(d["qlike_benchmark"])
            dm, gw = float(d["dm_p"]), float(d["gw_p"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(np.isfinite(v) for v in (qd, qb, dm, gw)):
            return None
        return {"deep_ahead": bool(qd < qb), "dm_sig": bool(dm < alpha),
                "gw_sig": bool(gw < alpha)}

    ref, cand = _facts(reference), _facts(candidate)
    out = {
        "deep_model": DEEP_MODEL,
        "benchmark": BENCHMARK,
        "alpha": float(alpha),
        "reference_asset": reference.get("asset", ""),
        "asset": candidate.get("asset", ""),
    }
    if ref is None or cand is None:
        out.update({
            "verdict": "indeterminate",
            "sign_agrees": "", "pooled_agrees": "", "conditional_agrees": "",
            "detail": ("a required quantity is missing or non-finite on "
                       f"{'the reference' if ref is None else 'this asset'}; no "
                       "comparison was made"),
        })
        return out

    sign_ok = ref["deep_ahead"] == cand["deep_ahead"]
    dm_ok = ref["dm_sig"] == cand["dm_sig"]
    gw_ok = ref["gw_sig"] == cand["gw_sig"]

    if sign_ok and dm_ok and gw_ok:
        verdict = "replicates"
        detail = ("all three agree with the reference: the same direction, the "
                  "same pooled verdict and the same conditional verdict")
    elif sign_ok and gw_ok and not dm_ok:
        verdict = "replicates conditionally"
        detail = ("direction and the conditional test agree; the pooled test "
                  "does not, so the finding transfers but its pooled visibility "
                  "differs")
    elif sign_ok and not gw_ok:
        verdict = "direction only"
        detail = ("the deep model is on the same side of the benchmark, but the "
                  "conditional test disagrees with the reference -- the "
                  "state-dependence the S&P result rests on does not reproduce")
    elif not sign_ok and (gw_ok or dm_ok):
        verdict = "does not replicate"
        detail = ("the sign of the level difference reverses; agreement on a "
                  "test of a difference running the other way is not transfer")
    else:
        verdict = "does not replicate"
        detail = "neither the direction nor either test agrees with the reference"

    out.update({
        "verdict": verdict,
        "sign_agrees": bool(sign_ok),
        "pooled_agrees": bool(dm_ok),
        "conditional_agrees": bool(gw_ok),
        "detail": detail,
        "ref_deep_ahead": ref["deep_ahead"], "ref_dm_sig": ref["dm_sig"],
        "ref_gw_sig": ref["gw_sig"],
        "deep_ahead": cand["deep_ahead"], "dm_sig": cand["dm_sig"],
        "gw_sig": cand["gw_sig"],
        "qlike_deep": float(candidate["qlike_deep"]),
        "qlike_benchmark": float(candidate["qlike_benchmark"]),
        "dm_p": float(candidate["dm_p"]), "gw_p": float(candidate["gw_p"]),
    })
    return out


def rank_agreement(ranks: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Spearman and Kendall agreement of each asset's QLIKE ordering with the S&P's.

    ``ranks`` is models x assets. Only models present on **both** assets enter a
    pair's statistic, and the count is reported beside it: the reduced design
    means the robustness boards carry five models against the S&P's fifteen, and
    a correlation whose overlap is not stated is unreadable.

    With five overlapping models a rank correlation is a weak instrument -- one
    swap moves it a long way -- so this is descriptive context for the per-pair
    tests, never the transfer evidence itself.
    """
    rows = []
    if reference not in ranks.columns:
        return pd.DataFrame(rows)
    for col in ranks.columns:
        if col == reference:
            continue
        pair = ranks[[reference, col]].dropna()
        n = int(len(pair))
        rho = tau = float("nan")
        if n >= 3:
            rho = float(pair[reference].corr(pair[col], method="spearman"))
            tau = float(pair[reference].corr(pair[col], method="kendall"))
        rows.append({
            "asset": col, "reference": reference, "n_models_in_common": n,
            "spearman": rho, "kendall": tau,
            "models_in_common": ", ".join(map(str, pair.index)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Loading one asset's finished artefacts                                       #
# --------------------------------------------------------------------------- #
def _load_board(pred_path: Path, regime_path: Path, state_col: str):
    """Return ``(y, models, state)`` for one asset's Phase-5-equivalent board.

    ``state`` is the **full-history** causal series from the Phase-4 parquet, not
    the copy sliced into the predictions file. The distinction cost this project
    a published inconsistency once already (2026-08-19 (iv)): lagging a sliced
    series loses the first out-of-sample day from every bucket, so the per-regime
    tables stop reconciling with the pooled ones. Passing the full series with
    ``index=y.index`` lets the ``t-1`` lag reach back across the window boundary.
    """
    preds = from_parquet(pred_path)
    if ACTUAL_COL not in preds.columns:
        raise KeyError(f"{pred_path.name}: no {ACTUAL_COL!r} column")
    y = preds[ACTUAL_COL].astype(float)
    models = preds.drop(columns=[c for c in (ACTUAL_COL, "reg_state")
                                 if c in preds.columns])
    regimes = from_parquet(regime_path)
    if state_col not in regimes.columns:
        raise KeyError(
            f"{regime_path.name}: missing {state_col!r}. Available causal state "
            f"columns: {[c for c in regimes.columns if c.endswith('_filt_state')]}"
        )
    return y, models, regimes[state_col].astype(float)


def _pair_tests(y, models, state, labels) -> pd.DataFrame:
    """DM and regime-conditional GW for the deep model against each baseline."""
    if DEEP_MODEL not in models.columns:
        log.warning("%s absent from this board; no pair tests", DEEP_MODEL)
        return pd.DataFrame()
    pairs = [(DEEP_MODEL, b) for b in TEST_BASELINES if b in models.columns]
    if not pairs:
        return pd.DataFrame()
    return joint_table(y, state, models, pairs, list(labels))


def _regime_summary(profile_name: str, tables: Path, state: pd.Series,
                    y_index: pd.Index, labels) -> dict:
    """One row describing the regime model this asset got, from its own artefacts.

    The point of RQ4's decision to re-estimate per asset (ROADMAP Phase 8) is
    that these numbers are allowed to differ, and a lambda or a BIC-preferred K
    that lands somewhere else is itself a finding about the method's portability.
    They are read from the per-asset Phase-4 tables rather than assumed to match.
    """
    row: dict = {"profile": profile_name}

    lam_path = tables / f"m04_regimes_{profile_name}_jump_lambda_train.csv"
    if not lam_path.exists():          # fall back to the full-sample sweep
        lam_path = tables / f"m04_regimes_{profile_name}_jump_lambda.csv"
    if lam_path.exists():
        lam = pd.read_csv(lam_path)
        row["lambda_table"] = lam_path.name
        lam_col = next((c for c in ("jump_penalty", "lambda", "penalty")
                        if c in lam.columns), None)
        dur_col = next((c for c in ("mean_duration", "mean_duration_days",
                                    "mean_regime_duration") if c in lam.columns), None)
        if lam_col:
            row["lambda_grid"] = ", ".join(f"{v:g}" for v in lam[lam_col])
        if dur_col:
            row["mean_duration_by_lambda"] = ", ".join(f"{v:.1f}" for v in lam[dur_col])

    # The SELECTED penalty comes from the causal-signals table, which records the
    # lambda the run actually used for the signal Phase 5 consumed -- not from
    # re-applying the selection rule here, which would be a second implementation
    # of it that could drift from the first. The sweep table carries no
    # `selected` flag, so reading it for one was silently yielding nothing: this
    # column was blank in the first RQ4 run, and it is the single most important
    # portability number in the phase (did each market need the same penalty?).
    sig_path = tables / f"m04_regimes_{profile_name}_causal_signals.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        jp = sig[sig["signal"] == "jump_penalised_hmm"] if "signal" in sig.columns else sig
        if len(jp) and "lambda" in jp.columns and pd.notna(jp["lambda"].iloc[0]):
            row["selected_lambda"] = float(jp["lambda"].iloc[0])
            for src, dst in (("test_mean_duration_days", "test_mean_duration_days"),
                             ("test_switches", "test_switches"),
                             ("mean_max_posterior_test", "mean_max_posterior_test")):
                if src in jp.columns:
                    row[dst] = float(jp[src].iloc[0])
    if "selected_lambda" not in row:
        log.warning(
            "no selected lambda found for %s (looked in %s). The RQ4 note's "
            "portability table needs it; check that Phase 4 wrote its "
            "causal-signals table for this profile.", profile_name, sig_path.name)

    bic_path = tables / f"m04_regimes_{profile_name}_bic.csv"
    if bic_path.exists():
        bic = pd.read_csv(bic_path)
        if {"K", "hmm_bic"} <= set(bic.columns):
            row["bic_preferred_K_hmm"] = int(bic.loc[bic["hmm_bic"].idxmin(), "K"])
        if {"K", "jump_bic"} <= set(bic.columns):
            row["bic_preferred_K_jump"] = int(bic.loc[bic["jump_bic"].idxmin(), "K"])

    # State shares over the evaluated window, on the t-1 label -- the timing every
    # per-regime number in this project uses (src.evaluation.regime_timing).
    lagged = align_regime_label(state, y_index, shift=DEFAULT_REGIME_SHIFT)
    row["timing"] = regime_timing_label(DEFAULT_REGIME_SHIFT)
    row["n_evaluated"] = int(len(y_index))
    row["n_labelled_at_t_minus_1"] = int(lagged.notna().sum())
    for k, lab in enumerate(labels):
        row[f"n_{lab}"] = int((lagged == float(k)).sum())
    return row


# --------------------------------------------------------------------------- #
# The phase                                                                    #
# --------------------------------------------------------------------------- #
def collect(data_cfg, cfg, *, labels, spx_profile: str) -> dict:
    """Read every finished artefact this phase needs and assemble the tables."""
    tables = repo_path("results", "tables")
    preds_dir = repo_path("results", "predictions")
    state_col = str(cfg.regime.state_col)

    boards: dict = {}

    # --- the reference: the primary asset's own Phase-5 board ---------------
    spx_pred = preds_dir / f"m05_regime_dl_{spx_profile}.parquet"
    spx_regime = preds_dir / f"m04_regimes_{spx_profile}.parquet"
    for p in (spx_pred, spx_regime):
        if not p.exists():
            raise SystemExit(
                f"RQ4 needs the primary-asset reference at {p}, which is missing. "
                "RQ4 is a comparison against the primary result; run Phases 4 and "
                "5 on the primary asset first."
            )
    y, models, state = _load_board(spx_pred, spx_regime, state_col)
    boards["SPX"] = {"profile": spx_profile, "asset": "SPX", "y": y,
                     "models": models, "state": state, "is_reference": True}

    # --- the robustness assets ----------------------------------------------
    for profile in cfg.profiles:
        name = str(profile.name)
        asset = str(profile.get("asset", "") or name)
        pred = preds_dir / f"m05_regime_dl_{name}.parquet"
        regime = repo_path(str(profile.get("regime_source", cfg.regime.source)))
        missing = [str(p) for p in (pred, regime) if not p.exists()]
        if missing:
            raise SystemExit(
                f"asset {asset}: missing {missing}. Run the Phase-8 chain first:\n"
                "  python -m src.experiments.run_econometric --config econometric_rq4\n"
                "  python -m src.experiments.run_regimes     --config hmm_rq4\n"
                "  python -m src.experiments.run_regime_lstm --config regime_lstm_rq4"
            )
        y, models, state = _load_board(pred, regime, state_col)
        boards[asset] = {"profile": name, "asset": asset, "y": y, "models": models,
                         "state": state, "is_reference": False}

    # --- per-asset tables ----------------------------------------------------
    metrics_rows, tests, per_regime, regime_rows, sample_rows = [], [], [], [], []
    for asset, b in boards.items():
        board = pd.concat([b["y"].rename(ACTUAL_COL), b["models"]], axis=1)
        met = evaluate_predictions(board).reset_index()
        met = met.rename(columns={met.columns[0]: "model"})
        met.insert(0, "asset", asset)
        met.insert(1, "profile", b["profile"])
        met["rank_qlike"] = met["qlike"].rank(method="min").astype(int)
        met["is_reference_asset"] = b["is_reference"]
        metrics_rows.append(met)

        t = _pair_tests(b["y"], b["models"], b["state"], labels)
        if len(t):
            t.insert(0, "asset", asset)
            t.insert(1, "profile", b["profile"])
            tests.append(t)

        cols = b["models"].columns
        pr_pairs = ([(DEEP_MODEL, BENCHMARK)]
                    if DEEP_MODEL in cols and BENCHMARK in cols else [])
        pr = per_regime_table(b["y"], b["state"], b["models"], pr_pairs, list(labels))
        if len(pr):
            pr.insert(0, "asset", asset)
            pr.insert(1, "profile", b["profile"])
            per_regime.append(pr)

        rs = _regime_summary(b["profile"], tables, b["state"], b["y"].index, labels)
        rs["asset"] = asset
        rs["is_reference_asset"] = b["is_reference"]
        regime_rows.append(rs)

        # Sample provenance. Phase 2 writes this per profile; the reference only
        # has one if Phase 2 has been re-run since 2026-09-01, so a missing file
        # is noted rather than fabricated.
        s_path = tables / f"m02_{b['profile']}_sample.csv"
        if s_path.exists():
            rec = pd.read_csv(s_path).iloc[0].to_dict()
            rec["asset"] = asset
            sample_rows.append(rec)
        elif not b["is_reference"]:
            log.warning("no sample-provenance file at %s", s_path)

    metrics = pd.concat(metrics_rows, ignore_index=True)
    ranks = pd.DataFrame()
    if "SPX" in set(metrics["asset"]):
        ranks = (metrics.pivot_table(index="model", columns="asset",
                                     values="rank_qlike").sort_values("SPX"))
    return {
        "boards": boards,
        "metrics": metrics,
        "ranks": ranks,
        "agreement": rank_agreement(ranks, "SPX") if len(ranks) else pd.DataFrame(),
        "tests": pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(),
        "per_regime": (pd.concat(per_regime, ignore_index=True) if per_regime
                       else pd.DataFrame()),
        "regimes": pd.DataFrame(regime_rows),
        "sample": pd.DataFrame(sample_rows),
        "labels": list(labels),
    }


def _facts_for(asset: str, metrics: pd.DataFrame, tests: pd.DataFrame) -> dict:
    """The four numbers :func:`transfer_verdict` grades, for one asset."""
    def _q(model: str) -> float:
        sel = metrics[(metrics["asset"] == asset) & (metrics["model"] == model)]
        return float(sel["qlike"].iloc[0]) if len(sel) else float("nan")

    row = pd.DataFrame()
    if len(tests):
        row = tests[(tests["asset"] == asset) & (tests["model_a"] == DEEP_MODEL)
                    & (tests["model_b"] == BENCHMARK)]
    return {
        "asset": asset,
        "qlike_deep": _q(DEEP_MODEL),
        "qlike_benchmark": _q(BENCHMARK),
        "dm_p": float(row["dm_p_pooled"].iloc[0]) if len(row) else float("nan"),
        "gw_p": float(row["gw_regime_p"].iloc[0]) if len(row) else float("nan"),
        "n": int(row["n"].iloc[0]) if len(row) else 0,
    }


def verdicts(ctx: dict, *, alpha: float = ALPHA) -> pd.DataFrame:
    """One verdict row per robustness asset, graded against the primary result."""
    reference = _facts_for("SPX", ctx["metrics"], ctx["tests"])
    rows = []
    for asset, b in ctx["boards"].items():
        if b["is_reference"]:
            continue
        cand = _facts_for(asset, ctx["metrics"], ctx["tests"])
        v = transfer_verdict(reference, cand, alpha=alpha)
        v["n"] = cand["n"]
        v["ref_qlike_deep"] = reference["qlike_deep"]
        v["ref_qlike_benchmark"] = reference["qlike_benchmark"]
        v["ref_dm_p"] = reference["dm_p"]
        v["ref_gw_p"] = reference["gw_p"]
        rows.append(v)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Milestone                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(df: pd.DataFrame, cols=None, nd: int = 6) -> str:
    """Markdown table, floats to ``nd`` significant figures, missing left blank."""
    if df is None or not len(df):
        return "_(not available)_\n"
    wanted = list(cols or df.columns)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        # Silently dropping a requested column is how a number goes missing from
        # a note while the CSV still has it -- which is exactly what happened to
        # the per-asset persistence figures on 2026-09-02.
        log.warning("milestone table omits %s: not in the frame (have %s)",
                    missing, list(df.columns))
    sub = df[[c for c in wanted if c in df.columns]].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}g}")
    head = "| " + " | ".join(map(str, sub.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(sub.columns)) + "|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |"
                     for r in sub.astype(str).itertuples(index=False))
    return f"{head}\n{rule}\n{body}\n"


def write_milestone(ctx: dict, verdict_tbl: pd.DataFrame, out_path: Path,
                    *, alpha: float) -> Path:
    """Write ``m08_rq4.md``.

    Every sentence below is generated from the tables in ``ctx``; no branch
    states a conclusion the tables do not carry, and the verdict wording comes
    from :func:`transfer_verdict`, which is unit-tested. This is the sixth
    milestone generator in the project and five defects of one family have come
    out of the others -- a hardcoded conclusion, a shadowed loop variable, an
    untested claim, a self-contradicting verdict, an over-read null. If a *word*
    here looks wrong, suspect this function before suspecting the numbers.
    """
    ensure_dir(out_path.parent)
    labels = ctx["labels"]
    ref = _facts_for("SPX", ctx["metrics"], ctx["tests"])

    L = []
    L.append("# m08 - RQ4: does the S&P 500 result transfer?\n")
    L.append(f"\n_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
             f"Every replication decision at alpha = {alpha:g}._\n")
    L.append(
        "\nRQ4 is pre-registered as **secondary** and answered on a reduced "
        f"design (Ch1 sec1.6; Ch3 sec3.9): the econometric baselines and "
        f"**{DEEP_MODEL}** only, on assets whose realized variance is built "
        "exactly as the S&P's is. The regime model is re-estimated on each asset "
        "and never transferred, so what is under test is the pipeline rather "
        "than a state map fitted elsewhere.\n"
    )

    L.append("\n## The reference result being tested for transfer\n")
    L.append(
        f"\nOn the S&P 500, {DEEP_MODEL} scores QLIKE {ref['qlike_deep']:.6g} "
        f"against {BENCHMARK}'s {ref['qlike_benchmark']:.6g} "
        f"(pooled DM p = {ref['dm_p']:.4g}; regime-conditional GW "
        f"p = {ref['gw_p']:.4g}; n = {ref['n']}). Those numbers are read from the "
        "committed Phase-5 artefacts at run time, never stored in this module.\n"
    )

    L.append("\n## Verdict\n\n")
    if not len(verdict_tbl):
        L.append("_No robustness asset produced a comparable board._\n")
    else:
        for r in verdict_tbl.to_dict("records"):
            L.append(
                f"- **{r['asset']}: {r['verdict']}** - {r['detail']}. "
                f"QLIKE {r.get('qlike_deep', float('nan')):.6g} vs "
                f"{r.get('qlike_benchmark', float('nan')):.6g}; "
                f"DM p = {r.get('dm_p', float('nan')):.4g}; "
                f"GW p = {r.get('gw_p', float('nan')):.4g}; n = {r.get('n', 0)}.\n"
            )
        L.append("\n" + _fmt(verdict_tbl, [
            "asset", "verdict", "sign_agrees", "pooled_agrees",
            "conditional_agrees", "qlike_deep", "qlike_benchmark", "dm_p",
            "gw_p", "n"]))

    L.append("\n## Pooled point accuracy, every asset\n\n")
    L.append(_fmt(ctx["metrics"], ["asset", "model", "rank_qlike", "qlike",
                                   "mse", "mae", "n"]))

    if len(ctx["agreement"]):
        L.append("\n## Rank agreement with the S&P ordering\n\n")
        L.append(_fmt(ctx["agreement"], ["asset", "n_models_in_common", "spearman",
                                         "kendall", "models_in_common"], nd=4))
        L.append(
            "\nDescriptive only. The reduced design puts five models on each "
            "robustness board against the S&P's fifteen, so the overlap is small "
            "and one swap moves a rank correlation a long way. The per-pair tests "
            "below carry the verdict.\n"
        )

    L.append("\n## The deep model against each baseline\n\n")
    L.append(_fmt(ctx["tests"], ["asset", "model_a", "model_b", "n",
                                 "dm_stat_pooled", "dm_p_pooled", "gw_regime_stat",
                                 "gw_regime_df", "gw_regime_p"]))
    L.append(
        "\nA **negative** DM statistic means the deep model has the lower loss. "
        "The GW column tests whether the better model is predictable from the "
        "state known at t-1 - RQ2's question, asked on a new market.\n"
    )

    if len(ctx["per_regime"]):
        L.append(f"\n## Per-state QLIKE ({regime_timing_label(DEFAULT_REGIME_SHIFT)})\n\n")
        L.append(_fmt(ctx["per_regime"], ["asset", "model_a", "model_b", "regime",
                                          "n", "qlike_a", "qlike_b",
                                          "mean_loss_diff", "dm_stat", "p_value",
                                          "better"]))

    L.append("\n## Each asset's own regime model\n\n")
    L.append(_fmt(ctx["regimes"], ["asset", "selected_lambda",
                                   "test_mean_duration_days", "test_switches",
                                   "bic_preferred_K_hmm", "bic_preferred_K_jump",
                                   "n_evaluated", "n_labelled_at_t_minus_1",
                                   *[f"n_{lab}" for lab in labels]], nd=4))
    L.append(
        "\nThe penalty was selected on each asset's **own training rows**, by the "
        "same rule applied to the same pre-registered grid (fixed 2026-05-15 and "
        "asserted identical by the test suite). A differing selected lambda is a "
        "result about the method's portability, not a free parameter.\n"
    )

    if len(ctx["sample"]):
        L.append("\n## Sample provenance\n\n")
        L.append(_fmt(ctx["sample"], ["asset", "session", "n_target_rows",
                                      "n_after_join", "n_modelling_rows",
                                      "lost_to_calendar_join", "pct_lost_to_join",
                                      "vix_joined", "first_date", "last_date"], nd=4))
        L.append(
            "\nThe modelling frame is an inner join across sources, so an asset "
            "trading on another exchange calendar yields a different sample. Ch3 "
            "sec3.9 commits to reporting those rows rather than absorbing them.\n"
        )

    L.append("\n## What this design cannot establish\n")
    L.append(
        "\nThe evaluation windows are identical by construction, so the same "
        "COVID-19 stress episode sits inside every asset's test block: agreement "
        "across assets is **not independent evidence**, and Ch3 sec3.9 says so "
        "before any result is read. Disagreement identifies a finding as "
        "market-specific without saying which feature of the market is "
        "responsible. And the reduced board carries no unconditional LSTM on the "
        "robustness assets, so RQ2's decomposition - how much of the edge is the "
        "architecture and how much the regime signal - is not reproduced here; "
        "only the composite deep model's transfer is.\n"
    )

    out_path.write_text("".join(L), encoding="utf-8")
    log.info("milestone -> %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_rq4")
    ap.add_argument("--config", default="regime_lstm_rq4",
                    help="Phase-8 deep config naming the robustness profiles.")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--spx-profile", default=SPX_PROFILE,
                    help="primary-asset profile used as the transfer reference")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="significance level for every replication decision")
    ap.add_argument("--refresh-sample", action="store_true",
                    help="rebuild each robustness asset's modelling frame to "
                         "refresh the calendar-join provenance (needs data/raw/)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "08_rq4",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    labels = list(cfg.regime.labels)
    ctx = collect(data_cfg, cfg, labels=labels, spx_profile=args.spx_profile)

    if args.refresh_sample:
        rows = []
        for profile in cfg.profiles:
            frame = frame_for_profile(data_cfg, profile, include_vix=True)
            rec = attrition_record(frame, profile_name=str(profile.name))
            rec["asset"] = str(profile.get("asset", ""))
            rows.append(rec)
        ctx["sample"] = pd.DataFrame(rows)

    verdict_tbl = verdicts(ctx, alpha=args.alpha)

    tables = repo_path("results", "tables")
    ensure_dir(tables)
    written = []
    for key, name in (("metrics", "m08_rq4_metrics.csv"),
                      ("agreement", "m08_rq4_ranks.csv"),
                      ("tests", "m08_rq4_tests.csv"),
                      ("per_regime", "m08_rq4_per_regime.csv"),
                      ("regimes", "m08_rq4_regimes.csv"),
                      ("sample", "m08_rq4_sample.csv")):
        df = ctx.get(key)
        if df is not None and len(df):
            df.to_csv(tables / name, index=False)
            written.append(name)
    if len(verdict_tbl):
        verdict_tbl.to_csv(tables / "m08_rq4_verdict.csv", index=False)
        written.append("m08_rq4_verdict.csv")
    if len(ctx["ranks"]):
        ctx["ranks"].to_csv(tables / "m08_rq4_rank_matrix.csv")
        written.append("m08_rq4_rank_matrix.csv")
    log.info("tables written: %s", ", ".join(written))

    write_milestone(ctx, verdict_tbl,
                    # `key` is this phase's own, NOT the deep config's
                    # `milestone_file` -- see milestone_path's docstring.
                    milestone_path(cfg, "m08_rq4.md", "rq4",
                                   key="rq4_milestone_file"),
                    alpha=args.alpha)

    for r in verdict_tbl.to_dict("records"):
        log.info("RQ4 %s: %s (%s)", r["asset"], r["verdict"], r["detail"])
    log.info("Phase 8 / RQ4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
