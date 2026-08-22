"""Regime-conditional forecast comparison — Giacomini & White (2006) (Ch2 §2.7).

What this closes
----------------
The Phase-5 milestone reports that the regime-aware models earn nearly all of
their edge in the crisis state (QLIKE 0.2976 vs 0.3193 on **70 days**), but backs
that claim with point estimates only: the pairwise Diebold–Mariano tests it does
report are *pooled* over the whole test window, and a pooled average is precisely
what hides a regime-dependent effect. Chapter 2 §2.7 names Giacomini & White
(2006) alongside DM as the dissertation's inferential apparatus; this script is
that commitment, applied to the predictions Phase 5 already produced.

It answers two distinct questions per model pair:

* **Joint (the proper test).** Conditioning on the *lagged* regime indicators,
  is the loss differential predictable — i.e. is one model reliably better in
  identifiable states? One χ² statistic on the full sample, correctly sized.
  Its per-state *moments* localise the effect: a negative moment means model *a*
  wins in that state.
* **Per-regime (descriptive).** A DM test inside each regime's subsample, which
  is what a reader expects to see next to a per-regime error table. Reported with
  its ``n`` because at ~70 crisis days it has little power — and because the
  subsample is not contiguous in calendar time, its HAC correction is an
  approximation. The joint test above is the one to quote; these are context.

Where the regime label comes from (2026-08-19 (iv))
---------------------------------------------------
The lagged indicators must come from the **full Phase-4 regime series**, not from
the ``reg_state`` column of the predictions parquet. That column is already sliced
to the evaluation window, so lagging it leaves the first out-of-sample day without
a ``t-1`` label and drops it: ``n`` falls to 783, the crisis bucket to 75, and the
QLIKE this script reports per bucket stops matching the one
``run_regime_lstm.per_regime_qlike`` reports for the same bucket (0.3107 vs
0.3070 for HAR-RV in crisis, in the run that surfaced this). Two published values
for one cell, differing by a single day of labelling.

This script therefore resolves the Phase-4 predictions file alongside its input
and takes the causal hard state from there, lagging it with
:func:`src.evaluation.regime_timing.align_regime_label` — the project's single
shift implementation, shared with the descriptive tables. ``--regime-source`` and
``--state-col`` override the resolution; if the Phase-4 file cannot be found the
script falls back to the parquet column and says so loudly, because that path
produces the off-by-one-day numbers.

Nothing is retrained: the script reads a predictions parquet, so it runs in
seconds and can be re-pointed at any phase's output.

Usage
-----
    # headline (jump-penalised-HMM-conditioned) Phase-5 predictions
    python -m src.experiments.report_gw

    # the Baum-Welch robustness variant, or any other predictions file
    python -m src.experiments.report_gw \
        --predictions results/predictions/m05_regime_dl_intraday_2019_2022_hmm.parquet

Outputs ``results/tables/<stem>_gw_joint.csv`` and ``<stem>_gw_per_regime.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    align_regime_label,
    label_reach,
    regime_timing_label,
)
from src.evaluation.significance import (
    diebold_mariano,
    giacomini_white,
    qlike_loss,
    regime_test_function,
)
from src.utils.config import repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("report_gw")

ACTUAL_COL = "y_true"
REGIME_STATE_COL = "reg_state"
DEFAULT_PREDICTIONS = "results/predictions/m05_regime_dl_intraday_2019_2022.parquet"
DEFAULT_LABELS = ("calm", "transitional", "crisis")

# Phase prefixes stripped from a predictions stem to recover the profile name,
# longest first so "m05_regime_dl_" wins over any shorter accidental match.
_PHASE_PREFIXES = ("m05_regime_dl_", "m04_regimes_", "m06_uq_", "m03_lstm_", "m02_")
# Causal hard-state column on the Phase-4 parquet, per regime estimator. The
# headline is the jump-penalised HMM (Ch2 §2.4/§2.8); the Baum-Welch posterior is
# the robustness comparator and lives under the "_hmm" profile.
_HEADLINE_STATE_COL = "jphmm_filt_state"
_COMPARATOR_STATE_COL = "hmm_filt_state"

# The Phase-5 head-to-head pairs (identical to the m05 DM table), plus the
# regime-aware vs HAR-RV contrast RQ2 ultimately cares about.
DEFAULT_PAIRS = [
    ("Regime-LSTM-A", "LSTM"),
    ("Regime-LSTM-B", "LSTM"),
    ("Regime-LSTM-A", "HAR-RV"),
    ("Regime-LSTM-B", "HAR-RV"),
    ("Regime-LSTM-A-RVonly", "LSTM-RVonly"),
]


def _profile_of(stem: str) -> str:
    """Recover the profile name from a predictions stem (``m05_regime_dl_X`` -> ``X``)."""
    for prefix in _PHASE_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def resolve_regime_source(
    path: Path, *, source: str | None = None, state_col: str | None = None
) -> tuple[Path | None, str]:
    """Locate the Phase-4 predictions file and the causal state column to read.

    ``m05_regime_dl_intraday_2019_2022`` -> ``m04_regimes_intraday_2019_2022.parquet``
    with ``jphmm_filt_state``; a ``_hmm`` profile maps to the *same* Phase-4 file
    (it carries both estimators) but to ``hmm_filt_state``. Explicit arguments win.
    Returns ``(path_or_None, state_col)``; ``None`` means the file is missing and
    the caller must fall back.
    """
    profile = _profile_of(path.stem)
    is_comparator = profile.endswith("_hmm")
    col = state_col or (_COMPARATOR_STATE_COL if is_comparator else _HEADLINE_STATE_COL)
    if source is not None:
        p = Path(source)
        return ((p if p.is_absolute() else repo_path(str(p))), col)
    base = profile[: -len("_hmm")] if is_comparator else profile
    p = repo_path("results", "predictions", f"m04_regimes_{base}.parquet")
    return (p if p.exists() else None), col


def _load(
    path: Path, *, regime_source: str | None = None, state_col: str | None = None
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return ``(y, state, models)`` with ``state`` the **full-history** regime series.

    ``state`` is deliberately *not* the predictions parquet's ``reg_state`` column
    unless the Phase-4 file is unavailable: see the module docstring.
    """
    preds = pd.read_parquet(path)
    if ACTUAL_COL not in preds.columns:
        raise KeyError(f"{path.name}: missing '{ACTUAL_COL}'")
    if REGIME_STATE_COL not in preds.columns:
        raise KeyError(
            f"{path.name}: missing '{REGIME_STATE_COL}' — this script needs a "
            "predictions file that carries the causal regime state (Phase 5+)"
        )
    y = preds[ACTUAL_COL].astype(float)
    models = preds.drop(columns=[ACTUAL_COL, REGIME_STATE_COL], errors="ignore")
    models = models.select_dtypes(include=[np.number])

    src, col = resolve_regime_source(path, source=regime_source, state_col=state_col)
    if src is not None and src.exists():
        regimes = pd.read_parquet(src)
        if col not in regimes.columns:
            raise KeyError(
                f"{src.name}: missing '{col}'. Available causal state columns: "
                f"{[c for c in regimes.columns if c.endswith('_filt_state')]}. "
                "Pass --state-col."
            )
        state = regimes[col].astype("float")
        reach = label_reach(state, y.index, shift=DEFAULT_REGIME_SHIFT)
        log.info(
            "regime label: %s::%s (%d rows, %d before the evaluation window) — %s",
            src.name, col, len(state), reach["n_available_before"],
            regime_timing_label(DEFAULT_REGIME_SHIFT),
        )
    else:
        state = preds[REGIME_STATE_COL].astype("float")
        log.warning(
            "Phase-4 regime file not found; falling back to %s::%s, which is "
            "already sliced to the evaluation window. The first out-of-sample day "
            "will lose its t-1 label and be dropped from every bucket, so these "
            "numbers will NOT match the descriptive per-regime table. Pass "
            "--regime-source to point at the Phase-4 predictions parquet.",
            path.name, REGIME_STATE_COL,
        )
    return y, state, models


def joint_table(
    y: pd.Series, state: pd.Series, models: pd.DataFrame, pairs, labels
) -> pd.DataFrame:
    """One row per pair: the regime-conditional GW test + the canonical GW test.

    ``state`` is the full-history causal regime series; passing ``index=y.index``
    is what lets the lag reach back across the start of the evaluation window.
    """
    h = regime_test_function(state, n_states=len(labels), labels=list(labels),
                             index=y.index)
    rows = []
    for a, b in pairs:
        if a not in models.columns or b not in models.columns:
            log.warning("skipping %s vs %s (column missing)", a, b)
            continue
        gw_reg = giacomini_white(y, models[a], models[b], h)
        gw_std = giacomini_white(y, models[a], models[b])  # h = (1, lagged diff)
        dm = diebold_mariano(y, models[a], models[b])
        row = {
            "model_a": a,
            "model_b": b,
            "n": gw_reg["n"],
            "dm_stat_pooled": dm["dm_stat"],
            "dm_p_pooled": dm["p_value"],
            "gw_regime_stat": gw_reg["gw_stat"],
            "gw_regime_df": gw_reg["df"],
            "gw_regime_p": gw_reg["p_value"],
            "gw_lagdiff_stat": gw_std["gw_stat"],
            "gw_lagdiff_df": gw_std["df"],
            "gw_lagdiff_p": gw_std["p_value"],
        }
        for lab, mom in zip(labels, gw_reg["moments"]):
            # E[1{state=k} * d]; negative => model_a has the lower loss in state k.
            row[f"moment_{lab}"] = float(mom)
        moments = np.asarray(gw_reg["moments"], dtype=float)
        # Where a is *relatively* strongest (may still be the losing model there),
        # and where it actually wins outright. Keeping these separate stops the
        # table from implying a win that the sign does not support.
        row["a_relative_edge_in"] = labels[int(np.argmin(moments))]
        wins = [lab for lab, m in zip(labels, moments) if m < 0]
        row["a_wins_in"] = ", ".join(wins) if wins else "none"
        rows.append(row)
    return pd.DataFrame(rows)


def per_regime_table(
    y: pd.Series, state: pd.Series, models: pd.DataFrame, pairs, labels
) -> pd.DataFrame:
    """Descriptive DM inside each regime subsample (lagged state, as GW uses).

    The lag goes through :func:`align_regime_label` rather than a local
    ``state.shift(1)`` so this table, the joint test above and
    ``run_regime_lstm.per_regime_qlike`` all bucket on the identical label. When
    they did not, ``qlike_a`` here and the same cell in
    ``m05_*_per_regime.csv`` disagreed by one day's worth of crisis observations.
    """
    lagged = align_regime_label(state, y.index, shift=DEFAULT_REGIME_SHIFT)
    # Every conditional table in the project carries its bucketing timing on the
    # row (2026-08-19 (ii)): a per-regime number quoted without it is not
    # interpretable, because the two timings disagree about which model wins calm
    # and crisis. This table is no exception.
    stamp = {"regime_shift": int(DEFAULT_REGIME_SHIFT),
             "timing": regime_timing_label(DEFAULT_REGIME_SHIFT)}
    rows = []
    for a, b in pairs:
        if a not in models.columns or b not in models.columns:
            continue
        for k, lab in enumerate(labels):
            mask = (lagged == float(k)).to_numpy()
            n_k = int(mask.sum())
            if n_k < 10:
                rows.append({**stamp, "model_a": a, "model_b": b, "regime": lab,
                             "n": n_k, "qlike_a": np.nan, "qlike_b": np.nan,
                             "mean_loss_diff": np.nan, "dm_stat": np.nan,
                             "p_value": np.nan, "better": "insufficient n"})
                continue
            ys, as_, bs = y[mask], models[a][mask], models[b][mask]
            la = float(qlike_loss(ys.to_numpy(), as_.to_numpy()).mean())
            lb = float(qlike_loss(ys.to_numpy(), bs.to_numpy()).mean())
            try:
                dm = diebold_mariano(ys, as_, bs)
                stat, p = dm["dm_stat"], dm["p_value"]
            except Exception as exc:  # noqa: BLE001 - a degenerate subsample is reportable
                log.warning("DM failed for %s vs %s in %s: %s", a, b, lab, exc)
                stat, p = np.nan, np.nan
            rows.append({
                **stamp,
                "model_a": a, "model_b": b, "regime": lab, "n": n_k,
                "qlike_a": la, "qlike_b": lb, "mean_loss_diff": la - lb,
                "dm_stat": stat, "p_value": p,
                "better": a if la < lb else b,
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.report_gw")
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                    help="predictions parquet carrying y_true, reg_state and model columns")
    ap.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    ap.add_argument("--pair", action="append", nargs=2, metavar=("A", "B"),
                    help="model pair to test (repeatable); defaults to the Phase-5 pairs")
    ap.add_argument("--regime-source", default=None,
                    help="Phase-4 predictions parquet holding the FULL causal regime "
                         "series (default: resolved from the predictions stem). The "
                         "lag must reach back before the evaluation window, so this "
                         "must not be a file already sliced to it.")
    ap.add_argument("--state-col", default=None,
                    help=f"causal hard-state column on the regime source (default: "
                         f"{_HEADLINE_STATE_COL}, or {_COMPARATOR_STATE_COL} for a "
                         f"'_hmm' profile)")
    args = ap.parse_args(argv)

    path = Path(args.predictions)
    if not path.is_absolute():
        path = repo_path(str(path))
    if not path.exists():
        raise SystemExit(
            f"predictions file not found: {path}\n"
            "Run the phase that produces it first, or pass --predictions."
        )

    y, state, models = _load(path, regime_source=args.regime_source,
                             state_col=args.state_col)
    pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
    labels = list(args.labels)
    log.info("loaded %s: %d obs, %d model columns, states %s",
             path.name, len(y), models.shape[1],
             sorted(state.dropna().unique().tolist()))

    joint = joint_table(y, state, models, pairs, labels)
    per = per_regime_table(y, state, models, pairs, labels)

    # Coverage invariant: the lagged buckets must span every evaluated day, so
    # these tables and the pooled DM/MCS results are computed on one sample. A
    # shortfall here is the 2026-08-19 (iv) defect resurfacing.
    lagged = align_regime_label(state, y.index, shift=DEFAULT_REGIME_SHIFT)
    n_bucketed = int(lagged.notna().sum())
    counts = {lab: int((lagged == float(k)).sum()) for k, lab in enumerate(labels)}
    if n_bucketed != len(y):
        log.warning(
            "%d of %d evaluated days have no t-1 regime label and are excluded "
            "from every bucket — the per-regime numbers below will not reconcile "
            "with the pooled ones. Check --regime-source.",
            len(y) - n_bucketed, len(y),
        )
    else:
        log.info("bucket coverage OK: %d/%d days labelled at t-1 %s",
                 n_bucketed, len(y), counts)

    stem = path.stem
    tables = repo_path("results", "tables")
    ensure_dir(tables)
    joint_path = tables / f"{stem}_gw_joint.csv"
    per_path = tables / f"{stem}_gw_per_regime.csv"
    joint.to_csv(joint_path, index=False)
    per.to_csv(per_path, index=False)

    pd.set_option("display.width", 160, "display.max_columns", 40)
    print("\n=== Giacomini-White, conditional on the lagged regime indicator ===")
    print(joint.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Descriptive DM within each regime subsample ===")
    print(per.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {joint_path}\n      {per_path}")

    sig = joint[joint["gw_regime_p"] < 0.05]
    if len(sig):
        print("\nRegime-conditional differences significant at 5%:")
        for _, r in sig.iterrows():
            print(f"  {r['model_a']} vs {r['model_b']}: chi2({int(r['gw_regime_df'])})="
                  f"{r['gw_regime_stat']:.2f}, p={r['gw_regime_p']:.4g}; "
                  f"relative edge in '{r['a_relative_edge_in']}', "
                  f"outright wins in: {r['a_wins_in']}")
    else:
        print("\nNo pair shows a regime-conditional difference at the 5% level — "
              "which is itself the finding to report for RQ2.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
