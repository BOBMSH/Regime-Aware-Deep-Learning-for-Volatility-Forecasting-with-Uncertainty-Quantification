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

# The Phase-5 head-to-head pairs (identical to the m05 DM table), plus the
# regime-aware vs HAR-RV contrast RQ2 ultimately cares about.
DEFAULT_PAIRS = [
    ("Regime-LSTM-A", "LSTM"),
    ("Regime-LSTM-B", "LSTM"),
    ("Regime-LSTM-A", "HAR-RV"),
    ("Regime-LSTM-B", "HAR-RV"),
    ("Regime-LSTM-A-RVonly", "LSTM-RVonly"),
]


def _load(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    preds = pd.read_parquet(path)
    if ACTUAL_COL not in preds.columns:
        raise KeyError(f"{path.name}: missing '{ACTUAL_COL}'")
    if REGIME_STATE_COL not in preds.columns:
        raise KeyError(
            f"{path.name}: missing '{REGIME_STATE_COL}' — this script needs a "
            "predictions file that carries the causal regime state (Phase 5+)"
        )
    y = preds[ACTUAL_COL].astype(float)
    state = preds[REGIME_STATE_COL].astype("float")
    models = preds.drop(columns=[ACTUAL_COL, REGIME_STATE_COL], errors="ignore")
    models = models.select_dtypes(include=[np.number])
    return y, state, models


def joint_table(
    y: pd.Series, state: pd.Series, models: pd.DataFrame, pairs, labels
) -> pd.DataFrame:
    """One row per pair: the regime-conditional GW test + the canonical GW test."""
    h = regime_test_function(state, n_states=len(labels), labels=list(labels))
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
    """Descriptive DM inside each regime subsample (lagged state, as GW uses)."""
    lagged = state.shift(1)
    rows = []
    for a, b in pairs:
        if a not in models.columns or b not in models.columns:
            continue
        for k, lab in enumerate(labels):
            mask = (lagged == float(k)).to_numpy()
            n_k = int(mask.sum())
            if n_k < 10:
                rows.append({"model_a": a, "model_b": b, "regime": lab, "n": n_k,
                             "qlike_a": np.nan, "qlike_b": np.nan,
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
    args = ap.parse_args(argv)

    path = Path(args.predictions)
    if not path.is_absolute():
        path = repo_path(str(path))
    if not path.exists():
        raise SystemExit(
            f"predictions file not found: {path}\n"
            "Run the phase that produces it first, or pass --predictions."
        )

    y, state, models = _load(path)
    pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
    labels = list(args.labels)
    log.info("loaded %s: %d obs, %d model columns, states %s",
             path.name, len(y), models.shape[1],
             sorted(state.dropna().unique().tolist()))

    joint = joint_table(y, state, models, pairs, labels)
    per = per_regime_table(y, state, models, pairs, labels)

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
