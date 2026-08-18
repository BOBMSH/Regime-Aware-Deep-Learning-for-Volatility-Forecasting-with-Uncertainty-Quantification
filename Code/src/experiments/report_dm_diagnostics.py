"""Diebold–Mariano regularity diagnostics (Ch2 §2.7 robustness exercise).

What this closes
----------------
Chapter 2 §2.7 states that "the test's regularity conditions, in particular weak
dependence in the loss differential, are inspected and reported as a robustness
exercise, since highly persistent forecast errors can degrade the test's power and
inflate its size". Nothing in the pipeline performed that inspection: every DM
p-value in Phases 3–5 was reported on trust in a single automatic HAC lag.

Three diagnostics per model pair, each answering a different question:

* **ACF of the loss differential** — is ``d_t`` even weakly dependent? DM assumes
  covariance-stationarity with summable autocovariances; the Bartlett band is the
  conservative test of each lag.
* **Ljung–Box** — the formal portmanteau version of the same question, up to
  lags 5, 10 and 20.
* **HAC-lag sensitivity** — does the *verdict* move as the Newey–West lag varies
  from 0 to 30? If a p-value crosses 0.05 across that grid, the conclusion is an
  artefact of the lag choice and must be reported as such. The ``lrv/γ₀`` column
  measures directly how much dependence inflates the standard error: ≈1 means the
  iid case is essentially adequate.

Nothing is retrained: this reads a predictions parquet and runs in seconds.

Usage
-----
    python -m src.experiments.report_dm_diagnostics
    python -m src.experiments.report_dm_diagnostics \
        --predictions results/predictions/m05_regime_dl_intraday_2019_2022_hmm.parquet

Outputs ``results/tables/<stem>_dm_{acf,ljungbox,lag_sensitivity,summary}.csv`` and
``results/figures/m05/<stem>_dm_acf.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.significance import (
    diebold_mariano,
    dm_lag_sensitivity,
    ljung_box,
    loss_differential,
    sample_acf,
)
from src.utils.config import repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger
from src.utils.plotting import save_fig, styled

log = get_logger("report_dm_diag")

ACTUAL_COL = "y_true"
DEFAULT_PREDICTIONS = "results/predictions/m05_regime_dl_intraday_2019_2022.parquet"

# The pairs whose DM verdicts the dissertation actually quotes.
DEFAULT_PAIRS = [
    ("LSTM", "HAR-RV"),
    ("LSTM-RVonly", "HAR-RV"),
    ("Regime-LSTM-A", "LSTM"),
    ("Regime-LSTM-B", "LSTM"),
    ("Regime-LSTM-B", "HAR-RV"),
]

NLAGS = 20


def diagnose(y: pd.Series, models: pd.DataFrame, pairs) -> dict[str, pd.DataFrame]:
    acf_rows, lb_rows, sens_rows, summary = [], [], [], []
    for a, b in pairs:
        if a not in models.columns or b not in models.columns:
            log.warning("skipping %s vs %s (column missing)", a, b)
            continue
        tag = f"{a} vs {b}"
        d = loss_differential(y, models[a], models[b])
        acf = sample_acf(d, nlags=NLAGS).assign(pair=tag)
        lb = ljung_box(d).assign(pair=tag)
        sens = dm_lag_sensitivity(y, models[a], models[b]).assign(pair=tag)
        acf_rows.append(acf)
        lb_rows.append(lb)
        sens_rows.append(sens)

        default = diebold_mariano(y, models[a], models[b])
        lb20 = lb[lb["lag"] == 20]
        p_lo, p_hi = float(sens["p_value"].min()), float(sens["p_value"].max())
        # Does the 5% verdict survive every HAC lag on the grid?
        stable = bool((sens["p_value"] < 0.05).all() or (sens["p_value"] >= 0.05).all())
        summary.append({
            "pair": tag,
            "n": int(d.size),
            "dm_stat_default": default["dm_stat"],
            "dm_p_default": default["p_value"],
            "default_hac_lag": default["lag"],
            "lrv_over_gamma0": default["lrv_over_gamma0"],
            "max_abs_acf": float(acf["acf"].abs().max()),
            "n_acf_sig_bartlett": int(acf["significant_bartlett"].sum()),
            "lb20_stat": float(lb20["lb_stat"].iloc[0]) if len(lb20) else np.nan,
            "lb20_p": float(lb20["p_value"].iloc[0]) if len(lb20) else np.nan,
            "dm_p_min_over_lags": p_lo,
            "dm_p_max_over_lags": p_hi,
            "verdict_stable_across_lags": stable,
        })
    return {
        "acf": pd.concat(acf_rows, ignore_index=True) if acf_rows else pd.DataFrame(),
        "ljungbox": pd.concat(lb_rows, ignore_index=True) if lb_rows else pd.DataFrame(),
        "lag_sensitivity": pd.concat(sens_rows, ignore_index=True) if sens_rows else pd.DataFrame(),
        "summary": pd.DataFrame(summary),
    }


def plot_acf(acf: pd.DataFrame, path: Path) -> Path:
    """ACF bars per pair with the 95% Bartlett band."""
    pairs = list(dict.fromkeys(acf["pair"]))
    with styled():
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(pairs), 1, figsize=(9, 2.1 * len(pairs)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, pair in zip(axes, pairs):
            sub = acf[acf["pair"] == pair]
            ax.bar(sub["lag"], sub["acf"], width=0.6, color="tab:blue", alpha=0.85)
            ax.plot(sub["lag"], 1.96 * sub["se_bartlett"], color="tab:red", lw=1.0, ls="--")
            ax.plot(sub["lag"], -1.96 * sub["se_bartlett"], color="tab:red", lw=1.0, ls="--")
            ax.axhline(0.0, color="0.3", lw=0.8)
            ax.set_ylabel("ACF")
            ax.set_title(f"QLIKE loss differential: {pair}", loc="left")
        axes[-1].set_xlabel("lag (trading days)")
        fig.suptitle("DM regularity check — autocorrelation of the loss differential "
                     "(dashed: 95% Bartlett band)", y=1.01)
        out = save_fig(fig, path)
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.report_dm_diagnostics")
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    ap.add_argument("--pair", action="append", nargs=2, metavar=("A", "B"))
    args = ap.parse_args(argv)

    path = Path(args.predictions)
    if not path.is_absolute():
        path = repo_path(str(path))
    if not path.exists():
        raise SystemExit(f"predictions file not found: {path}")

    preds = pd.read_parquet(path)
    if ACTUAL_COL not in preds.columns:
        raise KeyError(f"{path.name}: missing '{ACTUAL_COL}'")
    y = preds[ACTUAL_COL].astype(float)
    models = preds.drop(columns=[ACTUAL_COL], errors="ignore").select_dtypes(include=[np.number])
    pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
    log.info("loaded %s: %d obs, %d model columns", path.name, len(y), models.shape[1])

    out = diagnose(y, models, pairs)
    stem = path.stem
    tables = repo_path("results", "tables")
    ensure_dir(tables)
    for key, df in out.items():
        df.to_csv(tables / f"{stem}_dm_{key}.csv", index=False)

    figdir = repo_path("results", "figures", "m05")
    ensure_dir(figdir)
    fig_path = plot_acf(out["acf"], figdir / f"{stem}_dm_acf.png")

    pd.set_option("display.width", 190, "display.max_columns", 40)
    print("\n=== DM regularity summary (QLIKE loss differential) ===")
    print(out["summary"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== HAC-lag sensitivity of the DM p-value ===")
    piv = out["lag_sensitivity"].pivot(index="hac_lag", columns="pair", values="p_value")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {tables / f'{stem}_dm_summary.csv'} (+ acf, ljungbox, lag_sensitivity)")
    print(f"      {fig_path}")

    s = out["summary"]
    dependent = s[s["lb20_p"] < 0.05]
    unstable = s[~s["verdict_stable_across_lags"]]
    print("\nReading:")
    if len(dependent):
        print("  Loss differentials with significant autocorrelation up to lag 20 "
              "(HAC is doing real work here):")
        for _, r in dependent.iterrows():
            print(f"    {r['pair']}: LB(20)={r['lb20_stat']:.1f}, p={r['lb20_p']:.4g}, "
                  f"lrv/gamma0={r['lrv_over_gamma0']:.2f}")
    else:
        print("  No pair shows significant autocorrelation in the loss differential up to "
              "lag 20 — the DM regularity conditions are comfortably met.")
    if len(unstable):
        print("  WARNING — the 5% verdict changes with the HAC lag for:")
        for _, r in unstable.iterrows():
            print(f"    {r['pair']}: p ranges [{r['dm_p_min_over_lags']:.3f}, "
                  f"{r['dm_p_max_over_lags']:.3f}] across lags 0-30")
    else:
        print("  Every pair's 5% verdict is unchanged across HAC lags 0-30 — the inference "
              "does not depend on that choice.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
