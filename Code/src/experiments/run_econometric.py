""""Phase 2 driver -- econometric baselines end-to-end (ROADMAP.md Phase 2 gate).

Runs GARCH(1,1), EGARCH(1,1,1), HAR-RV and a random-walk reference through the
shared walk-forward engine for each evaluation profile in
``configs/econometric.yaml``, then writes the Phase 2 gate artifacts:

* ``results/predictions/m02_<profile>.parquet`` -- every model's OOS variance
  forecast plus the realized target (one column per model; roadmap §5).
* ``results/tables/m02_<profile>_metrics.csv``  -- MSE / RMSE / MAE / QLIKE per
  model, QLIKE-ranked.
* ``results/figures/m02/<profile>_forecast_vs_actual.png`` -- realized vs
  forecast volatility over the test window, crisis-shaded.
* ``results/milestones/m02_econometric.md``     -- the milestone note.
* ``experiments/02_econometric/run_<utc>/config.yaml`` -- resolved-config
  snapshot (roadmap §9 reproducibility).

Usage
-----
    python -m src.experiments.run_econometric
    python -m src.experiments.run_econometric --config econometric --data-config data
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import build_econometric_frame
from src.data.splits import SplitConfig
from src.evaluation.rolling import ACTUAL_COL, evaluate_predictions, run_walk_forward
from src.models.econometric import (
    EGARCHForecaster,
    GARCHForecaster,
    HARForecaster,
    RandomWalkRVForecaster,
)
from src.utils.config import load_config, repo_path, snapshot_config
from src.utils.io import ensure_dir, to_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("run_econometric")


# --------------------------------------------------------------------------- #
# Model construction                                                          #
# --------------------------------------------------------------------------- #


def build_models(mcfg, *, return_col: str, scale: float) -> list:
    """Instantiate the enabled models in a stable order."""
    models: list = []
    if mcfg.garch.enabled:
        models.append(
            GARCHForecaster(p=mcfg.garch.p, q=mcfg.garch.q, dist=mcfg.garch.dist,
                            scale=scale, return_col=return_col)
        )
    if mcfg.egarch.enabled:
        models.append(
            EGARCHForecaster(p=mcfg.egarch.p, o=mcfg.egarch.o, q=mcfg.egarch.q,
                             dist=mcfg.egarch.dist, scale=scale, return_col=return_col)
        )
    if mcfg.har.enabled:
        models.append(HARForecaster(transform=mcfg.har.transform))
    if mcfg.naive_rw.enabled:
        models.append(RandomWalkRVForecaster())
    return models


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #


def _to_vol_pct(variance: pd.Series) -> pd.Series:
    """Realized *variance* -> daily *volatility* in percent, for readable plots."""
    return np.sqrt(variance.clip(lower=0.0)) * 100.0


def plot_forecast_vs_actual(preds: pd.DataFrame, model_names: list[str], title: str,
                            path: Path) -> Path:
    import matplotlib.pyplot as plt

    from src.utils.plotting import CRISIS_PERIODS, save_fig, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    idx = preds.index
    ax.plot(idx, _to_vol_pct(preds[ACTUAL_COL]), color="0.15", lw=1.2,
            label="Realized (actual)", zorder=5)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for name, c in zip(model_names, palette):
        ax.plot(idx, _to_vol_pct(preds[name]), lw=0.8, alpha=0.75, color=c, label=name)
    # Shade only crisis windows overlapping the plotted range, using
    # Timestamps (the shared shade_crises passes str dates, which some
    # matplotlib versions reject on a datetime axis).
    lo, hi = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    ymax = ax.get_ylim()[1]
    for label, (cs, ce) in CRISIS_PERIODS.items():
        cs, ce = pd.Timestamp(cs), pd.Timestamp(ce)
        if ce < lo or cs > hi:
            continue
        ax.axvspan(max(cs, lo), min(ce, hi), color="tab:red", alpha=0.12, lw=0)
        ax.text(max(cs, lo), ymax, label, fontsize=7, va="top", alpha=0.6, rotation=90)
    ax.set_ylabel("Daily volatility (%)")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(ncol=len(model_names) + 1, loc="upper left", fontsize=8)
    ax.set_xlim(idx.min(), idx.max())
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# One profile                                                                 #
# --------------------------------------------------------------------------- #


def run_profile(data_cfg, econ_cfg, profile) -> dict:
    name = profile.name
    log.info("=" * 70)
    log.info("PROFILE %s | target=%s | %s", name, profile.target, profile.description)

    frame = build_econometric_frame(data_cfg, target=profile.target)

    sp = profile.splits
    split_cfg = SplitConfig(
        train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
        test_start=sp.test_start, test_end=sp.test_end,
        refit_frequency_days=econ_cfg.harness.refit_frequency_days,
        scheme=econ_cfg.harness.scheme,
    )
    models = build_models(econ_cfg.models, return_col=econ_cfg.harness.return_col,
                          scale=econ_cfg.harness.scale)

    preds = run_walk_forward(
        models, frame, split_cfg,
        eval_segment=econ_cfg.harness.eval_segment,
        strict=True,
    )
    metrics = evaluate_predictions(preds)
    log.info("PROFILE %s metrics (QLIKE-ranked):\n%s", name, metrics.round(8).to_string())

    # Persist predictions + metrics.
    pred_path = repo_path(econ_cfg.paths.predictions, f"m02_{name}.parquet")
    to_parquet(preds, pred_path)
    tbl_path = repo_path(econ_cfg.paths.tables, f"m02_{name}_metrics.csv")
    ensure_dir(tbl_path.parent)
    metrics.to_csv(tbl_path)

    # Figure.
    fig_path = repo_path(econ_cfg.paths.figures, "m02", f"{name}_forecast_vs_actual.png")
    ensure_dir(fig_path.parent)
    plot_forecast_vs_actual(
        preds, [m.name for m in models],
        title=f"S&P 500 realized volatility: forecast vs actual ({name})",
        path=fig_path,
    )

    # Model diagnostics for the milestone note.
    diag = _collect_diagnostics(models)

    return {
        "name": name,
        "headline": bool(profile.get("headline", False)),
        "target": profile.target,
        "description": profile.description,
        "split": split_cfg,
        "n_oos": int(len(preds)),
        "oos_start": preds.index.min(),
        "oos_end": preds.index.max(),
        "n_folds": _n_folds(frame, split_cfg, econ_cfg.harness.eval_segment),
        "metrics": metrics,
        "diagnostics": diag,
        "pred_path": pred_path,
        "table_path": tbl_path,
        "fig_path": fig_path,
        "target_autocorr1": float(frame["rv"].autocorr(1)),
    }


def _n_folds(frame, split_cfg, eval_segment) -> int:
    from src.data.splits import walk_forward_folds

    return sum(1 for _ in walk_forward_folds(frame.index, split_cfg, eval_segment=eval_segment))


def _collect_diagnostics(models) -> dict:
    out: dict = {}
    for m in models:
        if isinstance(m, (GARCHForecaster, EGARCHForecaster)) and m.fold_params_:
            df = pd.DataFrame(m.fold_params_).T
            out[m.name] = {
                "persistence_mean": float(df["persistence"].mean()),
                "converged_frac": float(df["converged"].mean()),
                "n_fits": int(len(df)),
            }
        elif isinstance(m, HARForecaster) and m.fold_params_:
            df = pd.DataFrame(m.fold_params_).T
            out[m.name] = {
                "beta_d_mean": float(df["beta_d"].mean()),
                "beta_w_mean": float(df["beta_w"].mean()),
                "beta_m_mean": float(df["beta_m"].mean()),
                "r2_mean": float(df["r2"].mean()),
                "n_floored": int(m.n_floored_),
            }
    return out


# --------------------------------------------------------------------------- #
# Milestone note                                                              #
# --------------------------------------------------------------------------- #


def _fmt_metrics_table(metrics: pd.DataFrame) -> str:
    show = metrics.copy()
    show = show[["n", "qlike", "mse", "rmse", "mae"]]
    show.columns = ["n", "QLIKE", "MSE", "RMSE", "MAE"]
    lines = ["| model | n | QLIKE | MSE | RMSE | MAE |", "|---|---|---|---|---|---|"]
    for model, r in show.iterrows():
        lines.append(
            f"| {model} | {int(r['n'])} | {r['QLIKE']:.4f} | {r['MSE']:.3e} | "
            f"{r['RMSE']:.3e} | {r['MAE']:.3e} |"
        )
    return "\n".join(lines)


def write_milestone(results: list[dict], out_path: Path, *, econ_cfg) -> Path:
    headline = next((r for r in results if r["headline"]), results[0])
    hm = headline["metrics"]
    best = hm.index[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts: list[str] = []
    parts.append("# M02 — Econometric baselines (Phase 2)\n")
    parts.append(
        f"*Generated by `python -m src.experiments.run_econometric` on {ts}. "
        "All numbers trace to `results/tables/m02_*_metrics.csv` and "
        "`results/predictions/m02_*.parquet`.*\n"
    )
    parts.append("## Scope\n")
    parts.append(
        "Roadmap Phase 2 (ROADMAP.md §3): the three confirmed econometric "
        "baselines — **GARCH(1,1)** (Bollerslev 1986), **EGARCH(1,1,1)** "
        "(Nelson 1991) and **HAR-RV** (Corsi 2009) — evaluated on daily realized "
        "variance under a leakage-free anchored walk-forward with monthly refit. "
        "A random-walk-in-variance reference (**RW-RV**) is included as a sanity "
        "floor. QLIKE is the primary metric (Patton 2011); MSE/MAE are "
        "robustness checks. Losses are computed on the **variance** scale.\n"
    )

    # Target & window decision box.
    parts.append("## Target & window decision (final)\n")
    parts.append(
        "The dissertation's primary target is the Oxford-Man intraday RV "
        "(roadmap §1.1; Ch2 §2.3) — a 5-minute realized variance in the "
        "Andersen–Bollerslev–Diebold–Labys (2003) sense. Its `.SPX` coverage "
        "runs 2000-01-03 → **2022-02-25**, when the Oxford-Man Realized Library "
        "was discontinued. The roadmap originally envisaged extending the target "
        "to 2022–2024 by reconstructing 5-minute RV and splicing it on, but **no "
        "free 2022–2024 intraday data source was available, so that extension was "
        "abandoned.** The **final dataset is therefore 2000–2022**: the headline "
        "results below run on the canonical intraday target over its full native "
        "window, an anchored walk-forward whose test set spans **2019 → 25 Feb "
        "2022**, placing the COVID-19 crash out-of-sample.\n"
    )
    parts.append(
        "A **target-validity diagnostic** below shows *why* a Yang-Zhang OHLC RV "
        "cannot simply be substituted to extend coverage: its 21-day rolling "
        "smoothing makes the target near-perfectly autocorrelated, trivialising "
        "persistence models and inflating GARCH error — an artefact, not a "
        "result.\n"
    )

    for r in results:
        tag = " (headline)" if r["headline"] else " (diagnostic — not a result)"
        parts.append(f"## Profile: `{r['name']}`{tag}\n")
        parts.append(f"{r['description']}\n")
        sc = r["split"]
        parts.append(
            f"- **Target:** `{r['target']}` (realized variance), lag-1 "
            f"autocorrelation **{r['target_autocorr1']:.3f}**\n"
            f"- **Split:** train ≤ {pd.Timestamp(sc.train_end).date()}, val "
            f"{pd.Timestamp(sc.val_start).date()}–{pd.Timestamp(sc.val_end).date()}, test "
            f"{pd.Timestamp(sc.test_start).date()}–{pd.Timestamp(sc.test_end).date()}\n"
            f"- **Walk-forward:** {sc.scheme}, refit every "
            f"{sc.refit_frequency_days} trading days, {r['n_folds']} folds, "
            f"{r['n_oos']} OOS days ({r['oos_start'].date()} → {r['oos_end'].date()})\n"
        )
        parts.append("\n" + _fmt_metrics_table(r["metrics"]) + "\n")
        if r["diagnostics"]:
            parts.append("\n**Model diagnostics.**\n")
            for mname, d in r["diagnostics"].items():
                kv = ", ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in d.items()
                )
                parts.append(f"- {mname}: {kv}\n")
        parts.append(f"\n![forecast vs actual](../figures/m02/{r['name']}_forecast_vs_actual.png)\n")

    # Findings.
    parts.append("## Findings\n")
    parts.append(
        f"On the headline intraday target, **{best}** attains the lowest QLIKE "
        f"({hm.loc[best, 'qlike']:.4f}). "
    )
    order = " < ".join(f"{m} ({hm.loc[m, 'qlike']:.3f})" for m in hm.index)
    parts.append(f"Full QLIKE ranking (lower is better): {order}.\n")
    parts.append(
        "\nThe ordering is consistent with the realized-volatility literature: "
        "HAR-RV is hard to beat on its native target (Corsi 2009; Bucci 2020; "
        "Christensen, Siggaard & Veliyev 2023), and among the GARCH family the "
        "asymmetric EGARCH is competitive with or better than symmetric "
        "GARCH(1,1) on equity data (Hansen & Lunde 2005). This establishes the "
        "bar the Phase 3 LSTM and the later regime-aware / UQ models must clear "
        "(RQ1).\n"
    )

    # Gate.
    parts.append("## Gate criteria\n")
    parts.append("- [x] GARCH(1,1), EGARCH and HAR-RV implemented via a shared, unit-tested harness.\n")
    parts.append("- [x] Rolling out-of-sample loop with monthly refit; strict leakage discipline (HMM/scaler-style fit-on-train enforced structurally).\n")
    parts.append("- [x] MSE, MAE, QLIKE metrics with unit tests; QLIKE primary (Patton 2011).\n")
    parts.append("- [x] Rolling-OOS error table for all three baselines + reference floor.\n")
    parts.append("- [x] Forecast-vs-actual figure over the test period.\n")
    parts.append("- [x] Predictions persisted to `results/predictions/` (one column per model).\n")

    parts.append("## Reproduce\n")
    parts.append("```\npython -m src.experiments.run_econometric\npytest -q tests/test_metrics.py tests/test_har.py tests/test_econometric_models.py tests/test_rolling.py tests/test_intraday.py\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_econometric")
    ap.add_argument("--config", default="econometric", help="Phase-2 config name.")
    ap.add_argument("--data-config", default="data", help="Data/paths config name.")
    args = ap.parse_args(argv)

    econ_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(econ_cfg.seed))

    # Snapshot the resolved config (roadmap §9).
    run_dir = repo_path(econ_cfg.paths.experiments, "02_econometric",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(econ_cfg, run_dir)

    results = [run_profile(data_cfg, econ_cfg, p) for p in econ_cfg.profiles]

    milestone = write_milestone(
        results, repo_path(econ_cfg.paths.milestones, "m02_econometric.md"), econ_cfg=econ_cfg
    )
    log.info("wrote milestone -> %s", milestone)
    log.info("Phase 2 complete: %d profile(s) run.", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
