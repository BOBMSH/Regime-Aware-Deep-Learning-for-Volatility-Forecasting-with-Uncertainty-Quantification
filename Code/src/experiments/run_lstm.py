"""Phase 3 driver -- LSTM baseline end-to-end (ROADMAP.md Phase 3 gate).

Selects the LSTM hyperparameters on the validation split, runs the chosen model
through the shared walk-forward engine on the test window, and compares it
head-to-head against the Phase 2 econometric baselines on the identical splits
and metrics (RQ1). Writes the Phase 3 gate artifacts:

* ``results/predictions/m03_lstm_<profile>.parquet`` -- LSTM + Phase 2 baselines
  + realized target, one column per model, over the test window.
* ``results/tables/m03_lstm_<profile>_metrics.csv`` -- MSE/RMSE/MAE/QLIKE per
  model, QLIKE-ranked (LSTM alongside GARCH/EGARCH/HAR-RV/RW-RV).
* ``results/tables/m03_lstm_<profile>_sweep.csv`` -- validation QLIKE for every
  swept (hidden, lookback, lr) combination.
* ``results/figures/m03/<profile>_{hp_heatmap,learning_curves,forecast_vs_actual}.png``.
* ``results/milestones/m03_lstm.md`` -- the milestone note.
* ``experiments/03_lstm/run_<utc>/{config,best_hparams}.yaml`` -- reproducibility.

Usage
-----
    python -m src.experiments.run_lstm                 # full protocol
    python -m src.experiments.run_lstm --fast          # quick pass (reduced epochs/grid)
    python -m src.experiments.run_lstm --no-sweep      # use configs/lstm_baseline model block
    python -m src.experiments.run_lstm --max-epochs 60 --refit-every 0
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.datasets import build_econometric_frame
from src.data.splits import Fold, SplitConfig
from src.evaluation.metrics import mae, mse, qlike
from src.evaluation.rolling import ACTUAL_COL, evaluate_predictions, run_walk_forward
from src.evaluation.significance import diebold_mariano
from src.models.deep import LSTMForecaster
from src.utils.config import load_config, repo_path, snapshot_config
from src.utils.io import ensure_dir, from_parquet, to_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("run_lstm")


# --------------------------------------------------------------------------- #
# Model construction                                                          #
# --------------------------------------------------------------------------- #

def _common_kwargs(mcfg, hcfg, *, max_epochs: int | None) -> dict:
    return dict(
        num_layers=int(mcfg.num_layers),
        dropout=float(mcfg.dropout),
        weight_decay=float(mcfg.weight_decay),
        batch_size=int(mcfg.batch_size),
        max_epochs=int(max_epochs if max_epochs is not None else mcfg.max_epochs),
        patience=int(mcfg.patience),
        grad_clip=float(mcfg.grad_clip),
        val_fraction=float(mcfg.val_fraction),
        smearing=bool(mcfg.smearing),
        refit_every_folds=int(hcfg.refit_every_folds),
    )


def make_forecaster(mcfg, hcfg, *, hidden, lookback, lr, seed, name="LSTM",
                    max_epochs=None, features=None) -> LSTMForecaster:
    feats = (tuple(features) if features is not None
             else tuple(OmegaConf.to_container(mcfg.features, resolve=True)))
    return LSTMForecaster(
        hidden_size=int(hidden), lookback=int(lookback), lr=float(lr), seed=int(seed),
        name=name, features=feats, **_common_kwargs(mcfg, hcfg, max_epochs=max_epochs),
    )


def _segment_fold(index: pd.DatetimeIndex, *, train_end, seg_start, seg_end) -> Fold:
    """A single 'fold' whose training window ends at ``train_end`` and whose
    prediction window is ``[seg_start, seg_end]`` -- used to train-once and score a
    whole segment (the validation sweep)."""
    idx = pd.DatetimeIndex(index)
    tend = idx[idx <= pd.Timestamp(train_end)][-1]
    pstart = idx[idx >= pd.Timestamp(seg_start)][0]
    pend = idx[idx <= pd.Timestamp(seg_end)][-1]
    return Fold(fold_idx=0, refit_date=pstart, train_start=idx[0], train_end=tend,
                predict_start=pstart, predict_end=pend)


# --------------------------------------------------------------------------- #
# Hyperparameter sweep (validation segment)                                   #
# --------------------------------------------------------------------------- #

def sweep_validation(frame, split, mcfg, hcfg, scfg, *, seed, max_epochs) -> tuple[dict, pd.DataFrame]:
    val_fold = _segment_fold(frame.index, train_end=split.train_end,
                             seg_start=split.val_start, seg_end=split.val_end)
    y_val = frame.loc[val_fold.predict_start: val_fold.predict_end, "rv"]

    grid = list(itertools.product(
        [int(h) for h in scfg.hidden_size],
        [int(l) for l in scfg.lookback],
        [float(r) for r in scfg.lr],
    ))
    log.info("sweep: %d combinations on validation %s -> %s (%d obs)",
             len(grid), val_fold.predict_start.date(), val_fold.predict_end.date(), len(y_val))

    rows = []
    for hidden, lookback, lr in grid:
        m = make_forecaster(mcfg, hcfg, hidden=hidden, lookback=lookback, lr=lr,
                            seed=seed, max_epochs=max_epochs)
        pred = m.forecast_fold(frame, val_fold)
        yv = y_val.loc[pred.index]
        rows.append({
            "hidden_size": hidden, "lookback": lookback, "lr": lr,
            "val_qlike": qlike(yv, pred), "val_mse": mse(yv, pred), "val_mae": mae(yv, pred),
            "epochs": int(m.fold_params_.get(0, {}).get("epochs_run", 0)),
        })
        log.info("  hidden=%3d lookback=%2d lr=%.0e -> val QLIKE %.4f (%d ep)",
                 hidden, lookback, lr, rows[-1]["val_qlike"], rows[-1]["epochs"])

    table = pd.DataFrame(rows).sort_values("val_qlike").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    log.info("sweep best: hidden=%d lookback=%d lr=%.0e (val QLIKE %.4f)",
             int(best["hidden_size"]), int(best["lookback"]), best["lr"], best["val_qlike"])
    return best, table


# --------------------------------------------------------------------------- #
# Significance                                                                 #
# --------------------------------------------------------------------------- #
# The Diebold–Mariano test now lives in src.evaluation.significance in its formal
# form (Newey–West HAC + Harvey–Leybourne–Newbold small-sample correction +
# Student-t), alongside the Model Confidence Set. It is imported above and
# re-exported here so existing callers (``from src.experiments.run_lstm import
# diebold_mariano``) keep working unchanged.
__all__ = ["diebold_mariano"]


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #

def plot_hp_heatmap(sweep: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    # best (min) validation QLIKE per (hidden, lookback) cell, minimised over lr.
    piv = sweep.groupby(["lookback", "hidden_size"])["val_qlike"].min().unstack("hidden_size")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(piv.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("hidden size"); ax.set_ylabel("lookback")
    ax.set_title("Validation QLIKE by (hidden size, lookback)  [min over lr]")
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="QLIKE (lower better)")
    return save_fig(fig, path)


def plot_learning_curves(history: list[dict], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(h["epoch"], h["train_mse"], label="train", lw=1.4)
    ax.plot(h["epoch"], h["val_mse"], label="inner-val", lw=1.4)
    best = int(h["val_mse"].idxmin())
    ax.axvline(best, color="0.5", ls="--", lw=0.9, label=f"best epoch ({best})")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised log-RV)")
    ax.set_title("LSTM learning curves (final fit on train+val)")
    ax.legend(fontsize=8)
    return save_fig(fig, path)


def plot_forecast_vs_actual(preds: pd.DataFrame, model_names: list[str], title: str,
                            path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import CRISIS_PERIODS, save_fig, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    idx = preds.index
    to_vol = lambda v: np.sqrt(np.clip(v, 0, None)) * 100.0
    ax.plot(idx, to_vol(preds[ACTUAL_COL]), color="0.15", lw=1.2, label="Realized", zorder=5)
    palette = {"LSTM": "tab:blue", "LSTM-RVonly": "tab:cyan", "HAR-RV": "tab:green",
               "GARCH": "tab:orange", "EGARCH": "tab:red", "RW-RV": "tab:purple"}
    for name in model_names:
        ax.plot(idx, to_vol(preds[name]), lw=0.8, alpha=0.8,
                color=palette.get(name, None), label=name)
    lo, hi = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
    ymax = ax.get_ylim()[1]
    for label, (cs, ce) in CRISIS_PERIODS.items():
        cs, ce = pd.Timestamp(cs), pd.Timestamp(ce)
        if ce < lo or cs > hi:
            continue
        ax.axvspan(max(cs, lo), min(ce, hi), color="tab:red", alpha=0.12, lw=0)
        ax.text(max(cs, lo), ymax, label, fontsize=7, va="top", alpha=0.6, rotation=90)
    ax.set_ylabel("Daily volatility (%)"); ax.set_xlabel("")
    ax.set_title(title); ax.legend(ncol=len(model_names) + 1, loc="upper left", fontsize=8)
    ax.set_xlim(idx.min(), idx.max())
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# Milestone note                                                              #
# --------------------------------------------------------------------------- #

def _fmt_metrics(metrics: pd.DataFrame) -> str:
    lines = ["| model | n | QLIKE | MSE | RMSE | MAE |", "|---|---|---|---|---|---|"]
    for model, r in metrics.iterrows():
        label = f"**{model}**" if model == "LSTM" else str(model)
        lines.append(
            f"| {label} | {int(r['n'])} | {r['qlike']:.4f} | {r['mse']:.3e} | "
            f"{r['rmse']:.3e} | {r['mae']:.3e} |"
        )
    return "\n".join(lines)


def write_milestone(ctx: dict, out_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m = ctx["metrics"]; best = m.index[0]
    lstm_rank = list(m.index).index("LSTM") + 1
    har_q = m.loc["HAR-RV", "qlike"]; lstm_q = m.loc["LSTM", "qlike"]
    b = ctx["best"]
    dm = ctx["dm_lstm_vs_har"]

    p: list[str] = []
    p.append("# M03 — LSTM baseline (Phase 3)\n")
    p.append(f"*Generated by `python -m src.experiments.run_lstm` on {ts}. "
             "Numbers trace to `results/tables/m03_lstm_*` and "
             "`results/predictions/m03_lstm_*.parquet`.*\n")
    if ctx.get("fast"):
        p.append("> ⚠️ **FAST pass** — reduced epochs/sweep for a quick CPU run. "
                 "Re-run `python -m src.experiments.run_lstm` (full config, GPU) for the "
                 "headline numbers before citing.\n")
    p.append("## Scope\n")
    p.append("Roadmap Phase 3: a vanilla **LSTM** (Hochreiter & Schmidhuber 1997; Ch2 §2.5) "
             "forecasting daily realized *variance*, selected on the validation split and "
             "run through the **same** leakage-free anchored walk-forward as the Phase 2 "
             "baselines. QLIKE is primary (Patton 2011). This answers **RQ1**: does a deep "
             "sequence model beat GARCH(1,1)/EGARCH/HAR-RV on the same protocol?\n")

    p.append("## Chosen configuration\n")
    p.append(f"- **Features:** `{ctx['features']}` (log-RV history + open-to-close return); "
             f"target = log realized variance, mapped back to variance with a log-normal "
             f"smearing correction.\n"
             f"- **Selected on validation:** hidden **{int(b['hidden_size'])}**, lookback "
             f"**{int(b['lookback'])}**, lr **{b['lr']:.0e}** (val QLIKE {b['val_qlike']:.4f}).\n"
             f"- **Network:** {ctx['num_layers']}-layer LSTM, dropout {ctx['dropout']}, "
             f"Adam, early stopping (patience {ctx['patience']}) on a time-ordered inner-val tail.\n"
             f"- **Refit:** {ctx['refit_desc']}; test folds every {ctx['refit_freq']} days "
             f"(matches Phase 2).\n"
             f"- **Split:** train ≤ {ctx['split'].train_end}, val {ctx['split'].val_start}–"
             f"{ctx['split'].val_end}, test {ctx['split'].test_start}–{ctx['split'].test_end} "
             f"({ctx['n_oos']} OOS days, {ctx['n_folds']} folds).\n")

    p.append("\n![hyperparameter heatmap](../figures/m03/%s_hp_heatmap.png)\n" % ctx["name"])
    p.append("![learning curves](../figures/m03/%s_learning_curves.png)\n" % ctx["name"])

    p.append("## Head-to-head vs Phase 2 (QLIKE-ranked)\n")
    p.append(_fmt_metrics(m) + "\n")
    p.append(f"\n![forecast vs actual](../figures/m03/{ctx['name']}_forecast_vs_actual.png)\n")

    p.append("## Findings (RQ1)\n")
    variants = [x for x in ("LSTM-RVonly", "LSTM") if x in m.index]
    best_lstm = min(variants, key=lambda x: m.loc[x, "qlike"])
    best_lstm_q = m.loc[best_lstm, "qlike"]
    best_dm = (ctx.get("rv_only_dm") if best_lstm == "LSTM-RVonly" else dm) or dm
    sig = "statistically significant" if best_dm["p_value"] < 0.05 else "not statistically significant"
    n_beat = sum(m.loc[v, "qlike"] < har_q for v in variants)
    lead = ("Both LSTM variants beat HAR-RV on QLIKE" if len(variants) > 1 and n_beat == len(variants)
            else ("The LSTM beats HAR-RV on QLIKE" if m.loc["LSTM", "qlike"] < har_q
                  else "The LSTM does not beat HAR-RV on QLIKE"))
    if m.loc[best_lstm, "mse"] < m.loc["HAR-RV", "mse"]:
        metric_txt = (f"the best LSTM leads HAR-RV on both QLIKE and MSE ({best_lstm} MSE "
                      f"{m.loc[best_lstm, 'mse']:.2e} vs {m.loc['HAR-RV', 'mse']:.2e}), with MAE "
                      f"near-identical across the top models")
    else:
        metric_txt = (f"the picture is loss-dependent — the best LSTM leads on QLIKE but HAR-RV is "
                      f"marginally ahead on MSE ({m.loc['HAR-RV', 'mse']:.2e} vs "
                      f"{m.loc[best_lstm, 'mse']:.2e}), with MAE near-identical across the top models")
    p.append(f"{lead}; the best model overall is **{best_lstm}** (QLIKE **{best_lstm_q:.4f}** vs "
             f"HAR-RV **{har_q:.4f}**). A formal Diebold–Mariano test (HLN-corrected, Student-t) on that "
             f"pair gives stat **{best_dm['dm_stat']:.2f}**, p≈**{best_dm['p_value']:.3f}** — the gap is "
             f"**{sig}** (the Model Confidence Set over the full model board, controlling for multiple "
             f"comparison, is reported in Phase 5, `m05`). The ranking is "
             f"loss-function-dependent: {metric_txt} — a Patton-2011 reminder that the chosen metric "
             f"can move the ranking.\n")
    p.append("\nThis is the expected shape for daily equity RV: HAR-RV is a demanding "
             "benchmark on its native target (Corsi 2009; Bucci 2020; Kılıç 2025), so the "
             "value of the deep model is not assumed — it is tested. The regime-aware "
             "(Phase 5) and uncertainty-aware (Phase 6) variants build on this exact LSTM.\n")

    if "LSTM-RVonly" in m.index:
        ro_q = m.loc["LSTM-RVonly", "qlike"]; har2 = m.loc["HAR-RV", "qlike"]
        rdm = ctx.get("rv_only_dm") or {"dm_stat": float("nan"), "p_value": float("nan")}
        if ro_q < har2:
            interp = ("Even with its inputs restricted to RV history — HAR-RV's exact information "
                      f"set — the LSTM architecture alone attains lower QLIKE than HAR-RV "
                      f"({ro_q:.4f} vs {har2:.4f}), so the deep model's edge is not merely the extra "
                      "return feature.")
        else:
            interp = ("With its inputs restricted to RV history — HAR-RV's exact information set — the "
                      f"LSTM no longer beats HAR-RV ({ro_q:.4f} vs {har2:.4f}); the full model's QLIKE "
                      "edge therefore comes chiefly from the added open-to-close return feature rather "
                      "than the architecture per se. That is an honest, reportable distinction, and it "
                      "is exactly what this control is designed to expose.")
        p.append("## Robustness — architecture vs. information (RQ1, Ch1 §1.5-ii)\n")
        p.append(f"The full LSTM uses RV history **and** the day's return — a superset of any single "
                 f"baseline's inputs. To separate the contribution of the *architecture* from that of "
                 f"the *inputs*, an identical-hyperparameter LSTM restricted to `{ctx['rv_only_feats']}` "
                 f"(HAR-RV's information set exactly) is benchmarked alongside it. RV-only LSTM QLIKE "
                 f"**{ro_q:.4f}** vs full LSTM {lstm_q:.4f} and HAR-RV {har2:.4f}. {interp} Formal "
                 f"DM (RV-only − HAR-RV): stat {rdm['dm_stat']:.2f}, p≈{rdm['p_value']:.3f}.\n")

    # --- Auxiliary-feature ablation: does the VIX earn its place? (Ch1 §1.8) ---
    if ctx.get("vix_feats"):
        m = ctx["metrics"]
        vq = float(m.loc["LSTM-VIX", "qlike"]); lq = float(m.loc["LSTM", "qlike"])
        vdm = ctx.get("vix_dm") or {"dm_stat": float("nan"), "p_value": float("nan")}
        better = vq < lq
        verdict = (
            f"adding it {'lowers' if better else 'raises'} QLIKE ({vq:.4f} vs {lq:.4f}), and the "
            f"difference is {'significant' if vdm['p_value'] < 0.05 else 'not significant'} "
            f"(DM {vdm['dm_stat']:.2f}, p≈{vdm['p_value']:.3f})"
        )
        p.append("## Auxiliary feature — the VIX (Ch1 §1.8)\n")
        p.append(f"Chapter 1 §1.8 lists the CBOE VIX as an *auxiliary feature*. It is evaluated here as "
                 f"a one-feature ablation: the identical architecture and hyperparameters with "
                 f"`vix_close` appended (`{ctx['vix_feats']}`), so the row prices the feature and "
                 f"nothing else. On the test window {verdict}. The window ends at t−1, so a forecast "
                 f"for RVₜ sees the VIX only up to t−1; and because the VIX enters as an exogenous "
                 f"predictor rather than as a competing implied-volatility model, this does not "
                 f"re-open the option-implied paradigm Chapter 2 §2.1 places out of scope.\n")

    p.append("## Gate criteria\n")
    p.append("- [x] PyTorch `Dataset`-style sliding windows; vanilla LSTM regressor (dropout, MSE).\n")
    p.append("- [x] Hyperparameter sweep on validation (hidden × lookback × lr); heatmap saved.\n")
    p.append("- [x] Walk-forward evaluation with early stopping; leakage discipline "
             "(train-only scaling, windows end before target, HMM-style fit-on-train).\n")
    p.append("- [x] Learning curves saved.\n")
    p.append("- [x] Head-to-head vs GARCH(1,1)/EGARCH/HAR-RV on identical splits; predictions persisted.\n")
    p.append("- [x] Robustness: RV-only (HAR-matched-input) LSTM benchmarked to isolate architecture from inputs.\n")
    if ctx.get("vix_feats"):
        p.append("- [x] Auxiliary-feature ablation: VIX-augmented LSTM benchmarked against the "
                 "identical network without it (Ch1 §1.8).\n")

    p.append("## Reproduce\n")
    p.append("```\npython -m src.experiments.run_lstm\n"
             "pytest -q tests/test_sequence.py tests/test_lstm.py\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# One profile                                                                  #
# --------------------------------------------------------------------------- #

def run_profile(data_cfg, lcfg, profile, args) -> dict:
    name = profile.name
    log.info("=" * 70)
    log.info("PROFILE %s | target=%s", name, profile.target)

    # VIX must be joined onto the frame if *any* variant asks for it -- the main
    # feature set or one of the robustness variants (the auxiliary-feature ablation).
    _rob = lcfg.get("robustness", {}) or {}
    _all_feats = list(profile_features(lcfg))
    for _key in ("rv_only_features", "vix_features"):
        if _key in _rob:
            _all_feats += list(OmegaConf.to_container(_rob[_key], resolve=True))
    # Both VIX feature builders read the same underlying `vix_close` column.
    include_vix = any(f in ("vix_close", "log_vix") for f in _all_feats)
    frame = build_econometric_frame(data_cfg, target=profile.target, include_vix=include_vix)

    sp = profile.splits
    split_cfg = SplitConfig(
        train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
        test_start=sp.test_start, test_end=sp.test_end,
        refit_frequency_days=int(lcfg.harness.refit_frequency_days),
        scheme=lcfg.harness.scheme,
    )
    max_epochs = args.max_epochs if args.max_epochs else (40 if args.fast else None)

    # --- 1. Hyperparameter selection on validation ---
    if lcfg.sweep.enabled and not args.no_sweep:
        scfg = lcfg.sweep
        if args.fast:  # trim the grid for a quick CPU pass
            scfg = OmegaConf.create({"hidden_size": [32, 64], "lookback": [10, 20], "lr": [1.0e-3]})
        best, sweep_tbl = sweep_validation(frame, sp, lcfg.model, lcfg.harness, scfg,
                                           seed=int(lcfg.seed), max_epochs=max_epochs)
    else:
        best = {"hidden_size": int(lcfg.model.hidden_size), "lookback": int(lcfg.model.lookback),
                "lr": float(lcfg.model.lr), "val_qlike": float("nan")}
        sweep_tbl = pd.DataFrame([best])

    # --- 2. Final model through the test walk-forward ---
    if args.refit_every is not None:
        lcfg.harness.refit_every_folds = int(args.refit_every)
    model = make_forecaster(lcfg.model, lcfg.harness, hidden=best["hidden_size"],
                            lookback=best["lookback"], lr=best["lr"], seed=int(lcfg.seed),
                            name="LSTM", max_epochs=max_epochs)
    lstm_preds = run_walk_forward([model], frame, split_cfg,
                                  eval_segment=lcfg.harness.eval_segment, strict=True)

    # --- 3. Robustness: RV-only LSTM (same HP, HAR-matched inputs) ---
    # Isolates the deep architecture from the richer input set: an LSTM whose inputs
    # are RV history ONLY (HAR-RV's exact information set), same hyperparameters as the
    # full model, so a win over HAR-RV is attributable to the architecture rather than
    # the extra open-to-close return feature (Ch1 §1.5 objective ii; RQ1).
    rob = lcfg.get("robustness", {}) or {}
    rv_only_feats = None
    if bool(rob.get("rv_only_lstm", False)):
        rv_only_feats = tuple(OmegaConf.to_container(rob.rv_only_features, resolve=True))
        m_ro = make_forecaster(lcfg.model, lcfg.harness, hidden=best["hidden_size"],
                               lookback=best["lookback"], lr=best["lr"], seed=int(lcfg.seed),
                               name="LSTM-RVonly", max_epochs=max_epochs, features=rv_only_feats)
        ro_preds = run_walk_forward([m_ro], frame, split_cfg,
                                    eval_segment=lcfg.harness.eval_segment, strict=True)

    # --- 3b. Auxiliary-feature ablation: + VIX (Ch1 §1.8) ---
    # Chapter 1 §1.8 lists the CBOE VIX as an *auxiliary feature*. It enters here
    # and nowhere else: the same architecture and hyperparameters as the main LSTM
    # with `vix_close` appended, so the row isolates what the auxiliary feature is
    # worth rather than confounding it with a different network. Causality is the
    # ordinary sliding-window rule -- the window ends at t−1, so a forecast for RVₜ
    # sees VIX up to t−1 only. Note for Ch3: the VIX enters as an exogenous
    # predictor, which is *not* the option-implied-volatility modelling paradigm
    # Ch2 §2.1 places out of scope.
    vix_feats = None
    if bool(rob.get("vix_lstm", False)):
        vix_feats = tuple(OmegaConf.to_container(rob.vix_features, resolve=True))
        m_vx = make_forecaster(lcfg.model, lcfg.harness, hidden=best["hidden_size"],
                               lookback=best["lookback"], lr=best["lr"], seed=int(lcfg.seed),
                               name="LSTM-VIX", max_epochs=max_epochs, features=vix_feats)
        vx_preds = run_walk_forward([m_vx], frame, split_cfg,
                                    eval_segment=lcfg.harness.eval_segment, strict=True)

    # --- 4. Head-to-head with Phase 2 baselines ---
    base_path = repo_path(profile.baseline_predictions)
    combined = from_parquet(base_path).copy()
    combined["LSTM"] = lstm_preds["LSTM"].reindex(combined.index)
    if rv_only_feats is not None:
        combined["LSTM-RVonly"] = ro_preds["LSTM-RVonly"].reindex(combined.index)
    if vix_feats is not None:
        combined["LSTM-VIX"] = vx_preds["LSTM-VIX"].reindex(combined.index)
    combined = combined.dropna(subset=["LSTM"])
    metrics = evaluate_predictions(combined)
    log.info("PROFILE %s metrics (QLIKE-ranked):\n%s", name, metrics.round(6).to_string())

    y = combined[ACTUAL_COL]
    dm = diebold_mariano(y, combined["LSTM"], combined["HAR-RV"])
    rv_only_dm = (diebold_mariano(y, combined["LSTM-RVonly"], combined["HAR-RV"])
                  if "LSTM-RVonly" in combined.columns else None)
    # Does the auxiliary feature earn its place? Tested against the identical
    # network without it, so the comparison is the feature and nothing else.
    vix_dm = (diebold_mariano(y, combined["LSTM-VIX"], combined["LSTM"])
              if "LSTM-VIX" in combined.columns else None)

    # --- 5. Persist + figures ---
    pred_path = repo_path(lcfg.paths.predictions, f"m03_lstm_{name}.parquet")
    to_parquet(combined, pred_path)
    ensure_dir(repo_path(lcfg.paths.tables))
    metrics.to_csv(repo_path(lcfg.paths.tables, f"m03_lstm_{name}_metrics.csv"))
    sweep_tbl.to_csv(repo_path(lcfg.paths.tables, f"m03_lstm_{name}_sweep.csv"), index=False)

    figdir = repo_path(lcfg.paths.figures, "m03"); ensure_dir(figdir)
    plot_hp_heatmap(sweep_tbl, figdir / f"{name}_hp_heatmap.png")
    plot_learning_curves(model.history_, figdir / f"{name}_learning_curves.png")
    ordered = [c for c in ["LSTM", "LSTM-RVonly", "HAR-RV", "GARCH", "EGARCH", "RW-RV"]
               if c in combined.columns]
    plot_forecast_vs_actual(combined, ordered,
                            title=f"S&P 500 realized volatility: LSTM vs baselines ({name})",
                            path=figdir / f"{name}_forecast_vs_actual.png")

    refit_desc = ("train once on all pre-test data (reused across OOS)"
                  if int(lcfg.harness.refit_every_folds) <= 0
                  else f"retrain every {int(lcfg.harness.refit_every_folds)} fold(s)")
    return {
        "name": name, "metrics": metrics, "best": best, "sweep": sweep_tbl,
        "dm_lstm_vs_har": dm, "split": split_cfg, "features": list(profile_features(lcfg)),
        "num_layers": int(lcfg.model.num_layers), "dropout": float(lcfg.model.dropout),
        "patience": int(lcfg.model.patience), "refit_desc": refit_desc,
        "refit_freq": int(lcfg.harness.refit_frequency_days),
        "n_oos": int(len(combined)), "n_folds": ctx_n_folds(frame, split_cfg, lcfg),
        "fast": bool(args.fast), "pred_path": pred_path,
        "rv_only_dm": rv_only_dm, "rv_only_feats": list(rv_only_feats) if rv_only_feats else None,
        "vix_dm": vix_dm, "vix_feats": list(vix_feats) if vix_feats else None,
    }


def profile_features(lcfg):
    return OmegaConf.to_container(lcfg.model.features, resolve=True)


def ctx_n_folds(frame, split_cfg, lcfg) -> int:
    from src.data.splits import walk_forward_folds
    return sum(1 for _ in walk_forward_folds(frame.index, split_cfg,
                                             eval_segment=lcfg.harness.eval_segment))


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_lstm")
    ap.add_argument("--config", default="lstm_baseline")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true", help="reduced epochs + trimmed sweep (quick CPU pass)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the sweep; use the config model block")
    ap.add_argument("--max-epochs", type=int, default=0, help="override max epochs (0 = config/fast default)")
    ap.add_argument("--refit-every", type=int, default=None, help="override refit_every_folds")
    args = ap.parse_args(argv)
    args.max_epochs = args.max_epochs or None

    lcfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(lcfg.seed))

    run_dir = repo_path(lcfg.paths.experiments, "03_lstm",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(lcfg, run_dir)

    results = [run_profile(data_cfg, lcfg, p, args) for p in lcfg.profiles]
    for r in results:
        OmegaConf.save(OmegaConf.create({k: (int(v) if isinstance(v, np.integer) else
                                             float(v) if isinstance(v, np.floating) else v)
                                         for k, v in r["best"].items()}),
                       run_dir / f"best_hparams_{r['name']}.yaml")
        write_milestone(r, repo_path(lcfg.paths.milestones, "m03_lstm.md"))
    log.info("Phase 3 complete: %d profile(s).", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
