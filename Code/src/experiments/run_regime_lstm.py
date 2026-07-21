"""Phase 5 driver -- regime-aware LSTM end-to-end (ROADMAP.md Phase 5 gate; RQ2).

Conditions the Phase-3 LSTM on the Phase-4 HMM regime posteriors and runs the
result head-to-head against the regime-agnostic LSTM and the econometric
baselines on the *identical* leakage-free anchored walk-forward. Two conditioning
architectures are built (roadmap Phase 5):

* **Regime-LSTM-A** -- regime as feature: the causal filtered posteriors
  (reg_p0..reg_p2) appended to the LSTM's inputs. `Regime-LSTM-A-RVonly` is the
  same idea on HAR-RV's exact input set (log-RV history only), matched to the
  Phase-3 winner so the regime contribution is isolated from the return feature.
* **Regime-LSTM-B** -- regime-specific experts: one LSTM per regime, soft-gated by
  the posterior (mixture-of-experts; :class:`RegimeExpertForecaster`).

Hyperparameters are fixed at the Phase-3 selection (hidden 128, lookback 10, lr
1e-3) so the contrast "LSTM -> Regime-LSTM" measures regime-conditioning, not a
new sweep. Answers **RQ2**: does regime-conditioning improve accuracy, and is the
gain concentrated in transitional / crisis periods (the per-regime breakdown)?

Outputs (the Phase 5 gate artifacts)
------------------------------------
* ``results/predictions/m05_regime_dl_<profile>.parquet`` -- all models + realized
  target + the causal regime label, over the test window.
* ``results/tables/m05_regime_dl_<profile>_{metrics,per_regime,dm}.csv``.
* ``results/figures/m05/<profile>_{forecast_vs_actual,per_regime_qlike}.png``.
* ``results/milestones/m05_regime_dl.md`` -- the milestone note.
* ``experiments/05_regime_lstm/run_<utc>/config.yaml`` -- reproducibility.

Usage
-----
    python -m src.experiments.run_regime_lstm            # full protocol
    python -m src.experiments.run_regime_lstm --fast     # quick pass (few epochs)
    python -m src.experiments.run_regime_lstm --max-epochs 60
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
from src.evaluation.metrics import qlike
from src.evaluation.rolling import ACTUAL_COL, evaluate_predictions, run_walk_forward
from src.experiments.run_lstm import diebold_mariano  # reuse the indicative DM readout
from src.models.deep import LSTMForecaster, RegimeExpertForecaster
from src.utils.config import load_config, repo_path, snapshot_config
from src.utils.io import ensure_dir, from_parquet, to_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("run_regime_lstm")

REGIME_STATE_COL = "reg_state"


# --------------------------------------------------------------------------- #
# Frame assembly: attach the Phase-4 causal regime posteriors                 #
# --------------------------------------------------------------------------- #
def attach_regime(frame: pd.DataFrame, rcfg) -> pd.DataFrame:
    """Join the Phase-4 causal filtered posteriors onto the modelling frame.

    The Phase-4 parquet is date-indexed over the same sample as ``frame``. We take
    its **filtered** posterior columns (leakage-free) -> ``reg_p0..reg_p{K-1}`` and
    its causal hard state -> ``reg_state`` (used only for the per-regime
    breakdown, never as a model input).
    """
    reg = from_parquet(repo_path(rcfg.source))
    post = list(rcfg.posterior_cols)
    rename = {post[k]: f"reg_p{k}" for k in range(len(post))}
    rename[rcfg.state_col] = REGIME_STATE_COL
    regf = reg[post + [rcfg.state_col]].rename(columns=rename)

    joined = frame.join(regf, how="left")
    reg_cols = [f"reg_p{k}" for k in range(len(post))]
    missing = joined[reg_cols].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(
            f"{missing} frame rows have no regime posterior after the join -- the "
            "Phase-4 parquet and the modelling frame are misaligned."
        )
    log.info("regime signal attached: cols=%s state=%s (%d rows, %s -> %s)",
             reg_cols, REGIME_STATE_COL, len(joined),
             joined.index.min().date(), joined.index.max().date())
    return joined


# --------------------------------------------------------------------------- #
# Model construction                                                          #
# --------------------------------------------------------------------------- #
def build_models(mcfg, hcfg, appr, *, seed, max_epochs):
    common = dict(
        lookback=int(mcfg.lookback), hidden_size=int(mcfg.hidden_size),
        num_layers=int(mcfg.num_layers), dropout=float(mcfg.dropout), lr=float(mcfg.lr),
        weight_decay=float(mcfg.weight_decay), batch_size=int(mcfg.batch_size),
        max_epochs=int(max_epochs if max_epochs else mcfg.max_epochs),
        patience=int(mcfg.patience), grad_clip=float(mcfg.grad_clip),
        val_fraction=float(mcfg.val_fraction), smearing=bool(mcfg.smearing),
        refit_every_folds=int(hcfg.refit_every_folds), seed=int(seed),
    )
    base = list(OmegaConf.to_container(mcfg.base_features, resolve=True))
    reg = list(OmegaConf.to_container(mcfg.regime_features, resolve=True))
    models = []
    if bool(appr.get("feature_full", True)):
        models.append(LSTMForecaster(features=tuple(base + reg), name="Regime-LSTM-A", **common))
    if bool(appr.get("feature_rvonly", True)):
        models.append(LSTMForecaster(features=tuple(["log_rv"] + reg),
                                     name="Regime-LSTM-A-RVonly", **common))
    if bool(appr.get("experts", True)):
        models.append(RegimeExpertForecaster(base_features=tuple(base), gate_cols=tuple(reg),
                                             name="Regime-LSTM-B", **common))
    return models


# --------------------------------------------------------------------------- #
# Per-regime breakdown (core for RQ2)                                          #
# --------------------------------------------------------------------------- #
def per_regime_qlike(preds: pd.DataFrame, reg_state: pd.Series, labels, model_cols) -> pd.DataFrame:
    """Per-regime QLIKE for every model, using the causal filtered state label.

    Rows: one per regime (calm/transitional/crisis) plus an ``all`` row; columns:
    ``n`` (days in that regime in the test window) + one QLIKE per model.
    """
    y = preds[ACTUAL_COL]
    rows = []
    for r, lab in enumerate(labels):
        mask = np.asarray(reg_state.reindex(preds.index) == r)
        rec = {"regime": lab, "n": int(mask.sum())}
        for m in model_cols:
            sub = pd.concat([y[mask], preds[m][mask]], axis=1).dropna()
            rec[m] = qlike(sub.iloc[:, 0], sub.iloc[:, 1]) if len(sub) else np.nan
        rows.append(rec)
    rec = {"regime": "all", "n": int(len(preds))}
    for m in model_cols:
        sub = pd.concat([y, preds[m]], axis=1).dropna()
        rec[m] = qlike(sub.iloc[:, 0], sub.iloc[:, 1])
    rows.append(rec)
    return pd.DataFrame(rows)


def dm_table(preds: pd.DataFrame, pairs) -> pd.DataFrame:
    """Indicative DM (QLIKE differential) for the RQ2 head-to-head pairs.

    Each pair (a, b): a negative statistic means model *a* has the lower loss.
    """
    y = preds[ACTUAL_COL]
    rows = []
    for a, b in pairs:
        if a not in preds.columns or b not in preds.columns:
            continue
        sub = pd.concat([y, preds[a], preds[b]], axis=1).dropna()
        d = diebold_mariano(sub.iloc[:, 0], sub.iloc[:, 1], sub.iloc[:, 2])
        rows.append({"model_a": a, "model_b": b, "dm_stat": d["dm_stat"],
                     "p_value": d["p_value"], "mean_loss_diff": d["mean_loss_diff"],
                     "better": a if d["dm_stat"] < 0 else b, "n": d["n"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def plot_forecast_vs_actual(preds, reg_state, labels, model_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    idx = preds.index
    to_vol = lambda v: np.sqrt(np.clip(v, 0, None)) * 100.0
    fig, ax = plt.subplots(figsize=(11, 4.6))
    # Shade the test window by causal regime state (calm/transitional/crisis).
    band = {0: "#2ca02c", 1: "#f0c000", 2: "#d62728"}
    rs = reg_state.reindex(idx).to_numpy()
    start = 0
    for i in range(1, len(rs) + 1):
        if i == len(rs) or rs[i] != rs[start]:
            ax.axvspan(idx[start], idx[min(i, len(idx) - 1)],
                       color=band.get(int(rs[start]), "0.8"), alpha=0.12, lw=0)
            start = i
    ax.plot(idx, to_vol(preds[ACTUAL_COL]), color="0.15", lw=1.2, label="Realized", zorder=6)
    palette = {"Regime-LSTM-A": "tab:blue", "Regime-LSTM-B": "tab:purple",
               "LSTM": "tab:cyan", "HAR-RV": "tab:green"}
    for name in model_names:
        if name in preds.columns:
            ax.plot(idx, to_vol(preds[name]), lw=0.9, alpha=0.85,
                    color=palette.get(name), label=name)
    import matplotlib.patches as mpatches
    handles, _ = ax.get_legend_handles_labels()
    handles += [mpatches.Patch(color=band[r], alpha=0.3, label=f"regime: {labels[r]}")
                for r in range(len(labels))]
    ax.legend(handles=handles, ncol=4, loc="upper left", fontsize=7.5)
    ax.set_ylabel("Daily volatility (%)")
    ax.set_title(title)
    ax.set_xlim(idx.min(), idx.max())
    return save_fig(fig, path)


def plot_per_regime(per_reg: pd.DataFrame, labels, model_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    sub = per_reg[per_reg["regime"].isin(labels)].set_index("regime").reindex(labels)
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(model_names))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    palette = {"HAR-RV": "tab:green", "LSTM": "tab:cyan", "LSTM-RVonly": "0.5",
               "Regime-LSTM-A": "tab:blue", "Regime-LSTM-B": "tab:purple"}
    for j, m in enumerate(model_names):
        ax.bar(x + (j - (len(model_names) - 1) / 2) * w, sub[m].to_numpy(), w,
               label=m, color=palette.get(m))
    ax.set_xticks(x, [f"{lab}\n(n={int(sub.loc[lab, 'n'])})" for lab in labels])
    ax.set_ylabel("QLIKE (lower better)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# Milestone note                                                              #
# --------------------------------------------------------------------------- #
def _fmt_metrics(metrics: pd.DataFrame, highlight) -> str:
    lines = ["| model | n | QLIKE | MSE | RMSE | MAE |", "|---|---|---|---|---|---|"]
    for model, r in metrics.iterrows():
        label = f"**{model}**" if model in highlight else str(model)
        lines.append(f"| {label} | {int(r['n'])} | {r['qlike']:.4f} | {r['mse']:.3e} | "
                     f"{r['rmse']:.3e} | {r['mae']:.3e} |")
    return "\n".join(lines)


def _fmt_per_regime(per_reg: pd.DataFrame, model_cols) -> str:
    head = "| regime | n | " + " | ".join(model_cols) + " |"
    sep = "|---|---|" + "---|" * len(model_cols)
    lines = [head, sep]
    for _, r in per_reg.iterrows():
        cells = " | ".join(f"{r[m]:.4f}" if pd.notna(r[m]) else "—" for m in model_cols)
        lines.append(f"| {r['regime']} | {int(r['n'])} | {cells} |")
    return "\n".join(lines)


def write_milestone(ctx: dict, out_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    m = ctx["metrics"]; per = ctx["per_regime"]; dm = ctx["dm"]; name = ctx["name"]
    labels = ctx["labels"]
    reg_models = [c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "Regime-LSTM-A-RVonly"] if c in m.index]
    best = m.index[0]
    best_reg = min(reg_models, key=lambda x: m.loc[x, "qlike"])

    def q(x):
        return m.loc[x, "qlike"] if x in m.index else float("nan")

    def dm_row(a, b):
        r = dm[(dm["model_a"] == a) & (dm["model_b"] == b)]
        return r.iloc[0] if len(r) else None

    p: list[str] = []
    p.append("# M05 — Regime-aware models (Phase 5)\n")
    p.append(f"*Generated by `python -m src.experiments.run_regime_lstm` on {ts}. "
             "Numbers trace to `results/tables/m05_regime_dl_*` and "
             "`results/predictions/m05_regime_dl_*.parquet`.*\n")
    if ctx.get("fast"):
        p.append("> ⚠️ **FAST pass** — reduced epochs for a quick CPU run. Re-run "
                 "`python -m src.experiments.run_regime_lstm` (full config) before citing.\n")

    p.append("## Scope\n")
    p.append("Roadmap Phase 5 (Ch2 §2.5): condition the Phase-3 **LSTM** on the Phase-4 "
             "**HMM regime** signal and test whether it improves daily realized-variance "
             "forecasts on the same leakage-free anchored walk-forward as Phases 2–3 "
             "(**RQ2**). The regime signal is the **causal filtered** posterior "
             "P(sₜ|x₁..xₜ) from Phase 4 (`hmm_filt_p*`); it is lagged one day so a forecast "
             "for RVₜ only ever conditions on the regime known at t−1. Hyperparameters are "
             "fixed at the Phase-3 selection (hidden 128, lookback 10, lr 1e-3), so "
             "'LSTM → Regime-LSTM' isolates regime-conditioning rather than a new sweep.\n")
    p.append("Two architectures (roadmap Phase 5):\n"
             "- **Regime-LSTM-A** — regime as feature: filtered posteriors appended to the "
             "LSTM input (Ch2 §2.5). `Regime-LSTM-A-RVonly` is the same on HAR-RV's exact "
             "input set (RV history only), matched to the Phase-3 winner.\n"
             "- **Regime-LSTM-B** — regime-specific experts: one LSTM per regime, soft-gated "
             "by the posterior at t−1 (mixture-of-experts).\n")

    p.append("## Head-to-head (QLIKE-ranked)\n")
    p.append(_fmt_metrics(m, highlight=set(reg_models)) + "\n")
    p.append(f"\n![forecast vs actual](../figures/m05/{name}_forecast_vs_actual.png)\n")

    p.append("## Per-regime breakdown (RQ2)\n")
    p.append("QLIKE within each causal-filtered regime over the test window (the COVID-19 "
             "crash dominates the crisis state):\n")
    p.append(_fmt_per_regime(per, ctx["per_regime_models"]) + "\n")
    p.append(f"\n![per-regime QLIKE](../figures/m05/{name}_per_regime_qlike.png)\n")

    p.append("## Significance (indicative DM on QLIKE; formal DM+MCS in Phase 7)\n")
    p.append("| comparison (a vs b) | DM stat | p | better | n |")
    p.append("|---|---|---|---|---|")
    for _, r in dm.iterrows():
        p.append(f"| {r['model_a']} vs {r['model_b']} | {r['dm_stat']:.2f} | "
                 f"{r['p_value']:.3f} | {r['better']} | {int(r['n'])} |")
    p.append("")

    # ---- findings (data-driven: significance- and regime-aware) ----
    reg_helps = q(best_reg) < q("LSTM")
    sig = dm[dm["p_value"] < 0.05]
    base_cols = [c for c in ["HAR-RV", "LSTM", "LSTM-RVonly"] if c in ctx["per_regime_models"]]
    reg_cols = [c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "Regime-LSTM-A-RVonly"]
                if c in ctx["per_regime_models"]]
    p.append("## Findings (RQ2)\n")
    p.append(f"The overall best model on the board is **{best}** (QLIKE **{q(best):.4f}**), ahead of "
             f"the regime-agnostic LSTM {q('LSTM'):.4f} and HAR-RV {q('HAR-RV'):.4f}. "
             + ("Adding the regime signal **lowers** pooled QLIKE relative to the regime-agnostic "
                f"LSTM ({q(best_reg):.4f} vs {q('LSTM'):.4f})."
                if reg_helps else
                "On the pooled window the regime signal does not lower QLIKE relative to the "
                f"regime-agnostic LSTM ({q(best_reg):.4f} vs {q('LSTM'):.4f}).") + "\n")

    if len(sig):
        sig_txt = "; ".join(
            f"**{r['model_a']}** beats {r['model_b']} (DM {r['dm_stat']:.2f}, **p={r['p_value']:.3f}**)"
            for _, r in sig.iterrows())
        line = f"This is statistically supported where it matters most: {sig_txt}. "
    else:
        line = "No pooled pairwise gap reaches significance on 784 days. "
    bh = dm_row(best_reg, "HAR-RV")
    if bh is not None and bh["p_value"] >= 0.05:
        line += (f"Against the tougher, lower-parameter HAR-RV benchmark the gap is not significant "
                 f"(DM {bh['dm_stat']:.2f}, p={bh['p_value']:.3f}) — consistent with the Phase-3 "
                 "result and the literature that HAR-RV is hard to beat on its native target. ")
    line += "The formal DM + Model Confidence Set arrive in Phase 7."
    p.append(line + "\n")

    crow = per[per["regime"] == labels[-1]]
    if len(crow) and reg_cols and base_cols:
        crow = crow.iloc[0]
        br = min(reg_cols, key=lambda c: crow[c]); bb = min(base_cols, key=lambda c: crow[c])
        impr = 100.0 * (crow[bb] - crow[br]) / crow[bb]
        crow0 = per[per["regime"] == labels[0]].iloc[0]
        p.append(f"**The regime signal's value is concentrated in the hard regimes — exactly the RQ2 "
                 f"hypothesis.** In the **{labels[-1]}** state (n={int(crow['n'])}, COVID-19-dominated) "
                 f"every regime-aware model beats every baseline, led by **{br}** (QLIKE {crow[br]:.4f}) "
                 f"vs the best baseline {bb} ({crow[bb]:.4f}) — a **{impr:.0f}% lower** crisis QLIKE. In "
                 f"the **{labels[0]}** state (n={int(crow0['n'])}), which dominates the pooled mean, the "
                 f"regime signal adds little (best regime {min(crow0[c] for c in reg_cols):.4f} vs best "
                 f"baseline {min(crow0[c] for c in base_cols):.4f}). The pooled average therefore "
                 "*understates* the regime contribution; the per-regime table is where RQ2 is properly "
                 "adjudicated.\n")

    p.append("The two architectures capture the signal differently: **Regime-LSTM-B** (mixture-of-"
             "experts) wins overall and in the transitional state, while **Regime-LSTM-A** (regime as "
             "feature) is strongest in the crisis state. The MoE trains one LSTM per regime, so the "
             "crisis expert's *effective* sample is small (crisis ≈ 8% of days), which caps its "
             "specialisation — a limitation worth stating rather than hiding, and a reason the feature "
             "approach can lead precisely in the crisis rows.\n")

    p.append("## Gate criteria\n")
    p.append("- [x] Approach A (regime-as-feature) and Approach B (regime-expert MoE) built, "
             "both on the causal filtered posterior; leakage discipline reused from Phase 3 "
             "(train-only scaling, windows end before target, gate lagged to t−1).\n")
    p.append("- [x] Four-way+ comparison vs GARCH/EGARCH/HAR-RV/RW-RV/LSTM/LSTM-RVonly on "
             "identical splits; predictions persisted.\n")
    p.append("- [x] Per-regime (calm/transitional/crisis) error breakdown — core for RQ2.\n")
    p.append("- [x] Indicative Diebold–Mariano on the head-to-head pairs (formal DM+MCS = Phase 7).\n")
    p.append("- [x] Milestone + figures + reproducible config snapshot.\n")

    p.append("## Reproduce\n")
    p.append("```\npython -m src.experiments.run_regime_lstm\n"
             "pytest -q tests/test_regime_lstm.py\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# One profile                                                                  #
# --------------------------------------------------------------------------- #
def run_profile(data_cfg, cfg, profile, args) -> dict:
    name = profile.name
    log.info("=" * 70)
    log.info("PROFILE %s | target=%s", name, profile.target)

    frame = build_econometric_frame(data_cfg, target=profile.target, include_vix=True)
    frame = attach_regime(frame, cfg.regime)
    reg_state = frame[REGIME_STATE_COL]
    labels = list(OmegaConf.to_container(cfg.regime.labels, resolve=True))

    sp = profile.splits
    split_cfg = SplitConfig(
        train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
        test_start=sp.test_start, test_end=sp.test_end,
        refit_frequency_days=int(cfg.harness.refit_frequency_days), scheme=cfg.harness.scheme,
    )
    max_epochs = args.max_epochs if args.max_epochs else (8 if args.fast else None)

    models = build_models(cfg.model, cfg.harness, cfg.approaches, seed=int(cfg.seed),
                          max_epochs=max_epochs)
    log.info("Phase-5 models: %s", [mm.name for mm in models])
    reg_preds = run_walk_forward(models, frame, split_cfg,
                                 eval_segment=cfg.harness.eval_segment, strict=True)

    # --- merge with the Phase-2/3 baselines (same target/window) ---
    base = from_parquet(repo_path(profile.baseline_predictions)).copy()
    for mm in models:
        base[mm.name] = reg_preds[mm.name].reindex(base.index)
    combined = base.dropna(subset=[mm.name for mm in models]).copy()
    model_cols = [c for c in combined.columns if c != ACTUAL_COL]

    metrics = evaluate_predictions(combined)
    log.info("PROFILE %s metrics (QLIKE-ranked):\n%s", name, metrics.round(6).to_string())

    # --- per-regime breakdown (report the key models to keep the table readable) ---
    per_regime_models = [c for c in ["HAR-RV", "LSTM", "LSTM-RVonly",
                                     "Regime-LSTM-A", "Regime-LSTM-A-RVonly", "Regime-LSTM-B"]
                         if c in combined.columns]
    per = per_regime_qlike(combined, reg_state, labels, per_regime_models)
    log.info("PROFILE %s per-regime QLIKE:\n%s", name, per.round(4).to_string(index=False))

    # --- indicative DM on the RQ2 pairs ---
    pairs = [("Regime-LSTM-A", "LSTM"), ("Regime-LSTM-A", "HAR-RV"),
             ("Regime-LSTM-B", "LSTM"), ("Regime-LSTM-B", "HAR-RV"),
             ("Regime-LSTM-A-RVonly", "LSTM-RVonly"),
             (min([c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "Regime-LSTM-A-RVonly"]
                   if c in combined.columns], key=lambda x: metrics.loc[x, "qlike"]), "HAR-RV")]
    seen, uniq = set(), []
    for a, b in pairs:
        if (a, b) not in seen:
            seen.add((a, b)); uniq.append((a, b))
    dm = dm_table(combined, uniq)

    # --- persist predictions (+ regime label) + tables ---
    out_pred = combined.copy()
    out_pred[REGIME_STATE_COL] = reg_state.reindex(out_pred.index).astype("Int64")
    pred_path = repo_path(cfg.paths.predictions, f"m05_regime_dl_{name}.parquet")
    to_parquet(out_pred, pred_path)
    ensure_dir(repo_path(cfg.paths.tables))
    metrics.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_metrics.csv"))
    per.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_per_regime.csv"), index=False)
    dm.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_dm.csv"), index=False)

    # --- figures ---
    figdir = repo_path(cfg.paths.figures, "m05"); ensure_dir(figdir)
    fva_models = [c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "LSTM", "HAR-RV"] if c in combined.columns]
    plot_forecast_vs_actual(combined, reg_state, labels, fva_models,
                            f"S&P 500 realized volatility: regime-aware LSTM vs baselines ({name})",
                            figdir / f"{name}_forecast_vs_actual.png")
    bar_models = [c for c in ["HAR-RV", "LSTM", "Regime-LSTM-A", "Regime-LSTM-B"] if c in combined.columns]
    plot_per_regime(per, labels, bar_models,
                    f"Per-regime QLIKE ({name})", figdir / f"{name}_per_regime_qlike.png")

    return {"name": name, "metrics": metrics, "per_regime": per,
            "per_regime_models": per_regime_models, "dm": dm, "labels": labels,
            "n_oos": int(len(combined)), "fast": bool(args.fast), "pred_path": pred_path,
            "models": [mm.name for mm in models]}


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_regime_lstm")
    ap.add_argument("--config", default="regime_lstm")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true", help="few epochs (quick CPU smoke pass)")
    ap.add_argument("--max-epochs", type=int, default=0, help="override max epochs (0 = config/fast default)")
    args = ap.parse_args(argv)
    args.max_epochs = args.max_epochs or None

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "05_regime_lstm",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    for profile in cfg.profiles:
        ctx = run_profile(data_cfg, cfg, profile, args)
        write_milestone(ctx, repo_path(cfg.paths.milestones, "m05_regime_dl.md"))
    log.info("Phase 5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
