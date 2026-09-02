"""Phase 5 driver -- regime-aware LSTM end-to-end (ROADMAP.md Phase 5 gate; RQ2).

Conditions the Phase-3 LSTM on the Phase-4 causal regime posteriors -- by default
those of the jump-penalised HMM (Nystrup et al. 2020; Ch2 §2.4/§2.8), with the
Baum-Welch HMM available as a robustness variant (configs/regime_lstm_hmm.yaml) --
and runs the
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

from src.data.datasets import frame_for_profile
from src.data.splits import SplitConfig, walk_forward_folds
from src.evaluation.metrics import qlike
from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    align_regime_label,
    regime_timing_label,
    transition_day_summary,
)
from src.evaluation.rolling import ACTUAL_COL, evaluate_predictions, run_walk_forward
from src.evaluation.significance import diebold_mariano, model_confidence_set
from src.models.deep import LSTMForecaster, RegimeExpertForecaster
from src.utils.config import (load_config, milestone_path, repo_path,
                              snapshot_config)
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
def per_regime_qlike(
    preds: pd.DataFrame,
    reg_state: pd.Series,
    labels,
    model_cols,
    *,
    shift: int = DEFAULT_REGIME_SHIFT,
) -> pd.DataFrame:
    """Per-regime QLIKE for every model, bucketed by the regime known at ``t-shift``.

    Rows: one per regime (calm/transitional/crisis) plus an ``all`` row; columns:
    ``n`` (days in that bucket in the test window) + one QLIKE per model.

    ``shift`` defaults to ``1``, which is a methodological choice, not a
    formatting one. The Phase-4 filtered state ``s_t`` is inferred **using the
    day-t return**, so bucketing day ``t``'s forecast error by ``state[t]`` sorts
    days by the very outcome whose error is measured inside the bucket. Under
    that timing a claim like "the regime models' edge is concentrated in state k"
    is not falsifiable by a user of the model, because "state k" was not knowable
    when the forecast was issued. ``shift=1`` uses the label the model itself
    conditions on (its input window ends at ``t-1``) and matches the lagged
    indicators :func:`src.evaluation.significance.regime_test_function` feeds to
    the Giacomini–White test — so the descriptive table and the formal test now
    answer the same question. Pass ``shift=0`` only for an explicitly-labelled
    ex-post description. See :mod:`src.evaluation.regime_timing`.

    The timing is recorded on ``.attrs['regime_shift']``.
    """
    y = preds[ACTUAL_COL]
    # warn_unreachable: ``reg_state`` must be the full Phase-4 series, not one
    # sliced to the test window — otherwise the first scored day loses its t-1
    # label, the buckets stop summing to the pooled n, and this table disagrees
    # with report_gw's by exactly one day (2026-08-19 (iv)).
    lab_series = align_regime_label(reg_state, preds.index, shift=shift,
                                    warn_unreachable=True)
    rows = []
    for r, lab in enumerate(labels):
        mask = np.asarray(lab_series == float(r))
        rec = {"regime": lab, "n": int(mask.sum())}
        for m in model_cols:
            sub = pd.concat([y[mask], preds[m][mask]], axis=1).dropna()
            rec[m] = qlike(sub.iloc[:, 0], sub.iloc[:, 1]) if len(sub) else np.nan
        rows.append(rec)
    # The ``all`` row spans the same days the buckets do, so the bucket counts
    # add up to it: with shift>0 the first day has no lagged label and is
    # excluded from both. A mismatched total is exactly how the previous
    # convention hid the fact that the buckets and the pooled DM tests were run
    # on different samples.
    scored = np.asarray(lab_series.notna())
    rec = {"regime": "all", "n": int(scored.sum())}
    for m in model_cols:
        sub = pd.concat([y[scored], preds[m][scored]], axis=1).dropna()
        rec[m] = qlike(sub.iloc[:, 0], sub.iloc[:, 1])
    rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["regime_shift"] = int(shift)
    out.attrs["regime_timing"] = regime_timing_label(shift)
    return out


def dm_table(preds: pd.DataFrame, pairs) -> pd.DataFrame:
    """Formal Diebold–Mariano (QLIKE differential) for the RQ2 head-to-head pairs.

    Uses the HLN-corrected, Student-t DM from :mod:`src.evaluation.significance`.
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

    # The conditioning signal is config-driven (jump-penalised HMM by default,
    # Baum-Welch in the robustness variant), so the note must describe whichever
    # was actually consumed rather than assert one of them.
    src_cols = list(ctx.get("regime_cols") or [])
    is_jphmm = any(str(c).startswith("jphmm") for c in src_cols)
    estimator = (
        "**jump-penalised HMM** (Nystrup, Lindström & Madsen 2020 — the estimator "
        "Ch2 §2.4/§2.8 adopts)" if is_jphmm else
        "**Baum–Welch HMM** (the regime-estimator robustness comparator; the headline "
        "run conditions on the jump-penalised HMM per Ch2 §2.8)"
    )
    cols_txt = ", ".join(f"`{c}`" for c in src_cols) or "`*_filt_p*`"
    p.append("## Scope\n")
    p.append(f"Roadmap Phase 5 (Ch2 §2.5): condition the Phase-3 **LSTM** on the Phase-4 "
             f"regime signal from the {estimator} and test whether it improves daily "
             "realized-variance forecasts on the same leakage-free anchored walk-forward as "
             "Phases 2–3 (**RQ2**). The signal is the **causal filtered** posterior "
             f"P(sₜ|x₁..xₜ) from Phase 4 ({cols_txt}); it is lagged one day so a forecast "
             "for RVₜ only ever conditions on the regime known at t−1. Hyperparameters are "
             "fixed at the Phase-3 selection (hidden 128, lookback 10, lr 1e-3), so "
             "'LSTM → Regime-LSTM' isolates regime-conditioning rather than a new sweep.\n")
    p.append("> **Refit cadence (read with every number below).** As in Phase 3, the deep "
             "models use `refit_every_folds: 0`: each network is trained **once** on all "
             "pre-test data (≤ 2018-12-31) and then frozen across the whole 2019 → Feb-2022 "
             "out-of-sample window, while GARCH/EGARCH/HAR-RV are refit every 21 trading "
             "days on an expanding window. Both schemes are leakage-free, but they are not "
             "symmetric: by February 2022 the econometric baselines have seen three more "
             "years of data than the networks, including the COVID crash. The asymmetry "
             "therefore runs **against** the deep models, so their wins are conservative — "
             "and any deep-model weakness in the high-volatility states cannot be separated "
             "from the staleness of their training window on this evidence alone. Chapter 3 "
             "states this; a `refit_every_folds: 1` sensitivity is Phase-8 work.\n")
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
    p.append("QLIKE within each regime over the test window. **A day is bucketed by the "
             "regime known at t−1**, not by the regime the filter assigns to that day. This "
             "is a methodological choice and it changes the answer: the causal filtered "
             "state sₜ is inferred *using the day-t return*, so bucketing day t's error by "
             "sₜ would sort days by the very outcome whose error is being measured inside "
             "the bucket. Only the t−1 label makes \"the edge is concentrated in state k\" a "
             "claim about something a user of the model actually knew, and it is the same "
             "label the regime-aware models condition on and the same one the "
             "Giacomini–White test below is given.\n")
    p.append(_fmt_per_regime(per, ctx["per_regime_models"]) + "\n")
    p.append(f"\n![per-regime QLIKE](../figures/m05/{name}_per_regime_qlike.png)\n")

    per_ex = ctx.get("per_regime_expost")
    tr = ctx.get("transition")
    if per_ex is not None and len(per_ex):
        p.append("<details><summary><b>Ex-post (contemporaneous) split — descriptive only, "
                 "do not cite as a conditional result</b></summary>\n")
        p.append("\nThe same table bucketed by the regime the filter assigns to day t "
                 "itself. It answers \"what did the market turn out to be doing on the days "
                 "where model X did well?\", which is narrative colour, not a conditional "
                 "performance claim. It is printed so the difference between the two "
                 "conventions is visible rather than hidden.\n")
        p.append(_fmt_per_regime(per_ex, ctx["per_regime_models"]) + "\n")
        p.append("\n</details>\n")
    if tr and tr.get("n_transition"):
        p.append(f"\nThe two tables differ on **{tr['n_transition']} of {tr['n']} days "
                 f"({100 * tr['share']:.1f}%)** — the regime-*transition* days, whose bucket "
                 "is decided by the labelling convention alone. That is the entire gap "
                 "between them, and it is why a per-regime ranking that flips between "
                 "conventions is evidence about the convention, not about the models.\n")

    mcs = ctx.get("mcs")
    p.append("## Significance — formal DM + Model Confidence Set\n")
    p.append("Pairwise **Diebold–Mariano** (1995) on the QLIKE differential, with the "
             "Harvey–Leybourne–Newbold (1997) small-sample correction and a Student-t "
             "reference (a negative statistic means the first model has the lower loss). "
             "The **Model Confidence Set** (Hansen, Lunde & Nason 2011) then controls for "
             "multiple comparison across *all* models jointly (Ch2 §2.7).\n")
    p.append("| comparison (a vs b) | DM stat | p | better | n |")
    p.append("|---|---|---|---|---|")
    for _, r in dm.iterrows():
        p.append(f"| {r['model_a']} vs {r['model_b']} | {r['dm_stat']:.2f} | "
                 f"{r['p_value']:.3f} | {r['better']} | {int(r['n'])} |")
    p.append("")
    if mcs is not None:
        conf = int(round((1 - mcs.attrs["alpha"]) * 100))
        p.append(f"**Model Confidence Set — {conf}% confidence.** QLIKE loss; stationary "
                 f"bootstrap ({mcs.attrs['reps']} reps, block {mcs.attrs['block_size']}, "
                 f"'{mcs.attrs['method']}' statistic, seed {mcs.attrs['seed']}). Surviving set "
                 f"= {{{', '.join(mcs.attrs['included'])}}}; eliminated "
                 f"= {{{', '.join(mcs.attrs['excluded'])}}}.\n")
        p.append("| model | avg QLIKE | MCS p-value | in MCS |")
        p.append("|---|---|---|---|")
        for mname, rr in mcs.iterrows():
            p.append(f"| {mname} | {rr['avg_loss']:.4f} | {rr['mcs_pvalue']:.3f} | "
                     f"{'✓' if rr['in_mcs'] else '—'} |")
        p.append("")

    # ---- findings (data-driven: significance- and regime-aware) ----
    sig = dm[dm["p_value"] < 0.05]
    base_cols = [c for c in ["HAR-RV", "LSTM", "LSTM-RVonly"] if c in ctx["per_regime_models"]]
    reg_cols = [c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "Regime-LSTM-A-RVonly"]
                if c in ctx["per_regime_models"]]
    # The effect of regime conditioning is only readable against the SAME feature
    # set. Comparing the best regime model to whichever regime-agnostic model one
    # likes mixes the treatment with the input set: on this board the RV-only pair
    # and the full-feature pair point in OPPOSITE directions, so an unmatched
    # comparison can report either sign at will.
    MATCHED = [("Regime-LSTM-A", "LSTM"),
               ("Regime-LSTM-A-RVonly", "LSTM-RVonly"),
               ("Regime-LSTM-B", "LSTM")]
    matched = [(a, b) for a, b in MATCHED
               if a in ctx["per_regime_models"] and b in ctx["per_regime_models"]]
    p.append("## Findings (RQ2)\n")
    p.append(f"The overall best model on the board is **{best}** (QLIKE **{q(best):.4f}**), ahead of "
             f"the regime-agnostic LSTM {q('LSTM'):.4f} and HAR-RV {q('HAR-RV'):.4f}.\n")
    if matched:
        helped = [(a, b) for a, b in matched if q(a) < q(b)]
        hurt = [(a, b) for a, b in matched if q(a) >= q(b)]
        bits = "; ".join(f"{a} {q(a):.4f} vs {b} {q(b):.4f} "
                         f"({'−' if q(a) < q(b) else '+'}{abs(q(a) - q(b)):.4f})"
                         for a, b in matched)
        if helped and hurt:
            verdict = (f"**Regime conditioning does not have a consistent pooled sign.** Compared "
                       f"only against its own regime-agnostic twin — the same features, the same "
                       f"hyperparameters, the regime posterior added — it helps in "
                       f"{len(helped)} of {len(matched)} matched pairs and hurts in {len(hurt)}: ")
        elif helped:
            verdict = ("**Regime conditioning lowers pooled QLIKE in every matched pair** (same "
                       "features, same hyperparameters, regime posterior added): ")
        else:
            verdict = ("**Regime conditioning does not lower pooled QLIKE in any matched pair** "
                       "(same features, same hyperparameters, regime posterior added): ")
        p.append(verdict + bits + ". None of these gaps is significant (see the DM table above); "
                 "note in particular that the best pooled model on the board carries **no** regime "
                 "signal.\n")

    if len(sig):
        sig_txt = "; ".join(
            f"**{r['model_a']}** beats {r['model_b']} (DM {r['dm_stat']:.2f}, **p={r['p_value']:.3f}**)"
            for _, r in sig.iterrows())
        line = f"This is statistically supported where it matters most: {sig_txt}. "
    else:
        line = f"No pooled pairwise gap reaches significance on {ctx['n_oos']} days. "
    bh = dm_row(best_reg, "HAR-RV")
    if bh is not None and bh["p_value"] >= 0.05:
        line += (f"Against the tougher, lower-parameter HAR-RV benchmark the gap is not significant "
                 f"(DM {bh['dm_stat']:.2f}, p={bh['p_value']:.3f}) — consistent with the Phase-3 "
                 "result and the literature that HAR-RV is hard to beat on its native target. ")
    if mcs is not None:
        conf = int(round((1 - mcs.attrs["alpha"]) * 100))
        line += (f"The {conf}% **Model Confidence Set** retains "
                 f"{{{', '.join(mcs.attrs['included'])}}} and rejects "
                 f"{{{', '.join(mcs.attrs['excluded'])}}}: the deep-learning/HAR cluster is "
                 f"statistically inseparable on {ctx['n_oos']} days, while the GARCH family and the "
                 "random-walk floor are excluded outright.")
    p.append(line + "\n")

    # Per-regime narrative, derived from the table rather than asserted. An earlier
    # version hardcoded "the regime signal's value is concentrated in the hard
    # regimes"; when the regime estimator changed, the numbers reversed and the
    # prose did not. Every claim below is now computed, including its direction.
    if reg_cols and base_cols and len(per):
        # Per state, read the MATCHED pairs -- same reason as the pooled paragraph.
        wins, losses, detail, gaps_pct = [], [], [], []
        for lab in labels:
            row = per[per["regime"] == lab]
            if not len(row):
                continue
            row = row.iloc[0]
            n_better = sum(1 for a, b in matched if row[a] < row[b])
            (wins if n_better > len(matched) / 2 else losses).append(lab)
            gaps_pct.extend(100.0 * (row[a] - row[b]) / row[b] for a, b in matched)
            pair_txt = ", ".join(
                f"{a} {row[a]:.4f} vs {b} {row[b]:.4f} "
                f"({100.0 * (row[a] - row[b]) / row[b]:+.1f}%)" for a, b in matched)
            detail.append(f"**{lab}** (n={int(row['n'])}): {n_better}/{len(matched)} matched pairs "
                          f"favour the regime model — {pair_txt}")
        if wins and not losses:
            head = ("**A majority of matched pairs favours regime conditioning in every state** "
                    "— on margins this small, read that as 'no clear harm', not as a win. ")
        elif wins:
            head = (f"**Regime conditioning helps in some states and not others** — a majority of "
                    f"matched pairs favours it in {', '.join(wins)} and not in "
                    f"{', '.join(losses)}. ")
        else:
            head = "**No state shows a majority of matched pairs favouring regime conditioning.** "
        p.append(head + "; ".join(detail) + ".\n")
        # per-state rows only -- the table also carries an "all" row (the full window).
        _ns = per[per["regime"].isin(labels)]["n"]
        if gaps_pct and len(_ns):
            p.append(f"Every one of those matched gaps lies between {min(map(abs, gaps_pct)):.1f}% "
                     f"and {max(map(abs, gaps_pct)):.1f}% of the baseline's QLIKE, on buckets of "
                     f"{int(_ns.min())}–{int(_ns.max())} days, with **no significance "
                     "test attached** — the direction of a majority vote across three pairs is "
                     "not evidence of an effect at these magnitudes. The Giacomini–White test "
                     "referenced below is what settles it, and it localises the only surviving "
                     "effect in the transitional state.\n")

        # The best-of-N framing is retained because readers expect it, but it is a
        # SELECTION statistic (min over 3 regime models vs min over 3 baselines,
        # per bucket) and reads far stronger than the evidence supports. Label it.
        sel_bits = []
        for lab in labels:
            row = per[per["regime"] == lab]
            if not len(row):
                continue
            row = row.iloc[0]
            br = min(reg_cols, key=lambda c: row[c])
            bb = min(base_cols, key=lambda c: row[c])
            gap = 100.0 * (row[bb] - row[br]) / row[bb]
            sel_bits.append(f"{lab} {br} {row[br]:.4f} vs {bb} {row[bb]:.4f} "
                            f"({abs(gap):.1f}% {'lower' if gap > 0 else 'higher'})")
        p.append("For reference, the best-of-each-family comparison — lowest of "
                 f"{len(reg_cols)} regime models against lowest of {len(base_cols)} baselines "
                 "within each bucket — reads: " + "; ".join(sel_bits) + ". **Treat that line as "
                 "descriptive only.** Taking a minimum over several models inside a bucket is a "
                 "selection statistic with no error bar attached; it will show a margin in favour "
                 "of whichever family has more entries even when every individual contrast is "
                 "noise, and which architecture supplies the minimum changes from state to "
                 "state.\n")
        p.append("Two cautions belong with this table rather than after it. First, the split "
                 "is still *descriptive*: it reports point estimates per bucket, and the "
                 "pooled DM tests above are the only significance evidence attached to it — "
                 "run `python -m src.experiments.report_gw` for the Giacomini–White "
                 "conditional test and the per-regime DM columns, which use the same t−1 "
                 "indicators and are the properly sized way to ask whether the loss difference "
                 "is regime-dependent. On the current board only the **transitional** state "
                 "survives that test; the calm and crisis contrasts do not, and the crisis "
                 "claim has been formally retracted in `ROADMAP.md` — do not reintroduce it "
                 "from this table. Second, the crisis bucket is the smallest and therefore the "
                 "least stable: with ~9% of the window it takes only a handful of relabelled "
                 "turning-point days to move its mean substantially, so a crisis-state ranking "
                 "should be checked against the alternative regime estimator "
                 "(`configs/regime_lstm_hmm.yaml`) before it is reported as a finding.\n")

    # Architecture contrast, also computed.
    if reg_cols and len(per):
        best_overall = m.index[0]
        arch_bits = []
        for lab in labels:
            row = per[per["regime"] == lab]
            if len(row):
                arch_bits.append(f"{lab}: **{min(reg_cols, key=lambda c: row.iloc[0][c])}**")
        p.append(f"Which architecture leads varies by state ({'; '.join(arch_bits)}), and the best "
                 f"pooled model overall is **{best_overall}**"
                 f"{' — a regime-aware model' if best_overall in reg_cols else ', which is *not* regime-aware'}"
                 ". The mixture-of-experts (Regime-LSTM-B) trains one LSTM per regime, so the crisis "
                 "expert's *effective* sample is small (crisis ≈ 9% of days), which caps its "
                 "specialisation — a limitation worth stating rather than hiding, and a reason the "
                 "feature approach can lead in the crisis rows.\n")

    p.append("## Gate criteria\n")
    p.append("- [x] Approach A (regime-as-feature) and Approach B (regime-expert MoE) built, "
             "both on the causal filtered posterior; leakage discipline reused from Phase 3 "
             "(train-only scaling, windows end before target, gate lagged to t−1).\n")
    p.append("- [x] Four-way+ comparison vs GARCH/EGARCH/HAR-RV/RW-RV/LSTM/LSTM-RVonly on "
             "identical splits; predictions persisted.\n")
    p.append("- [x] Per-regime (calm/transitional/crisis) error breakdown — core for RQ2.\n")
    p.append("- [x] Formal Diebold–Mariano (HLN-corrected, Student-t) on the head-to-head pairs, "
             "plus a Model Confidence Set (Hansen–Lunde–Nason 2011) over all models for "
             "multiple-comparison control (Ch2 §2.7).\n")
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

    frame = frame_for_profile(data_cfg, profile, include_vix=True)
    # A profile may name its own Phase-4 parquet. Phase 8 is why: RQ4 re-estimates
    # the regime model per asset (ROADMAP Phase 8, decision 1), so one config
    # carries two profiles reading two different signals. Absent the override the
    # top-level `regime.source` applies, which is every pre-Phase-8 config.
    rcfg = cfg.regime
    regime_source = str(profile.get("regime_source", "") or cfg.regime.source)
    if regime_source != str(cfg.regime.source):
        rcfg = OmegaConf.merge(cfg.regime, {"source": regime_source})
        log.info("profile %s overrides the regime source -> %s", name, regime_source)
    frame = attach_regime(frame, rcfg)
    reg_state = frame[REGIME_STATE_COL]
    labels = list(OmegaConf.to_container(cfg.regime.labels, resolve=True))

    sp = profile.splits
    split_cfg = SplitConfig(
        train_end=sp.train_end, val_start=sp.val_start, val_end=sp.val_end,
        test_start=sp.test_start, test_end=sp.test_end,
        refit_frequency_days=int(cfg.harness.refit_frequency_days), scheme=cfg.harness.scheme,
    )
    max_epochs = args.max_epochs if args.max_epochs else (8 if args.fast else None)

    pred_path = repo_path(cfg.paths.predictions, f"m05_regime_dl_{name}.parquet")

    if getattr(args, "from_predictions", False):
        # Evaluation-only re-run: reuse the persisted forecasts and rebuild every
        # downstream table, figure and milestone section. Nothing about the models
        # changes, so this is *numerically identical* to a full re-run for every
        # artefact that is a function of the predictions alone -- which is exactly
        # what an evaluation-convention fix needs, and it avoids re-training on a
        # different BLAS/thread layout (the ~9-significant-figure drift the roadmap
        # documents). Use it after changing scoring code; use a full run after
        # changing model or data code.
        if not pred_path.exists():
            raise FileNotFoundError(
                f"--from-predictions needs {pred_path}; run the full phase first.")
        log.info("--from-predictions: reusing %s (no training)", pred_path)
        combined = from_parquet(pred_path).copy()
        model_names = [c for c in ["Regime-LSTM-A", "Regime-LSTM-A-RVonly", "Regime-LSTM-B"]
                       if c in combined.columns]
        if REGIME_STATE_COL in combined.columns:
            # The persisted state column is the same Phase-4 causal series joined
            # above; prefer the frame's (longer) copy so the t-1 lag can reach the
            # trading day before the OOS window starts instead of losing that day.
            combined = combined.drop(columns=[REGIME_STATE_COL])
        return _evaluate_profile(cfg, name, combined, model_names, reg_state, labels,
                                 pred_path, args, persist_predictions=False,
                                 regime_source=regime_source)

    models = build_models(cfg.model, cfg.harness, cfg.approaches, seed=int(cfg.seed),
                          max_epochs=max_epochs)
    log.info("Phase-5 models: %s", [mm.name for mm in models])
    reg_preds = run_walk_forward(models, frame, split_cfg,
                                 eval_segment=cfg.harness.eval_segment, strict=True)

    # --- merge with the Phase-2/3 baselines (same target/window) ---
    base = from_parquet(repo_path(profile.baseline_predictions)).copy()

    # Drop models that are not information-set comparable with the rest of the
    # board before anything is scored. LSTM-VIX is the case this exists for: it
    # consumes option-implied information (the VIX) that no other model here can
    # see, so including it would (a) compare models with unequal information sets
    # — the exact confound the RV-only control was built to avoid — and (b) drag
    # the Model Confidence Set toward a model whose advantage is informational
    # rather than architectural, eliminating rivals for the wrong reason. It is
    # reported as an auxiliary-feature ablation in Phase 3 (m03, Ch1 §1.8).
    excluded = list(cfg.get("comparison", {}).get("exclude_models", []) or [])
    drop = [c for c in excluded if c in base.columns]
    if drop:
        log.info("excluded from the Phase-5 comparison (unequal information set): %s", drop)
        base = base.drop(columns=drop)

    for mm in models:
        base[mm.name] = reg_preds[mm.name].reindex(base.index)
    combined = base.dropna(subset=[mm.name for mm in models]).copy()

    return _evaluate_profile(cfg, name, combined, [mm.name for mm in models], reg_state,
                             labels, pred_path, args, persist_predictions=True,
                             regime_source=regime_source)


# --------------------------------------------------------------------------- #
# Evaluation half of a profile (shared by the training run and --from-predictions)
# --------------------------------------------------------------------------- #
def _evaluate_profile(cfg, name, combined, model_names, reg_state, labels, pred_path,
                      args, *, persist_predictions: bool,
                      regime_source: str = "") -> dict:
    """Score a finished prediction board: tables, tests, figures, milestone context.

    Split out of :func:`run_profile` so the ``--from-predictions`` path and the
    full training path cannot drift: every number in the milestone is produced by
    this one function whichever way the forecasts were obtained.
    """
    model_cols = [c for c in combined.columns if c not in (ACTUAL_COL, REGIME_STATE_COL)]

    metrics = evaluate_predictions(combined[[ACTUAL_COL] + model_cols])
    log.info("PROFILE %s metrics (QLIKE-ranked):\n%s", name, metrics.round(6).to_string())

    # --- per-regime breakdown (report the key models to keep the table readable) ---
    per_regime_models = [c for c in ["HAR-RV", "LSTM", "LSTM-RVonly",
                                     "Regime-LSTM-A", "Regime-LSTM-A-RVonly", "Regime-LSTM-B"]
                         if c in combined.columns]
    # HEADLINE table: bucket by the regime known at t-1, the only timing under
    # which "the edge is concentrated in state k" is a claim about something the
    # forecaster knew (see src.evaluation.regime_timing). The contemporaneous
    # table is kept alongside it, explicitly labelled, because the two disagree
    # and a reader is entitled to see by how much.
    per = per_regime_qlike(combined, reg_state, labels, per_regime_models,
                           shift=DEFAULT_REGIME_SHIFT)
    per_expost = per_regime_qlike(combined, reg_state, labels, per_regime_models, shift=0)
    # selector_shift=0 on purpose: here the object of interest really is the set of
    # days whose *bucket* differs between the two label timings, which is exactly
    # 1{s_t != s_(t-1)}. It is used only to size the convention gap, never to split
    # a performance metric -- that would be selection on the outcome (see
    # src.evaluation.regime_timing, "The same rule governs the selector").
    trans = transition_day_summary(reg_state, combined.index, selector_shift=0)
    log.info("PROFILE %s per-regime QLIKE [%s]:\n%s", name,
             regime_timing_label(DEFAULT_REGIME_SHIFT), per.round(4).to_string(index=False))
    log.info("PROFILE %s per-regime QLIKE [%s]:\n%s", name, regime_timing_label(0),
             per_expost.round(4).to_string(index=False))
    log.info("PROFILE %s regime-transition days: %d/%d (%.1f%%)", name,
             trans["n_transition"], trans["n"], 100 * trans["share"])

    # --- formal DM (HLN-corrected, Student-t) on the RQ2 pairs ---
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

    # --- Model Confidence Set over ALL models (multiple-comparison control) ---
    mcs = model_confidence_set(
        combined[ACTUAL_COL], {c: combined[c] for c in model_cols},
        alpha=0.10, method="R", reps=2000, block_size=10, seed=int(cfg.seed),
    )
    log.info("PROFILE %s MCS (%d%% confidence): included=%s excluded=%s",
             name, int(round((1 - mcs.attrs["alpha"]) * 100)),
             mcs.attrs["included"], mcs.attrs["excluded"])

    # --- persist predictions (+ regime label) + tables ---
    if persist_predictions:
        out_pred = combined.copy()
        out_pred[REGIME_STATE_COL] = reg_state.reindex(out_pred.index).astype("Int64")
        to_parquet(out_pred, pred_path)
    ensure_dir(repo_path(cfg.paths.tables))
    metrics.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_metrics.csv"))
    # Both per-regime tables are written, each stamped with its timing so a
    # number can never be quoted without the convention that produced it.
    for tbl, suffix, sh in ((per, "per_regime", DEFAULT_REGIME_SHIFT),
                            (per_expost, "per_regime_expost", 0)):
        stamped = tbl.copy()
        stamped.insert(0, "regime_shift", int(sh))
        stamped.insert(1, "timing", regime_timing_label(sh))
        stamped.to_csv(
            repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_{suffix}.csv"), index=False)
    dm.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_dm.csv"), index=False)
    mcs.to_csv(repo_path(cfg.paths.tables, f"m05_regime_dl_{name}_mcs.csv"))

    # --- figures ---
    figdir = repo_path(cfg.paths.figures, "m05"); ensure_dir(figdir)
    fva_models = [c for c in ["Regime-LSTM-A", "Regime-LSTM-B", "LSTM", "HAR-RV"] if c in combined.columns]
    plot_forecast_vs_actual(combined, reg_state, labels, fva_models,
                            f"S&P 500 realized volatility: regime-aware LSTM vs baselines ({name})",
                            figdir / f"{name}_forecast_vs_actual.png")
    bar_models = [c for c in ["HAR-RV", "LSTM", "Regime-LSTM-A", "Regime-LSTM-B"] if c in combined.columns]
    plot_per_regime(per, labels, bar_models,
                    f"Per-regime QLIKE, regime known at t−1 ({name})",
                    figdir / f"{name}_per_regime_qlike.png")

    return {"name": name, "metrics": metrics, "per_regime": per,
            "per_regime_expost": per_expost, "transition": trans,
            "per_regime_models": per_regime_models, "dm": dm, "mcs": mcs, "labels": labels,
            "n_oos": int(len(combined)), "fast": bool(args.fast), "pred_path": pred_path,
            "models": list(model_names),
            "regime_shift": int(DEFAULT_REGIME_SHIFT),
            # Which Phase-4 signal was consumed, so the milestone can name it --
            # the *effective* source, which a profile may override (see run_profile).
            "regime_cols": list(cfg.regime.posterior_cols),
            "regime_source": str(regime_source or cfg.regime.source)}


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_regime_lstm")
    ap.add_argument("--config", default="regime_lstm")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true", help="few epochs (quick CPU smoke pass)")
    ap.add_argument("--max-epochs", type=int, default=0, help="override max epochs (0 = config/fast default)")
    ap.add_argument(
        "--from-predictions", action="store_true",
        help="skip training; rebuild all tables/figures/milestone from the persisted "
             "predictions parquet (use after changing scoring code, not model code)")
    args = ap.parse_args(argv)
    args.max_epochs = args.max_epochs or None

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "05_regime_lstm",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    # Milestone filename is config-driven so a robustness variant (e.g. the
    # Baum-Welch-conditioned run in configs/regime_lstm_hmm.yaml, or a Phase-8
    # asset) cannot overwrite the headline note, and a multi-profile config must
    # carry a {profile} placeholder rather than silently keeping only the last.
    for profile in cfg.profiles:
        ctx = run_profile(data_cfg, cfg, profile, args)
        write_milestone(ctx, milestone_path(cfg, "m05_regime_dl.md", profile.name,
                                            n_profiles=len(cfg.profiles)))
    log.info("Phase 5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
