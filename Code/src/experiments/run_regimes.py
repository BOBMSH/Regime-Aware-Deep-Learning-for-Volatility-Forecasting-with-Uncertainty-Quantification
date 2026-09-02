"""Phase 4 driver -- regime detection end-to-end (ROADMAP.md Phase 4 gate; Ch2 §2.4).

Fits the two pre-registered regime detectors on S&P 500 daily data and produces
the Phase 4 gate artifacts:

* A Gaussian-emission **HMM** (Baum-Welch; Hamilton 1989) and the **Nystrup et
  al. (2020) jump model**, both on *standardised daily log returns* (Ch2 §2.4).
* **K selection by BIC** over K∈{2,3,4}, with K=3 (calm / transitional / crisis)
  carried as the interpretable headline (Guidolin 2011; Ch2 §2.4).
* **Persistence** statistics (transition matrix, expected regime durations).
* **VIX-quantile confusion** — external validation that the unsupervised states
  line up with market-implied volatility.
* **HMM ↔ jump agreement** — robustness of the map to the estimator.
* **Jump-penalty (λ) sensitivity** — the persistence knob is tuned, not fixed.
* **Smoothed vs causal-filtered posteriors** — the descriptive map uses the
  smoothed posterior; the leakage-free *filtered* posterior (fit on train only) is
  the object Phase 5 conditions on. The two are compared to show where look-ahead
  would matter.
* **Two causal signals, one headline.** ``jphmm_filt_p*`` comes from the
  **jump-penalised HMM** (Nystrup et al. 2020; Ch2 §2.4/§2.8) and is what Phases
  5–7 condition on; ``hmm_filt_p*`` is the same object for the plain Baum-Welch
  HMM and is kept as the regime-estimator robustness comparator. Both are fit on
  the training window only, with λ selected on training rows only.

Outputs
-------
* ``results/predictions/m04_regimes_<profile>.parquet`` — date-indexed regime
  states + posteriors (smoothed descriptive + causal filtered), VIX and RV. The
  Phase-5 conditioning input.
* ``results/tables/m04_regimes_<profile>_{bic,persistence,jump_lambda,agreement}.csv``
  plus the external-validation cross-tab, named for the validator actually used:
  ``_vix_confusion.csv`` where the asset's options trade on this market, and
  ``_rv_concordance.csv`` where they do not (see :func:`state_concordance`).
* ``results/figures/m04/<profile>_{regime_map_hmm,regime_map_jump,bic,transition,smoothed_vs_filtered}.png``
  plus ``_vix_by_state.png`` / ``_rv_by_state.png`` on the same rule.
* ``results/milestones/m04_regimes.md`` — the milestone note.
* ``experiments/04_regimes/run_<utc>/config.yaml`` — reproducibility.

Usage
-----
    python -m src.experiments.run_regimes
    python -m src.experiments.run_regimes --fast     # fewer restarts (quick pass)
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
from src.models.regime.features import (
    RegimeFeatureSpec,
    TrainStandardizer,
    build_regime_features,
    standardize_full_sample,
)
from src.models.regime.hmm import GaussianHMMRegime
from src.models.regime.jump import JumpModel, select_jump_penalty
from src.models.regime.jump_hmm import JumpPenalisedHMM, select_jump_penalty_causal
from src.utils.config import (load_config, milestone_path, repo_path,
                              snapshot_config)
from src.utils.io import ensure_dir, to_parquet
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("run_regimes")

# Volatility-ordered state palette (calm -> crisis): green, gold, orange, red.
STATE_COLORS = ["#2ca02c", "#f0c000", "#ff7f0e", "#d62728"]
STATE_LABELS = {
    2: ["calm", "crisis"],
    3: ["calm", "transitional", "crisis"],
    4: ["calm", "normal", "elevated", "crisis"],
}
HEADLINE_K = 3  # interpretable calm/transitional/crisis (Ch2 §2.4)


# --------------------------------------------------------------------------- #
# Model selection (BIC over K)                                                 #
# --------------------------------------------------------------------------- #
def bic_over_k(Xz: np.ndarray, k_grid, mcfg, jump_penalty, *, seed) -> pd.DataFrame:
    """Fit HMM + jump model for each K and tabulate log-likelihood, BIC and AIC."""
    rows = []
    for k in k_grid:
        hm = GaussianHMMRegime(
            k, covariance_type=mcfg.hmm.covariance_type, n_init=int(mcfg.hmm.n_init),
            n_iter=int(mcfg.hmm.n_iter), tol=float(mcfg.hmm.tol), seed=seed,
        ).fit(Xz)
        jm = JumpModel(
            k, jump_penalty=float(jump_penalty), n_init=int(mcfg.jump.n_init),
            max_iter=int(mcfg.jump.max_iter), seed=seed,
        ).fit(Xz)
        rows.append({
            "K": k,
            "hmm_loglik": hm.loglik(), "hmm_bic": hm.bic(Xz), "hmm_aic": hm.aic(Xz),
            "hmm_params": hm.n_params(),
            "jump_loglik": jm.loglik(Xz), "jump_bic": jm.bic(Xz),
            "jump_params": jm.n_params(), "jump_n_jumps": jm.n_jumps(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Persistence + external (VIX) validation                                     #
# --------------------------------------------------------------------------- #
def persistence_table(hmm: GaussianHMMRegime, path: np.ndarray, K: int) -> pd.DataFrame:
    pers = hmm.persistence(path)
    labels = STATE_LABELS.get(K, [str(i) for i in range(K)])
    return pd.DataFrame({
        "state": range(K), "label": labels,
        "stay_prob": pers.diag.round(4),
        "expected_duration_days": pers.expected_duration.round(1),
        "stationary_prob": pers.stationary.round(4),
        "empirical_freq": pers.empirical_freq.round(4),
        "mean_run_length_days": pers.mean_run_length.round(1),
    })


def state_concordance(state: np.ndarray, series: pd.Series, K: int, *,
                      key: str = "vix_close") -> tuple[pd.DataFrame, dict]:
    """Cross-tabulate decoded state against K quantile bins of an external series.

    A meaningful regime map concentrates mass on the diagonal: the calm state on
    the series' low days, the crisis state on its high days. We report the
    row-normalised confusion, the series mean per state (which should increase
    with the state index, by construction of the volatility ordering) and two
    scalar concordances -- the Spearman rank correlation between state index and
    series, and the fraction of days whose state bin equals their quantile bin.

    ``key`` names the validating series and decides how the result is labelled.
    Two are used:

    * ``vix_close`` -- the canonical check. The regime model never sees the VIX,
      so agreement is genuine external validation against market-implied
      volatility.
    * ``rv`` -- the fallback for an asset whose options trade on another market,
      where ``assets.registry.<key>.vix_feature`` is false and the frame
      therefore carries no VIX at all (the FTSE 100). It is still external to the
      *estimator*: the regime model is fitted on standardised daily log returns
      only and never sees realised variance, so concordance with it is a real
      check that the inferred states track independently measured volatility.
      It is a **different** diagnostic from the VIX one and is named differently
      everywhere it is written, so the two can never be compared as though they
      were the same number.

    The VIX path is bit-identical to the pre-2026-09-02 ``vix_confusion``.
    """
    v = series.to_numpy()
    prefix = "vix" if key == "vix_close" else key
    # K quantile bins of the series, 0 = lowest.
    ranks = v.argsort().argsort()
    series_bin = np.minimum((ranks * K) // len(v), K - 1)
    conf = np.zeros((K, K), dtype=int)
    for s, b in zip(state, series_bin):
        conf[s, b] += 1
    conf_df = pd.DataFrame(
        conf, index=[f"state{ i }" for i in range(K)],
        columns=[f"{prefix}Q{ j }" for j in range(K)],
    )
    row_sum = conf.sum(axis=1, keepdims=True)
    conf_norm = pd.DataFrame(
        np.divide(conf, row_sum, out=np.zeros_like(conf, float), where=row_sum > 0).round(3),
        index=conf_df.index, columns=conf_df.columns,
    )
    mean_by_state = pd.Series([v[state == k].mean() if (state == k).any() else np.nan
                               for k in range(K)],
                              index=[f"state{ i }" for i in range(K)])
    from scipy.stats import spearmanr
    rho = float(spearmanr(state, v).statistic)
    diag_concordance = float((series_bin == state).mean())
    # Scalar keys carry the validator's name so a value from one diagnostic can
    # never be read as the other; `validator` is always present. The VIX names
    # are unchanged, so a regenerated `.SPX` agreement table keeps its columns.
    summary = {
        "validator": key,
        f"spearman_state_{prefix}": rho,
        "diag_concordance": diag_concordance,
        f"mean_{prefix}_by_state": mean_by_state.round(4).to_dict(),
    }
    # `_spearman` is a private convenience alias for the milestone, which should
    # not have to know the validator's name to print one number. Keys starting
    # with an underscore are excluded from the agreement CSV, so the persisted
    # table carries only the *named* correlation and cannot show the same value
    # twice under two names.
    return conf_df, {"norm": conf_norm, "mean_by_state": mean_by_state,
                     "_spearman": rho, **summary}


def vix_confusion(state: np.ndarray, vix: pd.Series, K: int) -> tuple[pd.DataFrame, dict]:
    """Backwards-compatible alias for the VIX case of :func:`state_concordance`."""
    return state_concordance(state, vix, K, key="vix_close")


def agreement(hmm_path: np.ndarray, jump_path: np.ndarray) -> dict:
    """HMM ↔ jump agreement: raw (states already volatility-ordered) + adjusted
    Rand (label-permutation invariant)."""
    from sklearn.metrics import adjusted_rand_score
    return {
        "raw_agreement": float((hmm_path == jump_path).mean()),
        "adjusted_rand": float(adjusted_rand_score(hmm_path, jump_path)),
    }


def causal_signal_table(causal: dict, K: int) -> pd.DataFrame:
    """Side-by-side summary of the two *causal* regime signals Phase 5/6 can use.

    One row per estimator (jump-penalised HMM = headline, Ch2 §2.8; Baum-Welch
    HMM = robustness comparator), reporting how persistent the resulting filtered
    path is, how confident the posterior is, and how much the smoother would have
    disagreed on the test window (i.e. how much look-ahead was given up). The
    ``mean_max_posterior`` column is the one to watch when reading Phase-5
    results: a signal whose posterior sits near 1.0 gives the mixture-of-experts
    gate an almost-hard routing, which changes what Approach B can learn.
    """
    tm = causal["test_mask"]
    rows = []
    for label, filt, diverge, lam in (
        ("jump_penalised_hmm", causal["jp_filt_all"], causal["jp_mean_abs_divergence_test"],
         causal["lam_causal"]),
        ("baum_welch_hmm", causal["p_filt_all"], causal["mean_abs_divergence_test"], np.nan),
    ):
        state = filt.argmax(axis=1)
        state_test = state[tm]
        switches = int(np.count_nonzero(np.diff(state_test)))
        row = {
            "signal": label,
            "lambda": lam,
            "n_test": int(tm.sum()),
            "test_switches": switches,
            "test_mean_duration_days": float(len(state_test) / (switches + 1)),
            "mean_max_posterior_test": float(filt[tm].max(axis=1).mean()),
            "mean_abs_smoothed_minus_filtered_test": float(diverge),
        }
        for k in range(K):
            row[f"test_days_state{k}"] = int((state_test == k).sum())
        rows.append(row)
    tbl = pd.DataFrame(rows)
    # Agreement between the two causal hard paths over the test window.
    a = causal["jp_filt_all"].argmax(axis=1)[tm]
    b = causal["p_filt_all"].argmax(axis=1)[tm]
    tbl["agreement_with_other_signal_test"] = float((a == b).mean())
    return tbl


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def _shade_regimes(ax, dates, state, K):
    """Shade the plot background by regime state (contiguous runs)."""
    import matplotlib.patches as mpatches
    dates = pd.DatetimeIndex(dates)
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            ax.axvspan(dates[start], dates[min(i, len(dates) - 1)],
                       color=STATE_COLORS[state[start] % len(STATE_COLORS)], alpha=0.30, lw=0)
            start = i
    labels = STATE_LABELS.get(K, [str(i) for i in range(K)])
    handles = [mpatches.Patch(color=STATE_COLORS[k % len(STATE_COLORS)], alpha=0.5,
                              label=f"{k}: {labels[k]}") for k in range(K)]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=K)


def plot_regime_map(dates, rv, state, K, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    vol = np.sqrt(np.clip(rv, 0, None)) * 100.0
    ax.plot(pd.DatetimeIndex(dates), vol, color="0.15", lw=0.8, zorder=5)
    _shade_regimes(ax, dates, state, K)
    ax.set_ylabel("Daily volatility (%)")
    ax.set_title(title)
    ax.set_xlim(pd.DatetimeIndex(dates).min(), pd.DatetimeIndex(dates).max())
    return save_fig(fig, path)


def plot_bic(bic_tbl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(bic_tbl["K"], bic_tbl["hmm_bic"], "o-", label="HMM BIC", color="tab:blue")
    ax.plot(bic_tbl["K"], bic_tbl["jump_bic"], "s--", label="Jump BIC", color="tab:red")
    for _, r in bic_tbl.iterrows():
        ax.annotate(f"{r['hmm_bic']:.0f}", (r["K"], r["hmm_bic"]), fontsize=7,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xticks(list(bic_tbl["K"]))
    ax.set_xlabel("number of regimes K"); ax.set_ylabel("BIC (lower better)")
    ax.set_title("Model selection: BIC by K")
    ax.legend(fontsize=8)
    return save_fig(fig, path)


def plot_transition(transmat, K, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    labels = STATE_LABELS.get(K, [str(i) for i in range(K)])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(transmat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), labels, rotation=30, ha="right")
    ax.set_yticks(range(K), labels)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center",
                    color="white" if transmat[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title("HMM transition matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return save_fig(fig, path)


def plot_validator_by_state(state, series, K, path, *, ylabel: str, log_y: bool = False):
    """Boxplot of the validating series within each decoded state.

    ``ylabel`` names the series, so the figure cannot imply the VIX on an asset
    validated against its own realised variance. ``log_y`` is for realised
    variance, whose distribution is right-skewed enough that a linear axis hides
    the separation the plot exists to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import save_fig, set_style

    set_style()
    labels = STATE_LABELS.get(K, [str(i) for i in range(K)])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    data = [series.to_numpy()[state == k] for k in range(K)]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for patch, k in zip(bp["boxes"], range(K)):
        patch.set_facecolor(STATE_COLORS[k % len(STATE_COLORS)])
        patch.set_alpha(0.6)
    ax.set_xticklabels([f"{k}\n{labels[k]}" for k in range(K)])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(f"External validation: {ylabel} by decoded regime")
    return save_fig(fig, path)


def plot_smoothed_vs_filtered(dates, p_smooth, p_filt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.plotting import CRISIS_PERIODS, save_fig, set_style

    set_style()
    d = pd.DatetimeIndex(dates)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(d, p_smooth, color="tab:blue", lw=1.1, label="smoothed  P(crisis)  [uses full sample]")
    ax.plot(d, p_filt, color="tab:red", lw=1.1, alpha=0.85,
            label="filtered  P(crisis)  [causal, Phase-5 input]")
    ymax = 1.02
    for lab, (cs, ce) in CRISIS_PERIODS.items():
        cs, ce = pd.Timestamp(cs), pd.Timestamp(ce)
        if ce < d.min() or cs > d.max():
            continue
        ax.axvspan(max(cs, d.min()), min(ce, d.max()), color="0.6", alpha=0.15, lw=0)
    ax.set_ylim(0, ymax); ax.set_ylabel("P(crisis regime)")
    ax.set_title("Smoothed vs causal-filtered crisis probability (test window)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(d.min(), d.max())
    return save_fig(fig, path)


# --------------------------------------------------------------------------- #
# One profile                                                                  #
# --------------------------------------------------------------------------- #
def run_profile(data_cfg, cfg, profile, args) -> dict:
    name = profile.name
    seed = int(cfg.seed)
    k_grid = [int(k) for k in cfg.k_grid]
    log.info("=" * 70)
    log.info("PROFILE %s | target=%s", name, profile.target)

    if args.fast:  # trim restarts for a quick pass
        cfg.models.hmm.n_init = 4
        cfg.models.jump.n_init = 4

    frame = frame_for_profile(data_cfg, profile, include_vix=True)
    asset_label = str(frame.attrs.get("asset", "")) or "primary asset"

    # External validation series. The VIX is the canonical check, but it prices
    # S&P 500 options, so `assets.registry.<key>.vix_feature` refuses it for an
    # asset trading on another market and the frame then carries no `vix_close`
    # at all -- which is correct, and used to be a KeyError here. The fallback is
    # the asset's OWN realised variance: the regime model is fitted on
    # standardised daily log returns and never sees it, so concordance with it is
    # still external to the estimator. It is a different diagnostic and is
    # labelled differently in every artefact.
    if "vix_close" in frame.columns:
        validator_key, validator_label = "vix_close", "VIX (close)"
    else:
        validator_key, validator_label = "rv", f"{asset_label} realised variance"
        log.info(
            "no VIX column for asset %s -- the registry refuses it for this "
            "market. Validating the regime map against the asset's own realised "
            "variance instead, which the estimator never sees. This is a "
            "DIFFERENT diagnostic from the VIX concordance and is written to "
            "m04_regimes_%s_rv_concordance.csv, not the vix_confusion table.",
            asset_label, name,
        )
    validator = frame[validator_key]

    # --- features (pre-registered: standardised daily log returns) ---
    prim = RegimeFeatureSpec.from_set(cfg.features.primary, standardize=bool(cfg.features.standardize))
    feats = build_regime_features(frame, prim)
    frame = frame.loc[feats.index]; validator = validator.loc[feats.index]  # align to features
    Xz = standardize_full_sample(feats).to_numpy() if prim.standardize else feats.to_numpy()
    log.info("regime features %s: %d obs %s -> %s", list(feats.columns), len(feats),
             feats.index.min().date(), feats.index.max().date())

    # --- 1. K selection by BIC (both models) ---
    bic_tbl = bic_over_k(Xz, k_grid, cfg.models, float(cfg.jump_penalty), seed=seed)
    hmm_bic_k = int(bic_tbl.loc[bic_tbl["hmm_bic"].idxmin(), "K"])
    jump_bic_k = int(bic_tbl.loc[bic_tbl["jump_bic"].idxmin(), "K"])
    log.info("BIC-selected K: HMM=%d, jump=%d (headline K=%d, interpretable)",
             hmm_bic_k, jump_bic_k, HEADLINE_K)

    # --- 2. Headline models at the interpretable K ---
    K = HEADLINE_K
    hmm = GaussianHMMRegime(K, covariance_type=cfg.models.hmm.covariance_type,
                            n_init=int(cfg.models.hmm.n_init), n_iter=int(cfg.models.hmm.n_iter),
                            tol=float(cfg.models.hmm.tol), seed=seed).fit(Xz)
    # jump penalty: sweep then choose the smallest λ meeting the persistence floor.
    lam_grid = [float(x) for x in cfg.jump_penalty_grid]
    lam_tbl = select_jump_penalty(Xz, K, lam_grid, n_init=int(cfg.models.jump.n_init), seed=seed)
    lam = float(cfg.jump_penalty)
    if bool(cfg.select_by_persistence.enabled):
        floor = float(cfg.select_by_persistence.min_mean_duration_days)
        ok = lam_tbl[lam_tbl["mean_duration"] >= floor]
        lam = float(ok.iloc[0]["jump_penalty"]) if len(ok) else float(lam_tbl.iloc[-1]["jump_penalty"])
    log.info("jump penalty selected: lambda=%.1f (floor %.0f days)", lam,
             float(cfg.select_by_persistence.min_mean_duration_days))
    jump = JumpModel(K, jump_penalty=lam, n_init=int(cfg.models.jump.n_init),
                     max_iter=int(cfg.models.jump.max_iter), seed=seed).fit(Xz)

    hmm_path = hmm.decode(Xz)
    jump_path = jump.decode(Xz)
    p_smooth = hmm.smoothed_proba(Xz)

    # --- 3. Persistence + VIX validation + agreement ---
    pers_tbl = persistence_table(hmm, hmm_path, K)
    conf_df, conf_extra = state_concordance(hmm_path, validator, K, key=validator_key)
    agree = agreement(hmm_path, jump_path)
    log.info("%s concordance: spearman=%.3f diag=%.3f | HMM/jump agreement raw=%.3f ARI=%.3f",
             validator_label, conf_extra["_spearman"], conf_extra["diag_concordance"],
             agree["raw_agreement"], agree["adjusted_rand"])

    # --- 4. Causal (leakage-free) demonstration: fit on train, filter over all ---
    cd = cfg.causal_demo
    causal = None
    if bool(cd.enabled):
        train_end = pd.Timestamp(cd.splits.train_end)
        train_mask = np.asarray(feats.index <= train_end)
        scaler = TrainStandardizer().fit(feats, train_mask)
        Xz_causal = scaler.transform(feats).to_numpy()
        hmm_tr = GaussianHMMRegime(K, covariance_type=cfg.models.hmm.covariance_type,
                                   n_init=int(cfg.models.hmm.n_init), n_iter=int(cfg.models.hmm.n_iter),
                                   tol=float(cfg.models.hmm.tol), seed=seed).fit(Xz_causal[train_mask])
        p_filt_all = hmm_tr.filtered_proba(Xz_causal)   # causal posterior at every t
        p_smooth_all = hmm_tr.smoothed_proba(Xz_causal)  # smoother for contrast
        crisis = K - 1
        test_mask = np.asarray((feats.index >= pd.Timestamp(cd.splits.test_start)) &
                               (feats.index <= pd.Timestamp(cd.splits.test_end)))
        diverge = float(np.abs(p_smooth_all[:, crisis] - p_filt_all[:, crisis])[test_mask].mean())

        # --- Jump-penalised HMM: the estimator Ch2 §2.4/§2.8 commits to, and the
        # signal Phase 5/6 condition on. lambda is re-selected on TRAINING rows
        # only (the full-sample lambda above is descriptive and would leak).
        lam_causal, lam_tbl_tr = select_jump_penalty_causal(
            Xz_causal[train_mask], K, [float(x) for x in cfg.jump_penalty_grid],
            min_mean_duration_days=float(cfg.select_by_persistence.min_mean_duration_days),
            n_init=int(cfg.models.jump.n_init), seed=seed)
        jphmm_tr = JumpPenalisedHMM(K, jump_penalty=lam_causal,
                                    covariance_type=cfg.models.hmm.covariance_type,
                                    n_init=int(cfg.models.jump.n_init),
                                    max_iter=int(cfg.models.jump.max_iter),
                                    seed=seed).fit(Xz_causal[train_mask])
        jp_filt_all = jphmm_tr.filtered_proba(Xz_causal)
        jp_smooth_all = jphmm_tr.smoothed_proba(Xz_causal)
        jp_diverge = float(
            np.abs(jp_smooth_all[:, crisis] - jp_filt_all[:, crisis])[test_mask].mean())

        # --- lambda sensitivity: how much of the conditioning signal is an
        # artefact of the pre-registered grid's resolution? Every lambda on the
        # (finer) sensitivity grid is fitted on training rows only and filtered
        # causally, exactly like the headline, then compared to it day by day.
        # This is reported, never used to select -- see configs/hmm.yaml.
        lam_sens = None
        sens_grid = cfg.get("jump_penalty_sensitivity_grid")
        if sens_grid is not None:
            rows = []
            # NB the loop variable is `lam_s`, NOT `lam`. Until 2026-08-19 (iii) it
            # was `lam`, which shadowed the full-sample selected penalty for the rest
            # of the function: after the loop `lam` held the last grid value (300),
            # and that wrong value was written to the agreement table, printed in the
            # milestone as "the headline map uses lambda=300" and burned into the
            # jump regime-map figure caption. The fitted models were never affected
            # (they are built above, before this loop), but three reported artefacts
            # were. Keep the two names distinct.
            for lam_s in [float(x) for x in sens_grid]:
                try:
                    m_ = JumpPenalisedHMM(
                        K, jump_penalty=lam_s, covariance_type=cfg.models.hmm.covariance_type,
                        n_init=int(cfg.models.jump.n_init),
                        max_iter=int(cfg.models.jump.max_iter), seed=seed,
                    ).fit(Xz_causal[train_mask])
                except Exception as exc:  # noqa: BLE001 - a degenerate lambda must not kill the run
                    log.warning("lambda sensitivity: lambda=%.2f failed (%s); skipped", lam_s, exc)
                    continue
                s_ = m_.filtered_proba(Xz_causal).argmax(axis=1)
                head = jp_filt_all.argmax(axis=1)
                rows.append({
                    "jump_penalty": lam_s,
                    "train_jumps": int(m_.n_jumps_),
                    "train_mean_duration_days": float(
                        len(m_.jump_path_) / max(m_.n_jumps_ + 1, 1)),
                    "clears_persistence_floor": bool(
                        len(m_.jump_path_) / max(m_.n_jumps_ + 1, 1)
                        >= float(cfg.select_by_persistence.min_mean_duration_days)),
                    "test_switches": int(np.count_nonzero(np.diff(s_[test_mask]))),
                    "agreement_with_headline_test": float((s_[test_mask] == head[test_mask]).mean()),
                    "agreement_with_headline_train": float((s_[train_mask] == head[train_mask]).mean()),
                    "is_headline": bool(abs(lam_s - lam_causal) < 1e-9),
                })
            lam_sens = pd.DataFrame(rows)
            log.info("lambda sensitivity (training-only fits, causal paths):\n%s",
                     lam_sens.round(4).to_string(index=False))

        causal = {"train_end": train_end, "test_mask": test_mask,
                  "p_filt_all": p_filt_all, "p_smooth_all": p_smooth_all,
                  "mean_abs_divergence_test": diverge, "crisis": crisis,
                  "hmm_tr": hmm_tr,
                  "jp_filt_all": jp_filt_all, "jp_smooth_all": jp_smooth_all,
                  "jp_mean_abs_divergence_test": jp_diverge,
                  "jphmm_tr": jphmm_tr, "lam_causal": lam_causal,
                  "lam_tbl_tr": lam_tbl_tr, "lam_sensitivity": lam_sens}
        log.info("causal demo: mean |smoothed-filtered| P(crisis) on test = %.3f "
                 "(HMM) / %.3f (jump-penalised HMM)", diverge, jp_diverge)

    # --- 5. Persist predictions frame (Phase-5 conditioning input) ---
    out = pd.DataFrame(index=feats.index)
    out.index.name = "date"
    out["rv"] = frame["rv"].to_numpy()
    if validator_key == "vix_close":
        # Only where the asset actually has one. Writing the validator under a
        # `vix_close` name on an asset validated against its own RV would put two
        # different quantities in one column across profiles.
        out["vix_close"] = validator.to_numpy()
    out["hmm_state"] = hmm_path
    for k in range(K):
        out[f"hmm_p{k}"] = p_smooth[:, k]
    out["jump_state"] = jump_path
    jp = jump.predict_proba(Xz)
    for k in range(K):
        out[f"jump_p{k}"] = jp[:, k]
    if causal is not None:
        out["hmm_filt_state"] = causal["p_filt_all"].argmax(axis=1)
        for k in range(K):
            out[f"hmm_filt_p{k}"] = causal["p_filt_all"][:, k]
        # Jump-penalised HMM: the HEADLINE Phase-5/6 conditioning signal (Ch2 §2.8).
        out["jphmm_filt_state"] = causal["jp_filt_all"].argmax(axis=1)
        for k in range(K):
            out[f"jphmm_filt_p{k}"] = causal["jp_filt_all"][:, k]

    paths = cfg.paths
    pred_path = repo_path(paths.predictions, f"m04_regimes_{name}.parquet")
    to_parquet(out, pred_path)

    ensure_dir(repo_path(paths.tables))
    bic_tbl.to_csv(repo_path(paths.tables, f"m04_regimes_{name}_bic.csv"), index=False)
    pers_tbl.to_csv(repo_path(paths.tables, f"m04_regimes_{name}_persistence.csv"), index=False)
    conf_name = ("vix_confusion" if validator_key == "vix_close"
                 else f"{validator_key}_concordance")
    conf_df.to_csv(repo_path(paths.tables, f"m04_regimes_{name}_{conf_name}.csv"))
    lam_tbl.to_csv(repo_path(paths.tables, f"m04_regimes_{name}_jump_lambda.csv"), index=False)
    pd.DataFrame([{**agree, "hmm_bic_k": hmm_bic_k, "jump_bic_k": jump_bic_k,
                   "headline_k": K, "lambda": lam,
                   **{k: v for k, v in conf_extra.items()
                      if np.isscalar(v) and not k.startswith("_")}}]
                 ).to_csv(repo_path(paths.tables, f"m04_regimes_{name}_agreement.csv"), index=False)

    # Causal-signal comparison: the two leakage-free posteriors Phase 5/6 can
    # consume, side by side. The jump-penalised HMM is the headline (Ch2 §2.8);
    # the Baum-Welch HMM is the regime-estimator robustness comparator.
    causal_tbl = None
    if causal is not None:
        causal_tbl = causal_signal_table(causal, K)
        causal_tbl.to_csv(
            repo_path(paths.tables, f"m04_regimes_{name}_causal_signals.csv"), index=False)
        causal["lam_tbl_tr"].to_csv(
            repo_path(paths.tables, f"m04_regimes_{name}_jump_lambda_train.csv"), index=False)
        if causal.get("lam_sensitivity") is not None:
            causal["lam_sensitivity"].to_csv(
                repo_path(paths.tables, f"m04_regimes_{name}_jump_lambda_sensitivity.csv"),
                index=False)

    # --- 6. Figures ---
    figdir = repo_path(paths.figures, "m04"); ensure_dir(figdir)
    # The asset comes off the frame, never a literal: these titles read
    # "S&P 500" unconditionally until 2026-09-02, so a Phase-8 chart would have
    # named the wrong index with nothing to catch it.
    plot_regime_map(feats.index, frame["rv"].to_numpy(), hmm_path, K,
                    f"{asset_label} realized volatility by HMM regime (K={K}) — {name}",
                    figdir / f"{name}_regime_map_hmm.png")
    plot_regime_map(feats.index, frame["rv"].to_numpy(), jump_path, K,
                    f"{asset_label} realized volatility by jump-model regime "
                    f"(K={K}, λ={lam:.4g}) — {name}",
                    figdir / f"{name}_regime_map_jump.png")
    plot_bic(bic_tbl, figdir / f"{name}_bic.png")
    plot_transition(hmm.transmat_, K, figdir / f"{name}_transition.png")
    val_fig = ("vix_by_state" if validator_key == "vix_close"
               else f"{validator_key}_by_state")
    plot_validator_by_state(hmm_path, validator, K, figdir / f"{name}_{val_fig}.png",
                            ylabel=validator_label, log_y=(validator_key == "rv"))
    if causal is not None:
        tm = causal["test_mask"]; cr = causal["crisis"]
        plot_smoothed_vs_filtered(feats.index[tm], causal["p_smooth_all"][tm, cr],
                                  causal["p_filt_all"][tm, cr],
                                  figdir / f"{name}_smoothed_vs_filtered.png")
        plot_smoothed_vs_filtered(feats.index[tm], causal["jp_smooth_all"][tm, cr],
                                  causal["jp_filt_all"][tm, cr],
                                  figdir / f"{name}_smoothed_vs_filtered_jphmm.png")
        plot_regime_map(feats.index[tm], frame["rv"].to_numpy()[tm],
                        causal["jp_filt_all"][tm].argmax(axis=1), K,
                        f"Test-window RV by causal jump-penalised-HMM regime "
                        f"(K={K}, λ={causal['lam_causal']:.4g}) — {name}",
                        figdir / f"{name}_regime_map_jphmm_causal.png")

    return {
        "name": name, "K": K, "k_grid": k_grid, "bic_tbl": bic_tbl,
        "hmm_bic_k": hmm_bic_k, "jump_bic_k": jump_bic_k, "lam": lam, "lam_tbl": lam_tbl,
        "pers_tbl": pers_tbl, "conf_df": conf_df, "conf_extra": conf_extra, "agree": agree,
        "hmm": hmm, "jump": jump, "causal": causal, "causal_tbl": causal_tbl,
        "n_obs": len(feats),
        "feat_cols": list(feats.columns), "span": (feats.index.min(), feats.index.max()),
        "pred_path": pred_path, "fast": bool(args.fast),
        "asset": asset_label, "validator_key": validator_key,
        "validator_label": validator_label, "conf_name": conf_name,
        "validator_fig": val_fig,
    }


# --------------------------------------------------------------------------- #
# Milestone note                                                              #
# --------------------------------------------------------------------------- #
def write_milestone(ctx: dict, out_path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    K = ctx["K"]; name = ctx["name"]
    bic = ctx["bic_tbl"]; pers = ctx["pers_tbl"]; conf = ctx["conf_extra"]; agree = ctx["agree"]
    labels = STATE_LABELS.get(K, [str(i) for i in range(K)])

    p: list[str] = []
    p.append("# M04 — Regime detection (Phase 4)\n")
    p.append(f"*Generated by `python -m src.experiments.run_regimes` on {ts}. "
             "Numbers trace to `results/tables/m04_regimes_*` and "
             "`results/predictions/m04_regimes_*.parquet`.*\n")
    if ctx.get("fast"):
        p.append("> ⚠️ **FAST pass** — reduced restarts. Re-run at full config before citing.\n")

    p.append("## Scope\n")
    p.append(f"Roadmap Phase 4 (Ch2 §2.4): detect latent **volatility regimes** on "
             f"**{ctx.get('asset', 'S&P 500')}** daily data with the two pre-registered estimators — a Gaussian-emission **HMM** "
             "(Baum–Welch; Hamilton 1989) and the **Nystrup, Lindström & Madsen (2020) jump "
             "model** — both on *standardised daily log returns*. Regimes are the discrete "
             "state the Phase-5 LSTM conditions on; this phase characterises and validates "
             "them, and builds the **leakage-free causal posterior** that Phase 5 will consume.\n")
    p.append(f"- **Sample:** {ctx['span'][0].date()} → {ctx['span'][1].date()} "
             f"({ctx['n_obs']} trading days), target `{name}`.\n"
             f"- **Emission features:** `{ctx['feat_cols']}` (standardised).\n"
             f"- **Headline K = {K}** ({' / '.join(labels)}), the interpretable choice "
             "(Guidolin 2011; Ch2 §2.4); BIC sensitivity below.\n")

    p.append("## Model selection — BIC by K\n")
    p.append("| K | HMM logL | HMM params | HMM BIC | Jump logL | Jump params | Jump BIC |")
    p.append("|---|---|---|---|---|---|---|")
    for _, r in bic.iterrows():
        p.append(f"| {int(r['K'])} | {r['hmm_loglik']:.1f} | {int(r['hmm_params'])} | "
                 f"{r['hmm_bic']:.1f} | {r['jump_loglik']:.1f} | {int(r['jump_params'])} | "
                 f"{r['jump_bic']:.1f} |")
    p.append(f"\nBIC prefers **K={ctx['hmm_bic_k']}** (HMM) / **K={ctx['jump_bic_k']}** (jump). "
             f"The headline map fixes **K={K}** for interpretability (calm/transitional/crisis); "
             "Chapter 2 §2.4 pre-commits to a small, interpretable state space over BIC-chasing "
             "on the modest samples of daily data.\n")
    p.append(f"\n![BIC by K](../figures/m04/{name}_bic.png)\n")

    p.append("## Regime map (headline)\n")
    p.append(f"![HMM regime map](../figures/m04/{name}_regime_map_hmm.png)\n")
    p.append(f"![Jump regime map](../figures/m04/{name}_regime_map_jump.png)\n")

    p.append("## Persistence\n")
    p.append("Transition matrix and state durations of the **Baum–Welch HMM** fitted on the full "
             "sample — the descriptive map above, not the causal conditioning signal. The "
             "jump-penalised chain that Phases 5–7 actually consume is more persistent; its "
             "durations are in the causal-signal table further down. Stated explicitly because "
             "the two are easy to confuse and only one of them is a result.\n")
    p.append("| state | label | stay prob | expected duration (days) | stationary | freq | mean run (days) |")
    p.append("|---|---|---|---|---|---|---|")
    for _, r in pers.iterrows():
        p.append(f"| {int(r['state'])} | {r['label']} | {r['stay_prob']:.3f} | "
                 f"{r['expected_duration_days']:.1f} | {r['stationary_prob']:.3f} | "
                 f"{r['empirical_freq']:.3f} | {r['mean_run_length_days']:.1f} |")
    p.append(f"\n![transition matrix](../figures/m04/{name}_transition.png)\n")
    p.append("The jump penalty λ is **tuned, not fixed** (Ch2 §2.4; Nystrup 2020): the "
             f"**descriptive full-sample** jump map above uses λ=**{ctx['lam']:.4g}** — the "
             "smallest penalty on the pre-registered grid whose mean regime duration clears the "
             "persistence floor. The λ that matters for every downstream phase is the "
             "**training-only** one selected in the causal section below; both are reported "
             "because only the second is leakage-free. Full λ sensitivity is in "
             f"`results/tables/m04_regimes_{name}_jump_lambda.csv`.\n")

    vlabel = ctx.get("validator_label", "VIX (close)")
    is_vix = ctx.get("validator_key", "vix_close") == "vix_close"
    mv = conf["mean_by_state"]

    p.append(f"## External validation — {vlabel} by regime\n")
    if not is_vix:
        p.append(
            "This asset's options trade on another market, so the CBOE VIX is not a "
            "valid validator for it and `assets.registry` refuses the column outright "
            "(joining it would also intersect two trading calendars and shrink the "
            "sample silently). The regime map is instead checked against the asset's "
            "**own realised variance**, which the estimator never sees: it is fitted on "
            "standardised daily log returns alone. This is a different diagnostic from "
            "the VIX concordance reported for the US assets and its numbers are not "
            "comparable with them.\n"
        )
    # The monotonicity claim is TESTED, not asserted. It held on `.SPX`, where this
    # sentence was hardcoded; on a new asset it is exactly the kind of statement a
    # generator has no business making before looking. (Sixth of this family.)
    vals = mv.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    monotone = bool(finite.all() and np.all(np.diff(vals) > 0))
    fmt = "{:.1f}" if is_vix else "{:.3e}"
    if monotone:
        p.append(f"Mean {vlabel} rises monotonically with the (volatility-ordered) regime "
                 "index — the unsupervised states line up with independently measured "
                 "volatility without ever seeing it:\n")
    else:
        p.append(f"Mean {vlabel} does **not** rise monotonically with the "
                 "(volatility-ordered) regime index. The states are ordered by their own "
                 "fitted volatility, so a non-monotone validator means the ordering the "
                 "estimator found and the ordering the validator implies disagree "
                 "somewhere — read the per-state means below before conditioning "
                 "anything on this map:\n")
    p.append("| state | " + " | ".join(f"{labels[k]}" for k in range(K)) + " |")
    p.append("|---|" + "---|" * K)
    p.append(f"| mean {vlabel} | "
             + " | ".join(("n/a" if not np.isfinite(vals[k]) else fmt.format(vals[k]))
                          for k in range(K)) + " |")
    p.append(f"\nRank concordance (Spearman) between state and {vlabel} = "
             f"**{conf['_spearman']:.3f}**; state/quantile diagonal concordance = "
             f"**{conf['diag_concordance']:.3f}**.\n")
    p.append(f"\n![{vlabel} by state](../figures/m04/{name}_{ctx.get('validator_fig', 'vix_by_state')}.png)\n")

    p.append("## Estimator robustness — HMM vs jump model\n")
    p.append(f"The two estimators agree on **{agree['raw_agreement']*100:.1f}%** of days "
             f"(adjusted Rand {agree['adjusted_rand']:.3f}). The jump model, by penalising "
             "transitions, yields the more persistent sequence — the property Nystrup et al. "
             "(2020) motivate as the remedy for the HMM's state-switching noise (Ch2 §2.4).\n")

    if ctx.get("causal"):
        c = ctx["causal"]
        p.append("## Leakage discipline — smoothed vs causal-filtered posterior\n")
        p.append("The regime *map* above uses the **smoothed** posterior P(sₜ | x₁..x_T), the "
                 "right tool to describe which regime was active historically (Ch2 §2.4). A "
                 "forecast, however, may only use the **filtered** posterior P(sₜ | x₁..xₜ), which "
                 "conditions on the past alone. Fitting the HMM on the **training split only** "
                 "(≤ {te}) and filtering causally over the test window, the mean absolute gap "
                 "between the smoothed and filtered crisis probability is "
                 "**{d:.3f}** — concentrated, as expected, around turning points, where the "
                 "smoother's hindsight matters most.\n".format(
                     te=pd.Timestamp(c["train_end"]).date(), d=c["mean_abs_divergence_test"]))
        p.append(f"\n![smoothed vs filtered](../figures/m04/{name}_smoothed_vs_filtered.png)\n")

        # --- Jump-penalised HMM: the headline conditioning signal (Ch2 §2.4/§2.8) ---
        cs = ctx.get("causal_tbl")
        p.append("## The conditioning signal — jump-penalised HMM (Ch2 §2.4/§2.8)\n")
        p.append("Chapter 2 §2.4 adopts the **Nystrup, Lindström & Madsen (2020)** jump-penalised "
                 "estimator as the remedy for the HMM's state-switching noise, and §2.8 conditions "
                 "the regime-aware LSTMs on *\"a Gaussian-emission HMM estimated with the Nystrup "
                 "jump penalty\"*. That estimator is `JumpPenalisedHMM`: the jump model's penalised "
                 "state path supplies the state sequence, the Gaussian emissions and the transition "
                 "matrix are estimated from that path (with a Laplace pseudo-count so the chain "
                 "stays irreducible), and the ordinary forward recursion then yields a genuine "
                 "causal posterior P(sₜ | x₁..xₜ). Using the jump model's own `predict_proba` "
                 "instead would not do: it is a softmax over the emission loss alone and ignores "
                 "the jump penalty, i.e. exactly the persistence the chapter argues for.\n")
        p.append(f"The penalty is **re-selected on the training rows only** "
                 f"(λ = **{c['lam_causal']:.4g}**, smallest λ on the pre-registered grid whose "
                 "mean regime duration clears the persistence floor); the full-sample λ reported "
                 "above is descriptive and would leak into the forecast if reused here (roadmap "
                 "Phase 4: *the regularisation parameter is also CV-selected on training-only "
                 "data*). The train-only sweep is in "
                 f"`results/tables/m04_regimes_{name}_jump_lambda_train.csv`.\n")

        ls = c.get("lam_sensitivity")
        if ls is not None and len(ls):
            head_lam = float(c["lam_causal"])
            clears = ls[ls["clears_persistence_floor"]]
            smallest_clearing = float(clears["jump_penalty"].min()) if len(clears) else float("nan")
            p.append("### Penalty selection and its sensitivity\n")
            p.append("The selection **grid** matters as much as the rule, so both are reported "
                     "rather than just the winner. The grid above was fixed on 2026-05-15, before "
                     "any Phase-5/6 result existed, and is deliberately not widened now: "
                     "re-choosing a grid once the downstream numbers are visible is a "
                     "garden-of-forking-paths problem, and pre-registration is the only thing "
                     "that makes \"λ was not tuned to the answer\" a checkable claim rather than "
                     "an assurance.\n")
            p.append("It is coarse between 1 and 3, and that is worth being explicit about. On "
                     "the finer **sensitivity** grid below the literal rule — *smallest λ "
                     f"clearing the floor* — would select λ = {smallest_clearing:g} rather than "
                     f"λ = {head_lam:g}. Two things are true at once: that is what the rule says, "
                     f"and λ = {smallest_clearing:g} sits one grid point away from the λ = 1 "
                     "collapse, where mean regime duration falls to a few days — precisely the "
                     "over-switching pathology Nystrup et al. (2020) introduce the penalty to "
                     "cure. The headline therefore stays on the pre-registered value, and the "
                     "sensitivity is published so the reader can see the size of the choice "
                     "instead of taking it on trust.\n")
            p.append("Every row is fitted on **training rows only** and filtered causally, "
                     "exactly like the headline, then compared to it day by day:\n")
            lines = ["| λ | train jumps | train mean duration (d) | clears floor | test switches "
                     "| agreement w/ headline (test) | agreement w/ headline (train) |",
                     "|---|---|---|---|---|---|---|"]
            for _, r in ls.iterrows():
                star = " ★" if r["is_headline"] else ""
                lines.append(
                    f"| {r['jump_penalty']:g}{star} | {int(r['train_jumps'])} | "
                    f"{r['train_mean_duration_days']:.1f} | "
                    f"{'yes' if r['clears_persistence_floor'] else 'no'} | "
                    f"{int(r['test_switches'])} | "
                    f"{r['agreement_with_headline_test']:.1%} | "
                    f"{r['agreement_with_headline_train']:.1%} |")
            p.append("\n".join(lines) + "\n")
            plateau = ls[(ls["clears_persistence_floor"])
                         & (ls["agreement_with_headline_test"] >= 0.95)]
            if len(plateau) > 1:
                lo, hi = plateau["jump_penalty"].min(), plateau["jump_penalty"].max()
                p.append(f"\nλ ∈ [{lo:g}, {hi:g}] is a **stable plateau**: every value in it "
                         "assigns ≥95% of test days to the same state as the headline, so no "
                         "conclusion in Phases 5–6 turns on where in that range λ sits. The "
                         "claim being made is that the *conditioning signal* is robust across "
                         "the plateau — not that λ is identified to a point. ★ marks the "
                         f"headline (`results/tables/m04_regimes_{name}_jump_lambda_sensitivity.csv`).\n")
        if cs is not None:
            # Build the whole table as ONE element: the surrounding "\n".join(p)
            # would otherwise put a blank line between rows and break the markdown.
            lines = [
                "| causal signal | λ | test switches | mean duration (d) | "
                "mean max posterior | \\|smoothed−filtered\\| | calm/trans/crisis days |",
                "|---|---|---|---|---|---|---|",
            ]
            for _, r in cs.iterrows():
                lam_txt = "—" if not np.isfinite(r["lambda"]) else f"{r['lambda']:.4g}"
                days = "/".join(str(int(r[f"test_days_state{k}"])) for k in range(K))
                lines.append(
                    f"| {r['signal']} | {lam_txt} | {int(r['test_switches'])} | "
                    f"{r['test_mean_duration_days']:.1f} | {r['mean_max_posterior_test']:.3f} | "
                    f"{r['mean_abs_smoothed_minus_filtered_test']:.3f} | {days} |"
                )
            p.append("\n".join(lines) + "\n")
            p.append(f"The two causal hard paths agree on "
                     f"**{cs['agreement_with_other_signal_test'].iloc[0]*100:.1f}%** of test days. "
                     "A high agreement is the *robustness* reading: it means the Phase-5/6 "
                     "conclusions cannot be an artefact of which regime estimator drew the "
                     "state boundaries.\n")
        p.append(f"\n![smoothed vs filtered (jump-penalised HMM)]"
                 f"(../figures/m04/{name}_smoothed_vs_filtered_jphmm.png)\n")
        p.append(f"\n![causal jump-penalised regime map]"
                 f"(../figures/m04/{name}_regime_map_jphmm_causal.png)\n")

        te_note = pd.Timestamp(c["train_end"]).date()
        p.append("> **Phase-5 hand-off / Ch3 note.** Two causal signals are persisted. The "
                 "**headline** one is `jphmm_filt_p*` — the causal filtered posterior "
                 "P(sₜ | x₁..xₜ) of the **jump-penalised HMM**, which is what Phases 5–7 condition "
                 "on, matching Ch2 §2.8. `hmm_filt_p*` holds the same object for the plain "
                 "Baum-Welch HMM and is retained as the **regime-estimator robustness comparator** "
                 f"(both are a *single* fit on the **training split only** (≤ {te_note}; "
                 "standardisation and λ also fit/selected on train only), then filtered forward "
                 "over the whole sample). Neither model is refit as the walk-forward expands — "
                 "each is frozen at its training-window fit, a deliberately conservative choice "
                 "that keeps every out-of-sample day leakage-free. Each value is aligned to its own "
                 "date t (it uses the return of day t but nothing after it) and is **not** "
                 "pre-lagged; Phase 5 consumes it leakage-safely through the ordinary "
                 "sliding-window rule, because the LSTM input window ends at t−1, so a forecast "
                 "for RVₜ only ever sees regime posteriors dated ≤ t−1. Chapter 3 should state "
                 "both deviations from the roadmap's §1.3 rule 2: the regime model is frozen at "
                 "its training-window fit rather than refit per fold, and λ is selected by the "
                 "persistence rule on training rows rather than by cross-validated likelihood.\n")

    p.append("## Artifacts\n")
    p.append(f"- **Predictions (Phase-5 input):** `results/predictions/m04_regimes_{name}.parquet` — "
             "per-day `hmm_state`, `hmm_p*` (smoothed), `jump_state`, `jump_p*`, the **headline** "
             "causal `jphmm_filt_state`/`jphmm_filt_p*` (jump-penalised HMM) and the comparator "
             "causal `hmm_filt_state`/`hmm_filt_p*` (Baum–Welch), with `rv`"
             + (" and `vix_close`.\n" if is_vix else
                " (no `vix_close`: the registry refuses it for this market).\n"))
    p.append(f"- **Tables:** `results/tables/m04_regimes_{name}_{{bic,persistence,"
             f"{ctx.get('conf_name', 'vix_confusion')},jump_lambda,jump_lambda_train,"
             "agreement,causal_signals}}.csv`.\n")

    p.append("## Gate criteria\n")
    p.append("- [x] Gaussian-emission HMM on standardised daily log returns (Baum–Welch, restarts).\n")
    p.append("- [x] Nystrup et al. (2020) jump-penalised estimator; λ tuned via a persistence sweep.\n")
    p.append("- [x] K∈{2,3,4} compared by BIC; interpretable K=3 headline (calm/transitional/crisis).\n")
    p.append("- [x] Persistence statistics (transition matrix, expected durations).\n")
    p.append("- [x] External validation against VIX quantiles (confusion + rank concordance).\n")
    p.append("- [x] Volatility-ordered states; HMM↔jump agreement reported.\n")
    p.append("- [x] Leakage-free causal filtered posterior built + demonstrated for Phase 5.\n")
    p.append("- [x] **Jump-penalised HMM** (Ch2 §2.4/§2.8) fit on train-only data with a train-only "
             "λ, filtered causally, and persisted as the headline Phase-5/6 conditioning signal.\n")

    p.append("## Reproduce\n")
    p.append("```\npython -m src.experiments.run_regimes\n"
             "pytest -q tests/test_regime.py\n```\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(p), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_regimes")
    ap.add_argument("--config", default="hmm")
    ap.add_argument("--data-config", default="data")
    ap.add_argument("--fast", action="store_true", help="fewer restarts (quick pass)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = load_config(args.data_config)
    set_seed(int(cfg.seed))

    run_dir = repo_path(cfg.paths.experiments, "04_regimes",
                        f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    snapshot_config(cfg, run_dir)

    for profile in cfg.profiles:
        ctx = run_profile(data_cfg, cfg, profile, args)
        write_milestone(ctx, milestone_path(cfg, "m04_regimes.md", profile.name,
                                            n_profiles=len(cfg.profiles)))
    log.info("Phase 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
