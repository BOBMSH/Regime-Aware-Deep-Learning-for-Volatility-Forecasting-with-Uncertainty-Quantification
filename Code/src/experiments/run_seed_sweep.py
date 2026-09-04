"""C5 — is the ranking of the top four models a result, or an initialisation?

The question
------------
Pooled QLIKE separates the top four models by **0.0025**: LSTM-RVonly 0.25964,
Regime-LSTM-A-RVonly 0.26175, MC-Dropout-Regime-LSTM-B 0.26194, Regime-LSTM-B
0.26212. Every one of them was trained once, from one seed. Table 3.5 already
declares the exposure -- *"Reproducible is not robust: a small gap could be
initialisation"* -- and until now nothing measured it. This script does: it
retrains those four across five seeds and reports the spread each way.

What it is, and what it is not
------------------------------
Five seeds **bound** the initialisation spread and show whether the rank order
survives it. Five is not a sample from which to test anything, and no p-value is
computed here. The output is descriptive and the summary says so on its face.

Design decisions worth stating
------------------------------
* **Seeds 17-21** -- the published seed and the four following it. The rule is
  fixed here rather than chosen after seeing the outcome; the alternative
  (picking seeds until the story is clean) is the researcher-degrees-of-freedom
  threat Table 3.5 names.
* **Seed 17 is included on purpose.** Its result must reproduce the committed
  table, and the script reports the deviation for every model. That deviation is
  itself the measurement of *refit* noise at a fixed seed, which is a different
  quantity from initialisation noise and has to be separated from it -- see
  [environment reproducibility]: a same-environment refit reproduced Phase 7 to
  about 1e-6, a cross-interpreter one to 18%.
* **The MC-dropout mask seed is held at 17.** Varying it would fold inference
  noise into a measurement of initialisation noise. Monte Carlo error from the
  100 draws is separately bounded (it falls as 1/sqrt(B); Appendix A.3).
* **Published values are read from the committed tables**, never restated here:
  a static list of numbers that the code also computes is the exact shape of
  drift this project keeps finding.
* **Each model gets the frame its own phase builds.** ``include_vix`` changes the
  row count through an inner join, so Phase 7's frame is not Phase 5's; using one
  frame for all four would quietly change the training sample for two of them.

Usage
-----
    python -m src.experiments.run_seed_sweep

Outputs ``results/tables/m09_seed_sweep.csv`` (one row per model per seed) and
``m09_seed_sweep_summary.csv`` (one row per model). Runs roughly five minutes on
the configuration Appendix A.5 states; nothing else in the pipeline is touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.datasets import frame_for_profile
from src.data.splits import SplitConfig
from src.evaluation.metrics import point_metrics
from src.evaluation.rolling import ACTUAL_COL, run_walk_forward
from src.utils.config import load_config, repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_seed_sweep")

DEFAULT_SEEDS = (17, 18, 19, 20, 21)
PUBLISHED_SEED = 17
# The four models the pooled ranking cannot separate. Phase 5 supplies two of
# them, Phase 3 one and Phase 7 one, which is why three frames are built below.
TOP_FOUR = ("LSTM-RVonly", "Regime-LSTM-A-RVonly", "Regime-LSTM-B",
            "MC-Dropout-Regime-LSTM-B")
# Where each model's committed pooled QLIKE lives, so the seed-17 row can be
# checked against it rather than against a number retyped into this file.
PUBLISHED_TABLES = {
    "m05_regime_dl_intraday_2019_2022_metrics.csv": ("model", "qlike"),
    "m07_combined_intraday_2019_2022_master.csv": ("model", "qlike"),
}


def published_qlike() -> dict[str, float]:
    """Committed pooled QLIKE per model, read from the artefacts."""
    out: dict[str, float] = {}
    for name, (mcol, qcol) in PUBLISHED_TABLES.items():
        path = repo_path("results", "tables", name)
        if not path.exists():
            log.warning("%s not found; its models cannot be checked", name)
            continue
        tab = pd.read_csv(path)
        for m, q in zip(tab[mcol], tab[qcol]):
            out.setdefault(str(m), float(q))
    return out


def assert_shared_protocol(cfgs: dict) -> tuple[SplitConfig, str]:
    """The four models must share splits and refit cadence, or they are not comparable.

    They do today -- all three configs carry the identical ``harness`` block and
    the identical profile splits -- but a sweep that silently averaged over two
    protocols would be worse than no sweep, so the agreement is checked rather
    than assumed.
    """
    seen = {}
    for label, cfg in cfgs.items():
        p = cfg.profiles[0]
        sp = p.splits
        seen[label] = (
            str(sp.train_end), str(sp.val_start), str(sp.val_end),
            str(sp.test_start), str(sp.test_end),
            int(cfg.harness.refit_frequency_days), str(cfg.harness.scheme),
            str(cfg.harness.eval_segment), int(cfg.harness.refit_every_folds),
        )
    distinct = set(seen.values())
    if len(distinct) != 1:
        raise SystemExit(
            "the sweep's configs disagree on the evaluation protocol, so their "
            f"models are not comparable: {seen}"
        )
    (train_end, val_start, val_end, test_start, test_end,
     refit_days, scheme, eval_segment, _) = distinct.pop()
    return SplitConfig(train_end=train_end, val_start=val_start, val_end=val_end,
                       test_start=test_start, test_end=test_end,
                       refit_frequency_days=refit_days, scheme=scheme), eval_segment


def build_for_seed(seed: int, cfgs: dict, best: dict) -> list[tuple[str, object, str]]:
    """``(name, forecaster, frame_key)`` for the four models at one seed.

    Every forecaster comes from its own phase runner's factory, so this file
    states no hyperparameter and no feature set of its own.
    """
    from src.experiments import run_combined, run_lstm, run_regime_lstm

    lcfg, rcfg, ccfg = cfgs["lstm"], cfgs["regime"], cfgs["combined"]
    out: list[tuple[str, object, str]] = []

    rob = lcfg.get("robustness", {}) or {}
    feats = tuple(OmegaConf.to_container(rob.rv_only_features, resolve=True))
    out.append(("LSTM-RVonly",
                run_lstm.make_forecaster(lcfg.model, lcfg.harness, name="LSTM-RVonly",
                                         features=feats, seed=seed, **best),
                "vix"))

    for m in run_regime_lstm.build_models(rcfg.model, rcfg.harness, rcfg.approaches,
                                          seed=seed, max_epochs=None):
        if m.name in TOP_FOUR:
            out.append((m.name, m, "vix_regime"))

    # `build_model` reads cfg.seed, so the seed is overridden on a merged copy;
    # mc_seed keeps its configured value on purpose (see the module docstring).
    merged = OmegaConf.merge(ccfg, {"seed": int(seed)})
    m = run_combined.build_model(merged, max_epochs=None, mc_samples=None)
    out.append((m.name, m, "novix_regime"))

    names = [n for n, _, _ in out]
    missing = [n for n in TOP_FOUR if n not in names]
    if missing:
        raise SystemExit(f"the factories did not produce {missing}; check the configs")
    return out


def summarise(rows: pd.DataFrame, published: dict[str, float]) -> pd.DataFrame:
    """Per-model spread, and the seed-17 deviation from the committed value."""
    out = []
    for name, grp in rows.groupby("model", sort=False):
        q = grp["qlike"].to_numpy(dtype=float)
        at17 = grp.loc[grp["seed"] == PUBLISHED_SEED, "qlike"]
        pub = published.get(name, np.nan)
        out.append({
            "model": name,
            "n_seeds": int(len(q)),
            "qlike_mean": float(q.mean()),
            "qlike_sd": float(q.std(ddof=1)) if len(q) > 1 else np.nan,
            "qlike_min": float(q.min()),
            "qlike_max": float(q.max()),
            "qlike_range": float(q.max() - q.min()),
            "qlike_published": pub,
            "qlike_seed17": float(at17.iloc[0]) if len(at17) else np.nan,
            "refit_deviation": (float(abs(at17.iloc[0] - pub))
                                if len(at17) and np.isfinite(pub) else np.nan),
        })
    return pd.DataFrame(out).sort_values("qlike_mean").reset_index(drop=True)


def rank_stability(rows: pd.DataFrame) -> pd.DataFrame:
    """The ordering of the models under each seed, which is the actual question."""
    wide = rows.pivot(index="seed", columns="model", values="qlike")
    orders = wide.apply(lambda r: " < ".join(r.sort_values().index), axis=1)
    return pd.DataFrame({"seed": wide.index, "order_by_qlike": orders.to_numpy()})


def pairwise_signs(rows: pd.DataFrame) -> pd.DataFrame:
    """For each pair, how many seeds put A below B. 5 of 5 is stable; 3 of 5 is noise."""
    wide = rows.pivot(index="seed", columns="model", values="qlike")
    cols = list(wide.columns)
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            wins = int((wide[a] < wide[b]).sum())
            out.append({"model_a": a, "model_b": b, "seeds_a_better": wins,
                        "n_seeds": int(len(wide)),
                        "mean_gap": float((wide[a] - wide[b]).mean())})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.run_seed_sweep")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="cap epochs (diagnostic only; the artefact is then partial)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the artefact even from fewer than two seeds")
    args = ap.parse_args(argv)

    seeds = list(dict.fromkeys(int(s) for s in args.seeds))
    if len(seeds) < 2 and not args.allow_partial:
        raise SystemExit(
            f"a spread cannot be measured from {len(seeds)} seed(s). Pass more seeds, "
            "or --allow-partial to write a diagnostic artefact over the real one."
        )
    if args.max_epochs and not args.allow_partial:
        raise SystemExit(
            "--max-epochs produces under-trained models whose spread means nothing; "
            "pass --allow-partial if you intend to overwrite the artefact with one."
        )

    cfgs = {"lstm": load_config("lstm_baseline"),
            "regime": load_config("regime_lstm"),
            "combined": load_config("combined"),
            "data": load_config("data")}
    split_cfg, eval_segment = assert_shared_protocol(
        {k: v for k, v in cfgs.items() if k != "data"})

    profile = cfgs["lstm"].profiles[0]
    from src.experiments.run_regime_lstm import attach_regime
    log.info("building frames")
    f_vix = frame_for_profile(cfgs["data"], profile, include_vix=True)
    frames = {
        "vix": f_vix,
        "vix_regime": attach_regime(f_vix.copy(), cfgs["regime"].regime),
        "novix_regime": attach_regime(
            frame_for_profile(cfgs["data"], profile, include_vix=False),
            cfgs["combined"].regime),
    }
    for k, f in frames.items():
        log.info("  frame %-13s %d rows, %s -> %s", k, len(f),
                 f.index.min().date(), f.index.max().date())

    from src.experiments.report_compute import selected_hyperparameters
    sel = selected_hyperparameters(profile.name)
    best = {"hidden": sel["hidden"], "lookback": sel["lookback"], "lr": sel["lr"]}
    log.info("Phase-3 selected hyperparameters: %s", best)
    if args.max_epochs:
        log.warning("max_epochs capped at %d -- DIAGNOSTIC ONLY", args.max_epochs)

    rows = []
    for seed in seeds:
        log.info("=" * 62)
        log.info("SEED %d", seed)
        for name, fc, key in build_for_seed(seed, cfgs, best):
            if args.max_epochs:
                fc.max_epochs = int(args.max_epochs)
            preds = run_walk_forward([fc], frames[key], split_cfg,
                                     eval_segment=eval_segment, strict=True)
            met = point_metrics(preds[ACTUAL_COL], preds[name])
            rows.append({"model": name, "seed": seed, "n": int(len(preds)),
                         "qlike": float(met["qlike"]), "mse": float(met["mse"]),
                         "mae": float(met["mae"])})
            log.info("  %-26s QLIKE %.6f", name, met["qlike"])

    long = pd.DataFrame(rows)
    published = published_qlike()
    summary = summarise(long, published)
    ranks = rank_stability(long)
    pairs = pairwise_signs(long)

    tables = repo_path("results", "tables")
    ensure_dir(tables)
    long.to_csv(tables / "m09_seed_sweep.csv", index=False)
    summary.to_csv(tables / "m09_seed_sweep_summary.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== per model over the seeds ===")
    print(summary.to_string(index=False))
    print("\n=== ordering under each seed ===")
    print(ranks.to_string(index=False))
    print("\n=== pairwise: seeds in which A has the lower QLIKE ===")
    print(pairs.to_string(index=False))

    # The comparison the whole exercise exists to make, stated as measurement.
    between = float(summary["qlike_mean"].max() - summary["qlike_mean"].min())
    within = float(summary["qlike_range"].max())
    worst = summary.loc[summary["qlike_range"].idxmax(), "model"]
    stable = ranks["order_by_qlike"].nunique() == 1
    print(f"\nBetween-model spread of the seed means: {between:.6f}")
    print(f"Largest within-model range across seeds:  {within:.6f}  ({worst})")
    print(f"Ordering identical under every seed: {'yes' if stable else 'NO'}"
          f"  ({ranks['order_by_qlike'].nunique()} distinct orderings in {len(ranks)} seeds)")
    print("Five seeds bound the spread; they do not test it. No p-value is implied.")

    dev = summary["refit_deviation"].dropna()
    if len(dev):
        print(f"\nSeed-{PUBLISHED_SEED} rows reproduce the committed tables to "
              f"{dev.max():.2e} at worst ({len(dev)} of {len(summary)} models checked).")
    print(f"\nwrote {tables / 'm09_seed_sweep.csv'}\n      "
          f"{tables / 'm09_seed_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
