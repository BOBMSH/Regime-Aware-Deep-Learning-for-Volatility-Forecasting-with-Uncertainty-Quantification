"""Formal interval-coverage testing for any UQ predictions file (Phase 7; RQ3).

What this closes
----------------
Phase 6 reported PICP, MPIW, the Winkler score and CRPS — all point estimates,
none of them a test. RQ3 asks whether "empirical coverage matches nominal", and
answering it with "0.865 against 0.90" is not an answer: it neither rejects nor
fails to reject anything. The roadmap's Phase-7 entry therefore calls for the
Kupiec, Christoffersen and Engle–Manganelli DQ tests already cited in Ch2 §2.6.
This script is that commitment, and — like :mod:`src.experiments.report_gw` — it
reads persisted parquets only, so it retrains nothing, runs in seconds and can
be pointed at any phase's UQ output.

The one rule that governs everything here
-----------------------------------------
**Every test runs on the FULL sample, with every conditioning variable lagged
into the ``t−1`` information set.** Not on a transition subsample, not on a
per-regime subsample. Two independent reasons, both learned the hard way in this
project:

1. *A subsample selector chosen with the day-t outcome has no nominal size.*
   The ex-post transition indicator ``1{s_t ≠ s_(t−1)}`` is detected by the very
   one-day surprise that breaks an interval, so a Kupiec test on it returns
   p ≈ 1e-04 while the implementable ``1{s_(t−1) ≠ s_(t−2)}`` returns nothing
   (2026-08-19 (iii)). :func:`selector_is_implementable` guards this.
2. *A globally mis-levelled model fails a nominal test on every subsample.* The
   Quantile-LSTM covers 0.821 against 0.90, so Kupiec-against-nominal rejects on
   any subsample with power — measured p = 0.046 on the ex-ante transition days
   despite a −1.5 pp flagged-vs-rest gap. Reading that as "coverage fails at
   transitions" would report a **level** failure as a **timing** failure
   (2026-08-22 (iv)).

So conditioning enters as a *regressor* in the DQ test, where the null is
``E[hit_t | F_(t−1)] = 0`` on the whole sample, and any subsample statement is
made with :func:`two_sample_coverage_test` — a difference in coverage between
the flagged days and the rest — never against the nominal level.

What is produced
----------------
``results/tables/<stem>_coverage_tests.csv``
    Kupiec / Christoffersen (ind, cc) / DQ per method, per nominal level, for
    the pooled two-sided hit and for the upper tail (realized variance above the
    band — the risk-relevant miss).

``results/tables/<stem>_coverage_conditional.csv``
    DQ re-run with ``t−1`` conditioning variables appended one family at a time
    — the lagged regime indicators, the ex-ante transition dummy, the model's
    own (log) interval width — so a reader can see which, if any, predicts a
    miss. Plus the difference-in-coverage tests for every regime bucket and for
    the transition split, each stamped with its selector timing.

Usage
-----
    python -m src.experiments.report_coverage                    # Phase-6 headline
    python -m src.experiments.report_coverage \
        --predictions results/predictions/m06_uq_intraday_2019_2022_hmm.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.coverage_tests import (
    coverage_test_suite,
    dq_incremental,
    expected_rate,
    holm_adjust,
    interval_hits,
    two_sample_coverage_test,
)
from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    DEFAULT_SELECTOR_SHIFT,
    align_regime_label,
    regime_timing_label,
    selector_is_implementable,
    selector_timing_label,
    transition_mask,
)
from src.experiments.report_gw import resolve_regime_source
from src.utils.config import repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("report_coverage")

ACTUAL_COL = "y_true"
REGIME_STATE_COL = "reg_state"
DEFAULT_PREDICTIONS = "results/predictions/m06_uq_intraday_2019_2022.parquet"
DEFAULT_LABELS = ("calm", "transitional", "crisis")

_INTERVAL_RE = re.compile(r"^(?P<method>.+)_(?P<side>lo|hi)(?P<lvl>\d{2,3})$")


# --------------------------------------------------------------------------- #
# Discover the intervals carried by a predictions file                         #
# --------------------------------------------------------------------------- #
def discover_intervals(preds: pd.DataFrame) -> list[tuple[str, float]]:
    """Find every ``(method, level)`` pair with both a ``_loNN`` and ``_hiNN`` column.

    Written as discovery rather than a hardcoded list so the same script serves
    the Phase-6 methods and the Phase-7 combined model without editing, and so a
    method that silently stopped writing one endpoint disappears from the report
    instead of being scored against a stale column.
    """
    found: dict[tuple[str, float], set[str]] = {}
    for col in preds.columns:
        m = _INTERVAL_RE.match(str(col))
        if not m:
            continue
        key = (m.group("method"), float(m.group("lvl")) / 100.0)
        found.setdefault(key, set()).add(m.group("side"))
    pairs = sorted(k for k, sides in found.items() if sides == {"lo", "hi"})
    if not pairs:
        raise KeyError(
            "no interval columns found: expected pairs like "
            "'MC-Dropout-LSTM_lo90' / 'MC-Dropout-LSTM_hi90'"
        )
    return pairs


# --------------------------------------------------------------------------- #
# Load: predictions + the FULL-history regime series                           #
# --------------------------------------------------------------------------- #
def load_inputs(
    path: Path, *, regime_source: str | None = None, state_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(predictions, full_regime_state)``.

    The regime series is deliberately taken from the **Phase-4 parquet**, not the
    ``reg_state`` column of the predictions file: that column is already sliced
    to the evaluation window, so lagging it drops the first out-of-sample day and
    every conditioned test would silently run on a different sample from the
    unconditioned ones (2026-08-22 (iv)). Resolution is shared with
    :mod:`src.experiments.report_gw` so the two reports cannot disagree about
    which column is the causal state.
    """
    preds = pd.read_parquet(path)
    if ACTUAL_COL not in preds.columns:
        raise KeyError(f"{path.name}: missing '{ACTUAL_COL}'")

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
        log.info("regime label: %s::%s (%d rows) — %s", src.name, col, len(state),
                 regime_timing_label(DEFAULT_REGIME_SHIFT))
    elif REGIME_STATE_COL in preds.columns:
        state = preds[REGIME_STATE_COL].astype("float")
        log.warning(
            "Phase-4 regime file not found; falling back to %s::%s, which is "
            "already sliced to the evaluation window. The first evaluated day "
            "will lose its t-1 label, so the conditional tests will run on a "
            "different sample from the unconditional ones. Pass --regime-source.",
            path.name, REGIME_STATE_COL,
        )
    else:
        raise KeyError(
            f"{path.name} carries no '{REGIME_STATE_COL}' and no Phase-4 regime "
            "file could be resolved; pass --regime-source"
        )
    return preds, state


# --------------------------------------------------------------------------- #
# Conditioning variables, all measurable at t-1                                #
# --------------------------------------------------------------------------- #
def build_conditioners(
    preds: pd.DataFrame,
    state: pd.Series,
    labels: list[str],
    *,
    method: str,
    level: float,
) -> dict[str, pd.DataFrame]:
    """Families of ``t−1``-measurable DQ regressors, one family per DQ re-run.

    Families rather than one big design: with everything in at once a rejection
    says only "something predicts a miss", and Phase 7's question is *which*
    thing. Each family is small enough to read off the χ² directly.

    * ``regime`` — the lagged regime one-hots, **dropping the first state** as
      the reference category. The complete set plus the DQ constant is exactly
      collinear, and dropping a column is the standard remedy; ``calm`` is the
      natural reference because it is the modal state.
    * ``transition`` — the **ex-ante** selector ``1{s_(t−1) ≠ s_(t−2)}``. The
      ex-post version is not offered here at all: it is not in ``F_(t−1)``, and
      the point of this script is that everything in it is.
    * ``width`` — the model's own log interval width. Known at ``t−1`` (it *is*
      the forecast), and the sharpest test of whether the band widens when it
      should: if width predicted misses, the interval would be mis-*shaped*, not
      merely mis-levelled.
    """
    idx = preds.index
    out: dict[str, pd.DataFrame] = {}

    lagged = align_regime_label(state, idx, shift=DEFAULT_REGIME_SHIFT,
                                warn_unreachable=True)
    reg = pd.DataFrame(index=idx)
    for k, lab in enumerate(labels):
        if k == 0:
            continue                      # reference category (see docstring)
        col = (lagged == float(k)).astype(float)
        reg[f"regime_{lab}(t-1)"] = col.where(lagged.notna(), np.nan)
    if reg.shape[1] and reg.notna().any(axis=1).any():
        out["regime"] = reg

    mask, scored = transition_mask(state, idx, selector_shift=DEFAULT_SELECTOR_SHIFT)
    tr = pd.Series(np.where(scored, mask.astype(float), np.nan), index=idx)
    if np.nansum(tr.to_numpy()) > 0:
        out["transition"] = pd.DataFrame({"transition_exante(t-1)": tr})

    tag = f"{int(round(level * 100))}"
    lo, hi = f"{method}_lo{tag}", f"{method}_hi{tag}"
    if lo in preds.columns and hi in preds.columns:
        width = (preds[hi] - preds[lo]).astype(float)
        with np.errstate(divide="ignore"):
            logw = np.log(width.where(width > 0))
        out["width"] = pd.DataFrame({"log_interval_width": logw})

    return out


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def unconditional_table(preds: pd.DataFrame, pairs, *, dq_lags: int) -> pd.DataFrame:
    """Kupiec / Christoffersen / DQ for every (method, level), pooled and upper tail."""
    y = preds[ACTUAL_COL].astype(float)
    frames = []
    for method, level in pairs:
        tag = f"{int(round(level * 100))}"
        lo, hi = preds[f"{method}_lo{tag}"].astype(float), preds[f"{method}_hi{tag}"].astype(float)
        frames.append(coverage_test_suite(y, lo, hi, level=level, method=method,
                                          dq_lags=dq_lags))
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "sample", "full")
    return out


def conditional_table(
    preds: pd.DataFrame, state: pd.Series, pairs, labels, *, dq_lags: int
) -> pd.DataFrame:
    """Incremental DQ per conditioning family, plus difference-in-coverage tests.

    Two deliberate choices, both of which change what the table means:

    * **The DQ rows are INCREMENTAL** (:func:`dq_incremental`), not the raw
      statistic. The raw DQ design contains a constant, so it tests the level
      jointly with everything else — and every interval here is too narrow
      globally, so the raw DQ rejects for every method against every
      conditioner. That would say nothing about the regime. The incremental
      statistic asks the question RQ3 actually poses: *beyond the level being
      wrong, does anything knowable at t−1 say where the band will break?* The
      raw statistics are carried in ``detail`` so the decomposition is visible.
    * **p-values are Holm-adjusted within each (method, level, side) family.**
      Three regime buckets, a transition split and three DQ families is enough
      tests that one crosses 0.05 by chance; Holm controls the family-wise rate
      without assuming the tests are independent, which — sharing one hit series
      — they are not. Both raw and adjusted values are reported.

    Every row carries the timing of whatever it conditioned on, so a number can
    never be quoted without it: the convention adopted project-wide in
    2026-08-19 (ii), extended here to the coverage tests.
    """
    y = preds[ACTUAL_COL].astype(float)
    rows = []
    for method, level in pairs:
        tag = f"{int(round(level * 100))}"
        lo = preds[f"{method}_lo{tag}"].astype(float)
        hi = preds[f"{method}_hi{tag}"].astype(float)
        fams = build_conditioners(preds, state, labels, method=method, level=level)
        lagged = align_regime_label(state, preds.index, shift=DEFAULT_REGIME_SHIFT)
        mask, scored = transition_mask(state, preds.index,
                                       selector_shift=DEFAULT_SELECTOR_SHIFT)

        for side in ("both", "upper"):
            a = expected_rate(level, side)

            # ---- incremental DQ, one conditioning family at a time ----------
            for fam, ex in fams.items():
                sub = ex.dropna(how="any")
                if sub.empty:
                    continue
                idx = sub.index
                h = interval_hits(y.reindex(idx), lo.reindex(idx), hi.reindex(idx),
                                  side=side)
                try:
                    r = dq_incremental(h, a, sub.to_numpy(), lags=dq_lags,
                                       exog_names=list(sub.columns))
                except ValueError as exc:  # a degenerate design is reportable
                    log.warning("incremental DQ [%s|%s|%s] skipped: %s",
                                method, fam, side, exc)
                    continue
                rows.append({
                    "method": method, "nominal_level": float(level), "side": side,
                    "kind": "DQ incremental", "conditioner": fam,
                    "timing": "t-1 (implementable)",
                    "stat": r["stat"], "df": float(r["df"]), "p_value": r["p_value"],
                    "n": int(r["n"]),
                    "regressors": ", ".join(r["added_regressors"]),
                    "expected_violation_rate": a,
                    "detail": (f"DQ_full={r['dq_full']:.3f} (df {r['df_full']}, "
                               f"p={r['p_full']:.4g}) vs DQ_level+dynamics="
                               f"{r['dq_restricted']:.3f} (df {r['df_restricted']}, "
                               f"p={r['p_restricted']:.4g})"),
                })

            # ---- difference in coverage: each regime bucket vs the rest ------
            h_all = pd.Series(
                interval_hits(y, lo, hi, side=side), index=preds.index, dtype=float)
            keep = lagged.notna()
            for k, lab in enumerate(labels):
                g = (lagged == float(k))
                if int(g.sum()) < 5 or int((keep & ~g).sum()) < 5:
                    continue
                try:
                    r = two_sample_coverage_test(h_all[keep].to_numpy(),
                                                 g[keep].to_numpy().astype(float))
                except ValueError as exc:
                    log.warning("difference test [%s|%s|%s] skipped: %s",
                                method, lab, side, exc)
                    continue
                rows.append({
                    "method": method, "nominal_level": float(level), "side": side,
                    "kind": "difference in coverage", "conditioner": f"regime={lab}",
                    "timing": regime_timing_label(DEFAULT_REGIME_SHIFT),
                    "stat": r["z_hac"], "df": np.nan, "p_value": r["p_hac"],
                    "n": int(r["n_flagged"] + r["n_rest"]),
                    "regressors": "", "expected_violation_rate": a,
                    "detail": (f"PICP {r['picp_flagged']:.4f} (n={r['n_flagged']}) vs "
                               f"{r['picp_rest']:.4f} (n={r['n_rest']}); "
                               f"z_iid={r['z_iid']:.3f} p_iid={r['p_iid']:.4g}"),
                })

            # ---- difference in coverage: ex-ante transition vs stable --------
            if int(mask.sum()) >= 5 and int((scored & ~mask).sum()) >= 5:
                try:
                    r = two_sample_coverage_test(h_all[scored].to_numpy(),
                                                 mask[scored].to_numpy().astype(float))
                    rows.append({
                        "method": method, "nominal_level": float(level), "side": side,
                        "kind": "difference in coverage",
                        "conditioner": "regime transition (ex-ante)",
                        "timing": selector_timing_label(DEFAULT_SELECTOR_SHIFT),
                        "stat": r["z_hac"], "df": np.nan, "p_value": r["p_hac"],
                        "n": int(r["n_flagged"] + r["n_rest"]), "regressors": "",
                        "expected_violation_rate": a,
                        "detail": (f"PICP {r['picp_flagged']:.4f} (n={r['n_flagged']}) vs "
                                   f"{r['picp_rest']:.4f} (n={r['n_rest']}); "
                                   f"z_iid={r['z_iid']:.3f} p_iid={r['p_iid']:.4g}"),
                    })
                except ValueError as exc:
                    log.warning("transition difference test skipped: %s", exc)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm within each (method, level, side) family -- see the docstring.
    out["p_holm"] = np.nan
    for _, grp in out.groupby(["method", "nominal_level", "side"], sort=False):
        out.loc[grp.index, "p_holm"] = holm_adjust(grp["p_value"].to_numpy())
    out["n_tests_in_family"] = out.groupby(
        ["method", "nominal_level", "side"], sort=False)["p_value"].transform("size")
    # A standing reminder on every row that the selector is implementable: the
    # whole file would be meaningless otherwise, and two earlier audits both
    # turned on exactly this distinction.
    out["implementable"] = selector_is_implementable(DEFAULT_SELECTOR_SHIFT)
    return out


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.report_coverage")
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS,
                    help="UQ predictions parquet carrying y_true and <method>_lo/hi<NN> columns")
    ap.add_argument("--labels", nargs="*", default=list(DEFAULT_LABELS))
    ap.add_argument("--dq-lags", type=int, default=4,
                    help="lagged hits in the DQ design (Engle-Manganelli use 4)")
    ap.add_argument("--regime-source", default=None,
                    help="Phase-4 predictions parquet holding the FULL causal regime series")
    ap.add_argument("--state-col", default=None,
                    help="causal hard-state column on the regime source")
    args = ap.parse_args(argv)

    path = Path(args.predictions)
    if not path.is_absolute():
        path = repo_path(str(path))
    if not path.exists():
        raise SystemExit(
            f"predictions file not found: {path}\n"
            "Run the phase that produces it first, or pass --predictions."
        )

    preds, state = load_inputs(path, regime_source=args.regime_source,
                               state_col=args.state_col)
    pairs = discover_intervals(preds)
    labels = list(args.labels)
    log.info("loaded %s: %d obs; intervals %s", path.name, len(preds),
             [f"{m}@{int(l*100)}" for m, l in pairs])

    unc = unconditional_table(preds, pairs, dq_lags=int(args.dq_lags))
    con = conditional_table(preds, state, pairs, labels, dq_lags=int(args.dq_lags))

    stem = path.stem
    tables = repo_path("results", "tables")
    ensure_dir(tables)
    unc_path = tables / f"{stem}_coverage_tests.csv"
    con_path = tables / f"{stem}_coverage_conditional.csv"
    unc.to_csv(unc_path, index=False)
    con.to_csv(con_path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== Unconditional coverage tests (full sample) ===")
    show = unc[["method", "side", "nominal_level", "test", "stat", "df", "p_value",
                "violation_rate", "expected_violation_rate", "n"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Conditional tests — every conditioner measurable at t-1 ===")
    print("    (DQ rows are INCREMENTAL over level + hit dynamics; p_holm is "
          "family-wise adjusted)")
    if len(con):
        show2 = con[["method", "nominal_level", "side", "kind", "conditioner",
                     "stat", "df", "p_value", "p_holm", "n", "detail"]]
        print(show2.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("(no conditional tests could be formed)")
    print(f"\nwrote {unc_path}\n      {con_path}")

    # A plain-language read of the level-vs-timing question, computed rather
    # than asserted -- this is the sentence Chapter 4 needs and the one the
    # roadmap pre-registered an expectation for.
    print("\n--- Level vs timing ---")
    for method, level in pairs:
        sub = unc[(unc["method"] == method) & (unc["nominal_level"] == level)
                  & (unc["side"] == "both")]
        if not len(sub):
            continue
        uc = sub[sub["test"] == "Kupiec LR_uc"].iloc[0]
        ind = sub[sub["test"] == "Christoffersen LR_ind"].iloc[0]
        csub = con[(con["method"] == method) & (con["nominal_level"] == level)
                   & (con["side"] == "both") & (con["kind"] == "DQ incremental")]
        worst = csub.loc[csub["p_holm"].idxmin()] if len(csub) else None
        lvl_txt = ("REJECTED" if uc["p_value"] < 0.05 else "not rejected")
        tim_txt = ("REJECTED" if ind["p_value"] < 0.05 else "not rejected")
        line = (f"  {method} @ {level:.0%}: level {lvl_txt} (Kupiec p={uc['p_value']:.3g}, "
                f"PICP {1 - uc['violation_rate']:.4f}); clustering {tim_txt} "
                f"(LR_ind p={ind['p_value']:.3g})")
        if worst is not None:
            line += (f"; strongest t-1 conditioner '{worst['conditioner']}' "
                     f"incremental DQ p={worst['p_value']:.3g} "
                     f"(Holm {worst['p_holm']:.3g})")
        print(line)

    # The upper tail is reported separately because it is the economically
    # consequential miss for a variance forecast and because -- unlike the
    # pooled view -- it is where this project's subsample structure shows up.
    up = con[(con["side"] == "upper") & (con["kind"] == "difference in coverage")]
    if len(up):
        sig = up[up["p_holm"] < 0.05]
        print("\n--- Upper-tail coverage differences surviving Holm ---")
        if len(sig):
            for _, r in sig.iterrows():
                print(f"  {r['method']} @ {r['nominal_level']:.0%} | "
                      f"{r['conditioner']}: {r['detail']} | z_hac={r['stat']:.3f} "
                      f"p={r['p_value']:.4g} Holm={r['p_holm']:.4g}")
        else:
            print("  none — no t-1 subsample differs in upper-tail coverage once "
                  "the family-wise rate is controlled.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
