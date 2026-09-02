"""Phase 9 -- the reproducibility gate: re-derive every published number.

WHAT THIS IS FOR
----------------
The supervisor's brief is explicit that "the code should reproduce, as closely as
possible, the values reported in the dissertation", and the marking grid scores
reproducibility directly. Three audits have already checked that by hand --
(vi), (vii) and (ix) each re-derived the headline statistics with throwaway code
and threw the code away. This module is that discipline made permanent: one
command, run before submission and after any regeneration, that reads every
published table and recomputes its contents **from the prediction parquets**.

    python -m src.experiments.verify_claims          # exits 1 on any mismatch

It writes ``results/tables/m09_verification.csv`` (one row per checked value) and
``results/milestones/m09_verification.md``, and its exit code is the gate.

WHY THE FORMULAS ARE DUPLICATED HERE -- DO NOT "DRY THIS UP"
-----------------------------------------------------------
Every metric below is reimplemented inline, and this module imports **nothing**
from ``src.evaluation``. That is deliberate and it is the whole point.

Calling ``src.evaluation.metrics.qlike`` here would check that a CSV matches what
that function returns today -- catching a stale artefact, which is the common
failure, but agreeing enthusiastically with a regression in the metric itself. An
independent implementation checks both: the artefact against the data, *and* the
project's formula against a second reading of the definition. That is exactly
what audits (vi), (vii) and (ix) did by hand, and it is why they were worth
running.

If a future change makes these formulas disagree with ``src.evaluation``, that
disagreement is the finding. Resolve it by deciding which is right -- never by
importing one into the other.

WHAT IS CHECKED
---------------
* **Point metrics** -- ``mse``, ``rmse``, ``mae``, ``qlike`` in every
  ``*_metrics.csv`` and in the Phase-7 master table, against the model's forecast
  column in the matching predictions parquet.
* **Interval metrics** -- ``picp``, ``mpiw``, ``winkler``, ``coverage_error`` in
  every ``*_calibration.csv``, from the persisted ``<method>_lo<NN>`` /
  ``_hi<NN>`` bounds.
* **Per-regime point loss** -- every ``*_per_regime.csv`` written on the ``t-1``
  label, recomputed with the lag applied to the **full-history** Phase-4 state
  series, which is the timing convention the whole project uses.
* **Sample sizes** -- every ``n`` column against the rows actually scored.
* **Cross-artefact agreement** -- a model appearing in more than one table must
  carry the same value in all of them. This is the check that would have caught
  the 2026-09-01 CRPS defect, where three models shared one number because a
  table was assembled from the wrong frame.

WHAT IS NOT CHECKED, AND WHY
----------------------------
Test statistics (DM, GW, MCS, Kupiec, DQ) are not recomputed. Their published
values depend on HAC bandwidth rules, finite-sample corrections and bootstrap
draws whose *conventions* are the project's own choices, so an independent
reimplementation would be reimplementing those choices rather than checking them
-- it would test agreement with itself. They are covered instead by
``tests/test_significance.py`` against closed-form cases, and by the ``n`` and
input-column checks here: a test statistic computed on the right sample from
verified forecasts is the part this gate can honestly certify.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import repo_path
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger("verify")

ACTUAL_COL = "y_true"

#: Relative tolerance for a re-derived value. Artefacts are serialised at 16
#: significant figures, so agreement should be near machine precision; this is
#: loose enough to absorb the serialisation round-trip and nothing else.
RTOL = 1e-9
ATOL = 1e-15


# --------------------------------------------------------------------------- #
# Independent metric implementations -- see the module docstring               #
# --------------------------------------------------------------------------- #
def _qlike(y: np.ndarray, h: np.ndarray) -> float:
    """QLIKE on the variance scale, zero at a perfect forecast (Patton 2011).

    L = mean( y/h - log(y/h) - 1 ).
    """
    r = y / h
    return float(np.mean(r - np.log(r) - 1.0))


def _mse(y: np.ndarray, h: np.ndarray) -> float:
    return float(np.mean((y - h) ** 2))


def _rmse(y: np.ndarray, h: np.ndarray) -> float:
    return math.sqrt(_mse(y, h))


def _mae(y: np.ndarray, h: np.ndarray) -> float:
    return float(np.mean(np.abs(y - h)))


def _picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def _mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def _winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float) -> float:
    """Winkler (interval) score: width, plus a penalty for each exclusion.

    W = (u-l) + (2/a)(l-y) if y<l ; (u-l) + (2/a)(y-u) if y>u ; (u-l) otherwise,
    with a = 1 - level. Lower is better (Gneiting & Raftery 2007).
    """
    a = 1.0 - level
    w = hi - lo
    below = y < lo
    above = y > hi
    w = w + np.where(below, (2.0 / a) * (lo - y), 0.0)
    w = w + np.where(above, (2.0 / a) * (y - hi), 0.0)
    return float(np.mean(w))


def _lag_state(state: pd.Series, index: pd.Index) -> pd.Series:
    """The ``t-1`` regime label for each evaluated day.

    The shift is applied on the state series' **own** index and only then
    reindexed onto the evaluation dates, so the first out-of-sample day keeps the
    label it is entitled to. Pre-slicing the state to the evaluation window and
    shifting that loses it -- the defect of 2026-08-19 (iv), which published two
    values for one cell.
    """
    return state.sort_index().shift(1).reindex(index)


PointFn = {"mse": _mse, "rmse": _rmse, "mae": _mae, "qlike": _qlike}

#: Phase prefixes on an artefact stem. Stripping one leaves the profile, which is
#: what lets a Phase-7 table find the Phase-6 parquet its interval columns live
#: in -- the tables are assembled across phases and the checks must follow.
_PHASE_PREFIXES = ("m02_", "m03_lstm_", "m04_regimes_", "m05_regime_dl_",
                   "m06_uq_", "m07_combined_", "m08_")

#: Rows that are one forecast read a second way rather than a persisted series.
#: They have no column of their own, so they are reported as SKIP with the reason
#: rather than MISSING -- an absent forecast column and a derived row are
#: different facts and must not look the same in the report.
DERIVED_ROWS = {"Quantile-LSTM-mean"}


def _profile_of(stem: str) -> str:
    for pref in _PHASE_PREFIXES:
        if stem.startswith(pref):
            return stem[len(pref):]
    return stem


# --------------------------------------------------------------------------- #
# Locating the primary data behind a table                                     #
# --------------------------------------------------------------------------- #
def _load_parquets(preds_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(preds_dir.glob("*.parquet")):
        try:
            out[path.stem] = pd.read_parquet(path)
        except Exception as exc:                       # noqa: BLE001
            log.warning("could not read %s: %s", path.name, exc)
    return out


def _series_for(model: str, stems: list[str], parquets: dict[str, pd.DataFrame]):
    """The forecast series for *model*, searched over ``stems`` in order.

    Returns ``(series, y_true, stem)`` or ``None``. Searching several parquets is
    what lets the Phase-7 master table be checked at all: it assembles models
    from Phases 5, 6 and 7, and an assembly bug is precisely the failure this
    gate exists to catch.
    """
    for stem in stems:
        frame = parquets.get(stem)
        if frame is None or model not in frame.columns or ACTUAL_COL not in frame.columns:
            continue
        pair = frame[[ACTUAL_COL, model]].dropna()
        if len(pair):
            return pair[model].to_numpy(float), pair[ACTUAL_COL].to_numpy(float), stem
    return None


# --------------------------------------------------------------------------- #
# One comparison                                                               #
# --------------------------------------------------------------------------- #
def _check(rows: list, *, table: str, item: str, quantity: str,
           published, rederived, source: str, rtol: float = RTOL) -> bool:
    """Record one published-vs-re-derived comparison; return whether it passed.

    ``rel_diff`` is measured **against the published value** -- "how far off is
    the number in the table" -- so a table 5% high against a correct re-derivation
    reports 0.05/1.05 = 0.0476, not 0.05. The pass/fail decision uses
    ``np.isclose``'s own tolerance and does not depend on this choice.
    """
    try:
        pub = float(published)
        red = float(rederived)
    except (TypeError, ValueError):
        rows.append({"table": table, "item": item, "quantity": quantity,
                     "published": published, "rederived": rederived,
                     "abs_diff": np.nan, "rel_diff": np.nan, "source": source,
                     "status": "SKIP", "note": "not numeric"})
        return True
    if not (np.isfinite(pub) and np.isfinite(red)):
        rows.append({"table": table, "item": item, "quantity": quantity,
                     "published": pub, "rederived": red, "abs_diff": np.nan,
                     "rel_diff": np.nan, "source": source, "status": "SKIP",
                     "note": "non-finite"})
        return True
    diff = abs(pub - red)
    rel = diff / abs(pub) if pub else diff
    ok = bool(np.isclose(pub, red, rtol=rtol, atol=ATOL))
    rows.append({"table": table, "item": item, "quantity": quantity,
                 "published": pub, "rederived": red, "abs_diff": diff,
                 "rel_diff": rel, "source": source,
                 "status": "OK" if ok else "MISMATCH", "note": ""})
    return ok


# --------------------------------------------------------------------------- #
# The checks                                                                   #
# --------------------------------------------------------------------------- #
def check_point_metric_tables(tables: Path, parquets: dict, rows: list) -> None:
    """Every ``*_metrics.csv``: recompute each model's point losses."""
    for path in sorted(tables.glob("*_metrics.csv")):
        stem = path.name[: -len("_metrics.csv")]
        met = pd.read_csv(path)
        # Most metrics tables are one row per model with the model in column 0.
        # The RQ4 table is one row per (asset, model) and names its own source
        # profile, so the parquet is resolved per row rather than from the stem.
        model_col = "model" if "model" in met.columns else met.columns[0]
        for rec in met.to_dict("records"):
            model = str(rec[model_col])
            if "profile" in rec and pd.notna(rec["profile"]):
                stems = [f"m05_regime_dl_{rec['profile']}", f"m02_{rec['profile']}",
                         f"m03_lstm_{rec['profile']}"]
            else:
                stems = [stem]
            found = _series_for(model, stems, parquets)
            if found is None:
                rows.append({"table": path.name, "item": model, "quantity": "-",
                             "published": np.nan, "rederived": np.nan,
                             "abs_diff": np.nan, "rel_diff": np.nan,
                             "source": ", ".join(stems),
                             "status": "SKIP" if model in DERIVED_ROWS else "MISSING",
                             "note": ("derived row: one forecast read a second way, "
                                      "no persisted column"
                                      if model in DERIVED_ROWS
                                      else "no forecast column for this model")})
                continue
            h, y, src = found
            if "n" in rec:
                _check(rows, table=path.name, item=model, quantity="n",
                       published=rec["n"], rederived=len(y), source=f"{src}.parquet")
            for q, fn in PointFn.items():
                if q in rec:
                    _check(rows, table=path.name, item=model, quantity=q,
                           published=rec[q], rederived=fn(y, h),
                           source=f"{src}.parquet")


def check_master_tables(tables: Path, parquets: dict, rows: list) -> None:
    """The Phase-7 master table, whose models come from three different phases.

    This is the check that would have caught the 2026-09-01 CRPS defect: the
    table is assembled across parquets, and an assembly that reads the wrong
    frame produces plausible numbers attached to the wrong model.
    """
    for path in sorted(tables.glob("*_master.csv")):
        stem = path.name[: -len("_master.csv")]
        # Models may live in this profile's own parquet or in the phases it
        # inherits from; search the specific ones first, then anything sharing
        # the profile suffix.
        profile = _profile_of(stem)
        candidates = [stem] + [s for s in parquets
                               if _profile_of(s) == profile and s != stem]
        master = pd.read_csv(path)
        for rec in master.to_dict("records"):
            model = str(rec.get("model", ""))
            found = _series_for(model, candidates, parquets)
            if found is None:
                rows.append({"table": path.name, "item": model, "quantity": "-",
                             "published": np.nan, "rederived": np.nan,
                             "abs_diff": np.nan, "rel_diff": np.nan,
                             "source": ", ".join(candidates),
                             "status": "SKIP" if model in DERIVED_ROWS else "MISSING",
                             "note": ("derived row: the quantile head's own forecast "
                                      "retransformed to the mean scale, not a "
                                      "persisted series"
                                      if model in DERIVED_ROWS
                                      else "model column not found in any parquet")})
                continue
            h, y, src = found
            if "n" in rec:
                _check(rows, table=path.name, item=model, quantity="n",
                       published=rec["n"], rederived=len(y), source=f"{src}.parquet")
            for q, fn in PointFn.items():
                if q in rec and pd.notna(rec[q]):
                    _check(rows, table=path.name, item=model, quantity=q,
                           published=rec[q], rederived=fn(y, h),
                           source=f"{src}.parquet")


def check_interval_tables(tables: Path, parquets: dict, rows: list) -> None:
    """Every ``*_calibration.csv``: recompute coverage, width and Winkler."""
    for path in sorted(tables.glob("*_calibration.csv")):
        stem = path.name[: -len("_calibration.csv")]
        cal = pd.read_csv(path)
        profile = _profile_of(stem)
        candidates = [stem] + [s for s in parquets
                               if _profile_of(s) == profile and s != stem]
        for rec in cal.to_dict("records"):
            method = str(rec["method"])
            level = float(rec["nominal"])
            tag = f"{int(round(level * 100))}"
            lo_c, hi_c = f"{method}_lo{tag}", f"{method}_hi{tag}"
            frame = next((parquets[s] for s in candidates
                          if s in parquets and lo_c in parquets[s].columns), None)
            item = f"{method}@{tag}"
            if frame is None:
                rows.append({"table": path.name, "item": item, "quantity": "-",
                             "published": np.nan, "rederived": np.nan,
                             "abs_diff": np.nan, "rel_diff": np.nan,
                             "source": ", ".join(candidates), "status": "MISSING",
                             "note": f"no {lo_c}/{hi_c} columns"})
                continue
            sub = frame[[ACTUAL_COL, lo_c, hi_c]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            y = sub[ACTUAL_COL].to_numpy(float)
            lo = sub[lo_c].to_numpy(float)
            hi = sub[hi_c].to_numpy(float)
            src = next(s for s in candidates if s in parquets
                       and lo_c in parquets[s].columns)
            if "n" in rec:
                _check(rows, table=path.name, item=item, quantity="n",
                       published=rec["n"], rederived=len(y), source=f"{src}.parquet")
            for q, val in (("picp", _picp(y, lo, hi)),
                           ("mpiw", _mpiw(lo, hi)),
                           ("winkler", _winkler(y, lo, hi, level)),
                           ("coverage_error", _picp(y, lo, hi) - level)):
                if q in rec:
                    _check(rows, table=path.name, item=item, quantity=q,
                           published=rec[q], rederived=val, source=f"{src}.parquet")


def _regime_source(stem: str, parquets: dict):
    """The Phase-4 parquet and causal state column behind a profile.

    Two things this resolves. A ``_hmm`` profile is the **same** Phase-4 file read
    through the Baum-Welch state column rather than the jump-penalised one -- it
    has no Phase-4 parquet of its own, so looking for one left the whole
    comparator profile unverified. And a profile whose regimes were estimated for
    it (the Phase-8 assets) has its own file under its own name.
    """
    profile = _profile_of(stem)
    comparator = profile.endswith("_hmm")
    base = profile[: -len("_hmm")] if comparator else profile
    reg = parquets.get(f"m04_regimes_{base}")
    if reg is None:
        return None, None
    preferred = "hmm_filt_state" if comparator else "jphmm_filt_state"
    col = next((c for c in (preferred, "jphmm_filt_state", "hmm_filt_state")
                if c in reg.columns), None)
    return (reg, col) if col else (None, None)


def _per_regime_qlike(rows, *, table, preds, lagged, labels, regime, model,
                      published, item, source) -> None:
    """One per-regime cell: recompute QLIKE on the ``t-1`` bucket."""
    if model not in preds.columns or pd.isna(published):
        return
    mask = (np.ones(len(preds), dtype=bool) if regime == "all"
            else (lagged == float(labels.index(regime))).to_numpy())
    sub = preds.loc[mask, [ACTUAL_COL, model]].dropna()
    if not len(sub):
        return
    _check(rows, table=table, item=item, quantity="qlike", published=published,
           rederived=_qlike(sub[ACTUAL_COL].to_numpy(float),
                            sub[model].to_numpy(float)), source=source)


def check_per_regime_tables(tables: Path, parquets: dict, rows: list) -> None:
    """Every per-regime point-loss table, in all three shapes it is written in.

    The regime series always comes from the **Phase-4** parquet, never from the
    copy inside the predictions file: the lag has to reach back across the start
    of the evaluation window or the first day drops out of every bucket and the
    per-regime numbers stop reconciling with the pooled ones (the 2026-08-19 (iv)
    defect, which published two values for one cell).

    Three shapes exist and all three are checked, because a value published in
    two shapes is a value two artefacts can disagree about:

    * **wide** -- one row per regime, one column per model (``m05``/``m07``);
    * **pairwise** -- ``model_a``/``model_b`` with ``qlike_a``/``qlike_b``, the
      Giacomini-White per-regime table;
    * **cross-asset pairwise** -- the same, with an ``asset`` column (Phase 8),
      where each row resolves its own asset's parquets.
    """
    for path in sorted(tables.glob("*per_regime*.csv")):
        if path.name.endswith("_expost.csv"):
            continue          # the contemporaneous-label table; t-1 is the convention
        per = pd.read_csv(path)
        if "regime" not in per.columns:
            continue
        if "picp" in per.columns:
            continue          # interval-shaped: check_per_regime_interval_tables
        stem = path.name[: -len(".csv")]
        for suffix in ("_gw_per_regime", "_per_regime"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        pairwise = {"model_a", "model_b"} <= set(per.columns)
        cross_asset = "asset" in per.columns and "profile" in per.columns

        # Resolve the sources once per (profile) group.
        def _sources(profile_stem: str):
            preds = parquets.get(profile_stem)
            reg, col = _regime_source(profile_stem, parquets)
            if preds is None or reg is None:
                return None
            return preds, _lag_state(reg[col].astype(float), preds.index), col

        groups: dict[str, pd.DataFrame] = {}
        if cross_asset:
            for prof, grp in per.groupby("profile"):
                groups[f"m05_regime_dl_{prof}"] = grp
        else:
            groups[stem] = per

        for profile_stem, grp in groups.items():
            src = _sources(profile_stem)
            if src is None:
                rows.append({"table": path.name, "item": profile_stem,
                             "quantity": "-", "published": np.nan,
                             "rederived": np.nan, "abs_diff": np.nan,
                             "rel_diff": np.nan,
                             "source": f"{profile_stem}.parquet",
                             "status": "MISSING",
                             "note": "predictions or Phase-4 regime parquet not found"})
                continue
            preds, lagged, state_col = src
            labels = [r for r in dict.fromkeys(grp["regime"].astype(str)) if r != "all"]
            source = f"{profile_stem}.parquet x {state_col} (t-1)"

            for rec in grp.to_dict("records"):
                if "regime_shift" in rec and int(rec["regime_shift"]) != 1:
                    continue                     # only the t-1 convention
                regime = str(rec["regime"])
                if "n" in rec and pd.notna(rec["n"]):
                    n_red = (len(preds) if regime == "all"
                             else int((lagged == float(labels.index(regime))).sum()))
                    _check(rows, table=path.name,
                           item=f"{profile_stem}/{regime}", quantity="n",
                           published=rec["n"], rederived=n_red, source=source)
                if pairwise:
                    for side in ("a", "b"):
                        _per_regime_qlike(
                            rows, table=path.name, preds=preds, lagged=lagged,
                            labels=labels, regime=regime,
                            model=str(rec[f"model_{side}"]),
                            published=rec.get(f"qlike_{side}"),
                            item=f"{profile_stem}/{regime}/{rec[f'model_{side}']}",
                            source=source)
                else:
                    for model in [c for c in grp.columns
                                  if c not in ("regime_shift", "timing", "regime",
                                               "n", "asset", "profile")]:
                        _per_regime_qlike(
                            rows, table=path.name, preds=preds, lagged=lagged,
                            labels=labels, regime=regime, model=model,
                            published=rec.get(model),
                            item=f"{profile_stem}/{regime}/{model}", source=source)


#: Which method each per-regime interval table describes. The suffix is the only
#: place this is recorded, so the map lives here -- and a wrong entry makes the
#: check FAIL rather than pass quietly, because the numbers then come from the
#: wrong method's bounds.
PER_REGIME_INTERVAL_METHOD = {"mc": "MC-Dropout-LSTM", "q": "Quantile-LSTM"}


def check_per_regime_interval_tables(tables: Path, parquets: dict, rows: list) -> None:
    """``*_per_regime_{mc,q}.csv``: coverage, width and Winkler by ``t-1`` state.

    This is RQ3's central table -- "does the band's coverage depend on the market
    state?" -- so it is worth re-deriving from the persisted bounds rather than
    trusting the runner that wrote it. The level is the headline 90%: it is the
    only band both methods produce, which is why Chapter 3 makes it the headline.
    """
    level = 0.90
    tag = "90"
    for path in sorted(tables.glob("*_per_regime_*.csv")):
        if path.name.endswith("_expost.csv"):
            continue
        per = pd.read_csv(path)
        if "picp" not in per.columns or "regime" not in per.columns:
            continue
        stem, _, suffix = path.name[: -len(".csv")].rpartition("_per_regime_")
        method = PER_REGIME_INTERVAL_METHOD.get(suffix)
        if method is None:
            rows.append({"table": path.name, "item": suffix, "quantity": "-",
                         "published": np.nan, "rederived": np.nan,
                         "abs_diff": np.nan, "rel_diff": np.nan, "source": "-",
                         "status": "MISSING",
                         "note": f"no method mapped to the '{suffix}' suffix"})
            continue
        preds = parquets.get(stem)
        reg, state_col = _regime_source(stem, parquets)
        lo_c, hi_c = f"{method}_lo{tag}", f"{method}_hi{tag}"
        if preds is None or reg is None or lo_c not in (preds.columns if preds is not None else []):
            rows.append({"table": path.name, "item": method, "quantity": "-",
                         "published": np.nan, "rederived": np.nan,
                         "abs_diff": np.nan, "rel_diff": np.nan,
                         "source": f"{stem}.parquet", "status": "MISSING",
                         "note": f"no {lo_c}/{hi_c} or no Phase-4 regime series"})
            continue
        lagged = _lag_state(reg[state_col].astype(float), preds.index)
        labels = [r for r in dict.fromkeys(per["regime"].astype(str)) if r != "all"]
        source = f"{stem}.parquet {lo_c}/{hi_c} x {state_col} (t-1)"
        for rec in per.to_dict("records"):
            if "regime_shift" in rec and int(rec["regime_shift"]) != 1:
                continue
            regime = str(rec["regime"])
            mask = (np.ones(len(preds), dtype=bool) if regime == "all"
                    else (lagged == float(labels.index(regime))).to_numpy())
            sub = preds.loc[mask, [ACTUAL_COL, lo_c, hi_c]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if not len(sub):
                continue
            y = sub[ACTUAL_COL].to_numpy(float)
            lo = sub[lo_c].to_numpy(float)
            hi = sub[hi_c].to_numpy(float)
            item = f"{method}/{regime}"
            if "n" in rec:
                _check(rows, table=path.name, item=item, quantity="n",
                       published=rec["n"], rederived=len(y), source=source)
            for q, val in (("picp", _picp(y, lo, hi)),
                           ("mpiw", _mpiw(lo, hi)),
                           ("winkler", _winkler(y, lo, hi, level)),
                           ("coverage_error", _picp(y, lo, hi) - level)):
                if q in rec:
                    _check(rows, table=path.name, item=item, quantity=q,
                           published=rec[q], rederived=val, source=source)


def check_cross_table_agreement(tables: Path, rows: list) -> None:
    """A model's QLIKE must be the same wherever it is published.

    Two tables disagreeing about one model is the shape of every artefact defect
    this project has had: a value assembled from one input while a second bears
    on it, with nothing comparing the two.
    """
    seen: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for path in sorted(tables.glob("*_metrics.csv")) + sorted(tables.glob("*_master.csv")):
        df = pd.read_csv(path)
        col = "model" if "model" in df.columns else df.columns[0]
        if "qlike" not in df.columns:
            continue
        stem = path.name.replace("_metrics.csv", "").replace("_master.csv", "")
        table_profile = _profile_of(stem)
        for rec in df.to_dict("records"):
            if pd.isna(rec["qlike"]):
                continue
            # A cross-asset table (m08_rq4_metrics.csv) holds one row per
            # (asset, model), so the same model legitimately carries three
            # different values. Keying on the model alone reported that as a
            # disagreement -- the comparison unit is the *forecast*, which is
            # identified by profile and model together.
            profile = str(rec.get("profile") or rec.get("asset") or table_profile)
            seen.setdefault((profile, str(rec[col])), []).append(
                (path.name, float(rec["qlike"])))
    for (profile, model), entries in sorted(seen.items()):
        if len(entries) < 2:
            continue
        base_name, base_val = entries[0]
        for name, val in entries[1:]:
            if base_name == name:
                continue          # one table cannot disagree with itself
            _check(rows, table=f"{base_name} vs {name}", item=f"{profile}/{model}",
                   quantity="qlike agreement", published=base_val, rederived=val,
                   source="cross-table")


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def write_report(result: pd.DataFrame, out_path: Path, *, rtol: float) -> Path:
    """The verification note. Generated entirely from ``result`` -- no branch
    states a verdict the table does not carry."""
    ensure_dir(out_path.parent)
    n = len(result)
    counts = result["status"].value_counts().to_dict()
    bad = result[result["status"].isin(("MISMATCH", "MISSING"))]

    L = [f"# m09 - Reproducibility gate\n",
         f"\n_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
         f"Relative tolerance {rtol:g}._\n"]
    L.append(
        "\nEvery number below was recomputed **from the prediction parquets** by "
        "code that imports nothing from `src.evaluation`, and compared with what "
        "the published table says. The duplication of the metric formulas is "
        "deliberate: calling the project's own functions would check that a CSV "
        "matches what those functions return today, and agree enthusiastically "
        "with a regression in the functions themselves.\n"
    )

    if len(bad) == 0:
        L.append(f"\n## PASS - {n} values, all re-derive\n")
        L.append("\nEvery published point metric, interval metric, per-regime "
                 "loss and sample size reproduces from the primary data, and "
                 "every model carrying a value in more than one table carries "
                 "the same value in all of them.\n")
    else:
        L.append(f"\n## FAIL - {len(bad)} of {n} values do not re-derive\n\n")
        L.append("| table | item | quantity | published | re-derived | rel. diff | status |")
        L.append("\n|---|---|---|---|---|---|---|\n")
        for r in bad.to_dict("records"):
            L.append(f"| {r['table']} | {r['item']} | {r['quantity']} | "
                     f"{r['published']:.10g} | {r['rederived']:.10g} | "
                     f"{r['rel_diff']:.3g} | **{r['status']}** |\n")
        L.append("\nA MISMATCH means the table and the data disagree: either the "
                 "artefact is stale, or the code that wrote it changed. A MISSING "
                 "means a published row has no forecast column behind it at all, "
                 "which is worse.\n")

    L.append("\n## Coverage\n\n")
    L.append("| status | values |\n|---|---|\n")
    for k in sorted(counts):
        L.append(f"| {k} | {counts[k]} |\n")

    L.append("\n### By table\n\n| table | checked | mismatched |\n|---|---|---|\n")
    for tbl, grp in result.groupby("table", sort=True):
        L.append(f"| {tbl} | {len(grp)} | "
                 f"{int((grp['status'] == 'MISMATCH').sum())} |\n")

    L.append("\n## What this gate does not certify\n")
    L.append(
        "\nTest statistics (Diebold-Mariano, Giacomini-White, Model Confidence "
        "Set, Kupiec, Christoffersen, dynamic quantile) are **not** recomputed "
        "here. Their values depend on HAC bandwidth rules, finite-sample "
        "corrections and bootstrap draws whose conventions are this project's own "
        "choices, so a second implementation would be reimplementing those "
        "choices rather than checking them. They are covered by "
        "`tests/test_significance.py` against closed-form cases; what this gate "
        "certifies for them is that they were computed on the right sample from "
        "forecasts that do re-derive.\n"
    )
    out_path.write_text("".join(L), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def verify(tables: Path, preds_dir: Path, *, rtol: float = RTOL) -> pd.DataFrame:
    parquets = _load_parquets(preds_dir)
    log.info("loaded %d prediction parquets", len(parquets))
    rows: list = []
    check_point_metric_tables(tables, parquets, rows)
    check_master_tables(tables, parquets, rows)
    check_interval_tables(tables, parquets, rows)
    check_per_regime_tables(tables, parquets, rows)
    check_per_regime_interval_tables(tables, parquets, rows)
    check_cross_table_agreement(tables, rows)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.experiments.verify_claims")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--predictions", default="results/predictions")
    ap.add_argument("--rtol", type=float, default=RTOL,
                    help="relative tolerance for a re-derived value")
    ap.add_argument("--out", default="results/tables/m09_verification.csv")
    ap.add_argument("--report", default="results/milestones/m09_verification.md")
    args = ap.parse_args(argv)

    tables = repo_path(args.tables)
    preds = repo_path(args.predictions)
    for p in (tables, preds):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    result = verify(tables, preds, rtol=args.rtol)
    if not len(result):
        raise SystemExit("nothing to verify -- no published tables found")

    out = repo_path(args.out)
    ensure_dir(out.parent)
    result.to_csv(out, index=False)
    report = write_report(result, repo_path(args.report), rtol=args.rtol)

    bad = result[result["status"].isin(("MISMATCH", "MISSING"))]
    n_ok = int((result["status"] == "OK").sum())
    log.info("checked %d values across %d tables: %d OK, %d skipped, %d failed",
             len(result), result["table"].nunique(), n_ok,
             int((result["status"] == "SKIP").sum()), len(bad))
    log.info("detail -> %s | report -> %s", out, report)

    if len(bad):
        for r in bad.head(20).to_dict("records"):
            log.error("%s | %s | %s: published %r, re-derived %r (%s)",
                      r["table"], r["item"], r["quantity"], r["published"],
                      r["rederived"], r["status"])
        log.error("REPRODUCIBILITY GATE FAILED: %d value(s) do not re-derive "
                  "from the prediction parquets.", len(bad))
        return 1
    log.info("REPRODUCIBILITY GATE PASSED: every published value re-derives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
