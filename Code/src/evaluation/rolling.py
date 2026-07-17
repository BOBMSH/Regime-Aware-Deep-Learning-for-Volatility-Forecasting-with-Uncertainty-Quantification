"""Walk-forward backtest engine (roadmap §5 -- "the engine", built once).

Design contract
---------------
Every model in the dissertation, econometric or deep, is a :class:`VolForecaster`:
an object with a ``name`` and a single method

    forecast_fold(frame, fold) -> pd.Series      # variance forecasts, indexed by date

The engine owns everything else -- iterating the anchored walk-forward folds,
calling each model once per fold, stitching the per-fold forecasts into one
out-of-sample series per model, and persisting the result as a tidy
``predictions`` frame (one column per model plus the realized target). Once the
predictions frame exists, every downstream comparison (point error, Diebold-
Mariano, per-regime breakdown, calibration) is a pandas operation on that frame.
This is what lets Phases 3-6 drop in new models without touching the harness.

Leakage discipline (roadmap §1.3) is enforced structurally: the fold generator
guarantees ``fold.train_end < fold.predict_start`` and the engine re-asserts it
before every fit. A model only ever receives the full ``frame`` and a ``fold``;
it is the model's job to slice its own training window from ``fold`` -- but since
the only OOS dates that are ever collected are those inside the fold's predict
window, a model cannot leak a forecast onto a training date even if it tried.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.data.splits import Fold, SplitConfig, walk_forward_folds
from src.evaluation.metrics import point_metrics
from src.utils.logging import get_logger

log = get_logger("rolling")

# Column name used for the realized-variance target throughout the engine.
TARGET_COL = "rv"
# Column name under which the realized target is stored in the predictions frame.
ACTUAL_COL = "y_true"


@runtime_checkable
class VolForecaster(Protocol):
    """Structural type every model must satisfy to run through the engine."""

    name: str

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        """Fit on ``frame`` up to ``fold.train_end`` and return **variance**
        forecasts indexed by the dates in ``[fold.predict_start, fold.predict_end]``.
        """
        ...


def _validate_fold_output(pred: pd.Series, fold: Fold, model_name: str) -> pd.Series:
    """Guard a model's per-fold output: right dates, finite, positive-ish."""
    if not isinstance(pred, pd.Series):
        raise TypeError(f"{model_name}.forecast_fold must return a Series, got {type(pred)}")
    pred = pred.copy()
    pred.index = pd.DatetimeIndex(pred.index)
    # Any forecast dated on/before train_end would be leakage -- reject loudly.
    if (pred.index <= fold.train_end).any():
        raise AssertionError(
            f"{model_name} fold {fold.fold_idx}: forecast indexed on/before "
            f"train_end {fold.train_end.date()} -- leakage."
        )
    # Keep only dates inside the fold's prediction window.
    in_window = (pred.index >= fold.predict_start) & (pred.index <= fold.predict_end)
    return pred.loc[in_window]


def run_walk_forward(
    models: list[VolForecaster],
    frame: pd.DataFrame,
    cfg: SplitConfig,
    *,
    eval_segment: str = "test",
    target_col: str = TARGET_COL,
    strict: bool = True,
) -> pd.DataFrame:
    """Run every model across the walk-forward folds of the chosen OOS segment.

    Parameters
    ----------
    models : list of :class:`VolForecaster`.
    frame : aligned modelling frame indexed by trading date. Must contain
        ``target_col`` (realized *variance*) and whatever feature columns the
        models read (e.g. ``log_return`` for GARCH, ``rv_d/rv_w/rv_m`` for HAR).
    cfg : split configuration (defines train/val/test bounds and refit cadence).
    eval_segment : ``"test"`` (default), ``"val"`` or ``"val+test"``.
    target_col : name of the realized-variance column in ``frame``.
    strict : if True (default), a model error on any fold aborts the run. If
        False, the offending fold is logged and left as NaN so a single bad refit
        does not lose the whole backtest.

    Returns
    -------
    pd.DataFrame indexed by OOS trading date with one column per model
    (``model.name``) holding the variance forecast, plus a ``y_true`` column with
    the realized variance. This is the object persisted to
    ``results/predictions/`` and consumed by every comparison downstream.
    """
    if target_col not in frame.columns:
        raise KeyError(f"frame is missing target column {target_col!r}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index)

    folds = list(walk_forward_folds(frame.index, cfg, eval_segment=eval_segment))
    if not folds:
        raise RuntimeError("walk_forward_folds produced no folds for this frame/segment")

    oos_start, oos_end = folds[0].predict_start, folds[-1].predict_end
    log.info(
        "walk-forward: %d folds, %d models, OOS %s -> %s (%d obs), refit every %d days (%s)",
        len(folds),
        len(models),
        oos_start.date(),
        oos_end.date(),
        int(((frame.index >= oos_start) & (frame.index <= oos_end)).sum()),
        cfg.refit_frequency_days,
        cfg.scheme,
    )

    # Collect each model's forecasts fold-by-fold, then concatenate.
    per_model: dict[str, list[pd.Series]] = {m.name: [] for m in models}
    for m in models:
        t0 = time.perf_counter()
        for fold in folds:
            assert fold.train_end < fold.predict_start, "leakage: train_end >= predict_start"
            try:
                raw = m.forecast_fold(frame, fold)
                pred = _validate_fold_output(raw, fold, m.name)
            except Exception as exc:  # noqa: BLE001 -- we re-raise unless strict=False
                if strict:
                    log.error("%s failed on fold %d (%s): %s", m.name, fold.fold_idx,
                              fold.predict_start.date(), exc)
                    raise
                log.warning("%s fold %d (%s) errored, filling NaN: %s", m.name,
                            fold.fold_idx, fold.predict_start.date(), exc)
                idx = frame.loc[fold.predict_start:fold.predict_end].index
                pred = pd.Series(np.nan, index=idx)
            per_model[m.name].append(pred)
        dt = time.perf_counter() - t0
        n_pred = sum(len(s) for s in per_model[m.name])
        log.info("  %-10s done: %d forecasts in %.1fs", m.name, n_pred, dt)

    # Assemble the tidy predictions frame.
    out = pd.DataFrame(index=frame.loc[oos_start:oos_end].index)
    out.index.name = "date"
    for name, chunks in per_model.items():
        series = pd.concat(chunks).sort_index()
        series = series[~series.index.duplicated(keep="first")]
        out[name] = series.reindex(out.index)
    out[ACTUAL_COL] = frame.loc[oos_start:oos_end, target_col]

    _coverage_report(out, models)
    return out


def _coverage_report(out: pd.DataFrame, models: list[VolForecaster]) -> None:
    """Log how completely each model covered the OOS window (a leakage/omission
    tripwire: a healthy econometric run should be ~100%)."""
    n = len(out)
    for m in models:
        got = int(out[m.name].notna().sum())
        if got < n:
            log.warning("%s covered %d/%d OOS dates (%.1f%%)", m.name, got, n, 100 * got / n)


def evaluate_predictions(
    predictions: pd.DataFrame, *, actual_col: str = ACTUAL_COL
) -> pd.DataFrame:
    """Score a predictions frame: one row per model, columns = point metrics.

    Rows are sorted by QLIKE (the primary metric, Patton 2011) ascending, so the
    best model is first.
    """
    if actual_col not in predictions.columns:
        raise KeyError(f"predictions frame missing {actual_col!r}")
    y = predictions[actual_col]
    rows = {}
    for col in predictions.columns:
        if col == actual_col:
            continue
        paired = pd.concat([y, predictions[col]], axis=1).dropna()
        if paired.empty:
            continue
        rows[col] = point_metrics(paired[actual_col], paired[col])
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table = table.sort_values("qlike")
    # n as a nullable int for a clean table.
    if "n" in table.columns:
        table["n"] = table["n"].astype(int)
    return table
