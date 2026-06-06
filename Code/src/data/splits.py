"""Walk-forward / anchored split index generators (roadmap §1.3, §4).

Leakage discipline is the single biggest threat to dissertation credibility, so
these helpers exist to be:

1. Pure index manipulations (no model code) so they are trivially testable.
2. Deterministic given the same input index and config.
3. Cheap to call inside a tight refit loop.

Two split modes are supported:

* ``"anchored"`` -- training window grows: ``[t0, t]`` for increasing ``t``.
* ``"rolling"``  -- training window is a fixed length that slides forward.

In both modes a calendar of *refit boundaries* is emitted at the configured
cadence (default monthly, i.e. every ~21 trading days). Each fold reports the
train slice, the prediction window, and the boundary date itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd

SplitScheme = Literal["anchored", "rolling"]


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    fold_idx: int
    refit_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp        # inclusive
    predict_start: pd.Timestamp    # first OOS date predicted with this fit
    predict_end: pd.Timestamp      # inclusive

    def train_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.train_start) & (index <= self.train_end)

    def predict_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.predict_start) & (index <= self.predict_end)


@dataclass(frozen=True)
class SplitConfig:
    train_end: str | pd.Timestamp
    val_start: str | pd.Timestamp
    val_end: str | pd.Timestamp
    test_start: str | pd.Timestamp
    test_end: str | pd.Timestamp
    refit_frequency_days: int = 21
    scheme: SplitScheme = "anchored"
    min_train_size: int = 252  # require >= 1 trading year before issuing any prediction

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.refit_frequency_days < 1:
            raise ValueError("refit_frequency_days must be >= 1")
        if self.scheme not in ("anchored", "rolling"):
            raise ValueError(f"unknown scheme: {self.scheme!r}")
        # Logical order check.
        bounds = [
            pd.Timestamp(self.train_end),
            pd.Timestamp(self.val_start),
            pd.Timestamp(self.val_end),
            pd.Timestamp(self.test_start),
            pd.Timestamp(self.test_end),
        ]
        for a, b in zip(bounds, bounds[1:]):
            if a > b:
                raise ValueError(f"split bounds must be monotone non-decreasing; saw {a} > {b}")


def _coerce_index(index: pd.DatetimeIndex | pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(index, (pd.Series, pd.DataFrame)):
        idx = index.index
    else:
        idx = index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    if idx.has_duplicates:
        raise ValueError("index must be unique (deduplicate before splitting)")
    return idx


def fixed_three_way_split(
    index: pd.DatetimeIndex | pd.Series | pd.DataFrame, cfg: SplitConfig
) -> dict[str, pd.DatetimeIndex]:
    """Return the simple, dissertation-level train/val/test partition.

    This is **not** the walk-forward split (use ``walk_forward_folds`` for that)
    -- it is the high-level partition used for hyperparameter selection and for
    declaring which dates the model is ever allowed to see at each phase.
    """
    idx = _coerce_index(index)
    train = idx[idx <= pd.Timestamp(cfg.train_end)]
    val = idx[(idx >= pd.Timestamp(cfg.val_start)) & (idx <= pd.Timestamp(cfg.val_end))]
    test = idx[(idx >= pd.Timestamp(cfg.test_start)) & (idx <= pd.Timestamp(cfg.test_end))]
    return {"train": train, "val": val, "test": test}


def walk_forward_folds(
    index: pd.DatetimeIndex | pd.Series | pd.DataFrame,
    cfg: SplitConfig,
    *,
    eval_segment: Literal["test", "val", "val+test"] = "test",
) -> Iterator[Fold]:
    """Yield successive folds spanning the requested out-of-sample segment.

    Each fold's training window ends one trading day **before** the fold's
    ``predict_start`` -- this is the leakage-discipline rule from §1.3 of the
    roadmap: the model can only see data strictly prior to the day it forecasts.

    ``eval_segment`` selects which dates accumulate predictions:
      - ``"test"``      : 2022-2024 only (the headline OOS window).
      - ``"val"``       : 2020-2021 only (hyperparameter selection).
      - ``"val+test"``  : both, contiguous, with no gap.
    """
    idx = _coerce_index(index)

    if eval_segment == "test":
        oos_start, oos_end = pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.test_end)
    elif eval_segment == "val":
        oos_start, oos_end = pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.val_end)
    elif eval_segment == "val+test":
        oos_start, oos_end = pd.Timestamp(cfg.val_start), pd.Timestamp(cfg.test_end)
    else:
        raise ValueError(f"unknown eval_segment: {eval_segment!r}")

    oos_dates = idx[(idx >= oos_start) & (idx <= oos_end)]
    if len(oos_dates) == 0:
        return

    rf = cfg.refit_frequency_days
    # Refit at the start of the OOS window and every rf trading days thereafter.
    boundary_positions = list(range(0, len(oos_dates), rf))
    # Ensure the final fold reaches the end of the OOS window.
    if boundary_positions[-1] != len(oos_dates) - 1:
        boundary_positions.append(len(oos_dates))

    for fold_idx, b_pos in enumerate(boundary_positions[:-1]):
        next_pos = boundary_positions[fold_idx + 1]
        predict_start = oos_dates[b_pos]
        predict_end = oos_dates[next_pos - 1] if next_pos <= len(oos_dates) - 1 else oos_dates[-1]

        # Training data ends on the trading day before predict_start.
        train_end_pos = idx.searchsorted(predict_start) - 1
        if train_end_pos < 0:
            continue
        train_end_date = idx[train_end_pos]

        if cfg.scheme == "anchored":
            train_start_date = idx[0]
        else:  # rolling
            # Train length is fixed at the anchored length up to the first OOS date,
            # which gives the model the same look-back every fold.
            initial_train_len = int(idx.searchsorted(oos_start))
            start_pos = max(0, train_end_pos - initial_train_len + 1)
            train_start_date = idx[start_pos]

        train_len = idx.searchsorted(train_end_date, side="right") - idx.searchsorted(
            train_start_date
        )
        if train_len < cfg.min_train_size:
            continue

        yield Fold(
            fold_idx=fold_idx,
            refit_date=predict_start,
            train_start=train_start_date,
            train_end=train_end_date,
            predict_start=predict_start,
            predict_end=predict_end,
        )
