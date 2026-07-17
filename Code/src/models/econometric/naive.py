"""Random-walk reference for realized variance (roadmap Phase 2 harness check).

Not a dissertation baseline -- a *sanity floor*. The random-walk-in-variance
forecast is simply "tomorrow's variance equals today's":

    hat{RV}_{t} = RV_{t-1}

Because realized volatility is highly persistent, this trivial rule is
surprisingly hard to beat and is the natural anchor for reading the other
numbers: a model that cannot beat the random walk is not learning anything, and
GARCH/HAR should both clear it comfortably on QLIKE. Including it also validates
the walk-forward engine end-to-end with a model that has no estimation step.

Implementation is free: the ``rv_d`` column of the modelling frame is already
``RV.shift(1)`` (built by ``har_lagged_rv``), so ``rv_d`` on date ``t`` *is* the
random-walk forecast for ``t``.
"""

from __future__ import annotations

import pandas as pd

from src.data.splits import Fold


class RandomWalkRVForecaster:
    """Persistence / random-walk forecast: hat{RV}_t = RV_{t-1}."""

    def __init__(self, *, daily_lag_col: str = "rv_d", name: str = "RW-RV") -> None:
        self.daily_lag_col = daily_lag_col
        self.name = name

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        if self.daily_lag_col not in frame.columns:
            raise KeyError(f"{self.name}: frame missing {self.daily_lag_col!r}")
        oos = frame.loc[fold.predict_start : fold.predict_end]
        return oos[self.daily_lag_col].astype(float).rename(self.name)
