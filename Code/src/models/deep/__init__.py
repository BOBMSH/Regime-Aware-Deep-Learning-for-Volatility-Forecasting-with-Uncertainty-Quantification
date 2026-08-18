"""Deep-learning models (roadmap Phases 3+).

Phase 3 introduces the vanilla LSTM baseline
(:class:`~src.models.deep.lstm.LSTMForecaster`) -- the non-linear sequence model
that Chapter 2 §2.5 motivates as the deep counterpart to the econometric
baselines. It satisfies the same
:class:`src.evaluation.rolling.VolForecaster` protocol as the Phase 2 models, so
it drops straight into the shared walk-forward engine and is scored on identical
splits under the same QLIKE/MSE/MAE metrics (RQ1).

Later phases add the regime-aware and uncertainty-aware variants on top of this
same architecture (regime posteriors as extra input channels; MC-Dropout /
quantile heads), which is why the network keeps a head-level dropout even in the
Phase 3 baseline.
"""

from __future__ import annotations

from src.models.deep.lstm import LSTMForecaster, LSTMRegressor
from src.models.deep.regime_lstm import RegimeExpertForecaster
from src.models.deep.uncertainty import (
    MCDropoutLSTMForecaster,
    QuantileLSTMForecaster,
    QuantileLSTMRegressor,
)

__all__ = [
    "LSTMForecaster",
    "LSTMRegressor",
    "RegimeExpertForecaster",
    "MCDropoutLSTMForecaster",
    "QuantileLSTMForecaster",
    "QuantileLSTMRegressor",
]
