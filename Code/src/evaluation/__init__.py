"""Evaluation engine (roadmap §5).

Everything a model produces -- econometric or deep -- is scored through this
package. It is intentionally decoupled from the models: a model's only
responsibility is to emit a per-date **variance forecast**; this package owns the
walk-forward loop (``rolling``), the loss functions (``metrics``), and (from
later phases) the regime and calibration breakdowns.
"""

from __future__ import annotations

from src.evaluation.calibration import (
    interval_metrics,
    lognormal_interval,
    lognormal_mean,
    lognormal_quantile,
    per_regime_interval_metrics,
    reliability_curve,
)
from src.evaluation.metrics import (
    coverage_error,
    mae,
    mean_pinball_loss,
    mpiw,
    mse,
    picp,
    pinball_loss,
    point_metrics,
    qlike,
    rmse,
    winkler_score,
)
from src.evaluation.significance import (
    diebold_mariano,
    dm_lag_sensitivity,
    dm_pairs_table,
    giacomini_white,
    ljung_box,
    loss_differential,
    model_confidence_set,
    qlike_loss,
    regime_test_function,
    sample_acf,
)

__all__ = [
    "mse",
    "rmse",
    "mae",
    "qlike",
    "point_metrics",
    "picp",
    "mpiw",
    "winkler_score",
    "coverage_error",
    "pinball_loss",
    "mean_pinball_loss",
    "diebold_mariano",
    "dm_pairs_table",
    "giacomini_white",
    "regime_test_function",
    "model_confidence_set",
    "qlike_loss",
    "loss_differential",
    "sample_acf",
    "ljung_box",
    "dm_lag_sensitivity",
    "interval_metrics",
    "lognormal_interval",
    "lognormal_quantile",
    "lognormal_mean",
    "reliability_curve",
    "per_regime_interval_metrics",
]
