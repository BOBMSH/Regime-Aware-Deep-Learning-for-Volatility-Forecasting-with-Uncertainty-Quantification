"""Evaluation engine (roadmap §5).

Everything a model produces -- econometric or deep -- is scored through this
package. It is intentionally decoupled from the models: a model's only
responsibility is to emit a per-date **variance forecast**; this package owns the
walk-forward loop (``rolling``), the loss functions (``metrics``), and (from
later phases) the regime and calibration breakdowns.
"""

from __future__ import annotations

from src.evaluation.metrics import (
    mae,
    mpiw,
    mse,
    picp,
    point_metrics,
    qlike,
    rmse,
    winkler_score,
)
from src.evaluation.significance import (
    diebold_mariano,
    dm_pairs_table,
    model_confidence_set,
    qlike_loss,
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
    "diebold_mariano",
    "dm_pairs_table",
    "model_confidence_set",
    "qlike_loss",
]
