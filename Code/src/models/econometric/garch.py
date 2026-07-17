"""GARCH-family baselines via the ``arch`` package (roadmap Phase 2).

This module owns the shared machinery for every ``arch``-backed volatility model
(:class:`ArchVolForecaster`) and the concrete GARCH(1,1) baseline
(:class:`GARCHForecaster`). EGARCH lives in ``egarch.py`` but reuses this base.

The forecasting protocol (verified against ``arch`` 8.x)
-------------------------------------------------------
Within one walk-forward fold the model must produce a *daily, 1-step-ahead*
conditional-variance forecast for every date in the fold's prediction window,
while its parameters are estimated **once** on the fold's training window
(monthly refit, roadmap §1.3). We achieve this with ``arch``'s ``last_obs`` /
``forecast`` mechanism:

1. Returns are scaled by ``scale`` (default 100 -> percent). ``arch`` documents
   that returns near unit scale are numerically ill-conditioned; percent returns
   are the recommended input. The resulting variance forecast is later divided by
   ``scale**2`` to return to decimal-variance units that match the RV target.
2. ``arch_model(...).fit(last_obs=predict_start)`` estimates parameters using
   only observations *before* ``predict_start`` -- i.e. the training window --
   while retaining the later observations for filtering.
3. ``res.forecast(horizon=1, start=train_end)`` then rolls the conditional-
   variance recursion forward with **fixed parameters**, updating it with each
   realized return. The forecast whose *origin* is date ``t`` is the variance of
   ``t + 1``; we therefore map every origin to the next trading day to align the
   forecast with the date it predicts. Starting the forecast at ``train_end``
   (the last in-sample date) makes the first forecast land exactly on
   ``predict_start`` with no gap and no leakage.

Because parameters are frozen between refits but the variance recursion still
ingests realized returns, the forecast varies day-to-day within a fold -- the
standard "fixed-window" out-of-sample GARCH forecast.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data.splits import Fold

# ``arch`` emits convergence / data-scale warnings that are noise in a monthly
# refit loop; we silence them locally rather than globally.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from arch import arch_model

RETURN_COL = "log_return"


class ArchVolForecaster:
    """Base class for ``arch``-backed conditional-variance forecasters.

    Sub-classes fix the volatility process by setting ``vol`` and ``arch_kwargs``
    (e.g. ``vol="GARCH", arch_kwargs={"p": 1, "q": 1}``). Everything else -- the
    fit/forecast/rescale/align cycle -- is shared.

    Parameters
    ----------
    name : label used as the model's column in the predictions frame.
    vol : ``arch`` volatility process name ("GARCH", "EGARCH", ...).
    arch_kwargs : order arguments passed to ``arch_model`` (p, o, q).
    mean : conditional-mean model. "Constant" is standard for daily equity
        returns; the mean has negligible effect on the 1-step variance path.
    dist : innovation distribution. "normal" gives the canonical Bollerslev
        (1986) baseline; "t" (Student-t) is available as a fat-tailed robustness
        variant motivated by Chapter 2 §2.2 (Gaussian innovations understate tail
        risk) and can be selected from config without code changes.
    scale : multiplier applied to returns before fitting (100 -> percent).
    return_col : name of the decimal log-return column in the modelling frame.
    """

    def __init__(
        self,
        name: str,
        *,
        vol: str,
        arch_kwargs: dict,
        mean: str = "Constant",
        dist: str = "normal",
        scale: float = 100.0,
        return_col: str = RETURN_COL,
    ) -> None:
        self.name = name
        self.vol = vol
        self.arch_kwargs = dict(arch_kwargs)
        self.mean = mean
        self.dist = dist
        self.scale = float(scale)
        self.return_col = return_col
        # Diagnostics captured per fold (persistence, convergence) for the
        # milestone note. Keyed by fold index.
        self.fold_params_: dict[int, dict[str, float]] = {}

    # -- internal helpers ---------------------------------------------------- #

    def _returns(self, frame: pd.DataFrame) -> pd.Series:
        if self.return_col not in frame.columns:
            raise KeyError(f"{self.name}: frame missing return column {self.return_col!r}")
        r = frame[self.return_col].astype(float).dropna()
        return (r * self.scale).rename("ret")

    def _persistence(self, params: pd.Series) -> float:
        """Sum of ARCH+GARCH mass -- how close the process is to a unit root.

        For GARCH this is alpha+beta; for EGARCH the persistence is the beta on
        log-variance. We report whatever persistence parameters are present.
        """
        keys = [k for k in params.index if k.startswith(("alpha", "beta", "gamma"))]
        # Persistence is conventionally alpha+beta (GARCH) or beta (EGARCH log-var).
        if any(k.startswith("beta") for k in keys):
            beta = sum(v for k, v in params.items() if k.startswith("beta"))
            alpha = sum(v for k, v in params.items() if k.startswith("alpha"))
            return float(alpha + beta) if self.vol.upper() == "GARCH" else float(beta)
        return float("nan")

    # -- public API ---------------------------------------------------------- #

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        y = self._returns(frame)
        full_idx = y.index

        # Slice up to the end of the prediction window: params come from training
        # (via last_obs) and the tail is used only to filter the variance forward.
        y_fold = y.loc[: fold.predict_end]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                y_fold, mean=self.mean, vol=self.vol, dist=self.dist, **self.arch_kwargs
            )
            res = am.fit(last_obs=fold.predict_start, disp="off", show_warning=False)
            # Origin = train_end so the first 1-step forecast lands on predict_start.
            fc = res.forecast(horizon=1, start=fold.train_end, reindex=False)

        self.fold_params_[fold.fold_idx] = {
            **{k: float(v) for k, v in res.params.items()},
            "persistence": self._persistence(res.params),
            "converged": bool(getattr(res, "convergence_flag", 0) == 0),
        }

        var_scaled = fc.variance["h.1"]  # indexed by origin date; value = Var(origin+1)

        # Map each origin to the next trading day (the date it forecasts) using
        # integer positions on the full return index -- robust to calendar gaps.
        origin_pos = full_idx.get_indexer(var_scaled.index)
        target_pos = origin_pos + 1
        keep = target_pos < len(full_idx)
        target_dates = full_idx[target_pos[keep]]
        values = var_scaled.to_numpy()[keep] / (self.scale**2)  # -> decimal variance

        pred = pd.Series(values, index=target_dates, name=self.name)
        return pred.loc[fold.predict_start : fold.predict_end]


class GARCHForecaster(ArchVolForecaster):
    """GARCH(1,1) -- Bollerslev (1986), the primary econometric baseline.

    Conditional variance ``h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}``.
    Two parameters with a clean interpretation: ``alpha`` is the short-run
    shock response and ``beta`` the persistence of past variance. It is included
    not for sophistication but because it is the floor every alternative must
    clear (Hansen & Lunde 2005; Chapter 2 §2.2).
    """

    def __init__(
        self,
        *,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        dist: str = "normal",
        scale: float = 100.0,
        name: str = "GARCH",
        return_col: str = RETURN_COL,
    ) -> None:
        super().__init__(
            name,
            vol="GARCH",
            arch_kwargs={"p": p, "q": q},
            mean=mean,
            dist=dist,
            scale=scale,
            return_col=return_col,
        )
