"""HAR-RV baseline -- Corsi (2009) (roadmap Phase 2, §1.2).

The Heterogeneous AutoRegressive model of realized volatility regresses realized
variance on three lagged averages -- daily, weekly and monthly:

    RV_t = beta0 + beta_d * RV^{(d)}_{t-1} + beta_w * RV^{(w)}_{t-1}
                 + beta_m * RV^{(m)}_{t-1} + eps_t

where RV^{(d)} is yesterday's RV, RV^{(w)} the mean over the last 5 days and
RV^{(m)} the mean over the last 22 days (all built leakage-safe by
``src.data.features.har_lagged_rv``, which shifts by one day before averaging).
The cascade of horizons approximates the long memory of realized volatility with
only four parameters, and on realized-volatility targets it "routinely matches or
exceeds more elaborate fractionally integrated alternatives" (Chapter 2 §2.3).
A deep model that cannot beat HAR-RV on its native target is not a credible
alternative -- which is exactly why the viva will ask for it.

Estimation is ordinary least squares (statsmodels), refit once per fold on the
fold's training window. Two modelling choices worth flagging for the write-up:

* **Target scale.** We fit HAR on realized *variance* (Corsi's original object
  and the scale on which QLIKE/MSE are computed), so no cross-scale mapping is
  needed at evaluation time. A log-variance variant (``transform="log"``) is
  provided because logging tames the right skew of RV and is common in the modern
  HAR literature; it is offered as a sensitivity, not the default.
* **Positivity.** OLS on levels can in principle predict a negative variance;
  QLIKE is undefined there. Forecasts are floored at a small positive constant so
  the loss is always defined. In practice negative predictions are rare and the
  floor almost never binds; the milestone note reports how often it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.splits import Fold

HAR_LAG_COLS = ("rv_d", "rv_w", "rv_m")


class HARForecaster:
    """HAR-RV via OLS on daily/weekly/monthly realized-variance lags.

    Parameters
    ----------
    target_col : realized-variance column to regress (default ``"rv"``).
    lag_cols : the three HAR regressors, in order (daily, weekly, monthly).
    transform : ``"level"`` (default) fits RV directly; ``"log"`` fits log-RV and
        exponentiates the forecast (a bias-corrected exponential is used).
    floor : positivity floor applied to the variance forecast for QLIKE validity.
    """

    def __init__(
        self,
        *,
        target_col: str = "rv",
        lag_cols: tuple[str, ...] = HAR_LAG_COLS,
        transform: str = "level",
        floor: float = 1e-12,
        name: str = "HAR-RV",
    ) -> None:
        if transform not in ("level", "log"):
            raise ValueError(f"transform must be 'level' or 'log', got {transform!r}")
        self.target_col = target_col
        self.lag_cols = tuple(lag_cols)
        self.transform = transform
        self.floor = float(floor)
        self.name = name
        # Per-fold OLS coefficients, retained for the milestone note.
        self.fold_params_: dict[int, dict[str, float]] = {}
        self.n_floored_: int = 0

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = [self.target_col, *self.lag_cols]
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: frame missing columns {missing}")
        return frame[cols].astype(float)

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        data = self._prepare(frame)

        # Training window: rows with all lags + target present, up to train_end.
        train = data.loc[fold.train_start : fold.train_end].dropna()
        if train.empty:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: empty training window")

        y_tr = train[self.target_col].to_numpy()
        X_tr = train[list(self.lag_cols)].to_numpy()
        if self.transform == "log":
            y_tr = np.log(np.clip(y_tr, self.floor, None))

        X_tr = sm.add_constant(X_tr, has_constant="add")
        ols = sm.OLS(y_tr, X_tr).fit()

        self.fold_params_[fold.fold_idx] = {
            "const": float(ols.params[0]),
            "beta_d": float(ols.params[1]),
            "beta_w": float(ols.params[2]),
            "beta_m": float(ols.params[3]),
            "r2": float(ols.rsquared),
        }

        # Prediction window. Lags here look strictly backward (shift(1) inside
        # har_lagged_rv), so using rows dated in the predict window is leakage-safe:
        # the regressors are all known at the close of the prior day.
        oos = data.loc[fold.predict_start : fold.predict_end]
        X_oos = sm.add_constant(oos[list(self.lag_cols)].to_numpy(), has_constant="add")
        yhat = ols.predict(X_oos)

        if self.transform == "log":
            # Exponentiate with the parametric log-normal retransformation
            # correction exp(mu_hat + sigma_hat^2/2), which is exact under Gaussian
            # log-residuals. NB this is *not* Duan's (1983) smearing estimator,
            # which is the non-parametric mean(exp(resid)); the two coincide only
            # asymptotically under normality. The parametric form is used because
            # OLS already supplies mse_resid.
            yhat = np.exp(yhat + 0.5 * ols.mse_resid)

        floored = yhat < self.floor
        self.n_floored_ += int(np.count_nonzero(floored))
        yhat = np.clip(yhat, self.floor, None)

        return pd.Series(yhat, index=oos.index, name=self.name)
