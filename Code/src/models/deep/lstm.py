"""Vanilla LSTM baseline for realized-variance forecasting (roadmap Phase 3, §1.2/§3).

The LSTM (Hochreiter & Schmidhuber 1997; Chapter 2 §2.5) is the dissertation's
non-linear sequence baseline: it forecasts ``log(RV_t)`` from a ``lookback``
window of features dated ``t-L … t-1`` and maps the forecast back to the variance
scale for scoring. It exists to answer **RQ1** -- does a deep sequence model
improve on GARCH(1,1)/EGARCH/HAR-RV on daily S&P 500 realized volatility, under
the *same* leakage-free walk-forward protocol? -- and to be the backbone the
regime-aware (Phase 5) and uncertainty-aware (Phase 6) variants are built on.

Contract
--------
:class:`LSTMForecaster` is a :class:`src.evaluation.rolling.VolForecaster`: it has
a ``name`` and a ``forecast_fold(frame, fold) -> pd.Series`` returning **variance**
forecasts indexed by the fold's prediction dates. The engine owns the folds.

Leakage discipline (roadmap §1.3), enforced here:
* feature/target standardisers are fit on ``dates <= fold.train_end`` only;
* the network is trained on windows whose *target* date is ``<= fold.train_end``;
* early stopping uses a **time-ordered** tail of the training window (never a
  random split), so no future row informs the stopping decision;
* windows end strictly before their target date (see :mod:`src.data.sequence`).

Refit cadence. Retraining a neural net at every monthly fold is both expensive
and noisy, so refit is configurable via ``refit_every_folds``:
* ``<= 0`` (default) -- train **once** on the first fold (all pre-test data) and
  reuse it across the out-of-sample window. This is the standard, cheapest and
  most stable protocol for an LSTM RV baseline and keeps the comparison to the
  refit-every-fold econometric models fair on *information* (both see only
  pre-forecast data) while avoiding refit noise.
* ``k >= 1`` -- retrain every ``k`` folds (expanding window). ``1`` reproduces the
  econometric refit-every-fold cadence at the cost of ``n_folds`` trainings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # torch is a Phase-3 dependency; import at module load so failures are loud.
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the Phase 3 LSTM. Install it (see pyproject "
        "optional-dependency group 'deep', e.g. `pip install torch`)."
    ) from exc

from src.data.sequence import Standardizer, build_feature_target, make_windows
from src.data.splits import Fold
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("lstm")

# Positivity floor for the returned variance forecast (QLIKE needs h > 0).
_VAR_FLOOR = 1e-14
DEFAULT_FEATURES: tuple[str, ...] = ("log_rv", "oc_log_return")


class LSTMRegressor(nn.Module):
    """Single-output LSTM regressor: sequence -> scalar (standardised log-RV).

    ``num_layers``/``dropout`` follow the roadmap's "1-2 layers, dropout" spec.
    A head-level ``Dropout`` is kept even for the deterministic baseline because
    the Phase 6 MC-Dropout variant reuses this exact module with dropout left on
    at inference; here it is disabled by ``eval()`` at prediction time.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        last = out[:, -1, :]           # last time step's top-layer hidden state
        return self.head(self.head_dropout(last)).squeeze(-1)


class LSTMForecaster:
    """Walk-forward LSTM forecaster of daily realized *variance*.

    Parameters
    ----------
    features : ordered feature keys (see :data:`src.data.sequence.FEATURE_BUILDERS`).
        Default ``("log_rv", "oc_log_return")`` -- the RV history the HAR baseline
        uses plus the day's open-to-close return (sign/leverage information the
        econometric RV models cannot exploit).
    lookback : window length ``L`` (trading days).
    hidden_size, num_layers, dropout : network size.
    lr, weight_decay, batch_size, max_epochs, patience, grad_clip : optimisation.
    val_fraction : time-ordered tail of the training window held out for early
        stopping.
    smearing : if True, apply the parametric log-normal retransformation
        correction exp(mu_hat + sigma_hat^2/2) (exact under Gaussian log-residuals;
        not Duan's 1983 non-parametric smearing estimator mean(exp(resid)))
        ``exp(mu + 0.5 sigma^2)`` using the held-out residual variance so the
        variance forecast targets the conditional *mean* rather than the median.
    refit_every_folds : see module docstring (``<= 0`` trains once).
    seed : base seed; each refit uses ``seed + fold_idx`` for a reproducible but
        de-correlated initialisation.
    """

    def __init__(
        self,
        *,
        features: tuple[str, ...] = DEFAULT_FEATURES,
        lookback: int = 20,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 150,
        patience: int = 15,
        grad_clip: float = 1.0,
        val_fraction: float = 0.15,
        smearing: bool = True,
        refit_every_folds: int = 0,
        seed: int = 17,
        device: str | None = None,
        name: str = "LSTM",
    ) -> None:
        self.features = tuple(features)
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip = float(grad_clip)
        self.val_fraction = float(val_fraction)
        self.smearing = bool(smearing)
        self.refit_every_folds = int(refit_every_folds)
        self.seed = int(seed)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.name = name

        # Fitted state (per refit).
        self._model: LSTMRegressor | None = None
        self._feat_scaler: Standardizer | None = None
        self._targ_scaler: Standardizer | None = None
        self._smear_var: float = 0.0
        self._trained = False
        self._trained_at_fold: int | None = None
        # Diagnostics for the milestone note.
        self.fold_params_: dict[int, dict] = {}
        self.history_: list[dict] = []

    # ---- config summary (for milestone / config snapshot) ------------------- #
    def config(self) -> dict:
        return {
            "name": self.name, "features": list(self.features), "lookback": self.lookback,
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay,
            "batch_size": self.batch_size, "max_epochs": self.max_epochs,
            "patience": self.patience, "grad_clip": self.grad_clip,
            "val_fraction": self.val_fraction, "smearing": self.smearing,
            "refit_every_folds": self.refit_every_folds, "seed": self.seed,
            "device": str(self.device),
        }

    # ---- VolForecaster API -------------------------------------------------- #
    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        feats, target, index = build_feature_target(frame, self.features)
        if self._need_refit(fold):
            self._fit(feats, target, index, fold)
        return self._predict(feats, target, index, fold)

    # ---- internals ---------------------------------------------------------- #
    def _need_refit(self, fold: Fold) -> bool:
        if not self._trained:
            return True
        if self.refit_every_folds >= 1:
            return (fold.fold_idx % self.refit_every_folds) == 0
        return False  # <= 0: train once, reuse.

    def _fit(self, feats: pd.DataFrame, target: pd.Series, index: pd.DatetimeIndex,
             fold: Fold) -> None:
        set_seed(self.seed + fold.fold_idx)

        train_mask = np.asarray(index <= fold.train_end)
        n_train_rows = int(train_mask.sum())
        if n_train_rows <= self.lookback + 30:
            raise RuntimeError(
                f"{self.name} fold {fold.fold_idx}: only {n_train_rows} training rows "
                f"for lookback {self.lookback}."
            )

        # Standardise on training rows only, then window the whole series.
        self._feat_scaler = Standardizer().fit(feats.to_numpy()[train_mask])
        self._targ_scaler = Standardizer().fit(target.to_numpy()[train_mask].reshape(-1, 1))
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()

        X, y, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        tr = np.asarray(y_dates <= fold.train_end)
        X_tr, y_tr = X[tr], y[tr]
        if len(X_tr) < 50:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: too few training windows.")

        # Time-ordered inner validation tail for early stopping.
        n_val = max(1, int(round(len(X_tr) * self.val_fraction)))
        X_fit, y_fit = X_tr[: len(X_tr) - n_val], y_tr[: len(X_tr) - n_val]
        X_es, y_es = X_tr[len(X_tr) - n_val:], y_tr[len(X_tr) - n_val:]

        model, history, best_val = self._train(X_fit, y_fit, X_es, y_es, fold)

        # Log-normal smearing variance from held-out residuals (in log-RV units).
        model.eval()
        with torch.no_grad():
            pred_es = model(self._to_tensor(X_es)).cpu().numpy().ravel()
        resid_log = (y_es - pred_es) * float(self._targ_scaler.std_[0])
        self._smear_var = float(np.var(resid_log)) if self.smearing else 0.0

        self._model = model
        self._trained = True
        self._trained_at_fold = fold.fold_idx
        self.history_ = history
        self.fold_params_[fold.fold_idx] = {
            "n_train_windows": int(len(X_fit)),
            "n_es_windows": int(len(X_es)),
            "epochs_run": int(len(history)),
            "best_val_mse": float(best_val),
            "smear_var": self._smear_var,
            "hidden_size": self.hidden_size, "lookback": self.lookback, "lr": self.lr,
            "train_end": pd.Timestamp(fold.train_end).date().isoformat(),
        }

    def _train(self, X_fit, y_fit, X_es, y_es, fold):
        model = LSTMRegressor(
            input_size=X_fit.shape[2], hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xf, yf = self._to_tensor(X_fit), self._to_tensor(y_fit)
        Xv, yv = self._to_tensor(X_es), self._to_tensor(y_es)
        rng = np.random.default_rng(self.seed + fold.fold_idx)

        best_val, best_state, bad = float("inf"), None, 0
        history: list[dict] = []
        n = len(Xf)
        for epoch in range(self.max_epochs):
            model.train()
            perm = rng.permutation(n)
            run = 0.0
            for i in range(0, n, self.batch_size):
                bidx = torch.as_tensor(perm[i: i + self.batch_size], device=self.device)
                opt.zero_grad()
                out = model(Xf.index_select(0, bidx))
                loss = loss_fn(out, yf.index_select(0, bidx))
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                run += float(loss.item()) * len(bidx)
            train_mse = run / n
            model.eval()
            with torch.no_grad():
                val_mse = float(loss_fn(model(Xv), yv).item()) if len(Xv) else train_mse
            history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
            if val_mse < best_val - 1e-7:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history, best_val

    def _predict(self, feats, target, index, fold) -> pd.Series:
        if self._model is None or self._feat_scaler is None:
            raise RuntimeError(f"{self.name}: predict called before fit.")
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
        X, _, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        sel = np.asarray((y_dates >= fold.predict_start) & (y_dates <= fold.predict_end))
        if not sel.any():
            return pd.Series(dtype=float, name=self.name)

        self._model.eval()
        with torch.no_grad():
            z = self._model(self._to_tensor(X[sel])).cpu().numpy().ravel()
        log_rv = self._targ_scaler.inverse_transform(z.reshape(-1, 1)).ravel()
        var = np.exp(log_rv + 0.5 * self._smear_var)
        var = np.clip(var, _VAR_FLOOR, None)
        return pd.Series(var, index=y_dates[sel], name=self.name)

    def _to_tensor(self, a: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
