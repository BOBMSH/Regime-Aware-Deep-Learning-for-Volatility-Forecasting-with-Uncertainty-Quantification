"""Uncertainty-aware deep forecasters (roadmap Phase 6; Ch2 §2.6; RQ3).

Two ways to turn the Phase-3 LSTM into a *probabilistic* forecaster of daily
realized variance are built for Phase 6 and run head-to-head on the identical
leakage-free walk-forward as every earlier phase. They embody the two dominant,
philosophically distinct schools of neural-network uncertainty (roadmap §1.5):

* **MC-Dropout (primary) — Bayesian-flavoured.**
  :class:`MCDropoutLSTMForecaster` reuses the Phase-3 :class:`LSTMForecaster`
  *unchanged for training* — same architecture, same MSE objective, same
  leakage discipline — and only changes *inference*: the head dropout (which the
  Phase-3 module deliberately keeps) is left **active** at prediction time and
  the network is run ``T`` times, giving ``T`` stochastic forecasts. Following
  Gal & Ghahramani (2016), the spread across those passes estimates the **model
  (epistemic)** uncertainty; added to the **aleatoric** noise already estimated
  by the Phase-3 held-out residual variance (the log-normal retransformation
  term), it yields,
  per day, a Gaussian predictive law on log-variance -> a log-normal law on
  variance whose mean is the point forecast and whose central quantiles are the
  prediction interval.

* **Quantile regression (comparison) — distribution-free / frequentist.**
  :class:`QuantileLSTMForecaster` trains ONE LSTM with a multi-quantile head
  (default τ = 0.05, 0.50, 0.95) under the **pinball loss** (Koenker & Bassett
  1978; Engle & Manganelli 2004). The head is constructed to be **monotone by
  design** (cumulative soft-plus offsets), so the predicted quantiles never
  cross — the classic failure mode of independent quantile heads — and the
  interval is read straight off the outer quantiles with no distributional
  assumption.

Both satisfy the :class:`src.evaluation.rolling.VolForecaster` protocol (their
point forecast — the MC-Dropout predictive mean, the quantile median — runs
through the shared engine and is comparable on QLIKE to every earlier model),
and both expose an extra ``forecast_fold_uq`` returning the full per-day
predictive object the calibration metrics (PICP / MPIW / Winkler, per regime)
consume. Calibration itself lives in :mod:`src.evaluation.calibration`; this
module only *produces* the predictive laws.

Leakage discipline (roadmap §1.3) is inherited verbatim from Phase 3: scalers
and weights are fit on ``dates <= fold.train_end`` only, and every window ends
strictly before its target date, so a forecast for ``RV_t`` conditions only on
information dated ``<= t-1`` — the dropout masks and quantile heads change *what*
is emitted, never *when* the inputs are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # torch is a Phase-3+ dependency; import at module load so failures are loud.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the Phase 6 uncertainty models. Install it "
        "(pyproject optional-dependency group 'deep')."
    ) from exc

from src.data.sequence import Standardizer, build_feature_target, make_windows
from src.data.splits import Fold
from src.models.deep.lstm import _VAR_FLOOR, DEFAULT_FEATURES, LSTMForecaster, LSTMRegressor
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("uncertainty")

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)


# --------------------------------------------------------------------------- #
# MC-Dropout LSTM                                                             #
# --------------------------------------------------------------------------- #
def _enable_mc_dropout(model: nn.Module) -> None:
    """Put *only* the dropout layers of an eval-mode model into train mode.

    MC-Dropout needs stochastic dropout masks at inference while everything else
    stays deterministic. There is no BatchNorm in the LSTM head, so the sole
    train/eval-sensitive modules are the head :class:`~torch.nn.Dropout` (always)
    and the recurrent dropout inside a multi-layer :class:`~torch.nn.LSTM` (only
    when ``num_layers > 1`` gives it a non-zero rate). Call ``model.eval()``
    first, then this, to activate exactly those.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.LSTM) and float(getattr(m, "dropout", 0.0)) > 0.0:
            m.train()


class MCDropoutLSTMForecaster(LSTMForecaster):
    """MC-Dropout predictive LSTM: the Phase-3 LSTM with dropout left on at inference.

    Training is *identical* to :class:`LSTMForecaster` (inherited unchanged), so
    with the same hyperparameters and seed the fitted weights are the Phase-3
    weights — the MC-Dropout model adds *uncertainty*, not a different point
    model. Inference draws ``mc_samples`` stochastic forward passes and forms,
    per day, a Gaussian predictive law on log-variance:

    * predictive mean ``mu`` = mean of the ``T`` log-variance draws;
    * **epistemic** variance ``s2_epi`` = variance of the ``T`` draws;
    * **aleatoric** variance ``s2_ale`` = the held-out residual variance
      ``_smear_var`` estimated at fit time (the Phase-3 log-normal
      retransformation term);
    * total predictive std ``sigma = sqrt(s2_epi + s2_ale)`` (Gal & Ghahramani
      2016: the two sources add).

    The point forecast (variance scale) is the log-normal mean
    ``exp(mu + sigma^2/2)`` — the mean-unbiased conditional-variance forecast
    (Patton 2011), which reduces to the Phase-3 smeared forecast when the
    epistemic term is negligible. ``forecast_fold`` returns it (VolForecaster
    protocol); ``forecast_fold_uq`` returns the full predictive frame.

    Parameters (beyond :class:`LSTMForecaster`)
    -------------------------------------------
    mc_samples : number of stochastic forward passes ``T`` (roadmap: 100).
    mc_seed : torch RNG seed for the dropout masks (default = ``seed``), so the
        ``T`` passes — and hence every interval — are reproducible.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 100,
        mc_seed: int | None = None,
        features: tuple[str, ...] = DEFAULT_FEATURES,
        dropout: float = 0.1,
        name: str = "MC-Dropout-LSTM",
        **kwargs,
    ) -> None:
        super().__init__(features=features, dropout=dropout, name=name, **kwargs)
        if self.dropout <= 0.0:
            raise ValueError(
                "MC-Dropout needs dropout > 0 (the head dropout is the source of "
                "the epistemic samples); got dropout=0."
            )
        self.mc_samples = int(mc_samples)
        self.mc_seed = int(mc_seed) if mc_seed is not None else self.seed

    # ---- config summary (extends the base) --------------------------------- #
    def config(self) -> dict:
        c = super().config()
        c.update({"mc_samples": self.mc_samples, "mc_seed": self.mc_seed})
        return c

    # ---- UQ prediction ----------------------------------------------------- #
    def _predictive_frame(self, feats, target, index, fold) -> pd.DataFrame:
        if self._model is None or self._feat_scaler is None:
            raise RuntimeError(f"{self.name}: predict called before fit.")
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
        X, _, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        sel = np.asarray((y_dates >= fold.predict_start) & (y_dates <= fold.predict_end))
        if not sel.any():
            return pd.DataFrame(
                columns=["mean", "mu_log", "sigma_log", "sd_epistemic",
                         "sd_aleatoric", "mu_log_det"]
            )

        Xt = self._to_tensor(X[sel])
        model = self._model
        tmean = float(self._targ_scaler.mean_[0])
        tstd = float(self._targ_scaler.std_[0])

        # Deterministic pass (dropout OFF) -> the Phase-3 point in log space; used
        # for the aleatoric-only ("no MC") reference interval.
        model.eval()
        with torch.no_grad():
            z_det = model(Xt).cpu().numpy().ravel()
        mu_log_det = z_det * tstd + tmean

        # T stochastic passes (dropout ON). Seed torch once so the whole T×n mask
        # sequence — and every interval derived from it — is reproducible.
        model.eval()
        _enable_mc_dropout(model)
        torch.manual_seed(self.mc_seed)
        n = int(sel.sum())
        draws = np.empty((self.mc_samples, n), dtype=float)
        with torch.no_grad():
            for t in range(self.mc_samples):
                draws[t] = model(Xt).cpu().numpy().ravel()
        model.eval()  # restore deterministic mode

        logrv_draws = draws * tstd + tmean            # (T, n) log-variance samples
        mu = logrv_draws.mean(axis=0)                 # predictive mean (log scale)
        s2_epi = logrv_draws.var(axis=0, ddof=0)      # epistemic (MC) variance
        s2_ale = float(self._smear_var)               # aleatoric (held-out residual)
        sigma = np.sqrt(s2_epi + s2_ale)
        mean_var = np.clip(np.exp(mu + 0.5 * (s2_epi + s2_ale)), _VAR_FLOOR, None)

        return pd.DataFrame(
            {
                "mean": mean_var,
                "mu_log": mu,
                "sigma_log": sigma,
                "sd_epistemic": np.sqrt(s2_epi),
                "sd_aleatoric": np.full(n, np.sqrt(s2_ale)),
                "mu_log_det": mu_log_det,
            },
            index=pd.DatetimeIndex(y_dates[sel]),
        )

    def forecast_fold_uq(self, frame: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Full predictive frame for the fold: per-day ``mean`` (variance),
        ``mu_log``/``sigma_log`` (log-scale predictive Gaussian), the epistemic
        and aleatoric std components, and ``mu_log_det`` (dropout-off log mean)."""
        feats, target, index = build_feature_target(frame, self.features)
        if self._need_refit(fold):
            self._fit(feats, target, index, fold)
        return self._predictive_frame(feats, target, index, fold)

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        """VolForecaster point forecast = the log-normal predictive **mean**."""
        return self.forecast_fold_uq(frame, fold)["mean"].rename(self.name)


# --------------------------------------------------------------------------- #
# Quantile-regression LSTM                                                     #
# --------------------------------------------------------------------------- #
class QuantileLSTMRegressor(nn.Module):
    """LSTM with a monotone multi-quantile head (no quantile crossing).

    The LSTM body is identical to :class:`LSTMRegressor`; the head emits one raw
    score per quantile and maps them to **ordered** outputs by construction:

        q_0 = raw_0,   q_i = q_{i-1} + softplus(raw_i)   (i >= 1),

    so ``q_0 <= q_1 <= ... <= q_{K-1}`` for *any* weights — the network cannot
    produce a lower quantile above an upper one. The quantile levels must be
    passed in ascending order; the head returns a ``(batch, K)`` tensor aligned
    to them.
    """

    def __init__(
        self, input_size: int, hidden_size: int, n_quantiles: int,
        num_layers: int = 1, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_quantiles < 1:
            raise ValueError("need >= 1 quantile")
        self.n_quantiles = int(n_quantiles)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_size, self.n_quantiles)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        raw = self.head(self.head_dropout(last))          # (batch, K), unordered
        if self.n_quantiles == 1:
            return raw
        base = raw[:, :1]                                  # first quantile, free
        steps = F.softplus(raw[:, 1:])                     # non-negative gaps
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)


def _pinball_loss_torch(
    pred: "torch.Tensor", target: "torch.Tensor", taus: "torch.Tensor"
) -> "torch.Tensor":
    """Mean pinball loss over a ``(batch, K)`` quantile prediction.

    ``target`` is ``(batch,)``; ``taus`` is ``(K,)``. Averages the check loss
    ``max(tau*e, (tau-1)*e)`` (``e = target - pred``) over quantiles and batch.
    """
    e = target.unsqueeze(1) - pred                         # (batch, K)
    return torch.maximum(taus.unsqueeze(0) * e, (taus.unsqueeze(0) - 1.0) * e).mean()


class QuantileLSTMForecaster:
    """Walk-forward quantile-regression LSTM (pinball loss; VolForecaster).

    Trains one LSTM with a monotone :class:`QuantileLSTMRegressor` head under the
    pinball loss on the standardised log-variance target, then maps each fitted
    quantile back to the variance scale (``exp`` of the inverse-standardised
    output — monotone, so ordering is preserved). The **median** (τ=0.5, if
    present, else the central quantile) is the point forecast; the outer
    quantiles give the prediction interval directly, with no distributional
    assumption.

    Parameters mirror :class:`LSTMForecaster` (leakage discipline, early
    stopping, refit cadence identical), with:

    quantiles : ascending quantile levels to fit (default 0.05, 0.50, 0.95).
        The central interval reported is ``[min, max]`` of these — a
        ``(max-min)`` nominal-coverage band (0.90 for the default).
    """

    def __init__(
        self,
        *,
        features: tuple[str, ...] = DEFAULT_FEATURES,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        lookback: int = 10,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 90,
        patience: int = 15,
        grad_clip: float = 1.0,
        val_fraction: float = 0.15,
        refit_every_folds: int = 0,
        seed: int = 17,
        device: str | None = None,
        name: str = "Quantile-LSTM",
    ) -> None:
        q = tuple(float(x) for x in quantiles)
        if list(q) != sorted(q) or len(set(q)) != len(q):
            raise ValueError(f"quantiles must be strictly ascending, got {q}")
        if not all(0.0 < x < 1.0 for x in q):
            raise ValueError(f"quantiles must lie in (0, 1), got {q}")
        self.features = tuple(features)
        self.quantiles = q
        self.n_quantiles = len(q)
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
        self.refit_every_folds = int(refit_every_folds)
        self.seed = int(seed)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.name = name
        # median column index: exact 0.5 if present, else the quantile nearest 0.5.
        self._median_j = int(np.argmin(np.abs(np.asarray(q) - 0.5)))

        self._model: QuantileLSTMRegressor | None = None
        self._feat_scaler: Standardizer | None = None
        self._targ_scaler: Standardizer | None = None
        self._trained = False
        self._trained_at_fold: int | None = None
        self.fold_params_: dict[int, dict] = {}
        self.history_: list[dict] = []

    def config(self) -> dict:
        return {
            "name": self.name, "features": list(self.features),
            "quantiles": list(self.quantiles), "lookback": self.lookback,
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay,
            "batch_size": self.batch_size, "max_epochs": self.max_epochs,
            "patience": self.patience, "grad_clip": self.grad_clip,
            "val_fraction": self.val_fraction, "refit_every_folds": self.refit_every_folds,
            "seed": self.seed, "device": str(self.device),
        }

    # ---- VolForecaster API ------------------------------------------------- #
    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        """Point forecast = the median (τ=0.5) quantile on the variance scale."""
        q = self.forecast_fold_uq(frame, fold)
        med_col = f"q{self.quantiles[self._median_j]:g}"
        return q[med_col].rename(self.name)

    def forecast_fold_uq(self, frame: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Per-day fitted variance quantiles: columns ``q0.05``, ``q0.5``, ``q0.95``
        (one per level, in ascending order; non-crossing by construction)."""
        feats, target, index = build_feature_target(frame, self.features)
        if self._need_refit(fold):
            self._fit(feats, target, index, fold)
        return self._predict(feats, target, index, fold)

    # ---- internals --------------------------------------------------------- #
    def _need_refit(self, fold: Fold) -> bool:
        if not self._trained:
            return True
        if self.refit_every_folds >= 1:
            return (fold.fold_idx % self.refit_every_folds) == 0
        return False

    def _fit(self, feats, target, index, fold) -> None:
        set_seed(self.seed + fold.fold_idx)
        train_mask = np.asarray(index <= fold.train_end)
        if int(train_mask.sum()) <= self.lookback + 30:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: too few training rows.")

        self._feat_scaler = Standardizer().fit(feats.to_numpy()[train_mask])
        self._targ_scaler = Standardizer().fit(target.to_numpy()[train_mask].reshape(-1, 1))
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()

        X, y, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        tr = np.asarray(y_dates <= fold.train_end)
        X_tr, y_tr = X[tr], y[tr]
        if len(X_tr) < 50:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: too few training windows.")

        n_val = max(1, int(round(len(X_tr) * self.val_fraction)))
        cut = len(X_tr) - n_val
        X_fit, y_fit = X_tr[:cut], y_tr[:cut]
        X_es, y_es = X_tr[cut:], y_tr[cut:]

        model, history, best_val = self._train(X_fit, y_fit, X_es, y_es, fold)
        self._model = model
        self._trained = True
        self._trained_at_fold = fold.fold_idx
        self.history_ = history
        self.fold_params_[fold.fold_idx] = {
            "n_train_windows": int(len(X_fit)), "n_es_windows": int(len(X_es)),
            "epochs_run": int(len(history)), "best_val_pinball": float(best_val),
            "quantiles": list(self.quantiles),
            "train_end": pd.Timestamp(fold.train_end).date().isoformat(),
        }

    def build_network(self, input_size: int) -> "QuantileLSTMRegressor":
        """The exact module :meth:`_train` fits, at this forecaster's settings.

        The monotone head carries one extra output per quantile beyond the
        single-output baseline, which is the whole difference in size between
        this network and :class:`~src.models.deep.lstm.LSTMRegressor`.
        """
        return QuantileLSTMRegressor(
            input_size=int(input_size), hidden_size=self.hidden_size,
            n_quantiles=self.n_quantiles, num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def network_inputs(self) -> int:
        return len(self.features)

    def n_networks(self) -> int:
        return 1

    def _train(self, X_fit, y_fit, X_es, y_es, fold):
        model = self.build_network(X_fit.shape[2]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        taus = torch.as_tensor(self.quantiles, dtype=torch.float32, device=self.device)

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
                loss = _pinball_loss_torch(out, yf.index_select(0, bidx), taus)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                run += float(loss.item()) * len(bidx)
            train_pin = run / n
            model.eval()
            with torch.no_grad():
                val_pin = (float(_pinball_loss_torch(model(Xv), yv, taus).item())
                           if len(Xv) else train_pin)
            history.append({"epoch": epoch, "train_pinball": train_pin, "val_pinball": val_pin})
            if val_pin < best_val - 1e-7:
                best_val, bad = val_pin, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history, best_val

    def _predict(self, feats, target, index, fold) -> pd.DataFrame:
        if self._model is None or self._feat_scaler is None:
            raise RuntimeError(f"{self.name}: predict called before fit.")
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
        X, _, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        sel = np.asarray((y_dates >= fold.predict_start) & (y_dates <= fold.predict_end))
        cols = [f"q{t:g}" for t in self.quantiles]
        if not sel.any():
            return pd.DataFrame(columns=cols)

        self._model.eval()
        with torch.no_grad():
            z = self._model(self._to_tensor(X[sel])).cpu().numpy()   # (n, K) std log-RV
        # inverse-standardise each quantile column, then exp to the variance scale.
        log_rv = z * float(self._targ_scaler.std_[0]) + float(self._targ_scaler.mean_[0])
        var_q = np.clip(np.exp(log_rv), _VAR_FLOOR, None)
        return pd.DataFrame(var_q, columns=cols, index=pd.DatetimeIndex(y_dates[sel]))

    def _to_tensor(self, a: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
