"""Regime-aware deep forecasters (roadmap Phase 5; Ch2 §2.5; RQ2).

Two ways to condition the Phase-3 LSTM on the Phase-4 HMM regime signal are built
for Phase 5, and run head-to-head against the regime-agnostic LSTM and the
econometric baselines on the *identical* leakage-free walk-forward:

* **Approach A — regime as feature.** No new model class is needed: the vanilla
  :class:`src.models.deep.lstm.LSTMForecaster` is instantiated with the causal
  filtered regime posteriors (``reg_p0..reg_p{K-1}``) appended to its feature
  list. This is exactly what Ch2 §2.5 and the Phase-4 hand-off describe — passing
  the *filtered* state probabilities as an extra network input. Because the
  sliding window ends at ``t-1`` (:mod:`src.data.sequence`), the regime signal a
  forecast for ``RV_t`` sees is dated ``<= t-1``: leakage-free by construction.

* **Approach B — regime-specific experts (this module).**
  :class:`RegimeExpertForecaster` trains one LSTM *expert* per regime and mixes
  their forecasts with the current regime posterior as a soft gate — a
  mixture-of-experts (roadmap Phase 5, "Approach B"). Expert ``k`` is trained on
  every window but weighted by the (causal, one-day-lagged) probability of regime
  ``k`` on the forecast day, so it specialises in that regime's dynamics; at
  inference the experts are recombined by the same gate. This lets the model use
  a *different* non-linear map in calm vs crisis states rather than one map that
  must serve both — the hypothesis RQ2 tests.

Leakage discipline (roadmap §1.3), enforced here exactly as in the Phase-3 LSTM:
feature/target standardisers and the experts are fit on ``dates <= fold.train_end``
only; windows end strictly before their target date; and the gate for a forecast
on day ``t`` is the regime posterior on day ``t-1`` (``.shift(1)``), never day
``t`` — so the mixture weight, like every feature, is known at the close of
``t-1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # torch is a Phase-3+ dependency; import at module load so failures are loud.
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the Phase 5 regime-aware LSTM. Install it "
        "(pyproject optional-dependency group 'deep')."
    ) from exc

from src.data.sequence import Standardizer, build_feature_target, make_windows
from src.data.splits import Fold
from src.models.deep.lstm import _VAR_FLOOR, LSTMRegressor
from src.utils.logging import get_logger
from src.utils.seeding import set_seed

log = get_logger("regime_lstm")

DEFAULT_BASE_FEATURES: tuple[str, ...] = ("log_rv", "oc_log_return")
DEFAULT_GATE_COLS: tuple[str, ...] = ("reg_p0", "reg_p1", "reg_p2")


class RegimeExpertForecaster:
    """Mixture-of-experts LSTM: one expert per regime, soft-gated by the posterior.

    Implements the :class:`src.evaluation.rolling.VolForecaster` protocol
    (``name`` + ``forecast_fold``) so it runs through the shared walk-forward
    engine unchanged.

    Parameters
    ----------
    base_features : ordered feature keys for every expert (see
        :data:`src.data.sequence.FEATURE_BUILDERS`). The regime signal does *not*
        enter the experts as an input feature — it enters through the gate — so
        this is the same input set as the regime-agnostic LSTM it is compared to.
    gate_cols : the regime-posterior columns on the frame (``reg_p0..reg_p{K-1}``);
        ``K`` = ``len(gate_cols)``. These must be the **causal filtered**
        posteriors (Phase-4 ``jphmm_filt_p*`` for the headline jump-penalised HMM,
        ``hmm_filt_p*`` for the Baum-Welch comparator); they are lagged one day inside this
        class so the gate for day ``t`` uses only information known at ``t-1``.
    lookback, hidden_size, num_layers, dropout : per-expert network shape.
    lr, weight_decay, batch_size, max_epochs, patience, grad_clip, val_fraction :
        optimisation; shared across experts.
    smearing : log-normal (Duan 1983) bias correction, estimated from the
        *mixture* residuals on a held-out tail.
    refit_every_folds : ``<= 0`` trains once on the first fold and reuses; ``k>=1``
        retrains every ``k`` folds (same semantics as the Phase-3 LSTM).
    seed : base seed; expert ``k`` at fold ``f`` uses ``seed + 100*f + k`` for a
        reproducible but de-correlated initialisation.
    """

    def __init__(
        self,
        *,
        base_features: tuple[str, ...] = DEFAULT_BASE_FEATURES,
        gate_cols: tuple[str, ...] = DEFAULT_GATE_COLS,
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
        smearing: bool = True,
        refit_every_folds: int = 0,
        seed: int = 17,
        device: str | None = None,
        name: str = "Regime-LSTM-B",
    ) -> None:
        self.base_features = tuple(base_features)
        self.gate_cols = tuple(gate_cols)
        self.K = len(self.gate_cols)
        if self.K < 2:
            raise ValueError("need >= 2 regime gate columns")
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
        self._experts: list[LSTMRegressor] | None = None
        self._feat_scaler: Standardizer | None = None
        self._targ_scaler: Standardizer | None = None
        self._smear_var: float = 0.0
        self._trained = False
        self.fold_params_: dict[int, dict] = {}
        self.history_: list[list[dict]] = []  # per-expert learning curves (last fit)

    # ---- config summary ----------------------------------------------------- #
    def config(self) -> dict:
        return {
            "name": self.name, "base_features": list(self.base_features),
            "gate_cols": list(self.gate_cols), "K": self.K, "lookback": self.lookback,
            "hidden_size": self.hidden_size, "num_layers": self.num_layers,
            "dropout": self.dropout, "lr": self.lr, "batch_size": self.batch_size,
            "max_epochs": self.max_epochs, "patience": self.patience,
            "val_fraction": self.val_fraction, "smearing": self.smearing,
            "refit_every_folds": self.refit_every_folds, "seed": self.seed,
            "device": str(self.device),
        }

    # ---- VolForecaster API -------------------------------------------------- #
    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        feats, target, index = build_feature_target(frame, self.base_features)
        # Causal gate: the posterior known at t-1 drives the forecast for day t.
        # base features are NaN-free over the frame, so `index` == frame.index and
        # shift(1) means "previous trading day".
        gate = frame[list(self.gate_cols)].reindex(index).shift(1)
        if self._need_refit(fold):
            self._fit(feats, target, gate, index, fold)
        return self._predict(feats, target, gate, index, fold)

    # ---- internals ---------------------------------------------------------- #
    def _need_refit(self, fold: Fold) -> bool:
        if not self._trained:
            return True
        if self.refit_every_folds >= 1:
            return (fold.fold_idx % self.refit_every_folds) == 0
        return False

    def _gate_for(self, gate: pd.DataFrame, y_dates: pd.DatetimeIndex) -> np.ndarray:
        """(n, K) gate matrix aligned to the windowed target dates, NaN-safe and
        row-normalised to a probability. A row with no finite mass (the first
        shifted row, or a degenerate all-zero row) falls back to a uniform gate."""
        G = gate.reindex(y_dates).to_numpy(dtype=float)
        G = np.where(np.isfinite(G), G, 0.0)
        s = G.sum(axis=1, keepdims=True)
        degenerate = (s <= 1e-12).ravel()
        if degenerate.any():
            G[degenerate] = 1.0 / self.K
            s = G.sum(axis=1, keepdims=True)
        return G / s

    def _fit(self, feats, target, gate, index, fold) -> None:
        set_seed(self.seed + fold.fold_idx)
        train_mask = np.asarray(index <= fold.train_end)
        if int(train_mask.sum()) <= self.lookback + 30:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: too few training rows.")

        # Standardise on training rows only, then window the whole series.
        self._feat_scaler = Standardizer().fit(feats.to_numpy()[train_mask])
        self._targ_scaler = Standardizer().fit(target.to_numpy()[train_mask].reshape(-1, 1))
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()

        X, y, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        G = self._gate_for(gate, y_dates)
        tr = np.asarray(y_dates <= fold.train_end)
        X_tr, y_tr, G_tr = X[tr], y[tr], G[tr]
        if len(X_tr) < 50:
            raise RuntimeError(f"{self.name} fold {fold.fold_idx}: too few training windows.")

        # Time-ordered inner-validation tail for early stopping (never a random split).
        n_val = max(1, int(round(len(X_tr) * self.val_fraction)))
        cut = len(X_tr) - n_val
        X_fit, y_fit, G_fit = X_tr[:cut], y_tr[:cut], G_tr[:cut]
        X_es, y_es, G_es = X_tr[cut:], y_tr[cut:], G_tr[cut:]

        experts, histories = [], []
        for k in range(self.K):
            model, hist = self._train_expert(
                X_fit, y_fit, G_fit[:, k], X_es, y_es, G_es[:, k], fold, k
            )
            experts.append(model)
            histories.append(hist)
        self._experts = experts

        # Log-normal smearing from the *mixture* residuals on the held-out tail.
        z_es = self._mixture_pred(X_es, G_es)
        resid_log = (y_es - z_es) * float(self._targ_scaler.std_[0])
        self._smear_var = float(np.var(resid_log)) if self.smearing else 0.0

        self._trained = True
        self.history_ = histories
        self.fold_params_[fold.fold_idx] = {
            "n_train_windows": int(len(X_fit)),
            "n_es_windows": int(len(X_es)),
            "expert_effective_n": [float(G_fit[:, k].sum()) for k in range(self.K)],
            "epochs_run": [int(len(h)) for h in histories],
            "smear_var": self._smear_var,
            "train_end": pd.Timestamp(fold.train_end).date().isoformat(),
        }

    def _train_expert(self, X_fit, y_fit, w_fit, X_es, y_es, w_es, fold, k):
        """Train one expert with a per-sample-weighted MSE (weights = regime-k gate)."""
        set_seed(self.seed + 100 * fold.fold_idx + k)
        model = LSTMRegressor(
            input_size=X_fit.shape[2], hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xf, yf, wf = self._t(X_fit), self._t(y_fit), self._t(w_fit)
        Xv, yv, wv = self._t(X_es), self._t(y_es), self._t(w_es)
        rng = np.random.default_rng(self.seed + 100 * fold.fold_idx + k)

        def wmse(pred, targ, w):
            return (w * (pred - targ) ** 2).sum() / torch.clamp(w.sum(), min=1e-8)

        best_val, best_state, bad = float("inf"), None, 0
        history: list[dict] = []
        n = len(Xf)
        for epoch in range(self.max_epochs):
            model.train()
            perm = rng.permutation(n)
            run, seen = 0.0, 0.0
            for i in range(0, n, self.batch_size):
                bidx = torch.as_tensor(perm[i: i + self.batch_size], device=self.device)
                opt.zero_grad()
                out = model(Xf.index_select(0, bidx))
                wb = wf.index_select(0, bidx)
                loss = wmse(out, yf.index_select(0, bidx), wb)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                opt.step()
                run += float(loss.item()) * float(wb.sum().item())
                seen += float(wb.sum().item())
            train_wmse = run / max(seen, 1e-8)
            model.eval()
            with torch.no_grad():
                val_wmse = float(wmse(model(Xv), yv, wv).item()) if len(Xv) else train_wmse
            history.append({"epoch": epoch, "train_wmse": train_wmse, "val_wmse": val_wmse})
            if val_wmse < best_val - 1e-7:
                best_val, bad = val_wmse, 0
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def _mixture_pred(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Gate-weighted mixture of expert outputs, in standardised log-RV space."""
        if len(X) == 0:
            return np.empty((0,), dtype=float)
        cols = []
        Xt = self._t(X)
        for model in self._experts:
            model.eval()
            with torch.no_grad():
                cols.append(model(Xt).cpu().numpy().ravel())
        P = np.stack(cols, axis=1)          # (n, K) expert predictions
        return (P * G).sum(axis=1)          # gate-weighted mixture

    def _predict(self, feats, target, gate, index, fold) -> pd.Series:
        if self._experts is None or self._feat_scaler is None:
            raise RuntimeError(f"{self.name}: predict called before fit.")
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
        X, _, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        sel = np.asarray((y_dates >= fold.predict_start) & (y_dates <= fold.predict_end))
        if not sel.any():
            return pd.Series(dtype=float, name=self.name)
        G = self._gate_for(gate, y_dates)
        z = self._mixture_pred(X[sel], G[sel])
        log_rv = self._targ_scaler.inverse_transform(z.reshape(-1, 1)).ravel()
        var = np.clip(np.exp(log_rv + 0.5 * self._smear_var), _VAR_FLOOR, None)
        return pd.Series(var, index=y_dates[sel], name=self.name)

    def _t(self, a: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(self.device)
