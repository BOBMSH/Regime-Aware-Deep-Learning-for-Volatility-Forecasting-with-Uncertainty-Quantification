"""Combined regime-aware *and* uncertainty-aware forecaster (roadmap Phase 7).

What Phase 7 combines, and why it is a pre-registered null
---------------------------------------------------------
Phases 5 and 6 answered RQ2 and RQ3 separately and both answers point the same
way about what a combination can be expected to do:

* **Phase 5 (RQ2).** Conditioning on the regime buys nothing pooled (all DM
  p ≥ 0.19) but a great deal conditionally: Giacomini–White on the lagged regime
  indicator gives Regime-LSTM-B vs HAR-RV χ²(3) = 40.1, p = 1.0e-08, with the
  edge concentrated in the transitional state. So the regime signal carries
  information about *where the point forecast is hard*.
* **Phase 6 (RQ3).** The MC-Dropout interval is near-homoskedastic on the log
  scale — the aleatoric term is a single fitted constant and the epistemic
  spread is well under 1% of predictive variance — so the band is a fixed
  multiplicative collar on the point forecast. Its misses are not predictable
  from the regime *level*, nor from an implementable regime *change*, under
  either regime estimator.

Putting the two together is the obvious thing to try, and the roadmap records
the expected outcome *before* it is run: **combining them should move the
interval very little**, because the regime signal enters through the mean and
the interval's width is dominated by a term that does not depend on the mean.
If that holds, RQ3's answer is "miscalibration is a level problem, not a timing
problem", the combined model is a **pre-registered null**, and the deliverable
is a global width correction rather than a regime-conditional one.

Recording the prediction is what makes the null publishable. A null that was
expected is evidence; a null discovered after the fact reads as a failed
experiment.

The model
---------
:class:`MCDropoutRegimeExpertForecaster` is exactly
:class:`~src.models.deep.regime_lstm.RegimeExpertForecaster` — the Phase-5
mixture-of-experts, chosen as "best of Phase 5" on the transitional result —
with **training inherited unchanged** and only *inference* altered, in precise
parallel to how :class:`~src.models.deep.uncertainty.MCDropoutLSTMForecaster`
relates to the Phase-3 LSTM. With the same hyperparameters and seed the fitted
experts are the Phase-5 experts, so the combined model adds uncertainty, not a
different point model. That parallel is the whole design: it means any
difference in calibration between ``MC-Dropout-LSTM`` and
``MC-Dropout-Regime-LSTM-B`` is attributable to regime conditioning and nothing
else.

Per forecast day, ``T`` stochastic passes are drawn and each pass is recombined
**through the gate** before the moments are taken:

    z_t^(j) = Σ_k  g_k(t)  ·  f_k^(j)(x_t)        j = 1 … T

so the epistemic variance is the variance of the *mixture*, not the mixture of
per-expert variances. This is the right object and the difference is not
cosmetic: the two coincide only when the experts' dropout noise is independent
of the gate, which is exactly what a regime-specialised mixture is designed to
violate. Taking the variance of the mixture also means the gate's own
concentration shows up in the interval — on a day when the posterior is split
across two experts that disagree, the predictive spread widens, which is the
one mechanism by which this model *could* produce regime-dependent width.

Aleatoric noise is the inherited ``_smear_var``: the variance of the mixture's
log residuals on the held-out tail (:meth:`RegimeExpertForecaster._fit`), so the
predictive law is

    log RV_t | F_(t-1)  ~  N( mu_t , s2_epi(t) + s2_ale )

and, on the variance scale, log-normal — the identical contract Phase 6's
calibration code consumes, so every table, figure and test written for
``MC-Dropout-LSTM`` applies to this model with no change.

Leakage discipline is inherited verbatim (roadmap §1.3): scalers and experts fit
on ``dates <= fold.train_end`` only, windows ending strictly at ``t-1``, and the
gate for day ``t`` explicitly lagged to the posterior of ``t-1``. Dropout masks
change what is emitted, never when the inputs are known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # torch is a Phase-3+ dependency; import at module load so failures are loud.
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for the Phase 7 combined model. Install it "
        "(pyproject optional-dependency group 'deep')."
    ) from exc

from src.data.sequence import build_feature_target, make_windows
from src.data.splits import Fold
from src.models.deep.lstm import _VAR_FLOOR
from src.models.deep.regime_lstm import (
    DEFAULT_BASE_FEATURES,
    DEFAULT_GATE_COLS,
    RegimeExpertForecaster,
)
from src.models.deep.uncertainty import _enable_mc_dropout
from src.utils.logging import get_logger

log = get_logger("regime_uncertainty")


class MCDropoutRegimeExpertForecaster(RegimeExpertForecaster):
    """Regime mixture-of-experts with MC-Dropout intervals (Phase 7).

    Training is inherited from :class:`RegimeExpertForecaster` unchanged.
    Inference runs ``mc_samples`` stochastic passes with the head dropout left
    active in **every** expert, recombines each pass through the (lagged) regime
    gate, and reports the resulting Gaussian law on log-variance:

    * ``mu``          — mean of the ``T`` gate-mixed log-variance draws;
    * ``s2_epi``      — variance of those ``T`` mixtures (model uncertainty,
      including any disagreement the gate exposes between experts);
    * ``s2_ale``      — the inherited held-out mixture-residual variance;
    * ``sigma``       — ``sqrt(s2_epi + s2_ale)``.

    The point forecast is the log-normal mean ``exp(mu + sigma²/2)``, which is
    the mean-unbiased conditional-variance forecast (Patton 2011) and reduces to
    the Phase-5 smeared mixture forecast as the epistemic term vanishes.

    Parameters (beyond :class:`RegimeExpertForecaster`)
    ---------------------------------------------------
    mc_samples : number of stochastic passes ``T``. Keep this equal to Phase 6's
        (100) — the epistemic term is a variance over ``T`` draws and shrinks
        systematically with ``T``, so an unmatched ``T`` would make the Phase-6
        and Phase-7 intervals incomparable for reasons having nothing to do with
        regimes.
    mc_seed : torch RNG seed for the dropout masks, so every interval is
        reproducible.

    Notes
    -----
    ``dropout`` must be > 0: the head dropout is the only source of epistemic
    samples, and a zero rate would silently yield ``s2_epi = 0`` — an
    "uncertainty-aware" model whose uncertainty is a constant. That failure is
    quiet enough to survive a whole results chapter, so it raises here.
    """

    def __init__(
        self,
        *,
        mc_samples: int = 100,
        mc_seed: int | None = None,
        base_features: tuple[str, ...] = DEFAULT_BASE_FEATURES,
        gate_cols: tuple[str, ...] = DEFAULT_GATE_COLS,
        dropout: float = 0.1,
        name: str = "MC-Dropout-Regime-LSTM-B",
        **kwargs,
    ) -> None:
        super().__init__(
            base_features=base_features,
            gate_cols=gate_cols,
            dropout=dropout,
            name=name,
            **kwargs,
        )
        if self.dropout <= 0.0:
            raise ValueError(
                "MC-Dropout needs dropout > 0 (the head dropout is the source of "
                "the epistemic samples); got dropout=0."
            )
        self.mc_samples = int(mc_samples)
        if self.mc_samples < 2:
            raise ValueError("mc_samples must be >= 2 to form a predictive variance")
        self.mc_seed = int(mc_seed) if mc_seed is not None else self.seed

    # ---- config summary (extends the base) --------------------------------- #
    def config(self) -> dict:
        c = super().config()
        c.update({"mc_samples": self.mc_samples, "mc_seed": self.mc_seed})
        return c

    # ---- stochastic mixture ------------------------------------------------- #
    def _mixture_draws(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        """``(T, n)`` gate-mixed predictions with dropout active in every expert.

        Each of the ``T`` passes draws one dropout mask per expert and mixes the
        experts' outputs with the gate *within* that pass, so the returned rows
        are draws of the mixture. Taking the variance of these rows is the
        epistemic variance of the model actually being used; averaging per-expert
        variances instead would discard the gate's contribution entirely.
        """
        if len(X) == 0:
            return np.empty((self.mc_samples, 0), dtype=float)
        Xt = self._t(X)
        for model in self._experts:
            model.eval()
            _enable_mc_dropout(model)
        # Seed once for the whole T x K mask sequence, so the intervals are
        # reproducible run to run (matching MCDropoutLSTMForecaster).
        torch.manual_seed(self.mc_seed)
        draws = np.empty((self.mc_samples, len(X)), dtype=float)
        with torch.no_grad():
            for t in range(self.mc_samples):
                cols = [m(Xt).cpu().numpy().ravel() for m in self._experts]
                P = np.stack(cols, axis=1)              # (n, K) this pass
                draws[t] = (P * G).sum(axis=1)          # mix inside the pass
        for model in self._experts:
            model.eval()                                # restore determinism
        return draws

    # ---- predictive frame --------------------------------------------------- #
    def _predictive_frame(self, feats, target, gate, index, fold) -> pd.DataFrame:
        if self._experts is None or self._feat_scaler is None:
            raise RuntimeError(f"{self.name}: predict called before fit.")
        feats_s = self._feat_scaler.transform(feats.to_numpy())
        target_s = self._targ_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
        X, _, y_dates = make_windows(feats_s, target_s, index, self.lookback)
        sel = np.asarray((y_dates >= fold.predict_start) & (y_dates <= fold.predict_end))
        cols = ["mean", "mu_log", "sigma_log", "sd_epistemic", "sd_aleatoric",
                "mu_log_det", "gate_entropy"]
        if not sel.any():
            return pd.DataFrame(columns=cols)

        G_all = self._gate_for(gate, y_dates)
        G = G_all[sel]
        Xs = X[sel]
        tmean = float(self._targ_scaler.mean_[0])
        tstd = float(self._targ_scaler.std_[0])

        # Deterministic mixture (dropout OFF) -> the Phase-5 point in log space,
        # kept for the aleatoric-only reference interval and as the check that
        # this model's point forecast still is the Phase-5 one.
        z_det = self._mixture_pred(Xs, G)
        mu_log_det = z_det * tstd + tmean

        draws = self._mixture_draws(Xs, G) * tstd + tmean   # (T, n) log-variance
        mu = draws.mean(axis=0)
        s2_epi = draws.var(axis=0, ddof=0)
        s2_ale = float(self._smear_var)
        sigma = np.sqrt(s2_epi + s2_ale)
        mean_var = np.clip(np.exp(mu + 0.5 * (s2_epi + s2_ale)), _VAR_FLOOR, None)

        # Gate entropy (nats) is the diagnostic for the one mechanism by which
        # this model could be regime-sensitive in *width*: a split posterior
        # mixes disagreeing experts and should widen the epistemic term. Carried
        # into the predictions parquet so Phase 7 can test that directly instead
        # of asserting it.
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(np.where(G > 0, G * np.log(G), 0.0), axis=1)

        return pd.DataFrame(
            {
                "mean": mean_var,
                "mu_log": mu,
                "sigma_log": sigma,
                "sd_epistemic": np.sqrt(s2_epi),
                "sd_aleatoric": np.full(int(sel.sum()), np.sqrt(s2_ale)),
                "mu_log_det": mu_log_det,
                "gate_entropy": ent,
            },
            index=pd.DatetimeIndex(y_dates[sel]),
        )

    # ---- public API --------------------------------------------------------- #
    def forecast_fold_uq(self, frame: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Full predictive frame for the fold (same columns as the Phase-6
        MC-Dropout model, plus ``gate_entropy``)."""
        feats, target, index = build_feature_target(frame, self.base_features)
        gate = frame[list(self.gate_cols)].reindex(index).shift(1)
        if self._need_refit(fold):
            self._fit(feats, target, gate, index, fold)
        return self._predictive_frame(feats, target, gate, index, fold)

    def forecast_fold(self, frame: pd.DataFrame, fold: Fold) -> pd.Series:
        """VolForecaster point forecast = the log-normal predictive **mean**."""
        return self.forecast_fold_uq(frame, fold)["mean"].rename(self.name)
