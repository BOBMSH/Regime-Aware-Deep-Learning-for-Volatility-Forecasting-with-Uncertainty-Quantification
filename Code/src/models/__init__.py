"""Model implementations.

Sub-packages:

* ``econometric`` -- GARCH(1,1), EGARCH, HAR-RV and a random-walk reference
  (Phase 2). These are the baselines every later model is measured against.
* ``deep`` -- LSTM and its regime-aware / uncertainty-aware variants (Phases 3+).
* ``regime`` -- HMM regime detection (Phase 4+).

Every model, regardless of sub-package, satisfies the
:class:`src.evaluation.rolling.VolForecaster` protocol so it can be run through
the shared walk-forward engine.
"""
