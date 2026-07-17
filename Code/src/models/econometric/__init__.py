"""Econometric baselines (roadmap Phase 2, §1.2).

Three confirmed baselines plus one reference floor:

* :class:`~src.models.econometric.garch.GARCHForecaster` -- GARCH(1,1),
  Bollerslev (1986). The canonical short-memory conditional-variance model that
  any alternative must beat (Hansen & Lunde 2005).
* :class:`~src.models.econometric.egarch.EGARCHForecaster` -- EGARCH(1,1,1),
  Nelson (1991). Captures the leverage effect; included as a falsification check
  (Chapter 2 §2.2) so a regime-aware DL win over GARCH(1,1) cannot be dismissed
  as merely capturing asymmetric shock responses.
* :class:`~src.models.econometric.har.HARForecaster` -- HAR-RV, Corsi (2009).
  The de-facto benchmark on realized-volatility targets.
* :class:`~src.models.econometric.naive.RandomWalkRVForecaster` -- forecast
  tomorrow's variance with today's. Not a dissertation baseline; a sanity floor
  that validates the harness and anchors interpretation of the others.
"""

from __future__ import annotations

from src.models.econometric.egarch import EGARCHForecaster
from src.models.econometric.garch import GARCHForecaster
from src.models.econometric.har import HARForecaster
from src.models.econometric.naive import RandomWalkRVForecaster

__all__ = [
    "GARCHForecaster",
    "EGARCHForecaster",
    "HARForecaster",
    "RandomWalkRVForecaster",
]
