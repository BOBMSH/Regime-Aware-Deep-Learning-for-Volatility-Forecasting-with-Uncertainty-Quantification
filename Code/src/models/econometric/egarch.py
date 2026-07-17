"""EGARCH baseline via ``arch`` (roadmap Phase 2, §1.2).

EGARCH (Nelson 1991) models the *logarithm* of conditional variance, which
guarantees positivity without parameter constraints and lets innovations enter
through both their sign and magnitude -- so the leverage effect (negative shocks
raising future variance more than positive shocks of equal size) is captured by a
smooth functional form rather than a threshold.

Its role in the dissertation is a **falsification check** (Chapter 2 §2.2). If a
regime-aware deep model beats GARCH(1,1) but not EGARCH, the improvement may be
absorbed asymmetric-response behaviour rather than genuine regime-conditional
non-linearity -- a distinction central to RQ2. Implementation cost over GARCH is
one extra order argument (``o=1``, the asymmetric term), so it reuses the shared
:class:`~src.models.econometric.garch.ArchVolForecaster` base wholesale.
"""

from __future__ import annotations

from src.models.econometric.garch import RETURN_COL, ArchVolForecaster


class EGARCHForecaster(ArchVolForecaster):
    """EGARCH(1,1,1) -- Nelson (1991), second econometric baseline.

    Parameters mirror GARCH plus ``o`` (asymmetric order). ``p=o=q=1`` is the
    standard specification: one lag of the standardised innovation (magnitude),
    one asymmetric term (sign), and one lag of log-variance (persistence).
    """

    def __init__(
        self,
        *,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = "Constant",
        dist: str = "normal",
        scale: float = 100.0,
        name: str = "EGARCH",
        return_col: str = RETURN_COL,
    ) -> None:
        super().__init__(
            name,
            vol="EGARCH",
            arch_kwargs={"p": p, "o": o, "q": q},
            mean=mean,
            dist=dist,
            scale=scale,
            return_col=return_col,
        )
