"""Regime-label timing for per-regime evaluation (roadmap Phase 5/6; RQ2, RQ3).

Why this module exists
----------------------
Every per-regime table in this project answers a question of the form *"is the
model better / better-calibrated in state k?"*. Answering it requires bucketing
each out-of-sample day by a regime label — and **which** label you use decides
the answer.

The Phase-4 signal is the causal filtered posterior ``P(s_t | x_1..x_t)``. It is
leakage-free *as a forecast input*, because a forecast for ``RV_t`` only ever
sees the window ending at ``t-1``. But ``s_t`` itself is inferred **using the
day-t observation**. Bucketing day ``t``'s forecast error by ``state[t]`` therefore
sorts days using the very outcome whose error is being measured inside the
bucket — selection on the outcome. The forecasts stay clean; the *evaluation*
does not.

Two timings, two different questions
------------------------------------
* ``shift=1`` (**the default, and the only one that supports a conditional
  claim**). Bucket day ``t`` by ``state[t-1]`` — the state a user of the model
  actually knew when the forecast was issued. This is the same information set
  the regime-aware models condition on (their input window ends at ``t-1``) and
  the same one :func:`src.evaluation.significance.regime_test_function` passes to
  the Giacomini–White test, which is the formally-sized version of exactly this
  question. Statements of the form *"the gain is concentrated in state k"* are
  only meaningful under this timing, because only under this timing is "state k"
  something you could have acted on.

* ``shift=0`` (**ex-post description only**). Bucket day ``t`` by ``state[t]``.
  This describes what the market turned out to be doing on that day and is fine
  for narrative colour ("the COVID crash falls in the crisis state"), but it
  cannot support a conditional performance or calibration claim.

The difference is not academic. On the dissertation's test window the two
timings disagree about which model wins the calm and crisis buckets, and the
``shift=0`` table shows a monotone coverage degradation with regime severity that
``shift=1`` does not. The mechanism is that regime-*transition* days — whose label
is decided by the day-t observation, and which carry a far higher interval-miss
rate than non-transition days — are moved wholesale between buckets by the choice
of timing. :func:`transition_day_summary` quantifies exactly that, and is the
honest way to report the effect: not "coverage degrades with regime severity"
(an artefact) but "coverage collapses at regime transitions" (robust).

Nothing here is model code — it is pure pandas index arithmetic, so it is
unit-testable on its own and shared by the Phase-5 point tables and the Phase-6
calibration tables, which is what stops the two from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Default number of days to lag the regime label before bucketing. ``1`` puts
#: the label in the ``t-1`` information set; see the module docstring.
DEFAULT_REGIME_SHIFT = 1

#: Human-readable tag for each supported timing, used in table columns, figure
#: captions and milestone prose so a reader never has to guess which one a
#: number came from.
REGIME_TIMING_LABEL = {
    0: "contemporaneous (ex-post description)",
    1: "lagged t-1 (conditional / implementable)",
}


def regime_timing_label(shift: int) -> str:
    """One-line description of a bucketing timing, for tables and prose."""
    return REGIME_TIMING_LABEL.get(int(shift), f"lagged t-{int(shift)}")


def align_regime_label(
    reg_state: pd.Series,
    index: pd.Index,
    *,
    shift: int = DEFAULT_REGIME_SHIFT,
) -> pd.Series:
    """Return the regime label to bucket each date in ``index`` by.

    Parameters
    ----------
    reg_state : date-indexed causal hard regime label (Phase-4 ``*_filt_state``),
        dated at ``t`` — i.e. **not** pre-lagged.
    index : the dates being evaluated (the out-of-sample forecast dates).
    shift : days to lag the label. ``1`` (default) puts it in the ``t-1``
        information set; ``0`` keeps the contemporaneous label. See the module
        docstring for which question each answers.

    Returns
    -------
    A float Series on ``index``. Dates whose lagged label is unavailable (the
    first ``shift`` rows of the *full* series) are ``NaN`` and are dropped by the
    callers rather than silently bucketed.

    Notes
    -----
    The shift is applied on ``reg_state``'s **own** index before reindexing onto
    ``index``, so it means "the previous trading day in the regime series", not
    "the previous row of whatever subset is being scored". Reindexing first and
    shifting after would silently reach across a gap at the start of the
    evaluation window and mislabel its first day.
    """
    s = int(shift)
    if s < 0:
        raise ValueError(f"shift must be >= 0, got {shift}")
    lab = pd.Series(reg_state).astype("float")
    if s:
        lab = lab.shift(s)
    return lab.reindex(index)


def transition_day_summary(
    reg_state: pd.Series,
    index: pd.Index,
    *,
    shift: int = DEFAULT_REGIME_SHIFT,
) -> dict:
    """Describe the days whose bucket changes between the two timings.

    These are the regime-*transition* days: ``state[t] != state[t-shift]``. They
    are the entire difference between a ``shift=0`` and a ``shift=1`` per-regime
    table, and they are where prediction intervals actually fail — so reporting
    them directly is both more honest and more informative than reporting a
    per-regime split that silently depends on the convention.

    Returns a dict with ``n`` (evaluated days with both labels available),
    ``n_transition``, ``share`` and a boolean ``mask`` Series on ``index``.
    """
    cur = align_regime_label(reg_state, index, shift=0)
    lag = align_regime_label(reg_state, index, shift=shift)
    both = cur.notna() & lag.notna()
    mask = (cur != lag) & both
    n = int(both.sum())
    return {
        "n": n,
        "n_transition": int(mask.sum()),
        "share": float(mask.sum() / n) if n else float("nan"),
        "mask": mask.astype(bool),
    }


def bucket_masks(
    reg_state: pd.Series,
    index: pd.Index,
    n_states: int,
    *,
    shift: int = DEFAULT_REGIME_SHIFT,
) -> list[np.ndarray]:
    """Boolean masks over ``index``, one per state, under the chosen timing."""
    lab = align_regime_label(reg_state, index, shift=shift)
    return [np.asarray(lab == float(k)) for k in range(int(n_states))]
