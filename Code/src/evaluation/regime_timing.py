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
of timing. :func:`transition_day_summary` quantifies exactly that.

The same rule governs the *selector*, not just the label
--------------------------------------------------------
Lagging the bucket label is necessary but **not sufficient**. A statement of the
form *"the model fails on days of type X"* also requires the indicator for X to be
measurable at ``t-1``. The obvious candidate here — "days when the regime changed",
``1{s_t != s_{t-1}}`` — is **not**: ``s_t`` is inferred from the day-t return, so a
regime transition is *detected by* the very surprise that widens a forecast error
and breaks a prediction interval. Splitting on it is convention-*free* (both label
timings agree on which days those are) but it is still selection on the outcome —
the same error as above, one level down.

Measured on the Phase-6 predictions, that selection *is* the whole effect
(MC-Dropout, 90% nominal, 784 test days):

* ``1{s_t != s_{t-1}}``     — ex-post, NOT in F_{t-1}: PICP 0.712 flagged vs 0.876 rest (n=52)
* ``1{s_{t-1} != s_{t-2}}`` — ex-ante, in F_{t-1}:     PICP 0.885 flagged vs 0.863 rest (n=52)

Shifting the selector by one day removes the contrast entirely. Realized variance
on ex-post transition days is roughly twice the median of stable days, which is the
mechanism. Accordingly :func:`transition_mask` and :func:`transition_day_summary`
take an explicit ``selector_shift`` that **defaults to the implementable timing**;
``selector_shift=0`` is retained only for the descriptive convention-gap count and
is stamped ``implementable=False`` wherever it reaches disk or prose.

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

#: Default lag applied to the *transition selector* ``1{s_(t-k) != s_(t-k-1)}``.
#: ``1`` is the only value that keeps the selector inside the ``t-1`` information
#: set; ``0`` reads the day-t observation and is descriptive only. See the module
#: docstring, "The same rule governs the selector".
DEFAULT_SELECTOR_SHIFT = 1

#: Human-readable tag per selector timing, written into every table that splits on
#: transitions so a reader can never mistake the ex-post split for a conditional one.
SELECTOR_TIMING_LABEL = {
    0: "ex-post 1{s_t != s_(t-1)} — uses the day-t return; DESCRIPTIVE ONLY",
    1: "ex-ante 1{s_(t-1) != s_(t-2)} — known at t-1; implementable",
}


def regime_timing_label(shift: int) -> str:
    """One-line description of a bucketing timing, for tables and prose."""
    return REGIME_TIMING_LABEL.get(int(shift), f"lagged t-{int(shift)}")


def selector_timing_label(selector_shift: int) -> str:
    """One-line description of a transition-selector timing, for tables and prose."""
    k = int(selector_shift)
    return SELECTOR_TIMING_LABEL.get(k, f"ex-ante 1{{s_(t-{k}) != s_(t-{k + 1})}} — known at t-{k}")


def selector_is_implementable(selector_shift: int) -> bool:
    """Is a transition selector at this lag measurable with information at ``t-1``?

    ``True`` only for ``selector_shift >= 1``. Call this (or read the
    ``implementable`` field the helpers below return) before attaching any
    hypothesis test or conditional claim to a transition split: a Kupiec or
    Christoffersen test run on a subsample chosen with the day-t outcome does not
    have its nominal size, and will report a small p-value by construction.
    """
    return int(selector_shift) >= 1


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


def transition_mask(
    reg_state: pd.Series,
    index: pd.Index,
    *,
    selector_shift: int = DEFAULT_SELECTOR_SHIFT,
) -> tuple[pd.Series, pd.Series]:
    """"The state changed" indicator under an **explicit** selector timing.

    Returns ``(mask, scored)``: the boolean transition indicator on ``index``, and
    the boolean mask of rows where both labels the selector needs are available
    (so the caller can score "flagged" and "rest" over the same denominator).

    Parameters
    ----------
    selector_shift : lag applied to the state pair being compared.

        * ``1`` (default) → ``1{s_(t-1) != s_(t-2)}``. Both labels are known at
          ``t-1``, so this is the only timing on which a conditional claim, a
          coverage test, or a recalibration rule can be built.
        * ``0`` → ``1{s_t != s_(t-1)}``. Reads the day-t observation. It is also
          exactly the set of days whose *bucket* differs between the ``shift=0``
          and ``shift=1`` per-regime tables, which is the one legitimate use:
          quantifying the size of the labelling convention. It cannot support a
          performance or calibration claim — see the module docstring.
    """
    k = int(selector_shift)
    if k < 0:
        raise ValueError(f"selector_shift must be >= 0, got {selector_shift}")
    a = align_regime_label(reg_state, index, shift=k)
    b = align_regime_label(reg_state, index, shift=k + 1)
    scored = a.notna() & b.notna()
    mask = (a != b) & scored
    return mask.astype(bool), scored.astype(bool)


def transition_day_summary(
    reg_state: pd.Series,
    index: pd.Index,
    *,
    selector_shift: int = DEFAULT_SELECTOR_SHIFT,
    **legacy,
) -> dict:
    """Summarise the regime-*transition* days under an explicit selector timing.

    Returns a dict with ``n`` (evaluated days where the selector is defined),
    ``n_transition``, ``share``, a boolean ``mask`` and a boolean ``scored`` Series
    on ``index``, plus the provenance fields ``selector_shift``, ``selector`` and
    ``implementable``. Always carry those three through to any table or sentence
    built from ``mask`` — the ex-post split looks identical to the ex-ante one and
    means something completely different.

    At ``selector_shift=0`` this is also the count of days whose per-regime bucket
    differs between the two label timings, which is what the milestone notes use it
    for; that call must pass ``selector_shift=0`` explicitly.
    """
    if "shift" in legacy:
        # Pre-2026-08-19 signature. ``shift`` meant the *label* lag and produced
        # the ex-post selector, which is precisely the defect this rename exists to
        # surface. Fail loudly rather than silently answering a different question.
        raise TypeError(
            "transition_day_summary(shift=...) was renamed to selector_shift=... "
            "and its default changed from the ex-post selector 1{s_t != s_(t-1)} "
            "to the implementable 1{s_(t-1) != s_(t-2)}. Pass selector_shift=0 for "
            "the descriptive convention-gap count, selector_shift=1 for anything "
            "that supports a claim. See src/evaluation/regime_timing.py."
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {sorted(legacy)}")

    mask, scored = transition_mask(reg_state, index, selector_shift=selector_shift)
    n = int(scored.sum())
    return {
        "n": n,
        "n_transition": int(mask.sum()),
        "share": float(mask.sum() / n) if n else float("nan"),
        "mask": mask,
        "scored": scored,
        "selector_shift": int(selector_shift),
        "selector": selector_timing_label(selector_shift),
        "implementable": selector_is_implementable(selector_shift),
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
