"""Tests for regime-label timing in per-regime evaluation.

These guard the fix for the outcome-conditioning defect found in the 2026-08-19
audit: the Phase-4 filtered state ``s_t`` is inferred using the day-``t``
observation, so bucketing day ``t``'s forecast error by ``state[t]`` sorts days
by the very outcome whose error is measured inside the bucket. The default
everywhere is now ``shift=1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    align_regime_label,
    bucket_masks,
    regime_timing_label,
    transition_day_summary,
)


@pytest.fixture
def states() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.Series([0, 0, 0, 1, 1, 2, 2, 1, 0, 0], index=idx, dtype=float)


class TestDefault:
    def test_default_is_one(self):
        assert DEFAULT_REGIME_SHIFT == 1

    def test_label_text_distinguishes_the_two_conventions(self):
        assert "ex-post" in regime_timing_label(0)
        assert "t-1" in regime_timing_label(1)


class TestAlign:
    def test_shift_one_returns_previous_day_label(self, states):
        out = align_regime_label(states, states.index, shift=1)
        assert pd.isna(out.iloc[0])                      # no t-1 for the first row
        assert list(out.iloc[1:]) == list(states.iloc[:-1])

    def test_shift_zero_is_identity(self, states):
        out = align_regime_label(states, states.index, shift=0)
        pd.testing.assert_series_equal(out, states.astype(float), check_names=False)

    def test_negative_shift_rejected(self, states):
        with pytest.raises(ValueError, match="shift must be >= 0"):
            align_regime_label(states, states.index, shift=-1)

    def test_shift_applied_on_the_full_series_not_the_subset(self, states):
        """Lagging must reach back across the start of the evaluation window.

        If the label were reindexed onto the evaluation dates *first* and shifted
        after, the first evaluated day would silently take the label of whatever
        row happened to precede it in the subset -- or be dropped. Here the
        evaluation window starts at position 5, whose true t-1 label is the one
        at position 4, and that must survive.
        """
        eval_idx = states.index[5:]
        out = align_regime_label(states, eval_idx, shift=1)
        assert out.notna().all()
        assert out.iloc[0] == states.iloc[4]

    def test_missing_dates_come_back_nan(self, states):
        extra = states.index.append(pd.DatetimeIndex(["2030-01-01"]))
        out = align_regime_label(states, extra, shift=1)
        assert pd.isna(out.iloc[-1])


class TestBucketMasks:
    def test_masks_partition_the_labelled_days(self, states):
        masks = bucket_masks(states, states.index, 3, shift=1)
        total = sum(int(m.sum()) for m in masks)
        assert total == len(states) - 1          # first day has no t-1 label
        assert not np.any(masks[0] & masks[1])   # mutually exclusive

    def test_masks_differ_between_timings(self, states):
        lag = bucket_masks(states, states.index, 3, shift=1)
        ex = bucket_masks(states, states.index, 3, shift=0)
        assert not np.array_equal(lag[1], ex[1])


class TestTransitionDays:
    def test_counts_switch_days(self, states):
        # states: 0 0 0 1 1 2 2 1 0 0 -> switches at positions 3, 5, 7, 8
        out = transition_day_summary(states, states.index, shift=1)
        assert out["n"] == 9                     # first day has no t-1 label
        assert out["n_transition"] == 4
        assert out["share"] == pytest.approx(4 / 9)

    def test_mask_marks_exactly_the_switch_days(self, states):
        out = transition_day_summary(states, states.index, shift=1)
        marked = list(np.flatnonzero(out["mask"].to_numpy()))
        assert marked == [3, 5, 7, 8]

    def test_constant_state_has_no_transitions(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        s = pd.Series(1.0, index=idx)
        out = transition_day_summary(s, idx, shift=1)
        assert out["n_transition"] == 0
        assert out["share"] == pytest.approx(0.0)


class TestConsistencyWithTheGWTestFunction:
    def test_same_lag_as_regime_test_function(self, states):
        """The descriptive table and the formal test must condition identically.

        The whole point of the fix is that the per-regime table and the
        Giacomini-White test now answer the same question; if these two helpers
        ever disagree about what "state k at t-1" means, they do not.
        """
        from src.evaluation.significance import regime_test_function

        H = regime_test_function(states, n_states=3, shift=1,
                                 labels=["calm", "transitional", "crisis"])
        lab = align_regime_label(states, states.index, shift=1)
        for k, col in enumerate(["calm", "transitional", "crisis"]):
            from_gw = H[col].fillna(0.0).to_numpy().astype(bool)
            from_timing = np.asarray(lab == float(k))
            assert np.array_equal(from_gw, from_timing)
