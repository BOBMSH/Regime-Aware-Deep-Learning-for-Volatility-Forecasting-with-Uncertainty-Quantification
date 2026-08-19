"""Tests for regime-label **and selector** timing in per-regime evaluation.

These guard two fixes to the same defect, found in successive audits:

1. 2026-08-19 (ii) — the *label*. The Phase-4 filtered state ``s_t`` is inferred
   using the day-``t`` observation, so bucketing day ``t``'s forecast error by
   ``state[t]`` sorts days by the very outcome whose error is measured inside the
   bucket. ``DEFAULT_REGIME_SHIFT`` is now 1.
2. 2026-08-19 (iii) — the *selector*. Lagging the label is not enough: the
   transition indicator ``1{s_t != s_(t-1)}`` reads the day-t return too, so
   splitting coverage on it is the same error one level down.
   ``DEFAULT_SELECTOR_SHIFT`` is now 1, and every transition split carries an
   ``implementable`` flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.regime_timing import (
    DEFAULT_REGIME_SHIFT,
    DEFAULT_SELECTOR_SHIFT,
    align_regime_label,
    bucket_masks,
    regime_timing_label,
    selector_is_implementable,
    selector_timing_label,
    transition_day_summary,
    transition_mask,
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
        # states: 0 0 0 1 1 2 2 1 0 0 -> ex-post switches at positions 3, 5, 7, 8
        out = transition_day_summary(states, states.index, selector_shift=0)
        assert out["n"] == 9                     # first day has no t-1 label
        assert out["n_transition"] == 4
        assert out["share"] == pytest.approx(4 / 9)

    def test_mask_marks_exactly_the_switch_days(self, states):
        out = transition_day_summary(states, states.index, selector_shift=0)
        marked = list(np.flatnonzero(out["mask"].to_numpy()))
        assert marked == [3, 5, 7, 8]

    def test_constant_state_has_no_transitions(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        s = pd.Series(1.0, index=idx)
        out = transition_day_summary(s, idx, selector_shift=0)
        assert out["n_transition"] == 0
        assert out["share"] == pytest.approx(0.0)


class TestSelectorTiming:
    """The 2026-08-19 (iii) fix: the transition *selector* must also live in F_(t-1)."""

    def test_default_selector_is_implementable(self):
        assert DEFAULT_SELECTOR_SHIFT == 1
        assert selector_is_implementable(DEFAULT_SELECTOR_SHIFT)
        assert not selector_is_implementable(0)

    def test_selector_label_flags_the_expost_variant_loudly(self):
        assert "DESCRIPTIVE ONLY" in selector_timing_label(0)
        assert "implementable" in selector_timing_label(1)

    def test_exante_selector_is_the_expost_one_shifted_by_a_day(self, states):
        ex_post, _ = transition_mask(states, states.index, selector_shift=0)
        ex_ante, _ = transition_mask(states, states.index, selector_shift=1)
        # same rule, one day later: {3,5,7,8} -> {4,6,8,9}
        assert list(np.flatnonzero(ex_post.to_numpy())) == [3, 5, 7, 8]
        assert list(np.flatnonzero(ex_ante.to_numpy())) == [4, 6, 8, 9]

    def test_the_two_selectors_are_not_interchangeable(self, states):
        """The whole point of the fix: these flag different days, so a result
        computed on one is not evidence about the other."""
        ex_post, _ = transition_mask(states, states.index, selector_shift=0)
        ex_ante, _ = transition_mask(states, states.index, selector_shift=1)
        assert not ex_post.equals(ex_ante)

    def test_exante_selector_uses_no_information_after_t_minus_1(self, states):
        """Perturbing the state on day t must not change day t's ex-ante flag.

        This is the invariant the ex-post selector violates, and it is the reason
        a Kupiec/Christoffersen test on an ex-post transition subsample does not
        have its nominal size.
        """
        idx = states.index
        base_ante, _ = transition_mask(states, idx, selector_shift=1)
        for t in range(len(states)):
            for delta in (1, 2):
                bumped = states.copy()
                bumped.iloc[t] = (bumped.iloc[t] + delta) % 3
                ante, _ = transition_mask(bumped, idx, selector_shift=1)
                assert ante.iloc[t] == base_ante.iloc[t], (
                    f"ex-ante selector at t={t} changed when only day t changed")

        # ...whereas the ex-post selector is *fully determined* by day t given the
        # past. Assert the contrast explicitly so the test fails if the default is
        # ever re-pointed at the contemporaneous rule.
        for t in range(1, len(states)):
            same = states.copy()
            same.iloc[t] = same.iloc[t - 1]
            diff = states.copy()
            diff.iloc[t] = (same.iloc[t - 1] + 1) % 3
            post_same, _ = transition_mask(same, idx, selector_shift=0)
            post_diff, _ = transition_mask(diff, idx, selector_shift=0)
            assert not post_same.iloc[t]
            assert post_diff.iloc[t]

    def test_summary_carries_provenance(self, states):
        out = transition_day_summary(states, states.index, selector_shift=1)
        assert out["selector_shift"] == 1
        assert out["implementable"] is True
        assert "s_(t-1)" in out["selector"]
        assert set(out["scored"].to_numpy()[:2]) == {False}  # first two lack a t-2 label

    def test_legacy_shift_kwarg_is_rejected_not_silently_reinterpreted(self, states):
        """`shift=` used to mean the ex-post selector. Silently accepting it would
        answer a different question with the same call site."""
        with pytest.raises(TypeError, match="selector_shift"):
            transition_day_summary(states, states.index, shift=1)

    def test_expost_selector_equals_the_bucket_disagreement_set(self, states):
        """selector_shift=0 is exactly the set of days whose per-regime bucket
        differs between the two label timings -- its one legitimate use."""
        mask, _ = transition_mask(states, states.index, selector_shift=0)
        cur = align_regime_label(states, states.index, shift=0)
        lag = align_regime_label(states, states.index, shift=DEFAULT_REGIME_SHIFT)
        differs = (cur != lag) & cur.notna() & lag.notna()
        assert mask.equals(differs.astype(bool))


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
