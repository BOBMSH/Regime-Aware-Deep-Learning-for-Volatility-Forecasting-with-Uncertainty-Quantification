"""Unit tests for walk-forward split generation.

Leakage discipline is the single biggest threat to dissertation credibility
(roadmap §1.3), so these tests are deliberately paranoid: every fold's train
window must end strictly before its prediction window starts, no two folds'
prediction windows may overlap, and the union of prediction windows must
exactly cover the requested OOS segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    Fold,
    SplitConfig,
    fixed_three_way_split,
    walk_forward_folds,
)


@pytest.fixture
def trading_index() -> pd.DatetimeIndex:
    """All US business days 2000-01-03 .. 2024-12-31."""
    return pd.bdate_range("2000-01-03", "2024-12-31")


@pytest.fixture
def split_cfg() -> SplitConfig:
    return SplitConfig(
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2024-12-31",
        refit_frequency_days=21,
        scheme="anchored",
    )


class TestSplitConfig:
    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            SplitConfig(
                train_end="2019-12-31",
                val_start="2020-01-01",
                val_end="2019-06-30",
                test_start="2022-01-01",
                test_end="2024-12-31",
            )

    def test_rejects_bad_scheme(self):
        with pytest.raises(ValueError):
            SplitConfig(
                train_end="2019-12-31",
                val_start="2020-01-01",
                val_end="2021-12-31",
                test_start="2022-01-01",
                test_end="2024-12-31",
                scheme="random",  # type: ignore[arg-type]
            )

    def test_rejects_bad_refit_freq(self):
        with pytest.raises(ValueError):
            SplitConfig(
                train_end="2019-12-31",
                val_start="2020-01-01",
                val_end="2021-12-31",
                test_start="2022-01-01",
                test_end="2024-12-31",
                refit_frequency_days=0,
            )


class TestFixedThreeWaySplit:
    def test_partitions_are_disjoint(self, trading_index, split_cfg):
        parts = fixed_three_way_split(trading_index, split_cfg)
        train, val, test = parts["train"], parts["val"], parts["test"]
        assert train.intersection(val).empty
        assert val.intersection(test).empty
        assert train.intersection(test).empty

    def test_train_window_ends_at_2019(self, trading_index, split_cfg):
        parts = fixed_three_way_split(trading_index, split_cfg)
        assert parts["train"].max().year == 2019
        assert parts["val"].min().year == 2020
        assert parts["test"].min().year == 2022

    def test_handles_non_monotonic_input(self, split_cfg):
        idx = pd.DatetimeIndex(["2022-01-05", "2022-01-03", "2022-01-04"])
        parts = fixed_three_way_split(idx, split_cfg)
        assert list(parts["test"]) == sorted(parts["test"])

    def test_rejects_duplicate_dates(self, split_cfg):
        idx = pd.DatetimeIndex(["2022-01-03", "2022-01-03"])
        with pytest.raises(ValueError, match="unique"):
            fixed_three_way_split(idx, split_cfg)


class TestWalkForwardFolds:
    def _collect(self, idx, cfg, **kwargs) -> list[Fold]:
        return list(walk_forward_folds(idx, cfg, **kwargs))

    def test_produces_at_least_one_fold(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        assert len(folds) > 0

    def test_no_leakage_between_train_and_predict(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        for f in folds:
            # Strict: training ends BEFORE prediction starts.
            assert f.train_end < f.predict_start, (
                f"Fold {f.fold_idx} has train_end {f.train_end} >= predict_start {f.predict_start}"
            )

    def test_anchored_train_starts_at_index_start(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        anchor = trading_index[0]
        for f in folds:
            assert f.train_start == anchor

    def test_rolling_train_is_constant_length(self, trading_index, split_cfg):
        rolling_cfg = SplitConfig(
            train_end=split_cfg.train_end,
            val_start=split_cfg.val_start,
            val_end=split_cfg.val_end,
            test_start=split_cfg.test_start,
            test_end=split_cfg.test_end,
            refit_frequency_days=split_cfg.refit_frequency_days,
            scheme="rolling",
        )
        folds = self._collect(trading_index, rolling_cfg)
        lengths = []
        for f in folds:
            lengths.append(int(((trading_index >= f.train_start) & (trading_index <= f.train_end)).sum()))
        # All folds have within +/- 1 of the same training length.
        assert max(lengths) - min(lengths) <= 1

    def test_predict_windows_are_contiguous_and_cover_oos(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        windows = [(f.predict_start, f.predict_end) for f in folds]
        # No overlaps between consecutive fold predict windows.
        for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
            assert e1 < s2, f"fold predict windows overlap: ({s1}, {e1}) and ({s2}, {e2})"
        # Union covers the entire test segment.
        oos_dates = trading_index[
            (trading_index >= pd.Timestamp(split_cfg.test_start))
            & (trading_index <= pd.Timestamp(split_cfg.test_end))
        ]
        covered = pd.DatetimeIndex([])
        for f in folds:
            covered = covered.append(
                trading_index[(trading_index >= f.predict_start) & (trading_index <= f.predict_end)]
            )
        assert set(covered) == set(oos_dates)

    def test_refit_cadence_matches_config(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        # Each fold's predict window should contain at most refit_frequency_days dates.
        for f in folds:
            window_len = int(
                (
                    (trading_index >= f.predict_start) & (trading_index <= f.predict_end)
                ).sum()
            )
            assert window_len <= split_cfg.refit_frequency_days

    def test_eval_segment_val(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg, eval_segment="val")
        assert all(f.predict_start.year >= 2020 for f in folds)
        assert all(f.predict_end.year <= 2021 for f in folds)

    def test_eval_segment_val_plus_test(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg, eval_segment="val+test")
        assert min(f.predict_start for f in folds).year == 2020
        assert max(f.predict_end for f in folds).year == 2024

    def test_masks_are_correct(self, trading_index, split_cfg):
        folds = self._collect(trading_index, split_cfg)
        for f in folds[:3]:
            tm = f.train_mask(trading_index)
            pm = f.predict_mask(trading_index)
            # Train and predict masks must not both be True for any date.
            assert not np.any(tm & pm)

    def test_min_train_size_skips_undersized_folds(self, trading_index, split_cfg):
        tight = SplitConfig(
            train_end="2000-02-29",
            val_start="2000-03-01",
            val_end="2000-04-30",
            test_start="2000-05-01",
            test_end="2000-06-30",
            refit_frequency_days=5,
            scheme="anchored",
            min_train_size=10_000,  # impossible to reach in this synthetic window
        )
        folds = self._collect(trading_index, tight)
        assert folds == []
