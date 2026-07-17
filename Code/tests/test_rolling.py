"""Unit tests for the walk-forward engine (roadmap Phase 2, §5 "the engine").

The engine is the single most safety-critical piece after the split generator:
it must cover the OOS window exactly, never leak, align every model on the same
dates, and round-trip through parquet. A trivial deterministic model is used so
the expected forecasts are known exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import har_lagged_rv
from src.data.splits import Fold, SplitConfig
from src.evaluation.rolling import (
    ACTUAL_COL,
    evaluate_predictions,
    run_walk_forward,
)
from src.models.econometric import RandomWalkRVForecaster
from src.utils.io import from_parquet, to_parquet


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 1400
    idx = pd.bdate_range("2015-01-01", periods=n)
    rv = pd.Series(np.exp(rng.normal(-9, 0.4, size=n)), index=idx, name="rv")
    ret = pd.Series(rng.normal(0, 0.01, size=n), index=idx, name="log_return")
    return pd.concat([ret, rv, har_lagged_rv(rv)], axis=1).dropna()


@pytest.fixture
def split_cfg() -> SplitConfig:
    return SplitConfig(
        train_end="2017-12-31", val_start="2018-01-01", val_end="2018-12-31",
        test_start="2019-01-01", test_end="2020-06-30", refit_frequency_days=21,
    )


class _ConstModel:
    """Forecasts a constant -- used to check coverage/alignment deterministically."""

    name = "const"

    def __init__(self, value=1e-4):
        self.value = value

    def forecast_fold(self, frame, fold: Fold) -> pd.Series:
        idx = frame.loc[fold.predict_start : fold.predict_end].index
        return pd.Series(self.value, index=idx, name=self.name)


class TestEngine:
    def test_covers_oos_window_exactly(self, frame, split_cfg):
        preds = run_walk_forward([_ConstModel()], frame, split_cfg, eval_segment="test")
        oos = frame.loc["2019-01-01":"2020-06-30"].index
        assert list(preds.index) == list(oos)
        assert preds["const"].notna().all()

    def test_actual_column_is_the_target(self, frame, split_cfg):
        preds = run_walk_forward([_ConstModel()], frame, split_cfg, eval_segment="test")
        np.testing.assert_allclose(
            preds[ACTUAL_COL].values, frame.loc[preds.index, "rv"].values
        )

    def test_multiple_models_aligned(self, frame, split_cfg):
        preds = run_walk_forward(
            [_ConstModel(1e-4), RandomWalkRVForecaster()], frame, split_cfg,
            eval_segment="test",
        )
        assert {"const", "RW-RV", ACTUAL_COL} <= set(preds.columns)
        assert preds.notna().all().all()

    def test_random_walk_equals_lagged_rv(self, frame, split_cfg):
        preds = run_walk_forward([RandomWalkRVForecaster()], frame, split_cfg,
                                 eval_segment="test")
        expected = frame["rv"].shift(1).loc[preds.index]
        np.testing.assert_allclose(preds["RW-RV"].values, expected.values)

    def test_rejects_leaking_model(self, frame, split_cfg):
        class Leaker:
            name = "leaker"

            def forecast_fold(self, frame, fold):
                n_pred = len(frame.loc[fold.predict_start:fold.predict_end])
                bad = frame.loc[:fold.predict_end].index[-(n_pred + 1):]
                return pd.Series(1e-4, index=bad, name="leaker")

        with pytest.raises(AssertionError):
            run_walk_forward([Leaker()], frame, split_cfg, eval_segment="test")

    def test_parquet_round_trip(self, frame, split_cfg, tmp_path):
        preds = run_walk_forward([_ConstModel()], frame, split_cfg, eval_segment="test")
        p = tmp_path / "preds.parquet"
        to_parquet(preds, p)
        back = from_parquet(p)
        # parquet does not round-trip the index `freq` attribute (harmless
        # metadata); normalise it before comparing values/dtypes.
        lhs = preds.copy()
        lhs.index = pd.DatetimeIndex(lhs.index.to_numpy())
        back.index = pd.DatetimeIndex(back.index.to_numpy())
        pd.testing.assert_frame_equal(lhs, back)


class TestEvaluate:
    def test_ranks_by_qlike(self, frame, split_cfg):
        preds = run_walk_forward(
            [_ConstModel(1e-4), RandomWalkRVForecaster()], frame, split_cfg,
            eval_segment="test",
        )
        table = evaluate_predictions(preds)
        assert list(table.columns) == ["n", "mse", "rmse", "mae", "qlike"]
        assert table["qlike"].is_monotonic_increasing
        assert ACTUAL_COL not in table.index

    def test_missing_actual_raises(self, frame, split_cfg):
        preds = run_walk_forward([_ConstModel()], frame, split_cfg, eval_segment="test")
        with pytest.raises(KeyError):
            evaluate_predictions(preds.drop(columns=[ACTUAL_COL]))
