"""Unit tests for the sliding-window + standardisation layer (roadmap Phase 3).

Pure numpy/pandas -- no torch -- so these run fast and pin down the leakage
contract (windows end strictly before their target; scalers fit on train only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sequence import (
    Standardizer,
    build_feature_target,
    make_windows,
)


@pytest.fixture
def small_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    rv = np.exp(rng.normal(-9.0, 0.5, size=n))          # positive variance
    ret = rng.normal(0.0, np.sqrt(rv))
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame({"log_return": ret, "oc_log_return": ret, "rv": rv}, index=idx)


class TestBuildFeatureTarget:
    def test_columns_and_target(self, small_frame):
        feats, target, index = build_feature_target(small_frame, ["log_rv", "oc_log_return"])
        assert list(feats.columns) == ["log_rv", "oc_log_return"]
        # target is log realized variance, aligned to the frame.
        np.testing.assert_allclose(target.to_numpy(), np.log(small_frame["rv"].to_numpy()))
        np.testing.assert_allclose(feats["log_rv"].to_numpy(), np.log(small_frame["rv"].to_numpy()))
        assert not feats.isna().any().any() and not target.isna().any()
        assert len(index) == len(small_frame)

    def test_unknown_feature_raises(self, small_frame):
        with pytest.raises(KeyError):
            build_feature_target(small_frame, ["not_a_feature"])


class TestMakeWindows:
    def test_shapes_and_alignment(self, small_frame):
        feats, target, index = build_feature_target(small_frame, ["log_rv", "oc_log_return"])
        L = 10
        X, y, y_dates = make_windows(feats.to_numpy(), target.to_numpy(), index, L)
        assert X.shape == (len(index) - L, L, 2)
        assert y.shape == (len(index) - L,)
        # X[i] must equal feature rows i..i+L-1; y[i] the target on row i+L.
        np.testing.assert_allclose(X[0], feats.to_numpy()[0:L], rtol=1e-6)
        np.testing.assert_allclose(y[3], target.to_numpy()[L + 3], rtol=1e-6)
        assert y_dates[0] == index[L]

    def test_window_ends_strictly_before_target(self, small_frame):
        """No window may contain a feature row dated on/after its target day."""
        feats, target, index = build_feature_target(small_frame, ["log_rv"])
        L = 8
        X, y, y_dates = make_windows(feats.to_numpy(), target.to_numpy(), index, L)
        pos = {d: i for i, d in enumerate(index)}
        for i, td in enumerate(y_dates):
            last_feature_row = pos[td] - 1          # window ends at t-1
            assert last_feature_row == pos[y_dates[i]] - 1
            # the window's last value equals the log_rv on day t-1, not day t
            np.testing.assert_allclose(X[i, -1, 0], feats.to_numpy()[pos[td] - 1, 0], rtol=1e-6)

    def test_too_short_returns_empty(self, small_frame):
        feats, target, index = build_feature_target(small_frame, ["log_rv"])
        X, y, y_dates = make_windows(feats.to_numpy(), target.to_numpy(), index, len(index) + 5)
        assert len(X) == 0 and len(y) == 0 and len(y_dates) == 0


class TestStandardizer:
    def test_fit_on_slice_only(self):
        a = np.arange(100, dtype=float).reshape(-1, 1)
        sc = Standardizer().fit(a[:60])                 # fit on first 60 rows
        z = sc.transform(a[:60])
        assert abs(z.mean()) < 1e-9 and abs(z.std() - 1.0) < 1e-9
        # later rows are transformed with the *training* stats -> mean shifts up.
        assert sc.transform(a[60:]).mean() > 1.0

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(1)
        a = rng.normal(3.0, 2.0, size=(50, 3))
        sc = Standardizer().fit(a)
        np.testing.assert_allclose(sc.inverse_transform(sc.transform(a)), a, rtol=1e-6)

    def test_constant_column_safe(self):
        a = np.ones((20, 1))
        sc = Standardizer().fit(a)
        assert np.isfinite(sc.transform(a)).all()       # no divide-by-zero
