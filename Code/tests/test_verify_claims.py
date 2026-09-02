"""Tests for the Phase-9 reproducibility gate.

The gate re-derives every published number from the prediction parquets using
formulas reimplemented inside `verify_claims`, deliberately independent of
`src.evaluation` (see that module's docstring for why). That independence is
only worth anything if the reimplementations are right, so they are checked here
against **closed-form cases computed by hand**, not against the project's own
functions -- comparing the two implementations to each other would defeat the
point of having two.

The one place the two implementations *are* compared is
:func:`test_the_two_implementations_agree_on_random_data`, which is a
cross-check, not a definition: if it ever fails, one of them is wrong and the
closed-form tests above it say which.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.experiments import verify_claims as V


# --------------------------------------------------------------------------- #
# The metric formulas, against hand-computed values                            #
# --------------------------------------------------------------------------- #
def test_qlike_is_zero_at_a_perfect_forecast():
    """The zero-anchored form Chapter 3 Eq 3.6 defines: L = y/h - log(y/h) - 1."""
    y = np.array([1e-4, 2e-4, 5e-5])
    assert V._qlike(y, y.copy()) == pytest.approx(0.0, abs=1e-15)


def test_qlike_matches_a_hand_computed_case():
    y = np.array([2.0, 1.0])
    h = np.array([1.0, 2.0])
    # r = [2, 0.5] -> (2 - ln2 - 1) + (0.5 - ln0.5 - 1) = (1-ln2) + (ln2-0.5)
    expected = ((1.0 - math.log(2.0)) + (math.log(2.0) - 0.5)) / 2.0
    assert V._qlike(y, h) == pytest.approx(expected, rel=1e-12)


def test_qlike_punishes_under_prediction_harder_than_over_prediction():
    """The asymmetry Chapter 3 §3.7 relies on: halving the forecast must cost
    more than doubling it."""
    y = np.array([1.0])
    assert V._qlike(y, np.array([0.5])) > V._qlike(y, np.array([2.0]))


def test_point_metrics_against_hand_computed_values():
    y = np.array([1.0, 2.0, 3.0])
    h = np.array([1.5, 2.0, 4.0])
    assert V._mse(y, h) == pytest.approx((0.25 + 0.0 + 1.0) / 3)
    assert V._rmse(y, h) == pytest.approx(math.sqrt((0.25 + 0.0 + 1.0) / 3))
    assert V._mae(y, h) == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_picp_counts_the_closed_interval():
    """A realisation exactly on a bound is covered, not a violation."""
    y = np.array([1.0, 5.0, 10.0, 0.5])
    lo = np.array([1.0, 2.0, 2.0, 1.0])
    hi = np.array([9.0, 9.0, 9.0, 9.0])
    assert V._picp(y, lo, hi) == pytest.approx(0.5)     # 1.0 and 5.0 covered
    assert V._mpiw(lo, hi) == pytest.approx((8 + 7 + 7 + 8) / 4)


def test_winkler_is_the_width_when_nothing_escapes():
    y = np.array([5.0, 5.0])
    lo = np.array([4.0, 3.0])
    hi = np.array([6.0, 7.0])
    assert V._winkler(y, lo, hi, 0.9) == pytest.approx((2.0 + 4.0) / 2)


def test_winkler_penalty_scales_with_the_nominal_level():
    """A miss at 95% costs more than the same miss at 90%: the penalty is 2/alpha."""
    y = np.array([10.0])
    lo, hi = np.array([1.0]), np.array([5.0])
    w90 = V._winkler(y, lo, hi, 0.90)     # 4 + (2/0.10)*5 = 104
    w95 = V._winkler(y, lo, hi, 0.95)     # 4 + (2/0.05)*5 = 204
    assert w90 == pytest.approx(4.0 + 20.0 * 5.0)
    assert w95 == pytest.approx(4.0 + 40.0 * 5.0)
    assert w95 > w90


def test_winkler_penalises_a_miss_below_and_above_symmetrically():
    lo, hi = np.array([2.0]), np.array([4.0])
    below = V._winkler(np.array([1.0]), lo, hi, 0.9)
    above = V._winkler(np.array([5.0]), lo, hi, 0.9)
    assert below == pytest.approx(above)


def test_the_two_implementations_agree_on_random_data():
    """Cross-check, not a definition -- if this fails the closed-form tests above
    say which side is wrong."""
    from src.evaluation.metrics import mae, mse, picp, qlike
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(-9, 0.6, 300))
    h = np.exp(np.log(y) + rng.normal(0, 0.4, 300))
    assert V._qlike(y, h) == pytest.approx(qlike(y, h), rel=1e-12)
    assert V._mse(y, h) == pytest.approx(mse(y, h), rel=1e-12)
    assert V._mae(y, h) == pytest.approx(mae(y, h), rel=1e-12)
    lo, hi = h * 0.5, h * 1.8
    assert V._picp(y, lo, hi) == pytest.approx(picp(y, lo, hi), rel=1e-12)


# --------------------------------------------------------------------------- #
# The t-1 lag                                                                  #
# --------------------------------------------------------------------------- #
def test_the_lag_reaches_back_before_the_evaluation_window():
    """The whole reason the FULL Phase-4 series is passed rather than the slice
    inside the predictions file: the first evaluated day must keep its label."""
    full = pd.Series([0.0, 1.0, 2.0, 2.0, 1.0],
                     index=pd.bdate_range("2019-01-01", periods=5))
    evaluated = full.index[2:]                       # window starts on the third day
    lagged = V._lag_state(full, evaluated)
    assert lagged.notna().all()
    assert lagged.iloc[0] == 1.0                     # the label of the day before
    # ... whereas pre-slicing then shifting loses it, which is the 2026-08-19 (iv) defect
    naive = full.reindex(evaluated).shift(1)
    assert pd.isna(naive.iloc[0])


def test_the_lag_is_applied_on_the_state_index_not_the_evaluation_index():
    full = pd.Series([0.0, 1.0, 2.0], index=pd.to_datetime(
        ["2019-01-01", "2019-01-02", "2019-01-03"]))
    out = V._lag_state(full, pd.to_datetime(["2019-01-03"]))
    assert out.iloc[0] == 1.0


# --------------------------------------------------------------------------- #
# The comparison bookkeeping                                                   #
# --------------------------------------------------------------------------- #
def test_a_match_passes_and_a_mismatch_fails():
    rows = []
    assert V._check(rows, table="t", item="i", quantity="qlike",
                    published=0.274490, rederived=0.274490, source="s")
    assert not V._check(rows, table="t", item="i", quantity="qlike",
                        published=0.274490, rederived=0.274491, source="s")
    assert [r["status"] for r in rows] == ["OK", "MISMATCH"]
    assert rows[1]["rel_diff"] > 0


def test_serialisation_round_trip_is_within_tolerance():
    """Artefacts are written at 16 significant figures; that must not trip the gate."""
    rows = []
    value = 0.2744895722052716
    round_tripped = float(f"{value:.16g}")
    assert V._check(rows, table="t", item="i", quantity="qlike",
                    published=value, rederived=round_tripped, source="s")


def test_a_non_numeric_or_missing_value_is_skipped_not_silently_passed():
    rows = []
    V._check(rows, table="t", item="i", quantity="q", published="n/a",
             rederived=1.0, source="s")
    V._check(rows, table="t", item="i", quantity="q", published=np.nan,
             rederived=1.0, source="s")
    assert [r["status"] for r in rows] == ["SKIP", "SKIP"]


# --------------------------------------------------------------------------- #
# Profile resolution                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem,expected", [
    ("m02_intraday_2019_2022", "intraday_2019_2022"),
    ("m05_regime_dl_intraday_2019_2022", "intraday_2019_2022"),
    ("m06_uq_intraday_2019_2022", "intraday_2019_2022"),
    ("m07_combined_intraday_2019_2022", "intraday_2019_2022"),
    ("m07_combined_intraday_2019_2022_hmm", "intraday_2019_2022_hmm"),
    ("m05_regime_dl_rq4_ftse", "rq4_ftse"),
])
def test_profile_extraction_lets_a_table_find_another_phases_parquet(stem, expected):
    """A Phase-7 calibration table's interval columns live in the Phase-6 parquet;
    without this the whole combined profile went unchecked."""
    assert V._profile_of(stem) == expected


def test_the_comparator_profile_reads_the_baum_welch_state_column():
    """`_hmm` is the same Phase-4 file read through the other estimator's column,
    not a profile with a Phase-4 file of its own."""
    reg = pd.DataFrame({"jphmm_filt_state": [0.0], "hmm_filt_state": [1.0]},
                       index=pd.to_datetime(["2019-01-02"]))
    parquets = {"m04_regimes_intraday_2019_2022": reg}
    _, col = V._regime_source("m05_regime_dl_intraday_2019_2022_hmm", parquets)
    assert col == "hmm_filt_state"
    _, col = V._regime_source("m05_regime_dl_intraday_2019_2022", parquets)
    assert col == "jphmm_filt_state"


# --------------------------------------------------------------------------- #
# End to end, on a tree that contains a planted defect                         #
# --------------------------------------------------------------------------- #
def _tree(tmp_path, *, corrupt=False):
    tables = tmp_path / "tables"
    preds = tmp_path / "predictions"
    tables.mkdir(parents=True); preds.mkdir(parents=True)
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=200)
    y = np.exp(rng.normal(-9, 0.6, 200))
    a = np.exp(np.log(y) + rng.normal(0, 0.4, 200))
    b = np.exp(np.log(y) + rng.normal(0, 0.6, 200))
    frame = pd.DataFrame({"y_true": y, "HAR-RV": a, "LSTM": b}, index=idx)
    frame.index.name = "date"
    frame.to_parquet(preds / "m02_probe.parquet")
    rows = []
    for name, h in (("HAR-RV", a), ("LSTM", b)):
        q = V._qlike(y, h)
        rows.append({"model": name, "n": 200, "mse": V._mse(y, h),
                     "rmse": V._rmse(y, h), "mae": V._mae(y, h),
                     "qlike": q * (1.05 if corrupt and name == "LSTM" else 1.0)})
    pd.DataFrame(rows).to_csv(tables / "m02_probe_metrics.csv", index=False)
    return tables, preds


def test_the_gate_passes_on_a_consistent_tree(tmp_path):
    tables, preds = _tree(tmp_path)
    result = V.verify(tables, preds)
    assert len(result) > 0
    assert set(result["status"]) == {"OK"}


def test_the_gate_catches_a_table_that_disagrees_with_its_parquet(tmp_path):
    """The failure it exists for: an artefact left behind by a code change."""
    tables, preds = _tree(tmp_path, corrupt=True)
    result = V.verify(tables, preds)
    bad = result[result["status"] == "MISMATCH"]
    assert len(bad) == 1
    assert bad.iloc[0]["item"] == "LSTM"
    assert bad.iloc[0]["quantity"] == "qlike"
    # rel_diff is measured against the PUBLISHED value: a table 5% high against a
    # correct re-derivation reports 0.05/1.05, not 0.05.
    assert bad.iloc[0]["rel_diff"] == pytest.approx(0.05 / 1.05, rel=1e-6)


def test_the_report_states_pass_or_fail_from_the_table_not_from_a_branch(tmp_path):
    tables, preds = _tree(tmp_path, corrupt=True)
    result = V.verify(tables, preds)
    out = tmp_path / "m09.md"
    V.write_report(result, out, rtol=V.RTOL)
    text = out.read_text(encoding="utf-8")
    assert "## FAIL" in text and "MISMATCH" in text
    assert "## PASS" not in text

    clean = V.verify(*_tree(tmp_path / "clean", corrupt=False))
    out2 = tmp_path / "m09_ok.md"
    V.write_report(clean, out2, rtol=V.RTOL)
    assert "## PASS" in out2.read_text(encoding="utf-8")


def test_a_published_row_with_no_forecast_column_is_reported_as_missing(tmp_path):
    """Worse than a mismatch: a number with nothing behind it at all."""
    tables, preds = _tree(tmp_path)
    met = pd.read_csv(tables / "m02_probe_metrics.csv")
    met.loc[len(met)] = {"model": "Ghost-LSTM", "n": 200, "mse": 1.0,
                         "rmse": 1.0, "mae": 1.0, "qlike": 1.0}
    met.to_csv(tables / "m02_probe_metrics.csv", index=False)
    result = V.verify(tables, preds)
    missing = result[result["status"] == "MISSING"]
    assert list(missing["item"]) == ["Ghost-LSTM"]
