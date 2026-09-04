"""Tests for the Phase-9 compute report (``report_compute``).

Why this file exists
--------------------
Appendix A.4 states the parameter count as a formula and defers the instantiated
counts and the wall-clock to Chapter 4. Both of those are now artefacts, and the
two ways they could go wrong are exactly what is tested here.

* The formula and the code can disagree. ``parameter_rows`` refuses to write a
  table when they do, and :func:`test_parameter_rows_refuses_a_formula_mismatch`
  proves the guard fires -- otherwise the appendix's formula would be prose that
  merely sits next to a number.
* ``network_inputs()`` returns ``len(features)`` rather than measuring
  ``X.shape[2]``, which is only valid because ``build_feature_target`` emits one
  column per feature key. :func:`test_network_inputs_matches_the_real_builder`
  pins that against the real builder instead of leaving it as a comment.

The log parser gets its own tests because the reproduction logs are UTF-16 (a
PowerShell redirect wrote them) and one of them carries no timestamps at all --
two ways a naive reader would either crash or silently report nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sequence import build_feature_target
from src.experiments.report_compute import (
    closed_form_parameters,
    count_parameters,
    environment_rows,
    parameter_rows,
    runtime_rows,
    selected_hyperparameters,
)


# --------------------------------------------------------------- closed form
def test_closed_form_is_table_a4s_formula():
    """``4H(F + H) + 9H + 1`` at Q=1, quoted from Appendix A.4."""
    for h in (32, 64, 128):
        for f in (1, 2, 3, 5):
            assert closed_form_parameters(h, f) == 4 * h * (f + h) + 9 * h + 1


def test_closed_form_reproduces_the_appendix_worked_numbers():
    """The two figures the appendix's own text carries, at the selected width."""
    assert closed_form_parameters(128, 2) == 67_713
    assert 3 * closed_form_parameters(128, 2) == 203_139


def test_closed_form_matches_instantiated_modules():
    """The formula against torch itself -- the check ``parameter_rows`` runs."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.models.deep.lstm import LSTMRegressor
    from src.models.deep.uncertainty import QuantileLSTMRegressor

    total, trainable = count_parameters(LSTMRegressor(2, 128))
    assert total == trainable == closed_form_parameters(128, 2)

    total, _ = count_parameters(QuantileLSTMRegressor(2, 128, n_quantiles=3))
    assert total == closed_form_parameters(128, 2, outputs=3)


# ------------------------------------------------------- the input-width claim
def test_network_inputs_matches_the_real_builder():
    """``len(features)`` is a valid stand-in for ``X.shape[2]`` -- proven, not assumed."""
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "rv": np.exp(rng.normal(-9.0, 0.4, len(idx))),
        "log_return": rng.normal(0, 0.01, len(idx)),
        "oc_log_return": rng.normal(0, 0.01, len(idx)),
        "reg_p0": 0.6, "reg_p1": 0.3, "reg_p2": 0.1,
    }, index=idx)
    for features in (("log_rv",),
                     ("log_rv", "oc_log_return"),
                     ("log_rv", "oc_log_return", "reg_p0", "reg_p1", "reg_p2")):
        feats, _, _ = build_feature_target(frame, features)
        assert feats.shape[1] == len(features)


def test_forecasters_report_their_own_network_width():
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.models.deep.lstm import LSTMForecaster
    from src.models.deep.regime_lstm import RegimeExpertForecaster

    f = LSTMForecaster(features=("log_rv", "oc_log_return"), hidden_size=32)
    assert f.network_inputs() == 2 and f.n_networks() == 1
    assert count_parameters(f.build_network(f.network_inputs()))[0] == \
        closed_form_parameters(32, 2)

    # The experts see the base features; the three gate columns bypass them, so
    # K multiplies the count instead of widening the input.
    r = RegimeExpertForecaster(base_features=("log_rv", "oc_log_return"),
                               gate_cols=("reg_p0", "reg_p1", "reg_p2"), hidden_size=32)
    assert r.network_inputs() == 2 and r.n_networks() == 3


# ------------------------------------------------------------------ the guard
class _WrongSizeForecaster:
    """A forecaster whose network does not match the closed form."""

    hidden_size = 8
    name = "Broken"

    def network_inputs(self):
        return 2

    def n_networks(self):
        return 1

    def build_network(self, input_size):
        import torch.nn as nn
        return nn.Linear(int(input_size), 3)   # nothing like an LSTM


def test_parameter_rows_refuses_a_formula_mismatch(monkeypatch):
    """A table is not written when the appendix and the code disagree."""
    pytest.importorskip("torch")
    from src.experiments import report_compute

    monkeypatch.setattr(report_compute, "build_board",
                        lambda profile: [("Broken", _WrongSizeForecaster(), "x")])
    with pytest.raises(AssertionError, match="closed form"):
        report_compute.parameter_rows("any")


# ------------------------------------------------------------- sweep selection
def test_selected_hyperparameters_takes_the_best_validation_row(tmp_path, monkeypatch):
    from src.experiments import report_compute

    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame({"hidden_size": [32, 128, 64], "lookback": [40, 10, 20],
                  "lr": [1e-4, 1e-3, 1e-3],
                  "val_qlike": [0.26, 0.2343, 0.25]}).to_csv(
        tables / "m03_lstm_p_sweep.csv", index=False)
    monkeypatch.setattr(report_compute, "repo_path",
                        lambda *parts: tmp_path.joinpath(*parts))
    best = report_compute.selected_hyperparameters("p")
    assert best == {"hidden": 128, "lookback": 10, "lr": 1e-3, "val_qlike": 0.2343}


# --------------------------------------------------------------- log harvest
_LOG = """2026-08-24 05:25:32,881 [INFO] run_lstm: sweep: 18 combinations on validation
2026-08-24 05:25:38,089 [INFO] run_lstm:   hidden=  32 -> val QLIKE 0.2628 (48 ep)
2026-08-24 05:31:51,688 [INFO] run_lstm: sweep best: hidden=128 lookback=10 lr=1e-03
2026-08-24 05:31:58,154 [INFO] rolling:     LSTM       done: 784 forecasts in 6.5s
2026-08-24 05:32:14,649 [INFO] run_lstm: Phase 3 complete: 1 profile(s).
"""


def _write_utf16(path, text):
    path.write_bytes(text.encode("utf-16"))


def test_runtime_rows_parses_a_utf16_log(tmp_path):
    """The reproduction logs are UTF-16 with a BOM; reading them as UTF-8 yields nothing."""
    d = tmp_path / "rerun-20260824-052455"
    d.mkdir()
    _write_utf16(d / "02-phase3-lstm.log", _LOG)
    out = runtime_rows(d)

    total = out[out["item"] == "end to end"]["value"].iloc[0]
    assert total == pytest.approx(401.8, abs=0.1)

    sweep = out[out["item"].str.contains("hyperparameter sweep")]["value"].iloc[0]
    assert sweep == pytest.approx(378.8, abs=0.1)          # 05:25:32.881 -> 05:31:51.688

    wf = out[out["item"] == "walk-forward: LSTM [phase3-lstm, n=784]"]
    assert len(wf) == 1 and wf["value"].iloc[0] == pytest.approx(6.5)
    assert out["item"].is_unique
    assert (out["source"] == d.name).all()


def test_runtime_rows_skips_a_log_with_no_timestamps(tmp_path):
    """The pytest log is plain output; it must be skipped, not crash the harvest."""
    d = tmp_path / "rerun-x"
    d.mkdir()
    _write_utf16(d / "02-phase3-lstm.log", _LOG)
    _write_utf16(d / "13-pytest.log", "475 passed in 57.92s\n")
    out = runtime_rows(d)
    assert not out["item"].str.contains("pytest").any()
    assert (out["item"] == "end to end").sum() == 1
    assert out["item"].is_unique


def test_runtime_rows_on_an_empty_directory(tmp_path):
    d = tmp_path / "rerun-empty"
    d.mkdir()
    assert runtime_rows(d).empty


# --------------------------------------------------------------- environment
def test_environment_rows_carry_the_interpreter_and_the_processor():
    env = environment_rows().set_index("item")["value"]
    assert env["python"].startswith("3.")
    assert "processor" in env.index          # added so a timing has a machine
    assert env["accelerator"] == "none"
