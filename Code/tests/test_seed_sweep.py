"""Tests for the C5 seed sweep (``run_seed_sweep``).

Why this file exists
--------------------
The sweep exists to answer one question -- is the 0.0025 gap between the top four
models a result or an initialisation? -- and it can fail that question in three
ways that no amount of training would reveal.

* It could average over **two evaluation protocols**. The four models come from
  three configs; if their splits or refit cadence ever diverge, the sweep's
  spread mixes a protocol difference into a seed difference and reads as noise.
  :func:`test_shared_protocol_rejects_a_split_mismatch` pins the guard.
* It could **overwrite the real artefact with a diagnostic**. A reduced run once
  overwrote a headline run in this project; the CLI now refuses a one-seed or
  epoch-capped run unless the caller says so explicitly.
* Its **summary arithmetic** could be wrong, which is the one thing a reader
  cannot check without re-running five trainings.

Nothing here trains a model: the per-seed results are injected, so the tests run
in milliseconds and exercise exactly the logic that is not otherwise observable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.experiments.run_seed_sweep import (
    DEFAULT_SEEDS,
    PUBLISHED_SEED,
    TOP_FOUR,
    assert_shared_protocol,
    main,
    pairwise_signs,
    published_qlike,
    rank_stability,
    summarise,
)


def _cfg(test_end="2022-02-25", refit_days=21, scheme="anchored"):
    return OmegaConf.create({
        "harness": {"refit_every_folds": 0, "refit_frequency_days": refit_days,
                    "scheme": scheme, "eval_segment": "test"},
        "profiles": [{"name": "p", "splits": {
            "train_end": "2015-12-31", "val_start": "2016-01-01",
            "val_end": "2018-12-31", "test_start": "2019-01-01",
            "test_end": test_end}}],
    })


def _rows(qlike_by_model_seed):
    return pd.DataFrame([
        {"model": m, "seed": s, "n": 784, "qlike": q, "mse": 1e-8, "mae": 1e-5}
        for m, per in qlike_by_model_seed.items() for s, q in per.items()
    ])


# ------------------------------------------------------------ the protocol guard
def test_shared_protocol_accepts_agreeing_configs():
    split, seg = assert_shared_protocol({"a": _cfg(), "b": _cfg(), "c": _cfg()})
    assert seg == "test"
    assert str(split.test_end) == "2022-02-25"
    assert split.refit_frequency_days == 21 and split.scheme == "anchored"


@pytest.mark.parametrize("kwargs", [
    {"test_end": "2021-12-31"},          # a different evaluation window
    {"refit_days": 42},                  # a different refit cadence
    {"scheme": "rolling"},               # a different walk-forward scheme
])
def test_shared_protocol_rejects_a_mismatch(kwargs):
    """Four models compared under two protocols is not a seed sweep."""
    with pytest.raises(SystemExit, match="not comparable"):
        assert_shared_protocol({"a": _cfg(), "b": _cfg(**kwargs)})


# ------------------------------------------------------------------- the summary
def test_summarise_arithmetic_and_the_refit_deviation():
    rows = _rows({"A": {17: 0.2600, 18: 0.2610, 19: 0.2620},
                  "B": {17: 0.2650, 18: 0.2640, 19: 0.2660}})
    out = summarise(rows, {"A": 0.2599, "B": 0.2650}).set_index("model")

    assert out.loc["A", "n_seeds"] == 3
    assert out.loc["A", "qlike_mean"] == pytest.approx(0.2610)
    assert out.loc["A", "qlike_range"] == pytest.approx(0.0020)
    assert out.loc["A", "qlike_sd"] == pytest.approx(np.std([0.26, 0.261, 0.262], ddof=1))
    # The seed-17 row is checked against the committed table, not against itself.
    assert out.loc["A", "refit_deviation"] == pytest.approx(1e-4)
    assert out.loc["B", "refit_deviation"] == pytest.approx(0.0, abs=1e-15)
    # Sorted best-first, so the printed table reads as a ranking.
    assert list(out.index) == ["A", "B"]


def test_summarise_survives_a_model_absent_from_the_published_tables():
    rows = _rows({"New": {17: 0.26, 18: 0.27}})
    out = summarise(rows, {})
    assert np.isnan(out.loc[0, "qlike_published"])
    assert np.isnan(out.loc[0, "refit_deviation"])


# -------------------------------------------------------------- the real question
def test_rank_stability_reports_one_ordering_when_the_ranking_holds():
    rows = _rows({"A": {17: 0.260, 18: 0.261}, "B": {17: 0.265, 18: 0.266}})
    r = rank_stability(rows)
    assert len(r) == 2
    assert r["order_by_qlike"].nunique() == 1
    assert r["order_by_qlike"].iloc[0] == "A < B"


def test_rank_stability_catches_a_flip():
    """The failure mode C5 exists to detect: the order depends on the seed."""
    rows = _rows({"A": {17: 0.260, 18: 0.266}, "B": {17: 0.265, 18: 0.261}})
    r = rank_stability(rows)
    assert r["order_by_qlike"].nunique() == 2


def test_pairwise_signs_count_seeds_not_averages():
    rows = _rows({"A": {17: 0.260, 18: 0.266, 19: 0.259},
                  "B": {17: 0.265, 18: 0.261, 19: 0.264}})
    p = pairwise_signs(rows).set_index(["model_a", "model_b"])
    assert p.loc[("A", "B"), "seeds_a_better"] == 2
    assert p.loc[("A", "B"), "n_seeds"] == 3
    assert p.loc[("A", "B"), "mean_gap"] == pytest.approx(
        np.mean([0.260 - 0.265, 0.266 - 0.261, 0.259 - 0.264]))


# --------------------------------------------------- published values are READ
def test_published_qlike_reads_the_committed_tables(tmp_path, monkeypatch):
    from src.experiments import run_seed_sweep as rss

    tables = tmp_path / "results" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame({"model": ["LSTM-RVonly", "Regime-LSTM-B"],
                  "qlike": [0.25964, 0.26212]}).to_csv(
        tables / "m05_regime_dl_intraday_2019_2022_metrics.csv", index=False)
    pd.DataFrame({"model": ["MC-Dropout-Regime-LSTM-B", "LSTM-RVonly"],
                  "qlike": [0.26194, 0.25964]}).to_csv(
        tables / "m07_combined_intraday_2019_2022_master.csv", index=False)
    monkeypatch.setattr(rss, "repo_path", lambda *parts: tmp_path.joinpath(*parts))

    pub = rss.published_qlike()
    assert pub["LSTM-RVonly"] == pytest.approx(0.25964)
    assert pub["MC-Dropout-Regime-LSTM-B"] == pytest.approx(0.26194)


def test_published_qlike_tolerates_a_missing_table(tmp_path, monkeypatch):
    from src.experiments import run_seed_sweep as rss
    monkeypatch.setattr(rss, "repo_path", lambda *parts: tmp_path.joinpath(*parts))
    assert rss.published_qlike() == {}


# ----------------------------------------------------------- the artefact guard
def test_one_seed_is_refused():
    """A spread cannot be measured from one seed, and the artefact is not overwritten."""
    with pytest.raises(SystemExit, match="cannot be measured"):
        main(["--seeds", "17"])


def test_capped_epochs_are_refused_without_an_explicit_opt_in():
    with pytest.raises(SystemExit, match="under-trained"):
        main(["--max-epochs", "3"])


# ------------------------------------------------------------------ declarations
def test_the_declared_seeds_and_models_are_what_the_docstring_says():
    assert DEFAULT_SEEDS == (17, 18, 19, 20, 21)
    assert PUBLISHED_SEED == 17 and PUBLISHED_SEED in DEFAULT_SEEDS
    assert set(TOP_FOUR) == {"LSTM-RVonly", "Regime-LSTM-A-RVonly",
                             "Regime-LSTM-B", "MC-Dropout-Regime-LSTM-B"}
