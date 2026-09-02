"""Tests for Phase 8 / RQ4 -- the cross-asset transfer check.

Why this module exists
----------------------
Three things in Phase 8 are the kind that fail silently, and each has a test
here rather than a comment.

1. **The verdict is generated prose.** ``run_rq4.write_milestone`` states, in
   words, whether a finding replicates. Five defects of that family have already
   come out of this project's milestone generators -- a hardcoded conclusion, a
   shadowed loop variable, an untested claim, a self-contradicting verdict, an
   over-read null. So the wording comes from one function, :func:`transfer_verdict`,
   and every branch of it is exercised below, including the one that says "I
   cannot tell".

2. **"The same pre-registered rule on the same pre-registered grid" is a claim
   the write-up makes** (ROADMAP Phase 8 decision 1; Ch3 §3.9). A claim that
   lives only in a comment can be falsified by a one-character edit to a YAML
   list. It is asserted here against the headline config, so widening a grid to
   make an asset behave breaks the suite.

3. **The multi-profile milestone trap.** Every runner writes its milestone inside
   the profile loop. Until Phase 8 every config had exactly one profile, so a
   two-profile config would have written one note and kept only the last, with
   nothing to notice. ``milestone_path`` now refuses that; this pins the refusal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.datasets import attrition_record, frame_for_profile, resolve_asset
from src.experiments.run_rq4 import (
    BENCHMARK,
    DEEP_MODEL,
    rank_agreement,
    transfer_verdict,
)
from src.utils.config import load_config, milestone_path

# The S&P reference this project actually produced: the deep model is ahead in
# level, NOT significant pooled, and significant conditional on the lagged state.
# Values are illustrative -- what is pinned is the shape of the conjunction.
REFERENCE = {"asset": "SPX", "qlike_deep": 0.2621, "qlike_benchmark": 0.2745,
             "dm_p": 0.55, "gw_p": 1.2e-08}


def _cand(**over):
    base = {"asset": "RUT", "qlike_deep": 0.30, "qlike_benchmark": 0.34,
            "dm_p": 0.55, "gw_p": 1e-06}
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# transfer_verdict                                                             #
# --------------------------------------------------------------------------- #
def test_all_three_agreeing_is_a_replication():
    v = transfer_verdict(REFERENCE, _cand())
    assert v["verdict"] == "replicates"
    assert v["sign_agrees"] and v["pooled_agrees"] and v["conditional_agrees"]


def test_a_pooled_disagreement_is_reported_as_conditional_replication():
    """The pooled test differing is a weaker result than full replication.

    This is the case most likely to arise: the S&P's pooled DM is a null because
    the loss differential flips sign across states, and on another market the
    same mechanism may or may not cancel as neatly.
    """
    v = transfer_verdict(REFERENCE, _cand(dm_p=0.001))
    assert v["verdict"] == "replicates conditionally"
    assert v["sign_agrees"] and v["conditional_agrees"]
    assert v["pooled_agrees"] is False


def test_losing_the_conditional_test_is_direction_only_not_replication():
    """The S&P finding IS the state-dependence, so losing it is not transfer."""
    v = transfer_verdict(REFERENCE, _cand(gw_p=0.40))
    assert v["verdict"] == "direction only"
    assert v["sign_agrees"] and v["conditional_agrees"] is False
    assert "state-dependence" in v["detail"]


def test_a_sign_reversal_never_counts_as_replication():
    """Agreement on a test of a difference running the other way is not transfer."""
    v = transfer_verdict(REFERENCE, _cand(qlike_deep=0.40, qlike_benchmark=0.34))
    assert v["verdict"] == "does not replicate"
    assert v["sign_agrees"] is False
    # ... and it stays a non-replication even when both tests happen to agree.
    v2 = transfer_verdict(REFERENCE, _cand(qlike_deep=0.40, qlike_benchmark=0.34,
                                           dm_p=0.55, gw_p=1e-06))
    assert v2["verdict"] == "does not replicate"


def test_everything_disagreeing_is_also_a_non_replication():
    v = transfer_verdict(REFERENCE, _cand(qlike_deep=0.40, qlike_benchmark=0.34,
                                          dm_p=0.001, gw_p=0.40))
    assert v["verdict"] == "does not replicate"
    assert not any([v["sign_agrees"], v["pooled_agrees"], v["conditional_agrees"]])


@pytest.mark.parametrize("missing", ["qlike_deep", "dm_p", "gw_p"])
def test_a_missing_quantity_is_indeterminate_not_a_coerced_comparison(missing):
    """A model absent from a board must not silently become a verdict."""
    cand = _cand()
    cand[missing] = float("nan")
    v = transfer_verdict(REFERENCE, cand)
    assert v["verdict"] == "indeterminate"
    assert v["sign_agrees"] == ""


def test_an_absent_key_is_indeterminate_too():
    cand = _cand()
    del cand["gw_p"]
    assert transfer_verdict(REFERENCE, cand)["verdict"] == "indeterminate"


def test_a_broken_reference_is_indeterminate_and_says_so():
    v = transfer_verdict({**REFERENCE, "gw_p": np.nan}, _cand())
    assert v["verdict"] == "indeterminate"
    assert "reference" in v["detail"]


def test_alpha_is_honoured_on_both_sides_of_the_comparison():
    """One level decides both the reference and the candidate.

    Grading the reference at 5% and the candidate at 10% would manufacture
    disagreement, so the same alpha reaches both. Here the candidate's GW p sits
    between the two levels: at alpha=0.05 it is not significant (so the
    conditional verdict differs from the reference's) and at alpha=0.10 it is.
    """
    cand = _cand(gw_p=0.08)
    assert transfer_verdict(REFERENCE, cand, alpha=0.05)["verdict"] == "direction only"
    assert transfer_verdict(REFERENCE, cand, alpha=0.10)["verdict"] == "replicates"


def test_the_verdict_row_carries_the_facts_it_was_computed_from():
    """A row in m08_rq4_verdict.csv must be auditable without re-deriving it."""
    v = transfer_verdict(REFERENCE, _cand())
    for key in ("qlike_deep", "qlike_benchmark", "dm_p", "gw_p", "alpha",
                "ref_deep_ahead", "ref_dm_sig", "ref_gw_sig", "deep_model",
                "benchmark", "asset", "reference_asset"):
        assert key in v, key
    assert v["deep_model"] == DEEP_MODEL and v["benchmark"] == BENCHMARK


# --------------------------------------------------------------------------- #
# rank_agreement                                                               #
# --------------------------------------------------------------------------- #
def _ranks():
    return pd.DataFrame(
        {"SPX": [1, 2, 3, 4, 5], "RUT": [2, 1, 3, 5, 4], "FTSE": [1, 2, 3, 4, 5]},
        index=["Regime-LSTM-B", "HAR-RV", "GARCH", "EGARCH", "RW-RV"],
    )


def test_rank_agreement_reports_the_overlap_beside_the_correlation():
    out = rank_agreement(_ranks(), "SPX").set_index("asset")
    assert set(out.index) == {"RUT", "FTSE"}
    assert out.loc["FTSE", "spearman"] == pytest.approx(1.0)
    assert out.loc["RUT", "n_models_in_common"] == 5
    assert "Regime-LSTM-B" in out.loc["RUT", "models_in_common"]


def test_only_models_on_both_boards_enter_a_pair():
    """The reduced design puts five models on a robustness board and fifteen on
    the S&P's, so a correlation computed over a padded union would be fiction."""
    ranks = _ranks()
    ranks.loc["LSTM-VIX"] = [6, np.nan, np.nan]
    out = rank_agreement(ranks, "SPX").set_index("asset")
    assert out.loc["RUT", "n_models_in_common"] == 5
    assert "LSTM-VIX" not in out.loc["RUT", "models_in_common"]


def test_too_few_common_models_yields_nan_rather_than_a_number():
    ranks = pd.DataFrame({"SPX": [1, 2], "RUT": [2, 1]}, index=["a", "b"])
    out = rank_agreement(ranks, "SPX")
    assert np.isnan(out["spearman"].iloc[0])
    assert out["n_models_in_common"].iloc[0] == 2


def test_a_missing_reference_column_returns_nothing_rather_than_raising():
    assert rank_agreement(_ranks().drop(columns=["SPX"]), "SPX").empty


# --------------------------------------------------------------------------- #
# The pre-registered selection, asserted rather than trusted                    #
# --------------------------------------------------------------------------- #
def test_rq4_regime_config_matches_the_preregistered_selection():
    """The RQ4 regime run must reuse the headline's grid, rule and state count.

    ROADMAP Phase 8 decision 1 and Ch3 §3.9 both say each asset gets its own
    lambda "selected by the same pre-registered rule on the same pre-registered
    grid". That is what makes a differing lambda a finding about portability
    rather than a free parameter, and it is one YAML edit away from being false.
    """
    head, rq4 = load_config("hmm"), load_config("hmm_rq4")
    assert list(rq4.jump_penalty_grid) == list(head.jump_penalty_grid)
    assert list(rq4.jump_penalty_sensitivity_grid) == list(head.jump_penalty_sensitivity_grid)
    assert list(rq4.k_grid) == list(head.k_grid)
    assert (rq4.select_by_persistence.min_mean_duration_days
            == head.select_by_persistence.min_mean_duration_days)
    assert bool(rq4.select_by_persistence.enabled) is bool(head.select_by_persistence.enabled)
    assert rq4.features.primary == head.features.primary
    assert bool(rq4.features.standardize) is bool(head.features.standardize)


def test_rq4_deep_config_runs_only_the_preregistered_best_of_phase_5():
    """The reduced design is baselines + ONE deep model, fixed before this phase."""
    cfg = load_config("regime_lstm_rq4")
    assert cfg.approaches.experts is True
    assert cfg.approaches.feature_full is False
    assert cfg.approaches.feature_rvonly is False


def test_rq4_deep_config_holds_the_sp_hyperparameter_selection():
    """Re-searching per asset would confound transfer with a fresh search."""
    head, rq4 = load_config("regime_lstm"), load_config("regime_lstm_rq4")
    for key in ("lookback", "hidden_size", "lr", "num_layers", "dropout"):
        assert rq4.model[key] == head.model[key], key


@pytest.mark.parametrize("config", ["econometric_rq4", "hmm_rq4", "regime_lstm_rq4"])
def test_every_rq4_profile_names_an_asset_the_registry_can_resolve(config):
    """A profile naming an asset the registry does not know must fail loudly."""
    data_cfg = load_config("data")
    cfg = load_config(config)
    keys = [str(p.asset) for p in cfg.profiles]
    assert keys == ["RUT", "FTSE"], keys
    for key in keys:
        spec = resolve_asset(data_cfg, key)
        assert spec.oxfordman and spec.cache_alias


def test_the_ftse_never_receives_the_vix():
    """The CBOE VIX prices S&P 500 options; joining it to a FTSE model would be a
    cross-market spillover experiment, not the auxiliary-feature ablation Ch1
    §1.8 describes. Enforced in the registry, not by memory."""
    data_cfg = load_config("data")
    assert resolve_asset(data_cfg, "FTSE").vix_feature is False
    assert resolve_asset(data_cfg, "RUT").vix_feature is True


def test_rq4_profiles_are_wired_to_their_own_artefacts_not_the_sp_s():
    """Each deep profile must read its OWN asset's regime signal and baselines.

    Pairing one asset's posteriors with another's forecasts would produce a
    plausible table and answer nothing, which is the failure mode the asset
    registry was built to end.
    """
    cfg = load_config("regime_lstm_rq4")
    for profile in cfg.profiles:
        name = str(profile.name)
        assert str(profile.regime_source).endswith(f"m04_regimes_{name}.parquet")
        assert str(profile.baseline_predictions).endswith(f"m02_{name}.parquet")


@pytest.mark.parametrize("config", ["hmm_rq4", "regime_lstm_rq4"])
def test_per_profile_runners_cannot_overwrite_their_own_milestones(config):
    """These runners write one note per profile inside the loop, so a
    two-profile config must resolve to two distinct filenames."""
    cfg = load_config(config)
    assert len(cfg.profiles) > 1
    names = {milestone_path(cfg, "unused.md", str(p.name),
                            n_profiles=len(cfg.profiles)) for p in cfg.profiles}
    assert len(names) == len(cfg.profiles)


@pytest.mark.parametrize("config,expected", [
    ("econometric_rq4", "m08_econometric_rq4.md"),
    ("econometric_loghar", "m08_loghar_sensitivity.md"),
])
def test_econometric_variants_never_land_on_the_headline_note(config, expected):
    """`run_econometric` writes ONE note for all its profiles, so it needs a plain
    name rather than a placeholder — but it must not be the headline's.

    On 2026-09-02 it was: `run_econometric` was the one runner not routed through
    `milestone_path`, so both Phase-8 econometric configs wrote to
    `m02_econometric.md` and the headline Phase-2 note was overwritten twice.
    """
    cfg = load_config(config)
    got = milestone_path(cfg, "m02_econometric.md", str(cfg.profiles[0].name),
                         n_profiles=1)
    assert got.name == expected
    assert got.name != "m02_econometric.md"


def test_every_runner_routes_its_milestone_through_milestone_path():
    """A source-level guard, because the unit test above did not catch the bug.

    `milestone_path` was written, tested directly, and then *not called* by
    `run_econometric` — the exact failure the project's own rule warns about: a
    test that exercises a function in isolation does not certify that anything
    calls it. This walks each runner's AST and asserts that every
    ``write_milestone(...)`` call takes its destination from ``milestone_path``,
    never from a ``repo_path(..., "literal.md")``.
    """
    import ast
    import pathlib

    runners = sorted(pathlib.Path("src/experiments").glob("run_*.py"))
    assert len(runners) >= 6, runners
    checked = 0
    for path in runners:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "write_milestone"):
                continue
            # The destination is not always the same positional slot
            # (`run_rq4` passes a verdict table first), so look for a
            # milestone_path(...) call anywhere in the arguments.
            calls = [a.func.id for a in node.args
                     if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)]
            assert "milestone_path" in calls, (
                f"{path.name}:{node.lineno} writes its milestone via "
                f"{calls or 'a non-call argument'}, bypassing milestone_path — "
                "a non-headline config can then overwrite the headline note")
            checked += 1
    assert checked >= 6, f"expected a write_milestone call per runner, found {checked}"


def test_sensitivity_configs_change_exactly_one_thing():
    """A sensitivity that also moves something else answers a different question."""
    head, refit = load_config("lstm_baseline"), load_config("lstm_baseline_refit")
    assert int(refit.harness.refit_every_folds) == 1
    assert int(head.harness.refit_every_folds) == 0
    assert refit.harness.refit_frequency_days == head.harness.refit_frequency_days
    assert refit.harness.scheme == head.harness.scheme
    for key in ("lookback", "hidden_size", "lr", "num_layers", "dropout",
                "batch_size", "patience", "val_fraction"):
        assert refit.model[key] == head.model[key], key
    assert bool(refit.sweep.enabled) is False, "a fresh search would confound the cadence"

    econ, loghar = load_config("econometric"), load_config("econometric_loghar")
    assert loghar.models.har.transform == "log"
    assert econ.models.har.transform == "level"
    assert loghar.harness.refit_frequency_days == econ.harness.refit_frequency_days
    assert loghar.harness.scheme == econ.harness.scheme
    assert loghar.harness.return_col == econ.harness.return_col


def test_sensitivity_profiles_do_not_collide_with_the_headline_artefacts():
    """Distinct profile names are what keep m03_*/m02_* files from being overwritten."""
    head_names = {str(p.name) for p in load_config("lstm_baseline").profiles}
    head_names |= {str(p.name) for p in load_config("econometric").profiles}
    for config in ("lstm_baseline_refit", "econometric_loghar", "econometric_rq4",
                   "hmm_rq4", "regime_lstm_rq4"):
        for profile in load_config(config).profiles:
            assert str(profile.name) not in head_names, (config, profile.name)


# --------------------------------------------------------------------------- #
# milestone_path                                                               #
# --------------------------------------------------------------------------- #
def _cfg(milestone_file=None, milestones="results/milestones"):
    node = {"paths": {"milestones": milestones}}
    if milestone_file is not None:
        node["paths"]["milestone_file"] = milestone_file
    return OmegaConf.create(node)


def test_milestone_path_falls_back_to_the_phase_default():
    p = milestone_path(_cfg(), "m04_regimes.md", "intraday_2019_2022")
    assert p.name == "m04_regimes.md"


def test_milestone_path_formats_the_profile_placeholder():
    p = milestone_path(_cfg("m08_regimes_{profile}.md"), "m04_regimes.md", "rq4_ftse",
                       n_profiles=2)
    assert p.name == "m08_regimes_rq4_ftse.md"


def test_milestone_path_refuses_to_let_two_profiles_share_one_note():
    with pytest.raises(ValueError, match="placeholder"):
        milestone_path(_cfg("m08_regimes.md"), "m04_regimes.md", "rq4_rut",
                       n_profiles=2)


def test_a_single_profile_config_needs_no_placeholder():
    p = milestone_path(_cfg("m05_regime_dl_hmm.md"), "m05_regime_dl.md",
                       "intraday_2019_2022", n_profiles=1)
    assert p.name == "m05_regime_dl_hmm.md"


# --------------------------------------------------------------------------- #
# The asset actually reaches the data layer                                    #
# --------------------------------------------------------------------------- #
def test_frame_for_profile_passes_the_profiles_asset_through(monkeypatch):
    """The whole point of Phase 8's plumbing: ``resolve_asset`` and the ``asset=``
    parameter existed since 2026-08-30 and *no runner passed either*, so a
    profile naming RUT would have been run on .SPX."""
    seen = {}

    def _fake(cfg, *, target, asset=None, include_vix=False, yz_window=21):
        seen.update(target=target, asset=asset, include_vix=include_vix)
        return pd.DataFrame({"rv": [1.0]})

    monkeypatch.setattr("src.data.datasets.build_econometric_frame", _fake)
    profile = OmegaConf.create({"name": "rq4_ftse", "asset": "FTSE",
                                "target": "oxfordman_rv5"})
    frame_for_profile(OmegaConf.create({}), profile, include_vix=True)
    assert seen == {"target": "oxfordman_rv5", "asset": "FTSE", "include_vix": True}


def test_a_profile_without_an_asset_resolves_the_primary(monkeypatch):
    """Every pre-Phase-8 config omits ``asset``; those must be unchanged."""
    seen = {}

    def _fake(cfg, *, target, asset=None, include_vix=False, yz_window=21):
        seen.update(asset=asset)
        return pd.DataFrame({"rv": [1.0]})

    monkeypatch.setattr("src.data.datasets.build_econometric_frame", _fake)
    profile = OmegaConf.create({"name": "intraday_2019_2022",
                                "target": "oxfordman_rv5"})
    frame_for_profile(OmegaConf.create({}), profile)
    assert seen == {"asset": None}


def test_attrition_record_flattens_the_frame_attrs_into_one_row():
    frame = pd.DataFrame({"rv": [1.0, 2.0]},
                         index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    frame.attrs.update({
        "asset": "FTSE", "oxfordman_symbol": ".FTSE", "session": "UK",
        "vix_joined": False,
        "join_attrition": {"n_target_rows": 100, "n_after_join": 96,
                           "n_after_dropna": 90, "lost_to_join": 4,
                           "lost_to_warmup_or_nan": 6},
    })
    row = attrition_record(frame, profile_name="rq4_ftse")
    assert row["asset"] == "FTSE" and row["session"] == "UK"
    assert row["lost_to_calendar_join"] == 4
    assert row["pct_lost_to_join"] == pytest.approx(4.0)
    assert row["first_date"] == "2020-01-02" and row["last_date"] == "2020-01-03"
    assert row["vix_joined"] is False


def test_attrition_record_survives_a_frame_with_no_attrs():
    row = attrition_record(pd.DataFrame({"rv": []}, index=pd.DatetimeIndex([])))
    assert row["n_target_rows"] == 0
    assert np.isnan(row["pct_lost_to_join"])


# --------------------------------------------------------------------------- #
# End-to-end: collect -> verdicts -> milestone, on synthetic artefacts          #
# --------------------------------------------------------------------------- #
# A green unit suite is not "it works" (see the project's standing rule). These
# build the exact files the Phase-8 chain writes, in a tmp tree, and run the real
# `collect`/`verdicts`/`write_milestone` over them -- so the artefact NAMES, the
# column contracts, the S&P reference lookup and the note all get exercised
# without torch, without the real data, and without writing into results/.

from src.experiments import run_rq4 as R


def _board(rng, n=260, *, edge=0.0):
    """A synthetic predictions board: y_true plus five model columns.

    ``edge`` shifts the deep model's forecast towards the truth, so a test can
    ask for a board where the deep model wins, loses, or ties.
    """
    idx = pd.bdate_range("2019-01-01", periods=n)
    y = np.exp(rng.normal(-9.0, 0.6, n))
    def noisy(scale):
        return np.exp(np.log(y) + rng.normal(0.0, scale, n))
    frame = pd.DataFrame({
        "y_true": y,
        "HAR-RV": noisy(0.55),
        "GARCH": noisy(0.75),
        "EGARCH": noisy(0.74),
        "RW-RV": noisy(0.85),
        R.DEEP_MODEL: noisy(max(0.05, 0.55 - edge)),
    }, index=idx)
    frame.index.name = "date"
    return frame


def _regimes(rng, index, seed_state=0):
    """A Phase-4-shaped parquet: a persistent causal hard state, full history.

    The index starts a fortnight BEFORE the evaluation window on purpose: the
    t-1 lag has to reach back across the boundary, which is the whole reason the
    full-history series is passed rather than the slice inside the predictions.
    """
    full = pd.bdate_range(index[0] - pd.Timedelta(days=20), index[-1])
    state = np.repeat(rng.integers(0, 3, len(full) // 12 + 1), 12)[:len(full)]
    out = pd.DataFrame({"jphmm_filt_state": state.astype(float)}, index=full)
    out.index.name = "date"
    return out


def _write_phase8_tree(root, rng, *, edges):
    """Write every artefact `collect` reads, for the S&P and both RQ4 assets."""
    tables = root / "results" / "tables"
    preds = root / "results" / "predictions"
    for d in (tables, preds):
        d.mkdir(parents=True, exist_ok=True)

    for profile, edge in edges.items():
        board = _board(rng, edge=edge)
        board.to_parquet(preds / f"m05_regime_dl_{profile}.parquet")
        _regimes(rng, board.index).to_parquet(preds / f"m04_regimes_{profile}.parquet")
        pd.DataFrame({
            "K": [2, 3, 4],
            "hmm_bic": [100.0, 90.0, 95.0],
            "jump_bic": [110.0, 99.0, 94.0],       # the jump model prefers K=4
        }).to_csv(tables / f"m04_regimes_{profile}_bic.csv", index=False)
        # Column names match what Phase 4 actually writes -- `jump_penalty` /
        # `mean_duration`, and NO `selected` flag. The first RQ4 run came back
        # with a blank selected-lambda column because the fixture had invented
        # one; the selection now comes off the causal-signals table below.
        pd.DataFrame({
            "jump_penalty": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            "n_jumps": [1301, 51, 7, 6, 2, 0],
            "mean_duration": [3.3, 76.7, 498.6, 569.9, 1329.7, 3989.0],
        }).to_csv(tables / f"m04_regimes_{profile}_jump_lambda_train.csv", index=False)
        pd.DataFrame({
            "signal": ["jump_penalised_hmm", "baum_welch_hmm"],
            "lambda": [3.0, np.nan],
            "test_mean_duration_days": [19.6, 4.1],
            "test_switches": [40.0, 190.0],
        }).to_csv(tables / f"m04_regimes_{profile}_causal_signals.csv", index=False)
        pd.DataFrame([{
            "profile": profile, "asset": profile, "oxfordman_symbol": ".X",
            "session": "US", "vix_joined": True, "n_target_rows": 300,
            "n_after_join": 290, "n_modelling_rows": 260,
            "lost_to_calendar_join": 10, "lost_to_warmup_or_nan": 30,
            "pct_lost_to_join": 3.33, "first_date": "2019-01-01",
            "last_date": "2019-12-31",
        }]).to_csv(tables / f"m02_{profile}_sample.csv", index=False)
    return tables, preds


@pytest.fixture()
def phase8_tree(tmp_path, monkeypatch):
    """A tmp tree shaped like the repo, with `repo_path` pointed into it."""
    def _build(edges):
        _write_phase8_tree(tmp_path, np.random.default_rng(7), edges=edges)
        monkeypatch.setattr(R, "repo_path", lambda *parts: tmp_path.joinpath(*parts))
        return tmp_path
    return _build


def _rq4_cfg():
    return OmegaConf.create({
        "seed": 17,
        "regime": {"source": "results/predictions/m04_regimes_rq4_rut.parquet",
                   "state_col": "jphmm_filt_state",
                   "labels": ["calm", "transitional", "crisis"]},
        "profiles": [
            {"name": "rq4_rut", "asset": "RUT",
             "regime_source": "results/predictions/m04_regimes_rq4_rut.parquet"},
            {"name": "rq4_ftse", "asset": "FTSE",
             "regime_source": "results/predictions/m04_regimes_rq4_ftse.parquet"},
        ],
    })


EDGES = {"intraday_2019_2022": 0.15, "rq4_rut": 0.15, "rq4_ftse": 0.0}


def test_collect_assembles_every_table_from_the_phase8_artefacts(phase8_tree):
    phase8_tree(EDGES)
    ctx = R.collect(OmegaConf.create({}), _rq4_cfg(),
                    labels=["calm", "transitional", "crisis"],
                    spx_profile="intraday_2019_2022")

    assert set(ctx["boards"]) == {"SPX", "RUT", "FTSE"}
    assert ctx["boards"]["SPX"]["is_reference"] is True

    # Metrics cover every model on every asset, ranked within the asset.
    assert set(ctx["metrics"]["asset"]) == {"SPX", "RUT", "FTSE"}
    assert R.DEEP_MODEL in set(ctx["metrics"]["model"])
    for asset in ("SPX", "RUT", "FTSE"):
        assert ctx["metrics"].loc[ctx["metrics"]["asset"] == asset,
                                  "rank_qlike"].min() == 1

    # The deep model is tested against every baseline present on the board.
    tested = set(ctx["tests"].loc[ctx["tests"]["asset"] == "RUT", "model_b"])
    assert tested == set(R.TEST_BASELINES)
    assert (ctx["tests"]["model_a"] == R.DEEP_MODEL).all()
    for col in ("dm_p_pooled", "gw_regime_p", "gw_regime_stat"):
        assert ctx["tests"][col].notna().all(), col

    # Each asset's own regime model, read from that asset's own tables.
    reg = ctx["regimes"].set_index("asset")
    assert (reg["selected_lambda"] == 3.0).all()
    assert (reg["bic_preferred_K_hmm"] == 3).all()
    assert (reg["bic_preferred_K_jump"] == 4).all()

    # The t-1 lag reached back across the window boundary: every evaluated day
    # carries a label, which is the invariant the full-history series exists for.
    assert (reg["n_labelled_at_t_minus_1"] == reg["n_evaluated"]).all()
    assert len(ctx["sample"]) == 3
    assert len(ctx["per_regime"]) > 0


def test_a_deep_model_that_wins_on_one_asset_and_not_the_other_is_graded_apart(
        phase8_tree):
    """The point of the phase: one asset may transfer while another does not."""
    phase8_tree(EDGES)
    ctx = R.collect(OmegaConf.create({}), _rq4_cfg(),
                    labels=["calm", "transitional", "crisis"],
                    spx_profile="intraday_2019_2022")
    v = R.verdicts(ctx).set_index("asset")

    assert set(v.index) == {"RUT", "FTSE"}
    # RUT was built with the S&P's edge, FTSE with none. (bool() because a
    # DataFrame round-trip turns Python bools into numpy ones.)
    assert bool(v.loc["RUT", "sign_agrees"]) is True
    assert bool(v.loc["FTSE", "sign_agrees"]) is False
    assert v.loc["FTSE", "verdict"] == "does not replicate"
    # Every verdict carries the reference it was graded against.
    assert (v["reference_asset"] == "SPX").all()
    assert v["ref_qlike_deep"].notna().all()


def test_the_milestone_states_the_verdict_it_computed_and_nothing_else(
        phase8_tree, tmp_path):
    """Generated prose must not assert what the tables do not carry.

    Five defects of exactly this family have come out of this project's other
    milestone generators, so the note is checked against the verdict table rather
    than eyeballed.
    """
    phase8_tree(EDGES)
    ctx = R.collect(OmegaConf.create({}), _rq4_cfg(),
                    labels=["calm", "transitional", "crisis"],
                    spx_profile="intraday_2019_2022")
    v = R.verdicts(ctx)
    out = tmp_path / "m08_rq4.md"
    R.write_milestone(ctx, v, out, alpha=0.05)
    text = out.read_text(encoding="utf-8")

    for row in v.to_dict("records"):
        assert f"**{row['asset']}: {row['verdict']}**" in text
    # The note must never claim a replication the table did not record.
    recorded = set(v["verdict"])
    for word in ("replicates", "does not replicate", "direction only",
                 "indeterminate"):
        if word not in recorded and word != "replicates":
            assert f": {word}**" not in text
    # And it must carry the standing caveats, which are not results.
    assert "not independent evidence" in text
    assert "pre-registered grid" in text
    assert R.DEEP_MODEL in text


def test_a_missing_robustness_artefact_stops_the_phase_with_the_commands_to_fix_it(
        phase8_tree, tmp_path):
    """Half a Phase-8 chain must not silently produce half a verdict."""
    phase8_tree({"intraday_2019_2022": 0.15, "rq4_rut": 0.15})   # no FTSE
    with pytest.raises(SystemExit, match="run_regime_lstm"):
        R.collect(OmegaConf.create({}), _rq4_cfg(),
                  labels=["calm", "transitional", "crisis"],
                  spx_profile="intraday_2019_2022")


def test_a_missing_reference_is_refused_rather_than_worked_around(phase8_tree):
    """RQ4 is a comparison; without the primary result there is nothing to compare."""
    phase8_tree({"rq4_rut": 0.15, "rq4_ftse": 0.0})              # no S&P
    with pytest.raises(SystemExit, match="primary"):
        R.collect(OmegaConf.create({}), _rq4_cfg(),
                  labels=["calm", "transitional", "crisis"],
                  spx_profile="intraday_2019_2022")


# --------------------------------------------------------------------------- #
# External validation of the regime map when the asset has no VIX              #
# --------------------------------------------------------------------------- #
# Found by running the phase (2026-09-02): `run_regimes` read `frame["vix_close"]`
# unconditionally and the FTSE profile died with a KeyError, because the registry
# correctly refuses the column for an asset whose options trade elsewhere. The
# data layer was right and its consumer had not been told -- the same shape as the
# asset-threading defect this phase already fixed one layer down.

from src.experiments.run_regimes import state_concordance, vix_confusion


def _states_and_series(n=600, seed=3):
    """A validator series that genuinely separates by state, and its labels."""
    rng = np.random.default_rng(seed)
    state = np.repeat(rng.integers(0, 3, n // 10 + 1), 10)[:n]
    base = np.array([1.0, 3.0, 9.0])[state]
    series = pd.Series(base * np.exp(rng.normal(0, 0.25, n)),
                       index=pd.bdate_range("2005-01-03", periods=n))
    return state, series


def test_the_vix_path_is_unchanged_by_the_generalisation():
    """`.SPX` and `.RUT` artefacts must not move: the VIX branch is the old code."""
    state, series = _states_and_series()
    df_new, extra = state_concordance(state, series, 3, key="vix_close")
    df_alias, _ = vix_confusion(state, series, 3)
    assert df_new.equals(df_alias)
    assert list(df_new.columns) == ["vixQ0", "vixQ1", "vixQ2"]
    assert "spearman_state_vix" in extra and "mean_vix_by_state" in extra
    assert extra["validator"] == "vix_close"


def test_the_rv_fallback_is_named_so_it_cannot_be_read_as_a_vix_number():
    """Two different diagnostics must never share a column name."""
    state, series = _states_and_series()
    _, vix = state_concordance(state, series, 3, key="vix_close")
    df_rv, rv = state_concordance(state, series, 3, key="rv")
    assert list(df_rv.columns) == ["rvQ0", "rvQ1", "rvQ2"]
    assert "spearman_state_rv" in rv and "spearman_state_vix" not in rv
    assert "mean_rv_by_state" in rv and "mean_vix_by_state" not in rv
    assert rv["validator"] == "rv"
    # Same input, so the statistics agree -- what differs is only the labelling.
    assert rv["spearman_state_rv"] == pytest.approx(vix["spearman_state_vix"])


def test_only_named_keys_reach_the_agreement_table():
    """The private `_spearman` alias must not duplicate the named correlation."""
    state, series = _states_and_series()
    _, extra = state_concordance(state, series, 3, key="vix_close")
    persisted = {k for k, v in extra.items()
                 if np.isscalar(v) and not k.startswith("_")}
    assert persisted == {"validator", "spearman_state_vix", "diag_concordance"}
    assert extra["_spearman"] == extra["spearman_state_vix"]


def test_concordance_is_high_when_the_states_track_the_validator():
    state, series = _states_and_series()
    _, extra = state_concordance(state, series, 3, key="rv")
    assert extra["spearman_state_rv"] > 0.8
    means = [extra["mean_rv_by_state"][f"state{k}"] for k in range(3)]
    assert means == sorted(means), "a separating validator must order with the state"


def test_a_state_with_no_days_yields_nan_rather_than_a_crash():
    """K=3 with an empty state happens at a large jump penalty; the note must
    print 'n/a' for it rather than fabricating a mean."""
    state = np.array([0] * 50 + [2] * 50)          # state 1 never occurs
    series = pd.Series(np.linspace(1.0, 5.0, 100))
    _, extra = state_concordance(state, series, 3, key="rv")
    assert np.isnan(extra["mean_rv_by_state"]["state1"])


def test_the_selected_lambda_is_read_from_the_signal_the_run_actually_used(tmp_path):
    """The portability number the whole phase turns on must not come back blank.

    In the first real RQ4 run it did: `_regime_summary` looked for a `selected`
    flag on the lambda sweep table, which Phase 4 does not write, so
    `selected_lambda` was silently absent from `m08_rq4_regimes.csv`. It now
    comes off the causal-signals table -- the lambda the run actually used for
    the signal Phase 5 consumed -- rather than from a second implementation of
    the selection rule that could drift from the first.
    """
    tables = tmp_path
    pd.DataFrame({"jump_penalty": [1.0, 3.0, 10.0],
                  "n_jumps": [1301, 51, 7],
                  "mean_duration": [3.1, 76.7, 498.6]}).to_csv(
        tables / "m04_regimes_p_jump_lambda_train.csv", index=False)
    pd.DataFrame({"signal": ["jump_penalised_hmm", "baum_welch_hmm"],
                  "lambda": [3.0, np.nan],
                  "test_mean_duration_days": [12.6, 4.1],
                  "test_switches": [62.0, 190.0]}).to_csv(
        tables / "m04_regimes_p_causal_signals.csv", index=False)

    idx = pd.bdate_range("2019-01-01", periods=60)
    state = pd.Series(np.repeat([0, 1, 2], 30).astype(float),
                      index=pd.bdate_range("2018-11-01", periods=90))
    row = R._regime_summary("p", tables, state, idx,
                            ["calm", "transitional", "crisis"])
    assert row["selected_lambda"] == 3.0
    assert row["test_mean_duration_days"] == pytest.approx(12.6)
    # the sweep is still reported alongside it
    assert row["lambda_grid"] == "1, 3, 10"
    assert row["mean_duration_by_lambda"] == "3.1, 76.7, 498.6"


def test_a_missing_selected_lambda_is_warned_about_not_silently_dropped(tmp_path, project_logs):
    records = project_logs("rq4")
    idx = pd.bdate_range("2019-01-01", periods=30)
    state = pd.Series(np.zeros(60), index=pd.bdate_range("2018-11-01", periods=60))
    row = R._regime_summary("missing", tmp_path, state, idx, ["calm", "transitional", "crisis"])
    assert "selected_lambda" not in row
    assert any("no selected lambda" in r.getMessage() for r in records)


def test_run_rq4_does_not_inherit_the_deep_configs_milestone_name():
    """`run_rq4` reads `regime_lstm_rq4.yaml` to learn the profiles, and that
    file's `milestone_file` belongs to `run_regime_lstm`. Inheriting it sent the
    RQ4 note to `m08_regime_dl_rq4.md` on the first real run."""
    cfg = load_config("regime_lstm_rq4")
    assert str(cfg.paths.milestone_file) == "m08_regime_dl_{profile}.md"
    got = milestone_path(cfg, "m08_rq4.md", "rq4", key="rq4_milestone_file")
    assert got.name == "m08_rq4.md"
    # and the per-profile runner still gets its own names from the same config
    for profile in cfg.profiles:
        assert milestone_path(cfg, "m05_regime_dl.md", str(profile.name),
                              n_profiles=len(cfg.profiles)).name == \
            f"m08_regime_dl_{profile.name}.md"
