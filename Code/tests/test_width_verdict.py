"""Tests for the Phase-7 width-correction decision (``run_combined``).

Why this file exists
--------------------
The first Phase-7 run produced a milestone note that **contradicted itself**:
the verdict block concluded "the deliverable is the global width correction"
from the coverage tests, while the σ-scale section seven paragraphs later
concluded "the honest deliverable is the per-regime column" from the spread of
the per-regime scales. Both computations were individually correct; neither knew
the other existed.

That is the fourth instance in this project of one failure family -- a claim
assembled from one input while a second input bears on it. The decision is now
made **once**, by :func:`width_correction_verdict`, from both inputs, and the
four-way case analysis is pinned here.

The substantive point the function encodes: a wide spread in the per-regime
scales is a set of *point estimates* with no standard error, on buckets as small
as ~75 days. It is not, on its own, evidence of regime-dependent miscalibration.
Only the coverage tests can supply that.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.run_combined import (
    SIGMA_SPREAD_THRESHOLD,
    width_correction_verdict,
)


def _sigma(scales, ns=(379, 329, 76)) -> pd.DataFrame:
    """A minimal σ-scale table of the shape the generator consumes."""
    rows = [{"scope": "all", "n": int(sum(ns)),
             "coverage_calibrating_scale": 1.14, "crps_optimal_scale": 1.25}]
    for lab, n, s in zip(["calm", "transitional", "crisis"], ns, scales):
        rows.append({"scope": lab, "n": n,
                     "coverage_calibrating_scale": s, "crps_optimal_scale": s + 0.1})
    return pd.DataFrame(rows)


class TestFourCases:
    def test_narrow_spread_no_evidence_is_global(self):
        v = width_correction_verdict(_sigma([1.12, 1.10, 1.16]),
                                     conditional_evidence=False)
        assert v["recommendation"] == "global"
        assert v["material"] is False
        assert "one global scale is sufficient" in v["sentence"]
        assert "pre-registered deliverable stands" in v["sentence"]

    def test_wide_spread_with_evidence_is_per_regime(self):
        v = width_correction_verdict(_sigma([1.22, 1.04, 0.91]),
                                     conditional_evidence=True)
        assert v["recommendation"] == "per-regime"
        assert v["material"] is True
        assert "the deliverable is the per-regime column" in v["sentence"]

    def test_wide_spread_without_evidence_stays_global(self):
        """The case that actually occurred on the comparator profile, and the
        one the contradiction turned on: scales span 0.91-1.22 but no t-1
        conditioner survives Holm. Point estimates are not evidence."""
        v = width_correction_verdict(_sigma([1.22, 1.04, 0.91]),
                                     conditional_evidence=False)
        assert v["recommendation"] == "global, per-regime indicative"
        assert v["material"] is True
        assert "not** evidence" in v["sentence"] or "not** evidence" in v["sentence"]
        assert "defensible deliverable is the global scale" in v["sentence"]
        assert "indicative" in v["sentence"]

    def test_narrow_spread_with_evidence_is_a_shape_problem(self):
        """A conditioner predicts misses but every bucket needs the same width:
        no rescaling fixes that, so the note must not offer one."""
        v = width_correction_verdict(_sigma([1.12, 1.10, 1.16]),
                                     conditional_evidence=True)
        assert v["recommendation"] == "global, shape problem noted"
        assert "*shape*" in v["sentence"]
        assert "no rescaling" in v["sentence"]


class TestMechanics:
    def test_spread_is_max_minus_min_over_regimes_only(self):
        """The 'all' row must not enter the spread -- it is the thing the
        per-regime scales are being compared against."""
        v = width_correction_verdict(_sigma([1.22, 1.04, 0.91]),
                                     conditional_evidence=False)
        assert v["spread"] == pytest.approx(1.22 - 0.91)

    def test_threshold_boundary_is_exclusive(self):
        exact = width_correction_verdict(
            _sigma([1.00, 1.00, 1.00 + SIGMA_SPREAD_THRESHOLD]),
            conditional_evidence=False)
        assert exact["material"] is False          # equal to the threshold is not material
        over = width_correction_verdict(
            _sigma([1.00, 1.00, 1.00 + SIGMA_SPREAD_THRESHOLD + 0.01]),
            conditional_evidence=False)
        assert over["material"] is True

    def test_threshold_is_configurable(self):
        v = width_correction_verdict(_sigma([1.22, 1.04, 0.91]),
                                     conditional_evidence=False,
                                     spread_threshold=0.5)
        assert v["material"] is False
        assert v["recommendation"] == "global"

    def test_no_per_regime_rows_degrades_gracefully(self):
        only_all = pd.DataFrame([{"scope": "all", "n": 784,
                                  "coverage_calibrating_scale": 1.14,
                                  "crps_optimal_scale": 1.25}])
        v = width_correction_verdict(only_all, conditional_evidence=True)
        assert v["recommendation"] == "global"
        assert "No per-regime scales were estimable" in v["sentence"]

    def test_every_case_returns_a_non_empty_sentence(self):
        for scales in ([1.12, 1.10, 1.16], [1.22, 1.04, 0.91]):
            for ev in (True, False):
                v = width_correction_verdict(_sigma(scales), conditional_evidence=ev)
                assert v["sentence"].strip()
                assert v["recommendation"] in {
                    "global", "per-regime", "global, per-regime indicative",
                    "global, shape problem noted"}
                assert v["conditional_evidence"] is ev
