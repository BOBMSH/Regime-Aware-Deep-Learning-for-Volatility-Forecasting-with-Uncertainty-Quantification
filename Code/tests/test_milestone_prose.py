"""Tests for the milestone-prose primitives in ``src/experiments/run_uq.py``.

Why this file exists
--------------------
Three defects of one family have now been found in this project's generated
milestone notes:

1. a hardcoded conclusion ("coverage degrades in the crisis regime -- exactly
   the RQ3 hypothesis") that survived the result reversing;
2. a shadowed loop variable that printed ``lambda=300`` for a model fitted at
   ``lambda=3.0``;
3. (2026-08-22, audit vi) *"the ordering is non-monotone"* asserted in a branch
   that had only ever tested monotone-**decreasing**, which is false whenever
   coverage is monotone increasing -- as it is for the Baum-Welch comparator
   (0.848 / 0.880 / 0.887). That clause reached `m06_uq_hmm.md` and from there
   the roadmap changelog.

4. (2026-08-24, audit vii/viii) *"Both tails are worst in the same state, so
   the pooled view captures the structure adequately here"* -- emitted by
   ``run_combined.py`` from an ``argmax(upper_rate) == argmax(lower_rate)``
   comparison, in a note whose own verdict table reported **0 of 16** pooled
   subsample differences surviving Holm against **5 of 16** on the upper tail.
   The premise was true and the conclusion was refuted by the same document.

The pattern in all four: the *numbers* in these notes were computed, but the
qualitative *words* were templated into branches. The primitives that turn
numbers into claims are therefore module-level and tested here. The test that
would have caught (3) is :meth:`TestCoverageShape.test_monotone_increasing_is_not_called_non_monotone`;
the one that would have caught (4) is
:meth:`TestPooledViewVerdict.test_the_headline_counts_do_not_read_as_adequate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_combined import pooled_view_verdict
from src.experiments.run_uq import (
    MATERIAL_COVERAGE_GAP,
    coverage_degrades,
    coverage_shape,
    endpoint_gap_verdict,
    picp_standard_error,
)

LABELS = ["calm", "transitional", "crisis"]


def _pr(picps, ns=(379, 329, 76)) -> pd.DataFrame:
    """A minimal per-regime table of the shape the generator consumes."""
    rows = [{"regime": lab, "n": n, "picp": p}
            for lab, n, p in zip(LABELS, ns, picps)]
    rows.append({"regime": "all", "n": int(sum(ns)),
                 "picp": float(np.average(picps, weights=ns))})
    return pd.DataFrame(rows)


class TestCoverageShape:
    def test_monotone_increasing_is_not_called_non_monotone(self):
        """The audit-vi regression. These are the Baum-Welch comparator's actual
        numbers; the old generator described them as 'non-monotone'."""
        assert coverage_shape(_pr([0.848, 0.880, 0.887], (381, 332, 71)), LABELS) == \
            "monotone increasing"

    def test_headline_sequence_is_genuinely_non_monotone(self):
        """The headline profile's actual numbers, which *are* non-monotone."""
        assert coverage_shape(_pr([0.844, 0.894, 0.842]), LABELS) == "non-monotone"

    def test_monotone_decreasing_detected(self):
        assert coverage_shape(_pr([0.90, 0.86, 0.81]), LABELS) == "monotone decreasing"

    def test_constant_sequence_reads_as_decreasing(self):
        # weakly decreasing and weakly increasing both hold; the conservative
        # reading against the RQ3 hypothesis is reported first
        assert coverage_shape(_pr([0.87, 0.87, 0.87]), LABELS) == "monotone decreasing"

    def test_missing_bucket_is_undetermined_not_guessed(self):
        assert coverage_shape(_pr([0.85, np.nan, 0.84]), LABELS) == "undetermined"


class TestCoverageDegrades:
    def test_requires_the_whole_ordering_not_just_the_endpoints(self):
        """A big calm-to-crisis drop with a spike in the middle is not a
        monotone degradation, and must not be reported as one."""
        assert coverage_degrades(_pr([0.90, 0.97, 0.84]), LABELS) is False

    def test_requires_a_material_gap(self):
        # ordered, but a 1 pp end-to-end gap on a 76-day bucket is noise
        assert coverage_degrades(_pr([0.870, 0.865, 0.860]), LABELS) is False
        assert MATERIAL_COVERAGE_GAP == pytest.approx(0.02)

    def test_true_when_both_conditions_hold(self):
        assert coverage_degrades(_pr([0.90, 0.86, 0.81]), LABELS) is True

    def test_headline_and_comparator_both_reject_degradation(self):
        """The substantive RQ3 claim: neither regime estimator shows it."""
        assert coverage_degrades(_pr([0.844, 0.894, 0.842]), LABELS) is False
        assert coverage_degrades(_pr([0.848, 0.880, 0.887], (381, 332, 71)),
                                 LABELS) is False

    def test_missing_bucket_is_not_a_degradation(self):
        assert coverage_degrades(_pr([0.90, np.nan, 0.81]), LABELS) is False


class TestPicpStandardError:
    def test_matches_the_binomial_formula(self):
        pr = _pr([0.844, 0.894, 0.842])
        expect = np.sqrt(0.842 * (1 - 0.842) / 76)
        assert picp_standard_error(pr, "crisis") == pytest.approx(expect)

    def test_shrinks_with_n(self):
        small = picp_standard_error(_pr([0.85, 0.85, 0.85], (100, 100, 50)), "crisis")
        large = picp_standard_error(_pr([0.85, 0.85, 0.85], (100, 100, 500)), "crisis")
        assert large < small

    def test_absent_bucket_is_nan(self):
        assert np.isnan(picp_standard_error(_pr([0.85, 0.85, 0.85]), "nonexistent"))

    def test_degenerate_coverage_gives_zero_not_nan(self):
        assert picp_standard_error(_pr([1.0, 0.9, 1.0]), "calm") == pytest.approx(0.0)


class TestEndpointGapVerdict:
    def test_the_headline_02pp_gap_is_inside_noise(self):
        """calm 0.844 vs crisis 0.842 on n=379/76: two-bucket SE is ~4.3 pp, so
        a 0.2 pp gap is emphatically inside noise -- and now demonstrably so
        rather than asserted."""
        assert endpoint_gap_verdict(_pr([0.844, 0.894, 0.842]), LABELS) == \
            "well inside sampling noise"

    def test_the_comparator_39pp_gap_is_also_inside_noise(self):
        """The branch used to print 'well inside sampling noise' for this too,
        without computing anything. It happens to be true; now it is checked."""
        assert endpoint_gap_verdict(_pr([0.848, 0.880, 0.887], (381, 332, 71)),
                                    LABELS) == "well inside sampling noise"

    def test_a_large_gap_is_not_waved_through(self):
        assert endpoint_gap_verdict(_pr([0.95, 0.90, 0.55], (379, 329, 300)),
                                    LABELS) == "larger than sampling noise"

    def test_missing_bucket_is_not_assessable(self):
        assert endpoint_gap_verdict(_pr([np.nan, 0.89, 0.84]), LABELS) == \
            "not assessable"


def _diff(p_holms) -> pd.DataFrame:
    """The `difference in coverage` rows the verdict reads: only p_holm matters."""
    return pd.DataFrame({"p_holm": list(p_holms)})


class TestPooledViewVerdict:
    """The audit-vii defect: an argmax was used to license a claim about whether
    the pooled two-sided view is adequate. Adequacy is a claim about the tests,
    so every branch here is decided from Holm-adjusted counts."""

    def test_the_headline_counts_do_not_read_as_adequate(self):
        """The regression. Headline profile, 90% level: nothing survives Holm
        pooled, five of sixteen do on the upper tail. The old generator printed
        'the pooled view captures the structure adequately here'."""
        pooled = _diff([1.0] * 16)
        upper = _diff([0.0295, 0.0192, 0.0048, 0.0130, 0.0033] + [1.0] * 11)
        v = pooled_view_verdict(pooled, upper)
        assert v["verdict"] == "pooled understates"
        assert v["n_pooled_significant"] == 0 and v["n_upper_significant"] == 5
        assert "adequate" not in v["sentence"].lower()
        assert "0 of 16" in v["sentence"] and "5 of 16" in v["sentence"]

    def test_no_structure_anywhere_makes_no_adequacy_claim(self):
        """The comparator profile: 0 of 16 on both sides. The honest reading is
        that nothing is established either way -- the branch the argmax version
        could not express, because both argmaxes still land somewhere."""
        v = pooled_view_verdict(_diff([1.0] * 16), _diff([1.0] * 16))
        assert v["verdict"] == "no structure either way"
        assert "descriptive only" in v["sentence"]
        assert "adequate" not in v["sentence"].lower()

    def test_pooled_can_be_the_sharper_view(self):
        v = pooled_view_verdict(_diff([0.01, 0.02, 1.0]), _diff([1.0, 1.0, 1.0]))
        assert v["verdict"] == "pooled is sharper"
        assert v["n_pooled_significant"] == 2

    def test_equal_and_nonzero_counts_read_as_agreement(self):
        v = pooled_view_verdict(_diff([0.01, 1.0]), _diff([0.02, 1.0]))
        assert v["verdict"] == "views agree"

    def test_every_sentence_carries_its_own_counts(self):
        """No branch may assert without showing the evidence -- that is what
        made the previous four defects survive review."""
        cases = [(_diff([1.0] * 4), _diff([0.01] + [1.0] * 3)),
                 (_diff([1.0] * 4), _diff([1.0] * 4)),
                 (_diff([0.01, 1.0]), _diff([1.0, 1.0])),
                 (_diff([0.01, 1.0]), _diff([0.02, 1.0]))]
        for pooled, upper in cases:
            s = pooled_view_verdict(pooled, upper)["sentence"]
            assert "pooled" in s and "upper-tail" in s

    def test_empty_frames_do_not_raise(self):
        v = pooled_view_verdict(_diff([]), None)
        assert v["verdict"] == "no structure either way"
        assert v["n_pooled"] == 0 and v["n_upper"] == 0

    def test_alpha_is_respected(self):
        borderline = _diff([0.03])
        assert pooled_view_verdict(_diff([1.0]), borderline)["n_upper_significant"] == 1
        assert pooled_view_verdict(_diff([1.0]), borderline,
                                   alpha=0.01)["n_upper_significant"] == 0
