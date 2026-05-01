"""Unit tests for circle_packing's GEPA-style ASI helpers (Phase 1.2).

Tests target the pure helpers _build_per_instance_scores and _build_feedback
so we can pin their contracts without spinning up a subprocess evaluator.
A live end-to-end test (running evaluate() on the seed program) lives outside
the unit suite — see the live-test gate in the Phase 1.2 plan.
"""
import sys
from pathlib import Path

import pytest

# circle_packing's evaluator.py lives outside the package import path; add it
# to sys.path the same way the harness does at runtime.
BENCH_DIR = Path(__file__).resolve().parents[1] / "experiment/benchmarks/circle_packing"
sys.path.insert(0, str(BENCH_DIR))

from evaluator import (  # type: ignore
    _build_feedback,
    _build_per_instance_scores,
    TARGET_VALUE,
)


class TestPerInstanceScores:
    def test_perfect_packing_all_ones(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [],
            "overlap_pairs": [],
            "min_pairwise_slack": 1e-8,
        }
        s = _build_per_instance_scores(strict_pass=True, diagnostics=diag, target_ratio=1.0)
        assert s == {
            "validity": 1.0,
            "boundary": 1.0,
            "overlap": 1.0,
            "sum_radii_progress": 1.0,
        }

    def test_invalid_strict_zeros_validity(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [],
            "overlap_pairs": [],
            "min_pairwise_slack": 0.0,
        }
        s = _build_per_instance_scores(strict_pass=False, diagnostics=diag, target_ratio=0.94)
        assert s["validity"] == 0.0
        # But sub-constraints can still be 1.0 (e.g. tolerant pass, strict fails on float-noise)
        assert s["boundary"] == 1.0
        assert s["overlap"] == 1.0
        assert s["sum_radii_progress"] == pytest.approx(0.94)

    def test_boundary_violations_scale_with_count(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [{"circle_index": i, "slack": -1e-3} for i in range(13)],
            "overlap_pairs": [],
            "min_pairwise_slack": -1e-3,
        }
        s = _build_per_instance_scores(strict_pass=False, diagnostics=diag, target_ratio=0.5)
        assert s["boundary"] == pytest.approx(0.5)  # 13 of 26 inside
        assert s["overlap"] == 1.0

    def test_overlap_pairs_scale_with_total_pairs(self):
        n = 26
        total_pairs = n * (n - 1) // 2  # 325
        diag = {
            "n_circles_submitted": n,
            "boundary_violations": [],
            "overlap_pairs": [{"pair": [0, k], "interpenetration": 1e-3,
                               "center_distance": 0, "radius_sum": 0}
                              for k in range(1, 33)],  # 32 overlapping pairs
            "n_overlap_pairs_total": 32,
            "min_pairwise_slack": -1e-3,
        }
        s = _build_per_instance_scores(strict_pass=False, diagnostics=diag, target_ratio=0.0)
        assert s["overlap"] == pytest.approx(1.0 - 32 / total_pairs)

    def test_uses_uncapped_totals_when_present(self):
        # Diagnostics list is capped at 10 but the true count is 325. The
        # score must reflect ground truth, not the truncated visible list.
        n = 26
        total_pairs = n * (n - 1) // 2  # 325
        diag = {
            "n_circles_submitted": n,
            "boundary_violations": [],
            "overlap_pairs": [{"pair": [0, k], "interpenetration": 1e-1,
                               "center_distance": 0, "radius_sum": 0}
                              for k in range(1, 11)],  # capped to 10 visible
            "n_overlap_pairs_total": 325,  # uncapped truth
            "min_pairwise_slack": -1e-1,
        }
        s = _build_per_instance_scores(strict_pass=False, diagnostics=diag, target_ratio=0.0)
        assert s["overlap"] == pytest.approx(1.0 - 325 / total_pairs)
        assert s["overlap"] == 0.0

    def test_sum_radii_progress_clamped_low(self):
        diag = {"n_circles_submitted": 26, "boundary_violations": [], "overlap_pairs": []}
        s = _build_per_instance_scores(strict_pass=True, diagnostics=diag, target_ratio=-0.1)
        assert s["sum_radii_progress"] == 0.0

    def test_sum_radii_progress_clamped_high(self):
        diag = {"n_circles_submitted": 26, "boundary_violations": [], "overlap_pairs": []}
        s = _build_per_instance_scores(strict_pass=True, diagnostics=diag, target_ratio=10.0)
        assert s["sum_radii_progress"] == 1.5

    def test_handles_zero_submitted(self):
        # Defensive: division-by-zero guard
        diag = {"n_circles_submitted": 0, "boundary_violations": [], "overlap_pairs": []}
        s = _build_per_instance_scores(strict_pass=False, diagnostics=diag, target_ratio=0.0)
        assert 0.0 <= s["boundary"] <= 1.0
        assert 0.0 <= s["overlap"] <= 1.0


class TestFeedback:
    def test_valid_packing_summary(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [],
            "overlap_pairs": [],
            "min_pairwise_slack": 1.23e-8,
        }
        msg = _build_feedback(
            strict_pass=True, diagnostics=diag, sum_radii=2.481234, reported_sum=2.481234
        )
        assert "valid packing" in msg
        assert "2.481234" in msg
        assert str(TARGET_VALUE) in msg
        assert "1.230e-08" in msg

    def test_boundary_only_failure(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [
                {"circle_index": 17, "slack": -1.234e-2, "center": [0.0, 0.0], "radius": 0.1}
            ],
            "overlap_pairs": [],
            "min_pairwise_slack": 1e-3,
        }
        msg = _build_feedback(strict_pass=False, diagnostics=diag, sum_radii=0.0, reported_sum=2.41)
        assert msg.startswith("invalid")
        assert "boundary violation" in msg
        assert "circle 17" in msg
        assert "2.4100" in msg

    def test_overlap_only_failure(self):
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [],
            "overlap_pairs": [
                {"pair": [4, 11], "interpenetration": 2.34e-3,
                 "center_distance": 0.1, "radius_sum": 0.1023}
            ],
            "min_pairwise_slack": -2.34e-3,
        }
        msg = _build_feedback(strict_pass=False, diagnostics=diag, sum_radii=0.0, reported_sum=2.39)
        assert "overlap pair" in msg
        assert "between 4 and 11" in msg

    def test_wrong_circle_count(self):
        diag = {
            "n_circles_submitted": 25,
            "boundary_violations": [],
            "overlap_pairs": [],
            "min_pairwise_slack": 0.0,
        }
        msg = _build_feedback(strict_pass=False, diagnostics=diag, sum_radii=0.0, reported_sum=2.0)
        assert "got 25 circles" in msg

    def test_short_string(self):
        # Sanity: feedback shouldn't grow unbounded
        diag = {
            "n_circles_submitted": 26,
            "boundary_violations": [
                {"circle_index": i, "slack": -1e-3, "center": [0, 0], "radius": 0.1}
                for i in range(10)
            ],
            "overlap_pairs": [
                {"pair": [i, i + 1], "interpenetration": 1e-3,
                 "center_distance": 0.1, "radius_sum": 0.101}
                for i in range(10)
            ],
            "min_pairwise_slack": -1e-3,
        }
        msg = _build_feedback(strict_pass=False, diagnostics=diag, sum_radii=0.0, reported_sum=2.0)
        # Loose ceiling — the goal is "small enough to embed in a prompt"
        assert len(msg) < 600
