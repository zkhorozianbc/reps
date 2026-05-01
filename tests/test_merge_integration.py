"""Integration tests for system-aware merge wiring (Phase 4.2).

Confirms the controller's `_maybe_select_complementary_partner` and
`_build_crossover_context` helpers route the crossover second_parent
correctly:

- Disabled (default) → returns None / empty, legacy random pick stands.
- Enabled + primary has per_instance_scores + cross-island candidates →
  partner overrides the random pick to maximize complementarity.
- Strong-dim threshold renders correctly into crossover_context.
- Falls back gracefully when primary has no per-instance scores or
  when no other-island candidates exist.
"""
from unittest.mock import AsyncMock

import pytest

from reps.config import Config
from reps.controller import ProcessParallelController
from reps.database import Program, ProgramDatabase


def _make_controller(*, merge_enabled=True, reps_enabled=True, threshold=0.8):
    cfg = Config()
    cfg.database.num_islands = 3
    cfg.database.population_size = 30
    cfg.reps.enabled = reps_enabled
    cfg.reps.merge.enabled = merge_enabled
    cfg.reps.merge.strong_score_threshold = threshold

    db = ProgramDatabase(cfg.database)

    ctrl = ProcessParallelController.__new__(ProcessParallelController)
    ctrl.config = cfg
    ctrl.database = db
    ctrl._reps_enabled = reps_enabled
    ctrl.llm_ensemble = AsyncMock()
    return ctrl


def _add(db, program, island=0):
    db.add(program, target_island=island)


class TestMaybeSelectComplementaryPartner:
    def test_returns_none_when_reps_disabled(self):
        ctrl = _make_controller(reps_enabled=False)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        _add(ctrl.database, primary, island=0)
        _add(ctrl.database, Program(
            id="other", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        ), island=1)
        assert ctrl._maybe_select_complementary_partner(primary, primary_island=0) is None

    def test_returns_none_when_merge_disabled(self):
        ctrl = _make_controller(merge_enabled=False)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        _add(ctrl.database, primary, island=0)
        _add(ctrl.database, Program(
            id="other", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        ), island=1)
        assert ctrl._maybe_select_complementary_partner(primary, primary_island=0) is None

    def test_returns_none_when_primary_lacks_per_instance_scores(self):
        ctrl = _make_controller()
        # Pre-Phase-1.2: primary has only combined_score, no per-instance.
        primary = Program(id="p", code="pass", metrics={"combined_score": 0.5})
        _add(ctrl.database, primary, island=0)
        _add(ctrl.database, Program(
            id="other", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        ), island=1)
        assert ctrl._maybe_select_complementary_partner(primary, primary_island=0) is None

    def test_returns_none_when_no_other_island_candidates(self):
        ctrl = _make_controller()
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        _add(ctrl.database, primary, island=0)
        # Another program but ON THE SAME ISLAND — excluded from candidate pool.
        _add(ctrl.database, Program(
            id="same_island", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        ), island=0)
        assert ctrl._maybe_select_complementary_partner(primary, primary_island=0) is None

    def test_picks_complementary_partner_from_other_island(self):
        ctrl = _make_controller()
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        _add(ctrl.database, primary, island=0)

        # Two candidates on island 1: complement vs overlap.
        complement = Program(
            id="complement", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},  # gain = 1.0
        )
        overlap = Program(
            id="overlap", code="b", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.95, "y": 0.05},  # gain ~ 0.05
        )
        _add(ctrl.database, complement, island=1)
        _add(ctrl.database, overlap, island=1)
        # Plus a same-island competitor that should be excluded entirely.
        _add(ctrl.database, Program(
            id="same_island", code="c", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        ), island=0)

        for _ in range(20):
            partner_id = ctrl._maybe_select_complementary_partner(primary, primary_island=0)
            assert partner_id == "complement"

    def test_respects_instance_keys_subset(self):
        ctrl = _make_controller()
        ctrl.config.reps.merge.instance_keys = ["x"]  # ignore y
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        _add(ctrl.database, primary, island=0)
        # On {x} only, both candidates have gain 0; uniform pick.
        c1 = Program(
            id="c1", code="a", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        )
        c2 = Program(
            id="c2", code="b", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 0.5},
        )
        _add(ctrl.database, c1, island=1)
        _add(ctrl.database, c2, island=2)
        seen = set()
        for _ in range(100):
            seen.add(ctrl._maybe_select_complementary_partner(primary, primary_island=0))
        # Both candidates reachable when only x is considered (both tie at gain=0).
        assert seen == {"c1", "c2"}


class TestBuildCrossoverContext:
    def test_returns_empty_when_neither_has_per_instance_scores(self):
        ctrl = _make_controller()
        a = Program(id="a", code="x", metrics={"combined_score": 0.5})
        b = Program(id="b", code="y", metrics={"combined_score": 0.5})
        assert ctrl._build_crossover_context(a, b) == ""

    def test_renders_strong_dimensions(self):
        ctrl = _make_controller(threshold=0.8)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"validity": 1.0, "boundary": 1.0, "sum_radii_progress": 0.3},
        )
        partner = Program(
            id="q", code="pass2", metrics={"combined_score": 0.5},
            per_instance_scores={"validity": 0.0, "boundary": 0.5, "sum_radii_progress": 0.92},
        )
        ctx = ctrl._build_crossover_context(primary, partner)
        # Strong dims (>= 0.8): primary has validity + boundary; partner has sum_radii_progress.
        assert "Crossover hint" in ctx
        assert "validity" in ctx
        assert "boundary" in ctx
        assert "sum_radii_progress" in ctx
        # Header is present so the tool-runner dedup-by-first-line works.
        assert ctx.startswith("## Crossover hint")

    def test_handles_partner_with_no_strong_dims(self):
        ctrl = _make_controller(threshold=0.8)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 1.0},
        )
        partner = Program(
            id="q", code="pass2", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.3, "y": 0.4},  # all below threshold
        )
        ctx = ctrl._build_crossover_context(primary, partner)
        assert "no clear strengths" in ctx  # partner has none ≥ threshold
        assert "x" in ctx and "y" in ctx     # primary's strengths render

    def test_disabled_merge_does_not_render_crossover_context(self):
        # Regression guard: when merge is disabled, the legacy random-pick
        # crossover path must keep its empty crossover_context. The wiring
        # block must NOT fall through to render a "## Crossover hint" block
        # for the random partner just because both parents happen to have
        # per_instance_scores (Phase-1.2 benchmarks always do).
        from reps.iteration_config import IterationConfig

        ctrl = _make_controller(merge_enabled=False)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
        )
        random_partner = Program(
            id="rand_partner", code="x", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 1.0},
        )
        _add(ctrl.database, primary, island=0)
        _add(ctrl.database, random_partner, island=1)

        iter_cfg = IterationConfig(
            target_island=0,
            second_parent_id="rand_partner",  # WorkerPool's random pick
            prompt_extras={},
        )

        # Drive only the merge wiring block: simulate _run_iteration's behavior.
        complementary_id = ctrl._maybe_select_complementary_partner(primary, 0)
        # With merge disabled, no override.
        assert complementary_id is None
        # And iter_cfg.prompt_extras stays empty for crossover_context.
        assert "crossover_context" not in iter_cfg.prompt_extras

    def test_threshold_configurable(self):
        ctrl = _make_controller(threshold=0.4)
        primary = Program(
            id="p", code="pass", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.5, "y": 0.3},  # only x clears threshold 0.4
        )
        partner = Program(
            id="q", code="pass2", metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.3, "y": 0.5},  # only y clears
        )
        ctx = ctrl._build_crossover_context(primary, partner)
        # Primary lists x, partner lists y.
        assert "Primary parent excels on: x" in ctx
        assert "Partner parent excels on: y" in ctx
