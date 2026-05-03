"""Integration tests for Pareto sampling in ProgramDatabase + controller routing
(Phase 2.2)."""
import random

import pytest

from reps.config import Config
from reps.database import Program, ProgramDatabase


def _make_db(num_islands=2, seed=42):
    cfg = Config()
    cfg.database.num_islands = num_islands
    cfg.database.population_size = 20
    cfg.database.random_seed = seed
    return ProgramDatabase(cfg.database)


def _add(db, program, island=0):
    db.add(program, target_island=island)


class TestSampleParetoFromIsland:
    def test_picks_only_from_pareto_frontier(self):
        db = _make_db()
        # Frontier: a (best on x) and b (best on y). c is strictly dominated.
        a = Program(id="a", code="pass", metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 1.0, "y": 0.0})
        b = Program(id="b", code="pass2", metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 0.0, "y": 1.0})
        c = Program(id="c", code="pass3", metrics={"combined_score": 0.0},
                    per_instance_scores={"x": 0.0, "y": 0.0})
        _add(db, a, island=0)
        _add(db, b, island=0)
        _add(db, c, island=0)

        random.seed(0)
        picks = set()
        for _ in range(50):
            parent, _ = db.sample_pareto_from_island(island_id=0, num_inspirations=0)
            picks.add(parent.id)
        assert picks == {"a", "b"}  # never c

    def test_falls_back_to_sample_from_island_when_empty(self):
        db = _make_db()
        # No programs in island 0 — fallback path returns whatever sample()
        # produces (which itself uses any program when island is empty).
        # Add one program to island 1 so the fallback has something to pick.
        p = Program(id="only", code="pass",
                    metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 0.5})
        _add(db, p, island=1)
        parent, _ = db.sample_pareto_from_island(island_id=0, num_inspirations=0)
        assert parent is not None

    def test_inspirations_from_same_island_excluding_parent(self):
        db = _make_db()
        for i in range(5):
            _add(db, Program(
                id=f"p{i}", code=f"pass{i}",
                metrics={"combined_score": 0.5},
                per_instance_scores={"x": 1.0 if i == 0 else 0.0,
                                     "y": 0.0 if i == 0 else 1.0},
            ), island=0)

        random.seed(0)
        parent, inspirations = db.sample_pareto_from_island(
            island_id=0, num_inspirations=3
        )
        assert parent.id in {f"p{i}" for i in range(5)}
        assert len(inspirations) == 3
        for insp in inspirations:
            assert insp.id != parent.id

    def test_instance_keys_restricts_frontier(self):
        # Restricting Pareto to a subset of keys can change who counts as
        # frontier. Two programs that are complementary on (x, y) collapse
        # to a single winner when only x is considered.
        db = _make_db()
        a = Program(id="a", code="pass", metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 1.0, "y": 0.0})
        b = Program(id="b", code="pass2", metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 0.0, "y": 1.0})
        _add(db, a, island=0)
        _add(db, b, island=0)

        # Without restriction: both on frontier.
        random.seed(0)
        unrestricted = set()
        for _ in range(50):
            parent, _ = db.sample_pareto_from_island(island_id=0, num_inspirations=0)
            unrestricted.add(parent.id)
        assert unrestricted == {"a", "b"}

        # Restricted to {x}: a dominates b, only a survives.
        random.seed(0)
        restricted = set()
        for _ in range(50):
            parent, _ = db.sample_pareto_from_island(
                island_id=0, num_inspirations=0, instance_keys=["x"]
            )
            restricted.add(parent.id)
        assert restricted == {"a"}

    def test_works_without_per_instance_scores(self):
        # Pre-Phase-1.2 benchmarks have only combined_score. Pareto should
        # degenerate to "all top-combined-score tied programs are on the
        # frontier" via the broadcast fallback.
        db = _make_db()
        _add(db, Program(id="hi", code="a", metrics={"combined_score": 0.9}), island=0)
        _add(db, Program(id="lo", code="b", metrics={"combined_score": 0.1}), island=0)

        random.seed(0)
        for _ in range(20):
            parent, _ = db.sample_pareto_from_island(island_id=0, num_inspirations=0)
            assert parent.id == "hi"  # only frontier member


class TestControllerStrategyGating:
    """Verify _should_sample_pareto returns the right thing per strategy."""

    def _make_controller_stub(self, strategy, fraction=0.0):
        # Minimal stub to call _should_sample_pareto without spinning up the
        # full controller (which needs an evaluator file, LLM ensemble, etc.).
        from reps.controller import ProcessParallelController

        cfg = Config()
        cfg.database.selection_strategy = strategy
        cfg.database.pareto_fraction = fraction
        db = ProgramDatabase(cfg.database)

        # Bypass __init__ to avoid evaluator/LLM setup.
        ctrl = ProcessParallelController.__new__(ProcessParallelController)
        ctrl.database = db
        return ctrl

    def test_map_elites_never_pareto(self):
        ctrl = self._make_controller_stub("map_elites")
        for _ in range(20):
            assert ctrl._should_sample_pareto() is False

    def test_pareto_always_pareto(self):
        ctrl = self._make_controller_stub("pareto")
        for _ in range(20):
            assert ctrl._should_sample_pareto() is True

    def test_mixed_zero_fraction_never_pareto(self):
        ctrl = self._make_controller_stub("mixed", fraction=0.0)
        # With fraction=0.0 we should reproduce existing behavior exactly.
        for _ in range(50):
            assert ctrl._should_sample_pareto() is False

    def test_mixed_full_fraction_always_pareto(self):
        ctrl = self._make_controller_stub("mixed", fraction=1.0)
        for _ in range(20):
            assert ctrl._should_sample_pareto() is True

    def test_mixed_partial_fraction_distribution(self):
        ctrl = self._make_controller_stub("mixed", fraction=0.5)
        random.seed(0)
        n = 1000
        hits = sum(1 for _ in range(n) if ctrl._should_sample_pareto())
        # 0.5 ± 0.05 with n=1000 is well within the 95% CI.
        assert 400 < hits < 600

    def test_mixed_clamps_fraction_above_1(self):
        ctrl = self._make_controller_stub("mixed", fraction=2.0)
        for _ in range(20):
            assert ctrl._should_sample_pareto() is True

    def test_mixed_clamps_fraction_below_0(self):
        ctrl = self._make_controller_stub("mixed", fraction=-1.0)
        for _ in range(20):
            assert ctrl._should_sample_pareto() is False


class TestPickIterationInputsRoutesPareto:
    """End-to-end: when strategy=pareto, `_pick_iteration_inputs` calls
    `sample_pareto_from_island` (not `sample_from_island`) and the resulting
    parent is on the Pareto frontier of the target island."""

    def test_pareto_strategy_routes_through_sample_pareto(self):
        from unittest.mock import MagicMock
        from reps.controller import ProcessParallelController

        cfg = Config()
        cfg.database.num_islands = 2
        cfg.database.population_size = 20
        cfg.database.selection_strategy = "pareto"

        db = ProgramDatabase(cfg.database)
        # Build an island with one frontier member and one dominated member.
        front = Program(id="front", code="a", metrics={"combined_score": 0.5},
                        per_instance_scores={"x": 1.0, "y": 1.0})
        dom = Program(id="dom", code="b", metrics={"combined_score": 0.0},
                      per_instance_scores={"x": 0.0, "y": 0.0})
        db.add(front, target_island=0)
        db.add(dom, target_island=0)

        # Stub controller to avoid evaluator/LLM setup.
        ctrl = ProcessParallelController.__new__(ProcessParallelController)
        ctrl.database = db
        ctrl.config = cfg
        ctrl._plateau_boost_remaining = 0

        # Spy on both sampling paths.
        ctrl.database.sample_pareto_from_island = MagicMock(
            wraps=ctrl.database.sample_pareto_from_island
        )
        ctrl.database.sample_from_island = MagicMock(
            wraps=ctrl.database.sample_from_island
        )

        for _ in range(5):
            parent_id, _, _ = ctrl._pick_iteration_inputs(
                iteration=1, island_id=0, reps_iter_config=None
            )
            assert parent_id == "front"  # only frontier member

        assert ctrl.database.sample_pareto_from_island.called
        assert not ctrl.database.sample_from_island.called

    def test_map_elites_strategy_does_not_call_pareto(self):
        from unittest.mock import MagicMock
        from reps.controller import ProcessParallelController

        cfg = Config()
        cfg.database.num_islands = 2
        cfg.database.population_size = 20
        cfg.database.selection_strategy = "map_elites"  # default, but explicit

        db = ProgramDatabase(cfg.database)
        p = Program(id="only", code="a", metrics={"combined_score": 0.5},
                    per_instance_scores={"x": 1.0})
        db.add(p, target_island=0)

        ctrl = ProcessParallelController.__new__(ProcessParallelController)
        ctrl.database = db
        ctrl.config = cfg
        ctrl._plateau_boost_remaining = 0

        ctrl.database.sample_pareto_from_island = MagicMock()
        ctrl.database.sample_from_island = MagicMock(
            wraps=ctrl.database.sample_from_island
        )

        for _ in range(5):
            ctrl._pick_iteration_inputs(iteration=1, island_id=0, reps_iter_config=None)

        assert not ctrl.database.sample_pareto_from_island.called
        assert ctrl.database.sample_from_island.called
