"""Phase 6.3 — archive integrity tests for minibatch fidelity tagging.

`ProgramDatabase.add()` must respect `metrics["fidelity"] == "minibatch"`
under the safe-by-default `promoted_only` policy:

  - Minibatch-only programs are diverted to `_minibatch_only` side dict.
  - They never enter `programs`, the MAP-Elites archive, the islands, or
    the Pareto frontier.
  - Sampling paths are unaffected because they only see archived programs.
  - Promoted (full-eval) programs are archived normally.
"""

from __future__ import annotations

import pytest

from reps.config import DatabaseConfig, EvaluatorConfig
from reps.database import Program, ProgramDatabase


def _make_db(policy: str = "promoted_only", num_islands: int = 2) -> ProgramDatabase:
    cfg = DatabaseConfig(
        in_memory=True,
        archive_size=10,
        population_size=100,
        num_islands=num_islands,
        feature_dimensions=["complexity", "diversity"],
        feature_bins=4,
    )
    return ProgramDatabase(cfg, minibatch_archive_policy=policy)


def _full_program(pid: str, score: float = 0.8) -> Program:
    return Program(
        id=pid,
        code=f"def f_{pid}(): return {score}\n",
        metrics={"combined_score": score, "fidelity": "full"},
    )


def _minibatch_program(pid: str, score: float = 0.2) -> Program:
    return Program(
        id=pid,
        code=f"def g_{pid}(): return {score}\n",
        metrics={"combined_score": score, "fidelity": "minibatch"},
    )


class TestPromotedOnlyPolicy:
    def test_minibatch_program_skips_archive_and_islands(self):
        db = _make_db("promoted_only")
        prog = _minibatch_program("p1")
        db.add(prog, iteration=0)
        # Must NOT appear in main programs dict (sampler never sees it).
        assert "p1" not in db.programs
        # Side dict captures it for diagnostics.
        assert "p1" in db._minibatch_only
        # No island contains it.
        assert all("p1" not in island for island in db.islands)
        # Archive untouched.
        assert "p1" not in db.archive

    def test_minibatch_program_skipped_does_not_become_island_best(self):
        db = _make_db("promoted_only")
        db.add(_minibatch_program("mb1"), iteration=0)
        # No island best should be set, since no full-fidelity program
        # has been added yet.
        assert all(b is None for b in db.island_best_programs)
        assert db.best_program_id is None

    def test_full_eval_program_is_archived(self):
        db = _make_db("promoted_only")
        prog = _full_program("p1", score=0.8)
        db.add(prog, iteration=0)
        assert "p1" in db.programs
        assert "p1" not in db._minibatch_only
        # Either archive or island registers the program.
        assert "p1" in db.archive or any("p1" in island for island in db.islands)
        # Best tracking updated.
        assert db.best_program_id == "p1"

    def test_program_without_fidelity_tag_is_archived(self):
        # Backward compat: programs with no `fidelity` key behave like
        # full-eval — archive policy does not gate them out.
        db = _make_db("promoted_only")
        prog = Program(
            id="legacy",
            code="def h(): return 0.5\n",
            metrics={"combined_score": 0.5},  # no fidelity tag
        )
        db.add(prog, iteration=0)
        assert "legacy" in db.programs
        assert "legacy" not in db._minibatch_only

    def test_mixed_population_sampling_only_sees_promoted(self):
        db = _make_db("promoted_only")
        # 3 promoted, 2 minibatch-only.
        db.add(_full_program("f1", 0.6), iteration=0)
        db.add(_full_program("f2", 0.7), iteration=0)
        db.add(_full_program("f3", 0.8), iteration=0)
        db.add(_minibatch_program("m1", 0.2), iteration=1)
        db.add(_minibatch_program("m2", 0.3), iteration=1)

        # Only promoted programs reachable from sampling structures.
        all_island_pids = set().union(*db.islands)
        assert all_island_pids <= {"f1", "f2", "f3"}
        assert all_island_pids  # non-empty

        # Side dict captures both minibatch entries.
        assert set(db._minibatch_only.keys()) == {"m1", "m2"}

        # The standard sampler picks a parent from islands — must never
        # return a minibatch-only program.
        for _ in range(20):
            parent, _ = db.sample()
            assert parent.id in {"f1", "f2", "f3"}

    def test_pareto_frontier_excludes_minibatch_programs(self):
        # `sample_pareto_from_island` reads `self.islands[i]`, which never
        # contains minibatch-only programs under `promoted_only`. Verify
        # the sampler can't return one even when a minibatch-only program
        # would dominate on per-instance scores.
        db = _make_db("promoted_only")
        promoted = _full_program("good", 0.5)
        promoted.per_instance_scores = {"a": 0.5, "b": 0.5}
        db.add(promoted, iteration=0)

        sneaky = _minibatch_program("sneaky", 0.99)
        sneaky.per_instance_scores = {"a": 0.99, "b": 0.99}  # would dominate
        db.add(sneaky, iteration=1)

        # The sneaky program never made it into self.programs / islands,
        # so it can't be returned by the Pareto sampler.
        for _ in range(10):
            parent, _ = db.sample_pareto_from_island(0)
            assert parent.id == "good"


class TestAllWithTagPolicy:
    def test_minibatch_program_is_archived_under_all_with_tag(self):
        db = _make_db("all_with_tag")
        prog = _minibatch_program("p1", score=0.2)
        db.add(prog, iteration=0)
        # IS in the main population — selection treats it like any other
        # program until v1.5 stratification lands.
        assert "p1" in db.programs
        assert prog.metrics["fidelity"] == "minibatch"  # tag preserved
        # NOT routed to the side dict.
        assert "p1" not in db._minibatch_only

    def test_invalid_policy_rejected_at_construction(self):
        with pytest.raises(ValueError, match="minibatch_archive_policy"):
            _make_db("bogus")


class TestEvaluatorConfigField:
    def test_default_policy_is_promoted_only(self):
        cfg = EvaluatorConfig()
        assert cfg.minibatch_archive_policy == "promoted_only"

    def test_invalid_policy_at_evaluator_config_load(self):
        with pytest.raises(ValueError, match="minibatch_archive_policy"):
            EvaluatorConfig(minibatch_archive_policy="totally_invalid")
