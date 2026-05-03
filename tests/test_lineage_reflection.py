"""Tests for ancestry-aware trace reflection (Phase 5).

Covers:
- ProgramDatabase.walk_lineage edge cases (None id, max_depth, broken
  parent links, cycles, root-reached, oldest-first ordering).
- trace_reflection._build_lineage_block rendering.
- generate_directive accepting + serializing ancestors into the prompt.
- Controller helper threads lineage_depth through to generate_directive.
- lineage_depth=0 reproduces Phase 3 behavior (no lineage section).
"""
from typing import List, Optional
from unittest.mock import AsyncMock

import pytest

from reps.config import Config
from reps.controller import ProcessParallelController
from reps.database import Program, ProgramDatabase
from reps.trace_reflection import (
    _build_lineage_block,
    _build_user_message,
    generate_directive,
)


def _add(db: ProgramDatabase, program: Program, island: int = 0):
    db.add(program, target_island=island)


def _chain_db(num_islands: int = 1) -> ProgramDatabase:
    cfg = Config()
    cfg.database.num_islands = num_islands
    cfg.database.population_size = 30
    return ProgramDatabase(cfg.database)


class TestWalkLineage:
    def test_returns_empty_when_id_is_none(self):
        db = _chain_db()
        assert db.walk_lineage(None) == []

    def test_returns_empty_when_id_not_in_db(self):
        db = _chain_db()
        assert db.walk_lineage("ghost") == []

    def test_returns_empty_when_max_depth_zero(self):
        db = _chain_db()
        _add(db, Program(id="a", code="x", metrics={"combined_score": 0.5}))
        assert db.walk_lineage("a", max_depth=0) == []

    def test_single_program_no_parent(self):
        db = _chain_db()
        seed = Program(id="seed", code="x", metrics={"combined_score": 0.5})
        _add(db, seed)
        chain = db.walk_lineage("seed")
        assert [p.id for p in chain] == ["seed"]

    def test_chain_oldest_first(self):
        db = _chain_db()
        _add(db, Program(id="g0", code="x", generation=0))
        _add(db, Program(id="g1", code="y", generation=1, parent_id="g0"))
        _add(db, Program(id="g2", code="z", generation=2, parent_id="g1"))
        chain = db.walk_lineage("g2", max_depth=10)
        assert [p.id for p in chain] == ["g0", "g1", "g2"]

    def test_max_depth_truncates_to_most_recent(self):
        db = _chain_db()
        for i in range(6):
            _add(db, Program(
                id=f"g{i}", code="x", generation=i,
                parent_id=f"g{i-1}" if i > 0 else None,
            ))
        chain = db.walk_lineage("g5", max_depth=3)
        # max_depth=3 keeps the 3 most-recent (closest to start).
        assert [p.id for p in chain] == ["g3", "g4", "g5"]

    def test_broken_parent_link_stops_walk(self):
        db = _chain_db()
        _add(db, Program(id="g1", code="y", generation=1, parent_id="missing_g0"))
        _add(db, Program(id="g2", code="z", generation=2, parent_id="g1"))
        chain = db.walk_lineage("g2", max_depth=10)
        # Walk stops at g1 since g1.parent_id="missing_g0" doesn't resolve.
        assert [p.id for p in chain] == ["g1", "g2"]

    def test_cycle_protection(self):
        db = _chain_db()
        # Synthetic cycle: a → b → a (only possible by direct mutation).
        a = Program(id="a", code="x", generation=0, parent_id="b")
        b = Program(id="b", code="y", generation=0, parent_id="a")
        db.programs["a"] = a
        db.programs["b"] = b
        # Cycle protection ensures finite chain.
        chain = db.walk_lineage("a", max_depth=10)
        assert len(chain) == 2
        assert {p.id for p in chain} == {"a", "b"}


class TestBuildLineageBlock:
    def test_empty_returns_empty_string(self):
        assert _build_lineage_block([]) == ""

    def test_renders_oldest_first_with_score_and_changes(self):
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.30}, changes_description="initial"),
            Program(id="g1", code="x", generation=1,
                    metrics={"combined_score": 0.45}, changes_description="add boundary check"),
        ]
        block = _build_lineage_block(ancestors)
        assert "Recent ancestor lineage" in block
        # Oldest first: g0 line before g1 line.
        assert block.index("gen 0") < block.index("gen 1")
        assert "0.3000" in block  # g0 score
        assert "0.4500" in block  # g1 score
        assert "initial" in block
        assert "add boundary check" in block

    def test_includes_prior_directive_when_set(self):
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4},
                    changes_description="seed",
                    mutation_directive="reduce overlap area"),
        ]
        block = _build_lineage_block(ancestors)
        assert "prior directive" in block
        assert "reduce overlap area" in block

    def test_omits_directive_when_none(self):
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4}, changes_description="seed"),
        ]
        block = _build_lineage_block(ancestors)
        assert "prior directive" not in block

    def test_truncates_long_changes_description(self):
        long_desc = "X" * 300
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4}, changes_description=long_desc),
        ]
        block = _build_lineage_block(ancestors)
        assert "..." in block
        # Short enough to not blow up the prompt.
        assert len(block) < 250

    def test_truncates_long_directive(self):
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4},
                    changes_description="seed",
                    mutation_directive="Y" * 300),
        ]
        block = _build_lineage_block(ancestors)
        # Long directive truncated; total block stays compact.
        assert "..." in block
        assert len(block) < 400

    def test_combined_long_description_and_directive_bounded(self):
        # Both fields long simultaneously — independent truncation should
        # still keep the total per-ancestor block well under any reasonable
        # cap. Worst case: ~120 (desc) + ~100 (directive) + ~80 (frame).
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4},
                    changes_description="X" * 500,
                    mutation_directive="Y" * 500),
        ]
        block = _build_lineage_block(ancestors)
        # Tighter bound than either single-field test asserts on its own.
        assert len(block) < 450
        # Both fields ARE included (just truncated, not dropped).
        assert "XXX" in block
        assert "YYY" in block

    def test_handles_missing_score(self):
        # Programs without combined_score (e.g. evaluator returned an error).
        ancestors = [
            Program(id="g0", code="x", generation=0, metrics={}),
        ]
        block = _build_lineage_block(ancestors)
        assert "?" in block  # score render uses "?" sentinel


class TestBuildUserMessageWithLineage:
    def _parent(self):
        return Program(
            id="p", code="def f(): pass",
            metrics={"combined_score": 0.5},
            per_instance_scores={"x": 1.0, "y": 0.0},
            feedback="y dimension regressed because of foo bar baz",
        )

    def test_no_ancestors_omits_section(self):
        msg = _build_user_message(self._parent(), max_code_chars=4000, ancestors=None)
        assert "Recent ancestor lineage" not in msg

    def test_empty_ancestors_omits_section(self):
        msg = _build_user_message(self._parent(), max_code_chars=4000, ancestors=[])
        assert "Recent ancestor lineage" not in msg

    def test_with_ancestors_prepends_section(self):
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.30}, changes_description="seed"),
            Program(id="g1", code="x", generation=1,
                    metrics={"combined_score": 0.45}, changes_description="tweak"),
        ]
        msg = _build_user_message(
            self._parent(), max_code_chars=4000, ancestors=ancestors,
        )
        # Lineage prepended (so the LLM sees historical context first).
        assert msg.startswith("Recent ancestor lineage")
        # All standard sections still present.
        assert "Per-objective scores" in msg
        assert "Diagnostic feedback" in msg
        assert "Current program" in msg


class TestGenerateDirectivePassesAncestors:
    @pytest.mark.asyncio
    async def test_ancestors_appear_in_user_message(self):
        llm = AsyncMock()
        llm.generate_with_context.return_value = "fix it"
        parent = Program(
            id="p", code="pass", parent_id="g0",
            metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 0.5},
            feedback="x dimension regressed because of foo",
        )
        ancestors = [
            Program(id="g0", code="x", generation=0,
                    metrics={"combined_score": 0.4},
                    changes_description="initial seed",
                    mutation_directive="tighten the spacing"),
        ]
        await generate_directive(parent, llm, ancestors=ancestors)
        sent = llm.generate_with_context.call_args.kwargs["messages"][0]["content"]
        assert "initial seed" in sent
        assert "tighten the spacing" in sent

    @pytest.mark.asyncio
    async def test_no_ancestors_works_unchanged(self):
        llm = AsyncMock()
        llm.generate_with_context.return_value = "fix it"
        parent = Program(
            id="p", code="pass",
            metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 0.5},
            feedback="x dimension regressed because of foo",
        )
        await generate_directive(parent, llm)
        sent = llm.generate_with_context.call_args.kwargs["messages"][0]["content"]
        assert "Recent ancestor lineage" not in sent


class TestControllerLineageWiring:
    def _make_controller(self, *, lineage_depth: int = 3):
        cfg = Config()
        cfg.reps.enabled = True
        cfg.reps.trace_reflection.enabled = True
        cfg.reps.trace_reflection.min_feedback_length = 10
        cfg.reps.trace_reflection.lineage_depth = lineage_depth

        db = ProgramDatabase(cfg.database)
        ctrl = ProcessParallelController.__new__(ProcessParallelController)
        ctrl.config = cfg
        ctrl.database = db
        ctrl._reps_enabled = True
        ctrl.llm_ensemble = AsyncMock()
        return ctrl

    @pytest.mark.asyncio
    async def test_lineage_depth_zero_skips_lineage(self):
        ctrl = self._make_controller(lineage_depth=0)
        ctrl.llm_ensemble.generate_with_context.return_value = "ok"

        _add(ctrl.database, Program(
            id="g0", code="x", generation=0,
            metrics={"combined_score": 0.4},
            changes_description="seed",
        ))
        parent = Program(
            id="p", code="pass", parent_id="g0",
            metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 0.5},
            feedback="x dimension regressed because of foo",
        )
        _add(ctrl.database, parent)

        await ctrl._maybe_generate_trace_directive(parent)
        sent = ctrl.llm_ensemble.generate_with_context.call_args.kwargs["messages"][0]["content"]
        assert "Recent ancestor lineage" not in sent

    @pytest.mark.asyncio
    async def test_lineage_depth_propagates(self):
        ctrl = self._make_controller(lineage_depth=2)
        ctrl.llm_ensemble.generate_with_context.return_value = "ok"

        _add(ctrl.database, Program(
            id="g0", code="x", generation=0,
            metrics={"combined_score": 0.30},
            changes_description="oldest ancestor",
        ))
        _add(ctrl.database, Program(
            id="g1", code="y", generation=1, parent_id="g0",
            metrics={"combined_score": 0.40},
            changes_description="middle ancestor",
        ))
        _add(ctrl.database, Program(
            id="g2", code="z", generation=2, parent_id="g1",
            metrics={"combined_score": 0.45},
            changes_description="closest ancestor",
        ))
        parent = Program(
            id="p", code="pass", parent_id="g2",
            metrics={"combined_score": 0.5},
            per_instance_scores={"x": 0.0, "y": 0.5},
            feedback="x dimension regressed because of foo",
        )
        _add(ctrl.database, parent)

        await ctrl._maybe_generate_trace_directive(parent)
        sent = ctrl.llm_ensemble.generate_with_context.call_args.kwargs["messages"][0]["content"]
        # depth=2 keeps the 2 most-recent ancestors of `parent` (g1, g2),
        # NOT g0. Confirm by changes_description presence.
        assert "Recent ancestor lineage" in sent
        assert "middle ancestor" in sent
        assert "closest ancestor" in sent
        assert "oldest ancestor" not in sent