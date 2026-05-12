"""Integration tests for trace_reflection wiring (Phase 3.2).

Confirms the controller's `_maybe_generate_trace_directive` helper:
- Skips when reps or trace_reflection is disabled.
- Skips when the parent doesn't qualify (no feedback / no per_instance_scores).
- Calls the LLM exactly once per parent and caches the result.
- Returns prompt-ready text wrapping the directive with a header.

Also confirms the SingleCallWorker's append loop now propagates
`trace_directive` from prompt_extras into the user prompt.
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from reps.config import Config
from reps.controller import ProcessParallelController
from reps.database import Program, ProgramDatabase


def _make_controller(*, trace_enabled=True, reps_enabled=True):
    cfg = Config()
    cfg.reps.enabled = reps_enabled
    cfg.reps.trace_reflection.enabled = trace_enabled
    cfg.reps.trace_reflection.min_feedback_length = 10  # easier for tests

    db = ProgramDatabase(cfg.database)

    ctrl = ProcessParallelController.__new__(ProcessParallelController)
    ctrl.config = cfg
    ctrl.database = db
    ctrl._reps_enabled = reps_enabled
    ctrl.llm_ensemble = AsyncMock()
    return ctrl


def _qualifying_parent():
    return Program(
        id="p_qual",
        code="def f(): return 1",
        metrics={"combined_score": 0.5},
        per_instance_scores={"validity": 0.0, "boundary": 0.85},
        feedback="invalid; 3 boundary violations (worst slack -1.234e-2)",
    )


class TestMaybeGenerateTraceDirective:
    @pytest.mark.asyncio
    async def test_returns_empty_when_reps_disabled(self):
        ctrl = _make_controller(reps_enabled=False)
        ctrl.llm_ensemble.generate_with_context.return_value = "should not be called"
        out = await ctrl._maybe_generate_trace_directive(_qualifying_parent())
        assert out == ""
        ctrl.llm_ensemble.generate_with_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_when_trace_reflection_disabled(self):
        ctrl = _make_controller(trace_enabled=False)
        ctrl.llm_ensemble.generate_with_context.return_value = "should not be called"
        out = await ctrl._maybe_generate_trace_directive(_qualifying_parent())
        assert out == ""
        ctrl.llm_ensemble.generate_with_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_when_parent_lacks_signal(self):
        ctrl = _make_controller()
        ctrl.llm_ensemble.generate_with_context.return_value = "x"
        # No feedback, no scores.
        bare = Program(id="bare", code="pass", metrics={"combined_score": 0.5})
        out = await ctrl._maybe_generate_trace_directive(bare)
        assert out == ""
        ctrl.llm_ensemble.generate_with_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_formatted_block_and_caches_directive(self):
        ctrl = _make_controller()
        ctrl.llm_ensemble.generate_with_context.return_value = (
            "Tighten initial circle spacing to eliminate boundary slack."
        )
        parent = _qualifying_parent()
        assert parent.mutation_directive is None

        out = await ctrl._maybe_generate_trace_directive(parent)
        assert "## Suggested next change" in out
        assert "Tighten initial circle spacing" in out
        # Cached on the live Program.
        assert parent.mutation_directive == (
            "Tighten initial circle spacing to eliminate boundary slack."
        )
        ctrl.llm_ensemble.generate_with_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_recall_llm_when_cached(self):
        ctrl = _make_controller()
        ctrl.llm_ensemble.generate_with_context.return_value = "first directive"

        parent = _qualifying_parent()
        await ctrl._maybe_generate_trace_directive(parent)
        assert ctrl.llm_ensemble.generate_with_context.call_count == 1

        # Second call on the same parent: cached predicate returns False, so
        # the helper returns "" (cache short-circuits the LLM call).
        out2 = await ctrl._maybe_generate_trace_directive(parent)
        assert out2 == ""
        assert ctrl.llm_ensemble.generate_with_context.call_count == 1

    @pytest.mark.asyncio
    async def test_swallows_llm_exception(self):
        ctrl = _make_controller()
        ctrl.llm_ensemble.generate_with_context.side_effect = RuntimeError("boom")
        out = await ctrl._maybe_generate_trace_directive(_qualifying_parent())
        assert out == ""

    @pytest.mark.asyncio
    async def test_empty_directive_returns_empty_block(self):
        ctrl = _make_controller()
        # LLM returned only whitespace → directive is None → no header.
        ctrl.llm_ensemble.generate_with_context.return_value = "   "
        parent = _qualifying_parent()
        out = await ctrl._maybe_generate_trace_directive(parent)
        assert out == ""
        # Nothing cached (we don't cache empty directives so a future retry
        # could try again if conditions change).
        assert parent.mutation_directive is None


class TestSingleCallWorkerPropagatesTraceDirective:
    """The hardcoded extras list in single_call.py must include trace_directive
    so the prompt_extras["trace_directive"] block gets appended to the user
    prompt when no template placeholder consumes it."""

    def test_trace_directive_in_hardcoded_keys(self):
        # Read the source so a future refactor that drops the key fails loudly.
        import inspect
        from reps.workers import single_call
        src = inspect.getsource(single_call.SingleCallWorker.run)
        assert '"trace_directive"' in src

    def test_trace_directive_in_dspy_react_keys(self):
        pytest.importorskip("dspy")
        import inspect
        from reps.workers import dspy_react
        src = inspect.getsource(dspy_react._fmt_extras)
        assert '"trace_directive"' in src
