"""Unit tests for reps/trace_reflection.py (Phase 3.1).

Pure async function, mocked LLM. Tests the should_generate_directive
predicate and the generate_directive flow including failure paths.
"""
import asyncio
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from reps.database import Program
from reps.trace_reflection import (
    _build_user_message,
    _excerpt_code,
    generate_directive,
    should_generate_directive,
)


_UNSET = object()


def _parent(
    *,
    feedback: Optional[str] = "invalid; 3 boundary violations (worst slack -1.234e-2 at circle 17)",
    per_instance_scores=_UNSET,  # use sentinel so explicit None / {} aren't overridden
    code: str = "def f(): return 1",
    mutation_directive: Optional[str] = None,
):
    if per_instance_scores is _UNSET:
        per_instance_scores = {"validity": 0.0, "boundary": 0.85}
    return Program(
        id="parent_id",
        code=code,
        metrics={"combined_score": 0.5},
        per_instance_scores=per_instance_scores,
        feedback=feedback,
        mutation_directive=mutation_directive,
    )


class TestShouldGenerateDirective:
    def test_qualifies_with_feedback_and_scores(self):
        assert should_generate_directive(_parent(), min_feedback_length=20) is True

    def test_skips_when_no_feedback(self):
        assert should_generate_directive(_parent(feedback=None), min_feedback_length=20) is False

    def test_skips_when_feedback_too_short(self):
        assert should_generate_directive(_parent(feedback="oops"), min_feedback_length=20) is False

    def test_skips_when_feedback_only_whitespace(self):
        assert should_generate_directive(
            _parent(feedback="     " * 20), min_feedback_length=20
        ) is False

    def test_skips_when_no_per_instance_scores(self):
        assert should_generate_directive(
            _parent(per_instance_scores=None), min_feedback_length=20
        ) is False

    def test_skips_when_per_instance_scores_empty_dict(self):
        assert should_generate_directive(
            _parent(per_instance_scores={}), min_feedback_length=20
        ) is False

    def test_skips_when_directive_already_cached(self):
        assert should_generate_directive(
            _parent(mutation_directive="reduce overlap area"), min_feedback_length=20
        ) is False


class TestExcerptCode:
    def test_short_code_returned_verbatim(self):
        assert _excerpt_code("hello world", max_chars=100) == "hello world"

    def test_long_code_keeps_head_and_tail(self):
        code = "A" * 500 + "B" * 500
        out = _excerpt_code(code, max_chars=200)
        assert "[omitted middle]" in out
        # Head is mostly As, tail is mostly Bs.
        assert out.startswith("A")
        assert out.endswith("B")
        # Length is roughly capped (allowing a few chars for the marker).
        assert len(out) < 250

    def test_max_chars_zero_safe(self):
        # No crash even with degenerate cap; the elision marker may exceed cap.
        out = _excerpt_code("hello world hello world", max_chars=0)
        assert "[omitted middle]" in out


class TestBuildUserMessage:
    def test_includes_scores_feedback_and_code(self):
        p = _parent(
            per_instance_scores={"x": 0.9, "y": 0.1},
            feedback="y dimension regressed because of foo",
            code="def foo():\n    return 'bar'",
        )
        msg = _build_user_message(p, max_code_chars=4000)
        # All three signals appear.
        assert '"x": 0.9' in msg
        assert '"y": 0.1' in msg
        assert "y dimension regressed because of foo" in msg
        assert "def foo()" in msg

    def test_truncates_long_code(self):
        big = "X" * 10_000
        p = _parent(code=big)
        msg = _build_user_message(p, max_code_chars=200)
        assert "[omitted middle]" in msg
        assert len(msg) < 1000  # well under "send the entire 10K"


class TestGenerateDirective:
    @pytest.mark.asyncio
    async def test_returns_none_when_predicate_skips(self):
        llm = AsyncMock()
        out = await generate_directive(_parent(feedback=None), llm)
        assert out is None
        # Critical: when we skip, the LLM is never called.
        llm.generate_with_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_llm_with_system_and_user_message(self):
        llm = AsyncMock()
        llm.generate_with_context.return_value = (
            "Reduce sum_radii by tightening initial spacing."
        )
        out = await generate_directive(_parent(), llm)
        assert out == "Reduce sum_radii by tightening initial spacing."

        llm.generate_with_context.assert_called_once()
        call = llm.generate_with_context.call_args
        # Cache-friendly: system message is constant; user message carries the
        # per-parent payload.
        assert "system_message" in call.kwargs
        assert "messages" in call.kwargs
        assert call.kwargs["messages"][0]["role"] == "user"
        assert "boundary violations" in call.kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        llm = AsyncMock()
        llm.generate_with_context.return_value = "  \n  do the thing.  \n  "
        out = await generate_directive(_parent(), llm)
        assert out == "do the thing."

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self):
        llm = AsyncMock()
        llm.generate_with_context.return_value = "   \n  "
        out = await generate_directive(_parent(), llm)
        assert out is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_exception(self):
        # LLM errors should never crash the harness — they degrade silently.
        llm = AsyncMock()
        llm.generate_with_context.side_effect = RuntimeError("API down")
        out = await generate_directive(_parent(), llm)
        assert out is None

    @pytest.mark.asyncio
    async def test_min_feedback_length_threshold(self):
        llm = AsyncMock()
        # Default threshold is 20; "too short" triggers skip.
        out = await generate_directive(_parent(feedback="x"), llm, min_feedback_length=20)
        assert out is None
        llm.generate_with_context.assert_not_called()

        # Lower threshold → same parent now qualifies.
        llm.generate_with_context.return_value = "fix it"
        out = await generate_directive(_parent(feedback="oops"), llm, min_feedback_length=2)
        assert out == "fix it"

    @pytest.mark.asyncio
    async def test_handles_parent_without_id_in_logging_paths(self):
        # Defensive: both the LLM-failure path and the empty-response path
        # log the parent.id; both must tolerate a parent built without one.
        # (Program.id is typed `str` but tests construct ad-hoc parents.)
        p = Program(
            id="",  # falsy id — exercises the `if parent.id else "?"` branch
            code="pass",
            metrics={"combined_score": 0.0},
            per_instance_scores={"x": 0.0},
            feedback="x dimension regressed because of foo",
        )
        # Empty-response path
        llm = AsyncMock()
        llm.generate_with_context.return_value = ""
        out = await generate_directive(p, llm)
        assert out is None

        # Exception path
        llm.generate_with_context.side_effect = RuntimeError("boom")
        out = await generate_directive(p, llm)
        assert out is None

    @pytest.mark.asyncio
    async def test_handles_nan_in_per_instance_scores(self):
        # Failed evaluations can produce NaN in per_instance_scores. The
        # JSON-serialized payload that goes to the LLM should still be
        # produced without exception.
        llm = AsyncMock()
        llm.generate_with_context.return_value = "fix the NaN dimension"
        p = _parent(per_instance_scores={"x": float("nan"), "y": 0.5})
        out = await generate_directive(p, llm)
        assert out == "fix the NaN dimension"
        sent_user = llm.generate_with_context.call_args.kwargs["messages"][0]["content"]
        assert "NaN" in sent_user  # Python's json.dumps default

    @pytest.mark.asyncio
    async def test_max_code_chars_propagates(self):
        # Large code should be truncated before being sent.
        llm = AsyncMock()
        llm.generate_with_context.return_value = "ok"
        big_code = "A" * 50_000
        await generate_directive(
            _parent(code=big_code), llm, max_code_chars=500
        )
        sent_user = llm.generate_with_context.call_args.kwargs["messages"][0]["content"]
        assert "[omitted middle]" in sent_user
        # Sanity: the full 50K isn't being shipped.
        assert sent_user.count("A") < 1000


class TestProgramMutationDirectiveField:
    def test_default_is_none(self):
        p = Program(id="x", code="pass")
        assert p.mutation_directive is None

    def test_round_trip_through_dict(self):
        p = Program(id="x", code="pass", mutation_directive="do the thing")
        p2 = Program.from_dict(p.to_dict())
        assert p2.mutation_directive == "do the thing"


class TestREPSTraceReflectionConfig:
    def test_defaults(self):
        from reps.config import REPSTraceReflectionConfig
        c = REPSTraceReflectionConfig()
        assert c.enabled is False
        assert c.model is None
        assert c.min_feedback_length == 20
        assert c.max_code_chars == 4000

    def test_yaml_round_trip(self, tmp_path):
        from reps.runner import load_experiment_config
        cfg_text = (
            "harness: reps\n"
            "provider: openrouter\n"
            "max_iterations: 10\n"
            "llm:\n  primary_model: test\n  api_key: test\n"
            "reps:\n"
            "  enabled: true\n"
            "  trace_reflection:\n"
            "    enabled: true\n"
            "    min_feedback_length: 50\n"
            "    max_code_chars: 2000\n"
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(cfg_text)
        cfg = load_experiment_config(str(path))
        assert cfg.reps.trace_reflection.enabled is True
        assert cfg.reps.trace_reflection.min_feedback_length == 50
        assert cfg.reps.trace_reflection.max_code_chars == 2000
