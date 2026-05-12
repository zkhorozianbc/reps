"""Focused tests for native tool-runner worker contracts."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from reps.database import Program
from reps.workers.anthropic_tool_runner import AnthropicToolRunnerWorker
from reps.workers.base import WorkerConfig, WorkerContext, WorkerRequest
from reps.workers.openai_tool_runner import OpenAIToolRunnerWorker


class _PromptSampler:
    def build_prompt(self, **_kwargs):
        return {"system": "fallback system", "user": "user prompt"}


class _FailingEvaluator:
    async def evaluate_isolated(self, *_args, **_kwargs):
        raise RuntimeError("reeval exploded")


def _request() -> WorkerRequest:
    parent = Program(id="parent", code="x = 1\n", metrics={"combined_score": 0.1})
    return WorkerRequest(
        parent=parent,
        inspirations=[],
        top_programs=[],
        second_parent=None,
        iteration=7,
        language="python",
        feature_dimensions=[],
        generation_mode="diff",
        prompt_extras={},
    )


def _ctx(*, config=None, evaluator=None) -> WorkerContext:
    return WorkerContext(
        prompt_sampler=_PromptSampler(),
        llm_factory=lambda _name: None,
        dspy_lm_factory=lambda _cfg: None,
        evaluator=evaluator,
        scratch_id_factory=lambda: "scratch-id",
        final_child_id="child-id",
        config=config
        or SimpleNamespace(
            reasoning="high",
            llm=SimpleNamespace(
                api_key="shared-key",
                api_base="https://shared.example/v1",
                timeout=123,
                retries=4,
                retry_delay=9,
                max_tokens=321,
                reasoning_effort=None,
            ),
        ),
        iteration_config=SimpleNamespace(),
    )


def _openai_submit_child_response():
    code = "def solve():\n    return 2\n\nresult = solve()\n"
    call = SimpleNamespace(
        type="function_call",
        name="submit_child",
        call_id="call-1",
        arguments=json.dumps({"code": code, "changes_description": "change x"}),
    )
    return SimpleNamespace(
        id="resp-1",
        status="completed",
        output=[call],
        usage=None,
    )


def _anthropic_submit_child_response():
    code = "def solve():\n    return 2\n\nresult = solve()\n"
    raw = SimpleNamespace(
        type="tool_use",
        id="tool-1",
        name="submit_child",
        input={"code": code, "changes_description": "change x"},
    )
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[raw],
        usage=None,
    )


@pytest.mark.asyncio
@patch("reps.workers.openai_tool_runner.AsyncOpenAI")
async def test_openai_tool_runner_uses_shared_llm_config_with_impl_overrides(mock_client):
    cfg = WorkerConfig(
        name="openai-tools",
        impl="openai_tool_runner",
        model_id="gpt-4o",
        max_turns=1,
        tools=["submit_child"],
        impl_options={"api_key": "override-key", "timeout": 456},
    )
    worker = OpenAIToolRunnerWorker(cfg)
    worker._call_with_retry = AsyncMock(return_value=_openai_submit_child_response())

    result = await worker.run(_request(), _ctx())

    assert result.error is None
    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "override-key"
    assert kwargs["base_url"] == "https://shared.example/v1"
    assert kwargs["timeout"] == 456
    assert worker.retries == 4
    assert worker.max_tokens == 321
    assert worker.reasoning_effort == "high"


@pytest.mark.asyncio
@patch("reps.workers.anthropic_tool_runner.anthropic.AsyncAnthropic")
async def test_anthropic_tool_runner_uses_shared_llm_config_with_impl_overrides(mock_client):
    cfg = WorkerConfig(
        name="anthropic-tools",
        impl="anthropic_tool_runner",
        model_id="claude-sonnet-4.6",
        max_turns=1,
        tools=["submit_child"],
        impl_options={"api_key": "override-key", "timeout": 456, "thinking_effort": "medium"},
    )
    worker = AnthropicToolRunnerWorker(cfg)
    worker._call_with_retry = AsyncMock(return_value=_anthropic_submit_child_response())

    result = await worker.run(_request(), _ctx())

    assert result.error is None
    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "override-key"
    assert kwargs["base_url"] == "https://shared.example/v1"
    assert kwargs["timeout"] == 456
    assert worker.retries == 4
    assert worker.max_tokens == 321
    assert worker.thinking_effort == "medium"


@pytest.mark.asyncio
@patch("reps.workers.openai_tool_runner.AsyncOpenAI")
async def test_openai_tool_runner_reevaluation_error_is_not_accepted(mock_client):
    cfg = WorkerConfig(
        name="openai-tools",
        impl="openai_tool_runner",
        model_id="gpt-4o",
        max_turns=1,
        tools=["submit_child"],
    )
    worker = OpenAIToolRunnerWorker(cfg)
    worker._call_with_retry = AsyncMock(return_value=_openai_submit_child_response())

    result = await worker.run(_request(), _ctx(evaluator=_FailingEvaluator()))

    assert result.child_code == ""
    assert result.error is not None
    assert result.error.kind == "TOOL_ERROR"
    tool_blocks = result.turns[-1].blocks
    assert tool_blocks[0].tool_result_is_error is True
    assert "accepted" not in str(tool_blocks[0].tool_result_content)


@pytest.mark.asyncio
@patch("reps.workers.anthropic_tool_runner.anthropic.AsyncAnthropic")
async def test_anthropic_tool_runner_reevaluation_error_is_not_accepted(mock_client):
    cfg = WorkerConfig(
        name="anthropic-tools",
        impl="anthropic_tool_runner",
        model_id="claude-sonnet-4.6",
        max_turns=1,
        tools=["submit_child"],
    )
    worker = AnthropicToolRunnerWorker(cfg)
    worker._call_with_retry = AsyncMock(return_value=_anthropic_submit_child_response())

    result = await worker.run(_request(), _ctx(evaluator=_FailingEvaluator()))

    assert result.child_code == ""
    assert result.error is not None
    assert result.error.kind == "TOOL_ERROR"
    tool_blocks = result.turns[-1].blocks
    assert tool_blocks[0].tool_result_is_error is True
    assert "accepted" not in str(tool_blocks[0].tool_result_content)


@pytest.mark.parametrize("worker_cls", [OpenAIToolRunnerWorker, AnthropicToolRunnerWorker])
def test_explicit_tool_runner_template_miss_raises(worker_cls):
    cfg = WorkerConfig(
        name="tools",
        impl="tool_runner",
        system_prompt_template="definitely_missing_template",
    )
    with patch("reps.workers.openai_tool_runner.AsyncOpenAI"), patch(
        "reps.workers.anthropic_tool_runner.anthropic.AsyncAnthropic"
    ):
        worker = worker_cls(cfg)

        with pytest.raises(FileNotFoundError, match="definitely_missing_template"):
            worker._build_initial_prompt(_request(), _ctx())
