"""DSPyReActWorker — DSPy ReAct program over the native Anthropic provider.

DSPy is sync; we invoke it via asyncio.to_thread(...). Uses dspy.context(lm=)
for thread-safe LM scoping (NOT dspy.configure — that's process-global and
races under concurrent workers).

IMPORTANT: cache=False on dspy.LM. Default True would collapse evolutionary
diversity by returning stale completions."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import dspy

from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers.registry import register

logger = logging.getLogger(__name__)


def make_dspy_lm(config, worker_config: WorkerConfig) -> dspy.LM:
    """Build a per-invocation dspy.LM via the native Anthropic LiteLLM route."""
    kwargs: Dict[str, Any] = {
        "model": f"anthropic/{worker_config.model_id}",
        "api_key": config.llm.api_key,
        "max_tokens": config.llm.max_tokens,
        "cache": False,
    }
    if worker_config.temperature is not None:
        kwargs["temperature"] = worker_config.temperature
    thinking = worker_config.impl_options.get("thinking")
    if thinking:
        kwargs["thinking"] = thinking
    return dspy.LM(**kwargs)


class EvolveProgramFull(dspy.Signature):
    """Given a parent program and evolutionary context, produce an improved child program."""
    parent_code: str = dspy.InputField(desc="current parent program source")
    language: str = dspy.InputField()
    iteration: int = dspy.InputField()
    inspirations: str = dspy.InputField(desc="formatted inspirations block")
    top_programs: str = dspy.InputField(desc="formatted top-K programs block")
    second_parent_code: str = dspy.InputField(desc="optional crossover parent, or ''")
    feature_dimensions: str = dspy.InputField(desc="comma-separated MAP-Elites dims")
    extras: str = dspy.InputField(desc="reflection/SOTA/dead-end warnings")
    child_code: str = dspy.OutputField(desc="complete rewritten child program")
    changes_description: str = dspy.OutputField(desc="1-3 sentence summary of edits")


class EvolveProgramDiff(EvolveProgramFull):
    """Emit a unified diff against parent_code."""
    child_code: str = dspy.OutputField(desc="unified diff against parent_code")


def _fmt_programs(programs) -> str:
    lines = []
    for p in programs:
        score = p.metrics.get("combined_score", 0.0) if p.metrics else 0.0
        lines.append(f"--- id={p.id} score={score:.4f}\n{p.code}\n")
    return "\n".join(lines)


def _fmt_extras(extras: Dict[str, str]) -> str:
    parts = []
    for k in ("reflection", "sota_injection", "dead_end_warnings"):
        v = extras.get(k, "")
        if v:
            parts.append(f"[{k}]\n{v}")
    return "\n\n".join(parts)


@register("dspy_react")
class DSPyReActWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "DSPyReActWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        lm = ctx.dspy_lm_factory(self.config)

        sig = EvolveProgramDiff if request.generation_mode == "diff" else EvolveProgramFull

        # Build tools as dspy.Tool wrappers. Bridge async ctx.evaluator via
        # asyncio.run_coroutine_threadsafe on a captured main loop (DSPy is
        # sync inside to_thread).
        loop = asyncio.get_running_loop()

        def make_view_parent():
            def view_parent() -> str:
                """Return the parent program's source code."""
                return request.parent.code
            return dspy.Tool(view_parent)

        def make_run_tests():
            if ctx.evaluator is None:
                return None
            def run_tests(code: str) -> str:
                """Evaluate candidate code in isolation; returns metrics JSON string."""
                scratch_id = ctx.scratch_id_factory()
                fut = asyncio.run_coroutine_threadsafe(
                    ctx.evaluator.evaluate_isolated(code, program_id=scratch_id, scratch=True),
                    loop,
                )
                outcome = fut.result(timeout=self.config.expected_wall_clock_s + 60)
                return f"score={outcome.metrics.get('combined_score', 0.0):.4f} metrics={outcome.metrics}"
            return dspy.Tool(run_tests)

        tools = []
        if "view_parent" in self.config.tools:
            tools.append(make_view_parent())
        if "run_tests" in self.config.tools:
            rt = make_run_tests()
            if rt is not None:
                tools.append(rt)

        react = dspy.ReAct(signature=sig, tools=tools, max_iters=self.config.max_turns)

        def _invoke() -> Optional[dspy.Prediction]:
            with dspy.context(lm=lm):
                return react(
                    parent_code=request.parent.code,
                    language=request.language,
                    iteration=request.iteration,
                    inspirations=_fmt_programs(request.inspirations),
                    top_programs=_fmt_programs(request.top_programs),
                    second_parent_code=(request.second_parent.code if request.second_parent else ""),
                    feature_dimensions=",".join(request.feature_dimensions),
                    extras=_fmt_extras(request.prompt_extras),
                )

        error: Optional[WorkerError] = None
        pred: Optional[dspy.Prediction] = None
        try:
            pred = await asyncio.to_thread(_invoke)
        except asyncio.TimeoutError:
            error = WorkerError(kind="TIMEOUT", detail="asyncio timeout")
        except Exception as e:
            msg = str(e).lower()
            if "parse" in msg or "adapter" in msg:
                error = WorkerError(kind="PARSE_ERROR", detail=str(e))
            elif "tool" in msg:
                error = WorkerError(kind="TOOL_ERROR", detail=str(e))
            else:
                error = WorkerError(kind="INTERNAL", detail=str(e))

        turns, turn_count = _extract_turns(pred)
        if pred is not None and turn_count >= self.config.max_turns and not _finished(pred):
            error = WorkerError(kind="MAX_TURNS_HIT", detail=f"max_iters={self.config.max_turns}")

        child_code = getattr(pred, "child_code", "") if pred is not None else ""
        changes_description = getattr(pred, "changes_description", None) if pred is not None else None
        usage = _extract_usage(lm)

        return WorkerResult(
            child_code=child_code,
            changes_description=changes_description,
            changes_summary=(changes_description or "")[:120],
            applied_edit=child_code if request.generation_mode == "full" else (child_code or ""),
            turns=turns,
            turn_count=turn_count,
            usage=usage,
            wall_clock_seconds=time.monotonic() - t0,
            error=error,
        )


def _extract_turns(pred: Optional["dspy.Prediction"]) -> tuple[List[TurnRecord], int]:
    if pred is None or not hasattr(pred, "trajectory"):
        return [], 0
    traj: Dict[str, Any] = pred.trajectory or {}
    turns: List[TurnRecord] = []
    i = 0
    while f"thought_{i}" in traj:
        blocks: List[ContentBlock] = [
            ContentBlock(type="text", text=str(traj.get(f"thought_{i}", "")))
        ]
        tool_name = traj.get(f"tool_name_{i}")
        if tool_name:
            raw_args = traj.get(f"tool_args_{i}")
            tool_input = dict(raw_args) if isinstance(raw_args, dict) else {"args": raw_args}
            blocks.append(ContentBlock(
                type="tool_use",
                tool_use_id=f"dspy_{i}",
                tool_name=str(tool_name),
                tool_input=tool_input,
            ))
        turns.append(TurnRecord(
            index=len(turns),
            role="assistant",
            blocks=blocks,
            worker_type="dspy_react",
            impl_specific={"dspy_trace": dict(traj)} if i == 0 else {},
        ))
        obs = traj.get(f"observation_{i}")
        if obs is not None:
            turns.append(TurnRecord(
                index=len(turns),
                role="tool",
                blocks=[ContentBlock(
                    type="tool_result",
                    tool_use_id=f"dspy_{i}",
                    tool_result_for_id=f"dspy_{i}",
                    tool_result_content=str(obs),
                    tool_result_is_error=False,
                )],
                worker_type="dspy_react",
            ))
        i += 1
    return turns, i


def _finished(pred) -> bool:
    traj = getattr(pred, "trajectory", {}) or {}
    return any(v == "finish" for k, v in traj.items() if isinstance(k, str) and k.startswith("tool_name_"))


def _extract_usage(lm) -> Dict[str, int]:
    tin = tout = 0
    for h in getattr(lm, "history", []) or []:
        u = h.get("usage") or {}
        tin += int(u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0)
        tout += int(u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0)
    return {"input_tokens": tin, "output_tokens": tout, "calls": len(getattr(lm, "history", []) or [])}
