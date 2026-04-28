"""OpenAIToolRunnerWorker — native OpenAI Responses API tool-use loop.

Parallel to AnthropicToolRunnerWorker but Responses-API-native:
  - Streaming via client.responses.stream(...).get_final_response()
  - Multi-turn chained via previous_response_id (no full-history resend)
  - Tool results returned as function_call_output input items
  - reasoning={effort: "high"} for reasoning effort control
  - max_output_tokens for per-response ceiling

Deliberately duplicates some loop shape from the Anthropic worker rather than
sharing a base class — each provider's protocol is different enough that
abstraction now would be premature.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from reps.program_summarizer import format_summary_for_prompt
from reps.prompt_sampler import build_budget_block, build_siblings_block
from reps.workers.base import (
    apply_template_variations,
    ContentBlock,
    TurnRecord,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers._runner_common import (
    compute_applied_edit,
    reject_placeholder_submission,
    strip_full_rewrite_tail,
)
from reps.workers.edit_serializer import serialize_diff_blocks
from reps.workers.registry import register
from reps.workers.tools import (
    _format_metrics_precise,
    build_tool_impls,
    build_tool_schemas,
)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"


def _anthropic_schema_to_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a tool schema from Anthropic shape (input_schema) to OpenAI
    Responses function-tool shape (parameters)."""
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema.get("description", ""),
        "parameters": schema.get("input_schema") or {
            "type": "object", "properties": {}, "required": [],
        },
    }


@register("openai_tool_runner")
class OpenAIToolRunnerWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        api_key = config.impl_options.get("api_key") or os.environ.get("OPENAI_API_KEY")
        timeout = float(config.impl_options.get("timeout", 900.0))
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.retries = int(config.impl_options.get("retries", 3))
        self.retry_base_delay = float(config.impl_options.get("retry_base_delay", 1.0))
        # Wall-clock ceiling for a single responses.stream() call.
        self.wall_clock_timeout = float(config.impl_options.get("wall_clock_timeout", 1800.0))
        self.max_tokens = int(config.impl_options.get("max_tokens", 16000))
        # reasoning effort: "low" | "medium" | "high" (OpenAI canonical values).
        self.reasoning_effort: Optional[str] = config.impl_options.get("reasoning_effort") or None

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "OpenAIToolRunnerWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        model = self.config.model_id or request.model_id or ""

        system_prompt, user_prompt = self._build_initial_prompt(request, ctx)

        anth_schemas = build_tool_schemas(ctx, self.config.tools)
        tool_schemas = [_anthropic_schema_to_openai(s) for s in anth_schemas]

        child_code_holder = [request.parent.code]
        edit_accumulator: List[Tuple[str, str]] = []
        tool_impls = build_tool_impls(
            request, ctx, self.config.tools, edit_accumulator, child_code_holder
        )

        # TurnRecord tracks the user-facing trace (no full-history resend needed
        # in-call since we use previous_response_id).
        turns: List[TurnRecord] = [
            TurnRecord(
                index=0,
                role="user",
                blocks=[ContentBlock(type="text", text=user_prompt)],
                worker_type="openai_tool_runner",
                started_at_ms=0,
            )
        ]
        usage_total: Dict[str, int] = {}

        submitted_code: Optional[str] = None
        submitted_desc: Optional[str] = None
        converged_flag: bool = False
        converged_reason: Optional[str] = None

        previous_response_id: Optional[str] = None
        # `current_input` is what we hand to responses.stream for the next call.
        # First call: string user prompt. Subsequent calls: list of
        # function_call_output items.
        current_input: Any = user_prompt
        # instructions (system) are only used on the first call; subsequent calls
        # chain via previous_response_id which carries them.
        current_instructions: Optional[str] = system_prompt

        for turn_idx in range(self.config.max_turns):
            try:
                response = await self._call_with_retry(
                    model=model,
                    instructions=current_instructions,
                    input_payload=current_input,
                    tools=tool_schemas,
                    previous_response_id=previous_response_id,
                )
            except WorkerError as we:
                return self._fail(we, turns, usage_total, t0)

            self._accumulate_usage(usage_total, response.usage)

            # Build a single assistant TurnRecord combining reasoning + text +
            # function_call items for trace fidelity.
            assistant_blocks: List[ContentBlock] = []
            function_calls: List[Any] = []
            for item in response.output or []:
                itype = getattr(item, "type", None)
                if itype == "reasoning":
                    # `summary` is populated when the request sets
                    # reasoning.summary = "auto" (or "detailed"). This is
                    # the text content the per-program summarizer reads.
                    summary = getattr(item, "summary", None) or []
                    text_parts = []
                    for s in summary:
                        stext = getattr(s, "text", None)
                        if stext:
                            text_parts.append(stext)
                    assistant_blocks.append(ContentBlock(
                        type="thinking",
                        text="\n".join(text_parts) if text_parts else None,
                    ))
                elif itype == "message":
                    for block in getattr(item, "content", []) or []:
                        btype = getattr(block, "type", None)
                        if btype == "output_text":
                            assistant_blocks.append(ContentBlock(
                                type="text",
                                text=getattr(block, "text", "") or "",
                            ))
                elif itype == "function_call":
                    function_calls.append(item)
                    # Parse args eagerly for the trace; keep raw for dispatch.
                    try:
                        parsed = json.loads(getattr(item, "arguments", "") or "{}")
                    except Exception:
                        parsed = {"_raw": getattr(item, "arguments", "")}
                    assistant_blocks.append(ContentBlock(
                        type="tool_use",
                        tool_use_id=getattr(item, "call_id", None),
                        tool_name=getattr(item, "name", None),
                        tool_input=parsed if isinstance(parsed, dict) else {"_raw": parsed},
                    ))

            turns.append(TurnRecord(
                index=len(turns),
                role="assistant",
                blocks=assistant_blocks,
                model_id=model,
                usage=self._snapshot_usage(response.usage),
                stop_reason=self._derive_stop_reason(response),
                worker_type="openai_tool_runner",
            ))

            # Status dispatch.
            status = getattr(response, "status", None)
            if status == "incomplete":
                incomplete = getattr(response, "incomplete_details", None)
                reason = getattr(incomplete, "reason", None) if incomplete is not None else None
                if reason == "max_output_tokens":
                    return self._fail(
                        WorkerError(kind="INTERNAL", detail="max_output_tokens hit"),
                        turns, usage_total, t0,
                    )
                if reason == "content_filter":
                    return self._fail(
                        WorkerError(kind="REFUSED", detail="content_filter"),
                        turns, usage_total, t0,
                    )
                return self._fail(
                    WorkerError(kind="INTERNAL", detail=f"incomplete: {reason}"),
                    turns, usage_total, t0,
                )
            if status not in ("completed", None):
                return self._fail(
                    WorkerError(kind="INTERNAL", detail=f"unexpected status={status}"),
                    turns, usage_total, t0,
                )

            # Dispatch tool calls (if any).
            if not function_calls:
                # Model ended without a tool call. The only valid terminator in
                # this worker is submit_child — treat bare end as a parse error.
                return self._fail(
                    WorkerError(kind="PARSE_ERROR", detail="end without submit_child"),
                    turns, usage_total, t0,
                )

            tool_outputs: List[Dict[str, Any]] = []
            tool_result_turn_blocks: List[ContentBlock] = []
            terminated = False
            for call in function_calls:
                name = getattr(call, "name", None)
                call_id = getattr(call, "call_id", None)
                raw_args = getattr(call, "arguments", "") or "{}"
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}

                if name == "submit_child":
                    code = args.get("code", "")
                    desc = args.get("changes_description", "")
                    # Reject empty / placeholder / suspiciously short payloads
                    # BEFORE attempting to compile. Parallel to the Anthropic
                    # tool-runner safeguard — prevents clobbering the parent
                    # with sentinel strings ("...", "TODO", placeholder tokens).
                    err_text = reject_placeholder_submission(code)
                    if err_text is not None:
                        tool_outputs.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": err_text,
                        })
                        tool_result_turn_blocks.append(ContentBlock(
                            type="tool_result",
                            tool_use_id=call_id,
                            tool_result_for_id=call_id,
                            tool_result_content=err_text,
                            tool_result_is_error=True,
                        ))
                        # Do NOT terminate — let the model retry.
                        continue
                    try:
                        compile(code, "<submitted>", "exec")
                    except SyntaxError as se:
                        err_text = f"SyntaxError: {se}"
                        tool_outputs.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": err_text,
                        })
                        tool_result_turn_blocks.append(ContentBlock(
                            type="tool_result",
                            tool_use_id=call_id,
                            tool_result_for_id=call_id,
                            tool_result_content=err_text,
                            tool_result_is_error=True,
                        ))
                        continue
                    submitted_code = code
                    submitted_desc = desc
                    # Auto-reevaluate the submitted program synchronously so
                    # the model sees what it actually committed to. Does not
                    # reject on low scores — just surfaces the metrics.
                    accept_text = "accepted"
                    if ctx.evaluator is not None:
                        try:
                            scratch_id = ctx.scratch_id_factory()
                            outcome = await ctx.evaluator.evaluate_isolated(
                                code, program_id=scratch_id, scratch=True
                            )
                            metrics_precise = _format_metrics_precise(outcome.metrics)
                            accept_text = (
                                "accepted\n\n## Submitted program auto-reevaluation\n"
                                + json.dumps(metrics_precise, sort_keys=True)
                            )
                        except Exception as re_exc:
                            accept_text = (
                                "accepted\n\n## Submitted program auto-reevaluation\n"
                                f"ERROR: {type(re_exc).__name__}: {re_exc}"
                            )
                    else:
                        accept_text = (
                            "accepted\n\n## Submitted program auto-reevaluation\n"
                            "SKIPPED: no evaluator attached (uses_evaluator=False)"
                        )
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": accept_text,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=call_id,
                        tool_result_for_id=call_id,
                        tool_result_content=accept_text,
                        tool_result_is_error=False,
                    ))
                    terminated = True
                elif name == "mark_converged":
                    reason = args.get("reason", "") if isinstance(args, dict) else ""
                    converged_flag = True
                    converged_reason = reason
                    ack = "acknowledged: converged"
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": ack,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=call_id,
                        tool_result_for_id=call_id,
                        tool_result_content=ack,
                        tool_result_is_error=False,
                    ))
                    terminated = True
                else:
                    impl = tool_impls.get(name)
                    if impl is None:
                        out = f"ERROR: unknown tool '{name}'"
                        is_err = True
                    else:
                        try:
                            out = await impl(args)
                            is_err = False
                        except Exception as e:
                            out = f"{type(e).__name__}: {e}"
                            is_err = True
                    out_str = out if isinstance(out, str) else str(out)
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out_str,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=call_id,
                        tool_result_for_id=call_id,
                        tool_result_content=out_str,
                        tool_result_is_error=is_err,
                    ))

            turns.append(TurnRecord(
                index=len(turns),
                role="tool",
                blocks=tool_result_turn_blocks,
                worker_type="openai_tool_runner",
            ))

            if terminated and converged_flag:
                # Convergence short-circuit — see anthropic_tool_runner for
                # the controller-side contract. No child is returned.
                return WorkerResult(
                    child_code="",
                    turns=turns,
                    turn_count=len(turns),
                    usage=usage_total,
                    wall_clock_seconds=time.monotonic() - t0,
                    error=None,
                    converged=True,
                    converged_reason=converged_reason,
                )

            if terminated and submitted_code is not None:
                if edit_accumulator:
                    applied = serialize_diff_blocks(edit_accumulator)
                else:
                    applied = compute_applied_edit(
                        submitted_code,
                        request.parent.code,
                        request.parent.id,
                        self.config.generation_mode,
                    )
                return WorkerResult(
                    child_code=submitted_code,
                    changes_description=submitted_desc,
                    changes_summary=(submitted_desc or "")[:120],
                    applied_edit=applied,
                    turns=turns,
                    turn_count=len(turns),
                    usage=usage_total,
                    wall_clock_seconds=time.monotonic() - t0,
                    error=None,
                )

            # Prepare next turn: chain via previous_response_id, send tool outputs.
            previous_response_id = response.id
            current_input = tool_outputs
            current_instructions = None  # instructions are anchored on the original response
            # future: inject fresh budget block on each loop turn
            # The initial user prompt has "Turn: 1 / max_turns" (built in
            # _build_initial_prompt via build_budget_block). Later turns
            # don't refresh it, so the model can't see live Turn N /
            # max_turns or cumulative tokens out. A targeted extension
            # would prepend a fresh build_budget_block(current_turn=turn_idx+1,
            # max_turns=self.config.max_turns, cumulative_out=<sum from
            # usage_total>) as a text input item alongside tool_outputs.
            # Deferred — too invasive for the current PROMPT-team scope.

        # Turn budget exhausted without submit_child or mark_converged.
        # Auto-eject: synthesize a mark_converged result so the iteration
        # closes cleanly. (Parallel to anthropic_tool_runner's fix.)
        return WorkerResult(
            child_code="",
            turns=turns,
            turn_count=len(turns),
            usage=usage_total,
            wall_clock_seconds=time.monotonic() - t0,
            error=None,
            converged=True,
            converged_reason=(
                f"auto-ejected: turn budget {self.config.max_turns} "
                "exhausted without submit_child or mark_converged"
            ),
        )

    # -------------------------------------------- helpers
    def _build_initial_prompt(
        self, request: WorkerRequest, ctx: WorkerContext
    ) -> Tuple[str, str]:
        template_key = self.config.system_prompt_template or "system_message_tool_runner"
        sampler = ctx.prompt_sampler
        prompt = sampler.build_prompt(
            current_program=request.parent.code,
            parent_program=request.parent.code,
            program_metrics=request.parent.metrics,
            previous_programs=[p.to_dict() for p in request.recent_iterations],
            top_programs=[p.to_dict() for p in request.top_programs],
            inspirations=[p.to_dict() for p in request.inspirations],
            language=request.language,
            evolution_round=request.iteration,
            diff_based_evolution=False,
            program_artifacts=request.parent_artifacts,
            feature_dimensions=request.feature_dimensions,
            current_changes_description=None,
            **request.prompt_extras,
        )
        try:
            system_text = (_TEMPLATE_DIR / f"{template_key}.txt").read_text()
        except Exception:
            system_text = prompt.get("system", "")

        bc = (self.config.baseline_context or "").strip()
        bc_block = f"\n{bc}\n" if bc else ""
        if "{baseline_context}" in system_text:
            system_text = system_text.replace("{baseline_context}", bc_block)
        elif bc_block:
            system_text = system_text.rstrip() + "\n" + bc_block

        # Apply per-worker template_variations (role_directive, etc.),
        # seeded by iteration for reproducibility. Mirrors the Anthropic
        # tool-runner wiring.
        system_text = apply_template_variations(
            system_text, self.config.template_variations, request.iteration
        )

        user_text = prompt.get("user", "")
        user_text = apply_template_variations(
            user_text, self.config.template_variations, request.iteration
        )

        # Drop the sampler's full-rewrite task framing; the tool-runner uses
        # edit_file. See strip_full_rewrite_tail docstring for rationale.
        user_text = strip_full_rewrite_tail(user_text)

        # Append un-consumed prompt_extras (mirrors single_call.py:80-84).
        # Skip when the value is already substring-present in user_text
        # (sampler may have rendered it via a template slot). For multi-line
        # values (unexplored_directions / recent_avoids ship as
        # "## Header\n- ..."), a raw substring check can drop the block when
        # whitespace differs — probe by first non-whitespace line instead.
        for key, value in (request.prompt_extras or {}).items():
            if not isinstance(value, str):
                continue
            v = value.strip()
            if not v:
                continue
            probe = v.splitlines()[0].strip()
            if probe and probe in user_text:
                continue
            if v.lstrip().startswith("## "):
                # Already-formatted block — inject verbatim, don't add a
                # derived "## Title" header.
                user_text = user_text + "\n\n" + v + "\n"
            else:
                title = key.replace("_", " ").title()
                user_text = user_text + f"\n\n## {title}\n{value}\n"

        # Append a '## Siblings you can view_program' bullet list so the
        # model sees short 8-char ids + one-liners alongside the tool
        # description. Mirrors AnthropicToolRunnerWorker._build_initial_prompt.
        top_dicts = [p.to_dict() for p in request.top_programs]
        insp_dicts = [p.to_dict() for p in request.inspirations]
        siblings_block = build_siblings_block(top_dicts, insp_dicts, limit=8)
        if siblings_block:
            user_text = user_text.rstrip() + "\n\n" + siblings_block + "\n"

        parent_summary = (
            request.parent.metadata.get("reps_annotations", {}).get("summary")
            if request.parent and request.parent.metadata
            else None
        )
        if parent_summary:
            insights_text = format_summary_for_prompt(parent_summary, label="Parent's notebook")
            user_text = insights_text + "\n\n" + user_text

        # Prepend iteration-budget block at the very top (above the parent's
        # notebook). Displays Turn: 1 / max_turns for the initial prompt;
        # live per-turn updates are deferred — see the
        # "# future: inject fresh budget block on each loop turn" comment
        # in run() near the previous_response_id chain logic.
        budget_block = build_budget_block(current_turn=1, max_turns=self.config.max_turns)
        user_text = budget_block + "\n\n" + user_text

        return system_text, user_text

    async def _call_with_retry(
        self,
        *,
        model: str,
        instructions: Optional[str],
        input_payload: Any,
        tools: List[Dict[str, Any]],
        previous_response_id: Optional[str],
    ):
        params: Dict[str, Any] = {
            "model": model,
            "input": input_payload,
            "tools": tools,
            "max_output_tokens": self.max_tokens,
        }
        if instructions is not None:
            params["instructions"] = instructions
        if previous_response_id is not None:
            params["previous_response_id"] = previous_response_id
        if self.reasoning_effort:
            # summary="auto" surfaces human-readable reasoning text in
            # the response's reasoning items so it reaches the trace
            # sidecar and the per-program summarizer. Without it, reasoning
            # items are billed but carry no visible content.
            params["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}

        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.wait_for(
                    self._stream_to_final(params),
                    timeout=self.wall_clock_timeout,
                )
            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning(
                    "responses.stream wall-clock timeout after %.0fs (attempt %d/%d)",
                    self.wall_clock_timeout, attempt + 1, self.retries + 1,
                )
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                httpx.TimeoutException,
                httpx.TransportError,
            ) as e:
                last_exc = e
                logger.warning(
                    "responses.stream transport error on attempt %d/%d: %s",
                    attempt + 1, self.retries + 1, type(e).__name__,
                )
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < self.retries:
                    last_exc = e
                    await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                    continue
                raise WorkerError(
                    kind="INTERNAL", detail=f"{e.status_code}: {e.message}"
                ) from e
        kind = (
            "TIMEOUT"
            if isinstance(
                last_exc,
                (APITimeoutError, asyncio.TimeoutError, httpx.TimeoutException),
            )
            else "INTERNAL"
        )
        raise WorkerError(kind=kind, detail=repr(last_exc))

    async def _stream_to_final(self, params: Dict[str, Any]):
        """Open responses.stream(), iterate events for logging, return the
        aggregated final Response."""
        async with self.client.responses.stream(**params) as stream:
            async for event in stream:
                et = getattr(event, "type", None)
                if et == "response.output_item.done":
                    self._log_completed_item(getattr(event, "item", None))
            final = await stream.get_final_response()
        return final

    def _log_completed_item(self, item) -> None:
        if item is None:
            return
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            summary = getattr(item, "summary", None) or []
            logger.info("item: reasoning (%d summary blocks)", len(summary))
        elif itype == "message":
            content = getattr(item, "content", None) or []
            n_text = sum(1 for b in content if getattr(b, "type", None) == "output_text")
            total_chars = sum(
                len(getattr(b, "text", "") or "") for b in content
                if getattr(b, "type", None) == "output_text"
            )
            logger.info("item: message (%d text blocks, %d chars)", n_text, total_chars)
        elif itype == "function_call":
            logger.info(
                "item: function_call name=%s call_id=%s",
                getattr(item, "name", "?"),
                getattr(item, "call_id", "?"),
            )
        else:
            logger.info("item: type=%s", itype)

    @staticmethod
    def _derive_stop_reason(response) -> Optional[str]:
        """Map Responses status → a label compatible with the trace renderer."""
        status = getattr(response, "status", None)
        if status == "completed":
            # If any function_call in output → tool_use-like; else end_turn-like.
            for item in getattr(response, "output", None) or []:
                if getattr(item, "type", None) == "function_call":
                    return "tool_use"
            return "end_turn"
        if status == "incomplete":
            incomplete = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete, "reason", None) if incomplete is not None else None
            if reason == "max_output_tokens":
                return "max_tokens"
            return f"incomplete:{reason}" if reason else "incomplete"
        return status

    def _accumulate_usage(self, total: Dict[str, int], usage) -> None:
        if usage is None:
            return
        for f in ("input_tokens", "output_tokens"):
            v = getattr(usage, f, 0) or 0
            total[f] = total.get(f, 0) + int(v)
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            r = getattr(details, "reasoning_tokens", 0) or 0
            if r:
                total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + int(r)
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details is not None:
            cached = getattr(in_details, "cached_tokens", 0) or 0
            if cached:
                total["cache_read_input_tokens"] = (
                    total.get("cache_read_input_tokens", 0) + int(cached)
                )

    def _snapshot_usage(self, usage) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        out = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            r = getattr(details, "reasoning_tokens", 0) or 0
            out["reasoning_tokens"] = int(r)
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details is not None:
            cached = getattr(in_details, "cached_tokens", 0) or 0
            out["cache_read_input_tokens"] = int(cached)
        return out

    def _fail(self, we: WorkerError, turns, usage_total, t0) -> WorkerResult:
        return WorkerResult(
            child_code="",
            turns=turns,
            turn_count=len(turns),
            usage=usage_total,
            wall_clock_seconds=time.monotonic() - t0,
            error=we,
        )
