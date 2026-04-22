"""AnthropicToolRunnerWorker — native Anthropic tool-use loop.

Bypasses LLMInterface because we need raw content blocks (thinking, tool_use,
tool_result) and native semantics. Reuses one AsyncAnthropic client per worker
instance for connection pooling across iterations."""
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import httpx
from anthropic import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from reps.llm.anthropic import REASONING_MODEL_PATTERNS
from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers.edit_serializer import serialize_diff_blocks
from reps.workers.registry import register
from reps.workers.tools import build_tool_impls, build_tool_schemas

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"


@register("anthropic_tool_runner")
class AnthropicToolRunnerWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        api_key = config.impl_options.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        # httpx per-chunk read timeout. Opus 4.7 with thinking_effort=high +
        # code_execution can go ~5min between chunks during silent computation;
        # bump to 900s to avoid false positives.
        timeout = float(config.impl_options.get("timeout", 900.0))
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.retries = int(config.impl_options.get("retries", 3))
        self.retry_base_delay = float(config.impl_options.get("retry_base_delay", 1.0))
        # Hard wall-clock ceiling per API call. Guards against the httpx
        # per-chunk timeout (which resets every byte) — a stalled stream
        # would otherwise block forever.
        self.wall_clock_timeout = float(config.impl_options.get("wall_clock_timeout", 1800.0))
        # Opus 4.7 coding/agentic sweet spot is "xhigh" (per claude-api skill).
        self.thinking_effort = str(config.impl_options.get("thinking_effort", "xhigh"))
        # Opus 4.7 omits thinking content by default; set "summarized" to see reasoning text.
        self.thinking_display = str(config.impl_options.get("thinking_display", "summarized"))
        self.max_tokens = int(config.impl_options.get("max_tokens", 16000))
        # Opt-in code execution server tool → enables programmatic tool calling.
        self.use_code_execution = bool(config.impl_options.get("code_execution", False))

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "AnthropicToolRunnerWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        model = self.config.model_id or request.model_id or ""
        is_reasoning = any(p in model.lower() for p in REASONING_MODEL_PATTERNS)

        # Build the initial user message via PromptSampler using the tool-runner template.
        system_prompt, user_prompt = self._build_initial_prompt(request, ctx)

        tool_schemas = build_tool_schemas(ctx, self.config.tools)
        if self.use_code_execution:
            # Server-side code execution tool. Also enables programmatic tool calling
            # for any custom tool that declares `allowed_callers: ["code_execution_20260120"]`.
            tool_schemas.insert(0, {"type": "code_execution_20260120", "name": "code_execution"})
        container_id: Optional[str] = None  # reused across turns for state persistence

        # In-flight child code starts as parent; edit_file mutates it.
        child_code_holder = [request.parent.code]
        edit_accumulator: List[Tuple[str, str]] = []
        tool_impls = build_tool_impls(
            request, ctx, self.config.tools, edit_accumulator, child_code_holder
        )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        turns: List[TurnRecord] = [
            TurnRecord(
                index=0,
                role="user",
                blocks=[ContentBlock(type="text", text=user_prompt)],
                worker_type="anthropic_tool_runner",
                started_at_ms=0,
            )
        ]
        usage_total: Dict[str, int] = {}

        submitted_code: Optional[str] = None
        submitted_desc: Optional[str] = None

        for turn_idx in range(self.config.max_turns):
            try:
                response = await self._call_with_retry(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tool_schemas,
                    is_reasoning=is_reasoning,
                    container=container_id,
                )
            except WorkerError as we:
                return self._fail(we, turns, usage_total, t0)

            # Capture container id for reuse across turns (code-execution sessions).
            # _stream_to_final stashes it on the Message when the streaming path
            # captures it from message_start; also check the standard `container`
            # field in case the non-streaming shape is present.
            new_cid = (
                getattr(response, "_captured_container_id", None)
                or self._extract_container_id(response)
            )
            if new_cid and new_cid != container_id:
                logger.info("captured container_id=%s", new_cid)
                container_id = new_cid

            self._accumulate_usage(usage_total, response.usage)

            assistant_blocks_for_turn: List[ContentBlock] = []
            assistant_blocks_for_message: List[Dict[str, Any]] = []
            for raw in response.content:
                block_dict = self._raw_block_to_message_dict(raw)
                assistant_blocks_for_message.append(block_dict)
                assistant_blocks_for_turn.append(self._raw_block_to_content_block(raw))

            turns.append(TurnRecord(
                index=len(turns),
                role="assistant",
                blocks=assistant_blocks_for_turn,
                model_id=model,
                usage=self._snapshot_usage(response.usage),
                stop_reason=response.stop_reason,
                worker_type="anthropic_tool_runner",
            ))

            stop_reason = response.stop_reason

            if stop_reason == "refusal":
                return self._fail(
                    WorkerError(kind="REFUSED", detail="model refused"),
                    turns, usage_total, t0,
                )

            if stop_reason == "model_context_window_exceeded":
                return self._fail(
                    WorkerError(kind="INTERNAL", detail="context window exhausted"),
                    turns, usage_total, t0,
                )

            if stop_reason == "max_tokens":
                # If the tail block is an incomplete tool_use, the model was mid-call.
                last = response.content[-1] if response.content else None
                incomplete_tool = last is not None and getattr(last, "type", None) == "tool_use"
                detail = "max_tokens hit mid-tool_use" if incomplete_tool else "max_tokens hit"
                return self._fail(
                    WorkerError(kind="INTERNAL", detail=detail),
                    turns, usage_total, t0,
                )

            if stop_reason == "pause_turn":
                # Server-side loop (code_execution / web search) reached its iteration cap.
                # Re-send the conversation as-is to resume — no tool dispatch required.
                messages.append({"role": "assistant", "content": assistant_blocks_for_message})
                continue

            if stop_reason == "stop_sequence":
                return self._fail(
                    WorkerError(kind="PARSE_ERROR", detail="unexpected stop_sequence"),
                    turns, usage_total, t0,
                )

            if stop_reason == "end_turn":
                # Defensive: if submit_child was called (e.g. from inside a
                # code_execution block) and the final stop_reason wraps up as
                # end_turn, let the tool_use dispatch path below process it.
                has_submit = any(
                    getattr(raw, "type", None) == "tool_use" and raw.name == "submit_child"
                    for raw in response.content
                )
                if not has_submit:
                    return self._fail(
                        WorkerError(kind="PARSE_ERROR", detail="end_turn without submit_child"),
                        turns, usage_total, t0,
                    )
                # Fall through into the tool_use dispatch block below.

            if stop_reason != "tool_use":
                return self._fail(
                    WorkerError(kind="PARSE_ERROR", detail=f"unexpected stop_reason={stop_reason}"),
                    turns, usage_total, t0,
                )

            # Echo assistant turn verbatim (thinking blocks + signatures preserved).
            messages.append({"role": "assistant", "content": assistant_blocks_for_message})

            # Dispatch every tool_use block in this turn.
            tool_result_blocks: List[Dict[str, Any]] = []
            tool_result_turn_blocks: List[ContentBlock] = []
            terminated = False
            for raw in response.content:
                if getattr(raw, "type", None) != "tool_use":
                    continue
                name = raw.name
                tid = raw.id
                args = raw.input or {}

                if name == "submit_child":
                    code = args.get("code", "")
                    desc = args.get("changes_description", "")
                    try:
                        compile(code, "<submitted>", "exec")
                    except SyntaxError as se:
                        err_text = f"SyntaxError: {se}"
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": err_text,
                            "is_error": True,
                        })
                        tool_result_turn_blocks.append(ContentBlock(
                            type="tool_result",
                            tool_use_id=tid,
                            tool_result_for_id=tid,
                            tool_result_content=err_text,
                            tool_result_is_error=True,
                        ))
                        continue
                    submitted_code = code
                    submitted_desc = desc
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "accepted",
                        "is_error": False,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=tid,
                        tool_result_for_id=tid,
                        tool_result_content="accepted",
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
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": out if isinstance(out, str) else str(out),
                        "is_error": is_err,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=tid,
                        tool_result_for_id=tid,
                        tool_result_content=out if isinstance(out, str) else str(out),
                        tool_result_is_error=is_err,
                    ))

            turns.append(TurnRecord(
                index=len(turns),
                role="tool",
                blocks=tool_result_turn_blocks,
                worker_type="anthropic_tool_runner",
            ))
            messages.append({"role": "user", "content": tool_result_blocks})

            if terminated and submitted_code is not None:
                # Compute applied_edit per D1 hybrid rule.
                if edit_accumulator:
                    applied = serialize_diff_blocks(edit_accumulator)
                else:
                    applied = self._compute_applied_edit(submitted_code, request)
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

        # Turn budget exhausted without submit_child. Give the model one final
        # nudge to ship whatever it has before we fail.
        nudge_text = (
            f"You have reached the turn limit (max_turns={self.config.max_turns}). "
            "Your next and final response MUST call submit_child with your best "
            "working program so far — whatever you have. Do not call any other "
            "tools. Do not do any more editing or testing. Submit now."
        )
        messages.append({"role": "user", "content": [{"type": "text", "text": nudge_text}]})
        turns.append(TurnRecord(
            index=len(turns),
            role="user",
            blocks=[ContentBlock(type="text", text=nudge_text)],
            worker_type="anthropic_tool_runner",
        ))
        try:
            final_resp = await self._call_with_retry(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                tools=tool_schemas,
                is_reasoning=is_reasoning,
                container=container_id,
            )
        except WorkerError as we:
            return self._fail(we, turns, usage_total, t0)

        self._accumulate_usage(usage_total, final_resp.usage)
        final_blocks_turn = [self._raw_block_to_content_block(r) for r in final_resp.content]
        turns.append(TurnRecord(
            index=len(turns),
            role="assistant",
            blocks=final_blocks_turn,
            model_id=model,
            usage=self._snapshot_usage(final_resp.usage),
            stop_reason=final_resp.stop_reason,
            worker_type="anthropic_tool_runner",
        ))
        # Extract submit_child call from the final response, if present.
        for raw in final_resp.content:
            if getattr(raw, "type", None) == "tool_use" and raw.name == "submit_child":
                code = (raw.input or {}).get("code", "")
                desc = (raw.input or {}).get("changes_description", "")
                try:
                    compile(code, "<submitted>", "exec")
                except SyntaxError as se:
                    return self._fail(
                        WorkerError(kind="PARSE_ERROR", detail=f"final submit SyntaxError: {se}"),
                        turns, usage_total, t0,
                    )
                applied = (
                    serialize_diff_blocks(edit_accumulator)
                    if edit_accumulator
                    else self._compute_applied_edit(code, request)
                )
                return WorkerResult(
                    child_code=code,
                    changes_description=desc,
                    changes_summary=(desc or "")[:120],
                    applied_edit=applied,
                    turns=turns,
                    turn_count=len(turns),
                    usage=usage_total,
                    wall_clock_seconds=time.monotonic() - t0,
                    error=None,
                )
        # Final nudge still didn't yield a submit_child.
        return self._fail(
            WorkerError(kind="MAX_TURNS_HIT", detail=f"max_turns={self.config.max_turns} + final nudge ignored"),
            turns, usage_total, t0,
        )

    # -------------------------------------------- helpers
    def _build_initial_prompt(
        self, request: WorkerRequest, ctx: WorkerContext
    ) -> Tuple[str, str]:
        template_key = self.config.system_prompt_template or "system_message_tool_runner"
        sampler = ctx.prompt_sampler
        # Use PromptSampler's existing build_prompt for user-message body, then
        # override the system message with the tool-runner template.
        prompt = sampler.build_prompt(
            current_program=request.parent.code,
            parent_program=request.parent.code,
            program_metrics=request.parent.metrics,
            previous_programs=[p.to_dict() for p in request.top_programs],
            top_programs=[p.to_dict() for p in (request.top_programs + request.inspirations)],
            inspirations=[p.to_dict() for p in request.inspirations],
            language=request.language,
            evolution_round=request.iteration,
            diff_based_evolution=False,  # tool runner handles its own edit mechanics
            program_artifacts=None,
            feature_dimensions=request.feature_dimensions,
            current_changes_description=None,
            **request.prompt_extras,
        )
        # Try to load the tool-runner template from the prompt_templates dir;
        # PromptSampler exposes no public `load_template`, so we read directly.
        try:
            system_text = (_TEMPLATE_DIR / f"{template_key}.txt").read_text()
        except Exception:
            system_text = prompt.get("system", "")
        return system_text, prompt.get("user", "")

    async def _call_with_retry(self, *, model, system_prompt, messages, tools, is_reasoning, container=None):
        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
        }
        if container is not None:
            params["container"] = container
        if is_reasoning:
            # Opus 4.7+ uses adaptive thinking + output_config.effort (not budget_tokens).
            # display="summarized" restores reasoning text (Opus 4.7 defaults to omitted).
            params["thinking"] = {"type": "adaptive", "display": self.thinking_display}
            params["output_config"] = {"effort": self.thinking_effort}
        else:
            if self.config.temperature is not None:
                params["temperature"] = self.config.temperature

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
                    "messages.stream wall-clock timeout after %.0fs (attempt %d/%d)",
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
                    "messages.stream transport error on attempt %d/%d: %s",
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
        """Open a streaming messages.create, log each content block as it
        completes, then return the aggregated final Message (same shape as a
        non-streaming .create()). The SDK internally accumulates deltas into
        blocks; we tap content_block_stop events for block-level visibility
        without handling raw chunks ourselves.

        Also captures the code-execution container id robustly: the final
        Message may not carry `container` in the streaming path, but the
        `message_start` event does. We stash it on the returned message for
        the caller to read.
        """
        container_id_seen: Optional[str] = None
        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                et = getattr(event, "type", None)
                if et == "content_block_stop":
                    self._log_completed_block(getattr(event, "content_block", None))
                elif et == "message_start":
                    msg = getattr(event, "message", None)
                    cid = self._extract_container_id(msg)
                    if cid:
                        container_id_seen = cid
                elif et == "message_delta":
                    # Container id lands here in the streaming path — the
                    # SDK does NOT aggregate it onto the final Message.
                    delta = getattr(event, "delta", None)
                    cid = self._extract_container_id(delta)
                    if cid:
                        container_id_seen = cid
            final = await stream.get_final_message()
        # Prefer container id from final Message if present; fall back to what
        # we saw on message_start.
        cid_final = self._extract_container_id(final)
        resolved = cid_final or container_id_seen
        if resolved is not None:
            try:
                object.__setattr__(final, "_captured_container_id", resolved)
            except Exception:
                pass
        return final

    @staticmethod
    def _extract_container_id(obj) -> Optional[str]:
        if obj is None:
            return None
        c = getattr(obj, "container", None)
        if c is None and isinstance(obj, dict):
            c = obj.get("container")
        if c is None:
            return None
        cid = getattr(c, "id", None)
        if not cid and isinstance(c, dict):
            cid = c.get("id")
        return cid

    def _log_completed_block(self, block) -> None:
        if block is None:
            return
        bt = getattr(block, "type", None)
        if bt == "text":
            txt = getattr(block, "text", "") or ""
            logger.info("block: text (%d chars)", len(txt))
        elif bt == "thinking":
            txt = getattr(block, "thinking", "") or ""
            signed = bool(getattr(block, "signature", None))
            logger.info("block: thinking %s (%d chars)", "signed" if signed else "unsigned", len(txt))
        elif bt == "redacted_thinking":
            logger.info("block: redacted_thinking")
        elif bt == "tool_use":
            name = getattr(block, "name", "?")
            caller = getattr(block, "caller", None)
            ctype = getattr(caller, "type", None) if caller is not None else None
            logger.info("block: tool_use name=%s caller=%s", name, ctype or "direct")
        elif bt == "server_tool_use":
            logger.info("block: server_tool_use name=%s", getattr(block, "name", "?"))
        elif bt == "code_execution_tool_result":
            logger.info("block: code_execution_tool_result")
        else:
            logger.info("block: type=%s", bt)

    def _raw_block_to_message_dict(self, raw) -> Dict[str, Any]:
        """Canonicalize a response content block into the dict shape Anthropic
        expects echoed back. Preserves thinking signatures verbatim."""
        t = getattr(raw, "type", None)
        if t == "text":
            return {"type": "text", "text": raw.text}
        if t == "thinking":
            d = {"type": "thinking", "thinking": raw.thinking}
            sig = getattr(raw, "signature", None)
            if sig is not None:
                d["signature"] = sig
            return d
        if t == "redacted_thinking":
            return {"type": "redacted_thinking", "data": raw.data}
        if t == "tool_use":
            d: Dict[str, Any] = {"type": "tool_use", "id": raw.id, "name": raw.name, "input": raw.input}
            # Preserve caller field (programmatic tool calling) for echo-back.
            caller = getattr(raw, "caller", None)
            if caller is not None:
                d["caller"] = caller.model_dump() if hasattr(caller, "model_dump") else dict(caller)
            return d
        if t == "server_tool_use":
            return {"type": "server_tool_use", "id": raw.id, "name": raw.name, "input": raw.input}
        if t == "code_execution_tool_result":
            if hasattr(raw, "model_dump"):
                return raw.model_dump()
            return {"type": "code_execution_tool_result", "tool_use_id": getattr(raw, "tool_use_id", None)}
        # fallback: best-effort dict
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        return {"type": t or "unknown"}

    def _raw_block_to_content_block(self, raw) -> ContentBlock:
        t = getattr(raw, "type", None)
        if t == "text":
            return ContentBlock(type="text", text=raw.text)
        if t == "thinking":
            return ContentBlock(
                type="thinking",
                text=getattr(raw, "thinking", None),
                signature=getattr(raw, "signature", None),
            )
        if t == "redacted_thinking":
            return ContentBlock(type="redacted_thinking", data=getattr(raw, "data", None))
        if t == "tool_use":
            caller = getattr(raw, "caller", None)
            extras: Dict[str, Any] = {}
            if caller is not None:
                extras["caller"] = caller.model_dump() if hasattr(caller, "model_dump") else dict(caller)
            return ContentBlock(
                type="tool_use",
                tool_use_id=raw.id,
                tool_name=raw.name,
                tool_input=dict(raw.input) if raw.input else {},
                provider_extras=extras,
            )
        if t == "server_tool_use":
            # Persist server-tool invocations (code_execution). Not echoed in our ContentBlock
            # taxonomy natively; stash in provider_extras so trace-render can show it.
            return ContentBlock(
                type="tool_use",
                tool_use_id=getattr(raw, "id", None),
                tool_name=getattr(raw, "name", "code_execution"),
                tool_input=dict(getattr(raw, "input", {}) or {}),
                provider_extras={"server_tool_use": True},
            )
        if t == "code_execution_tool_result":
            content_payload = getattr(raw, "content", None)
            body = content_payload.model_dump() if hasattr(content_payload, "model_dump") else content_payload
            return ContentBlock(
                type="tool_result",
                tool_result_for_id=getattr(raw, "tool_use_id", None),
                tool_result_content=body if isinstance(body, (str, list)) else str(body),
                provider_extras={"code_execution_tool_result": True},
            )
        return ContentBlock(type="text", text=str(raw))

    def _accumulate_usage(self, total: Dict[str, int], usage) -> None:
        if usage is None:
            return
        for f in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            v = getattr(usage, f, 0) or 0
            total[f] = total.get(f, 0) + int(v)

    def _snapshot_usage(self, usage) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        return {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }

    def _compute_applied_edit(self, code: str, request: WorkerRequest) -> str:
        if self.config.generation_mode == "full":
            return code
        parent = request.parent.code.splitlines(keepends=True)
        child = code.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(
            parent, child,
            fromfile=f"parent/{request.parent.id}",
            tofile="child/new",
            n=3,
        ))
        return diff or "# no textual change"

    def _fail(self, we: WorkerError, turns, usage_total, t0) -> WorkerResult:
        return WorkerResult(
            child_code="",
            turns=turns,
            turn_count=len(turns),
            usage=usage_total,
            wall_clock_seconds=time.monotonic() - t0,
            error=we,
        )
