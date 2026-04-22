"""SingleCallWorker: port of today's one-shot LLM call path under the Worker
primitive. Uses PromptSampler for prompt construction, LLMEnsemble for the
call, and reps.utils for diff/full-rewrite parsing."""
from __future__ import annotations

import time
from typing import Dict, List

from reps.program_summarizer import format_summary_for_prompt
from reps.utils import (
    apply_diff,
    apply_diff_blocks,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
    split_diffs_by_target,
)
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


@register("single_call")
class SingleCallWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "SingleCallWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        ensemble = ctx.llm_factory(self.config.model_id)

        # Build prompt via PromptSampler (existing path)
        changes_description_text = None
        if ctx.config.prompt.programs_as_changes_description:
            changes_description_text = (
                request.parent.changes_description
                or ctx.config.prompt.initial_changes_description
            )

        use_diff = _decide_diff_mode(request.generation_mode, ctx.config.diff_based_evolution)

        prompt = ctx.prompt_sampler.build_prompt(
            current_program=request.parent.code,
            parent_program=request.parent.code,
            program_metrics=request.parent.metrics,
            previous_programs=[p.to_dict() for p in request.top_programs],
            top_programs=[p.to_dict() for p in (request.top_programs + request.inspirations)],
            inspirations=[p.to_dict() for p in request.inspirations],
            language=request.language,
            evolution_round=request.iteration,
            diff_based_evolution=use_diff,
            program_artifacts=None,
            feature_dimensions=request.feature_dimensions,
            current_changes_description=changes_description_text,
            **request.prompt_extras,
        )

        # Prepend parent's per-program summary if available.
        parent_summary = (
            request.parent.metadata.get("reps_annotations", {}).get("summary")
            if request.parent and request.parent.metadata
            else None
        )
        if parent_summary:
            insights_text = format_summary_for_prompt(parent_summary, label="Parent's notebook")
            prompt["user"] = insights_text + "\n\n" + prompt["user"]

        # Append un-consumed prompt_extras (existing behavior from controller.py:287)
        for key in ("reflection", "sota_injection", "dead_end_warnings"):
            text = request.prompt_extras.get(key, "")
            if text and "{" + key + "}" not in prompt.get("user", ""):
                prompt["user"] = prompt["user"] + "\n\n" + text

        gen_kwargs = {}
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.model_id is not None:
            gen_kwargs["model"] = request.model_id

        # Make the call
        try:
            llm_response = await ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
                **gen_kwargs,
            )
        except Exception as e:
            return WorkerResult(
                child_code="",
                turns=_build_turns_for_error(prompt),
                error=WorkerError(kind="INTERNAL", detail=f"LLM call failed: {e}"),
                wall_clock_seconds=time.monotonic() - t0,
            )

        if llm_response is None:
            return WorkerResult(
                child_code="",
                turns=_build_turns_for_error(prompt),
                error=WorkerError(kind="INTERNAL", detail="LLM returned None"),
                wall_clock_seconds=time.monotonic() - t0,
            )

        # Parse
        child_code: str = ""
        changes_summary: str = ""
        changes_description_out = changes_description_text
        applied_edit: str = ""

        if use_diff:
            diff_blocks = extract_diffs(llm_response, ctx.config.diff_pattern)
            if not diff_blocks:
                return WorkerResult(
                    child_code="",
                    turns=_build_turns(prompt, llm_response),
                    error=WorkerError(kind="PARSE_ERROR", detail="No valid diffs"),
                    wall_clock_seconds=time.monotonic() - t0,
                )
            if ctx.config.prompt.programs_as_changes_description:
                code_blocks, desc_blocks, _unmatched = split_diffs_by_target(
                    diff_blocks,
                    code_text=request.parent.code,
                    changes_description_text=changes_description_text,
                )
                child_code, _ = apply_diff_blocks(request.parent.code, code_blocks)
                new_desc, desc_applied = apply_diff_blocks(changes_description_text, desc_blocks)
                if desc_applied == 0 or not new_desc.strip() or new_desc.strip() == (changes_description_text or "").strip():
                    return WorkerResult(
                        child_code="",
                        turns=_build_turns(prompt, llm_response),
                        error=WorkerError(kind="PARSE_ERROR", detail="changes_description not updated"),
                        wall_clock_seconds=time.monotonic() - t0,
                    )
                changes_description_out = new_desc
                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=ctx.config.prompt.diff_summary_max_line_len,
                    max_lines=ctx.config.prompt.diff_summary_max_lines,
                )
                applied_edit = serialize_diff_blocks(code_blocks)
            else:
                child_code = apply_diff(request.parent.code, llm_response, ctx.config.diff_pattern)
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=ctx.config.prompt.diff_summary_max_line_len,
                    max_lines=ctx.config.prompt.diff_summary_max_lines,
                )
                applied_edit = serialize_diff_blocks(diff_blocks)
        else:
            new_code = parse_full_rewrite(llm_response, request.language)
            if not new_code:
                return WorkerResult(
                    child_code="",
                    turns=_build_turns(prompt, llm_response),
                    error=WorkerError(kind="PARSE_ERROR", detail="No valid code in response"),
                    wall_clock_seconds=time.monotonic() - t0,
                )
            child_code = new_code
            changes_summary = "Full rewrite"
            applied_edit = new_code

        usage = getattr(ensemble, "last_usage", {}) or {}
        return WorkerResult(
            child_code=child_code,
            changes_description=changes_description_out,
            changes_summary=changes_summary,
            applied_edit=applied_edit,
            turns=_build_turns(prompt, llm_response),
            turn_count=2,
            usage=dict(usage),
            wall_clock_seconds=time.monotonic() - t0,
            error=None,
        )


def _decide_diff_mode(request_mode: str, global_diff_enabled: bool) -> bool:
    if not global_diff_enabled:
        return False
    if request_mode == "full":
        return False
    if request_mode == "diff":
        return True
    return global_diff_enabled


def _build_turns(prompt: Dict[str, str], llm_response: str) -> List[TurnRecord]:
    return [
        TurnRecord(
            index=0,
            role="user",
            blocks=[
                ContentBlock(type="text", text=prompt.get("system", "")),
                ContentBlock(type="text", text=prompt.get("user", "")),
            ],
            worker_type="single_call",
        ),
        TurnRecord(
            index=1,
            role="assistant",
            blocks=[ContentBlock(type="text", text=llm_response)],
            worker_type="single_call",
        ),
    ]


def _build_turns_for_error(prompt: Dict[str, str]) -> List[TurnRecord]:
    return [
        TurnRecord(
            index=0,
            role="user",
            blocks=[
                ContentBlock(type="text", text=prompt.get("system", "")),
                ContentBlock(type="text", text=prompt.get("user", "")),
            ],
            worker_type="single_call",
        )
    ]
