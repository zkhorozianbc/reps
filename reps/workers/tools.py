"""Shared tool implementations used by AnthropicToolRunnerWorker and
DSPyReActWorker. Each tool is a pair: (json_schema, async callable).

Tools take a `WorkerRequest` + `WorkerContext` closure for state. They return
strings (tool_result content) or structured dicts."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

from reps.workers.base import WorkerContext, WorkerRequest


ToolSchema = Dict[str, Any]
ToolImpl = Callable[[Dict[str, Any]], Any]  # async-or-sync returning str/dict


def submit_child_schema() -> ToolSchema:
    return {
        "name": "submit_child",
        "description": (
            "Submit the final child program. Call exactly once to end the "
            "iteration. `code` must be a complete program in the target language. "
            "Callable directly (for one-shot full rewrites) or from inside a "
            "code_execution block (the common chaining pattern)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full program source."},
                "changes_description": {
                    "type": "string",
                    "description": "One-to-three-sentence summary of changes vs parent.",
                },
            },
            "required": ["code", "changes_description"],
        },
        "allowed_callers": ["direct", "code_execution_20260120"],
    }


def edit_file_schema() -> ToolSchema:
    return {
        "name": "edit_file",
        "description": (
            "Apply a SEARCH/REPLACE edit to the in-flight child program. The "
            "`search` substring must appear exactly once in the current child "
            "code; it will be replaced with `replace`. Multiple edit_file calls "
            "may be made before submit_child. Callable from code_execution so "
            "many edits can be chained in one Python block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {"type": "string"},
                "replace": {"type": "string"},
            },
            "required": ["search", "replace"],
        },
        "allowed_callers": ["code_execution_20260120"],
    }


def view_parent_schema() -> ToolSchema:
    return {
        "name": "view_parent",
        "description": "Return the parent program's current source code (returns a string).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "allowed_callers": ["code_execution_20260120"],
    }


def view_program_schema() -> ToolSchema:
    return {
        "name": "view_program",
        "description": "Return the source of an inspiration or top program by id (returns a string).",
        "input_schema": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
        },
        "allowed_callers": ["code_execution_20260120"],
    }


def run_tests_schema() -> ToolSchema:
    return {
        "name": "run_tests",
        "description": (
            "Evaluate candidate code in an isolated scratch workspace. Returns a "
            "JSON string with metrics and a truncated artifacts summary. "
            "Intermediate artifacts are NOT persisted to the final child "
            "program's record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "allowed_callers": ["code_execution_20260120"],
    }


def build_tool_impls(
    request: WorkerRequest,
    ctx: WorkerContext,
    tool_names: List[str],
    edit_accumulator: List[Tuple[str, str]],
    child_code_holder: List[str],          # single-element list used as mutable ref
) -> Dict[str, ToolImpl]:
    """Build a dict of {tool_name: async-callable} for the requested tool_names.

    `edit_accumulator` is the list that edit_file appends to; the worker reads
    it after submit_child to produce applied_edit.
    `child_code_holder[0]` is the current in-flight child code (starts as parent);
    edit_file mutates it, view_parent reads it.
    """
    lookup = {request.parent.id: request.parent.code}
    for p in request.inspirations:
        lookup[p.id] = p.code
    for p in request.top_programs:
        lookup[p.id] = p.code
    if request.second_parent is not None:
        lookup[request.second_parent.id] = request.second_parent.code

    async def view_parent(_args):
        return child_code_holder[0]

    async def view_program(args):
        pid = args.get("program_id", "")
        return lookup.get(pid, f"ERROR: program_id '{pid}' not available")

    async def edit_file(args):
        search = args["search"]
        replace = args["replace"]
        current = child_code_holder[0]
        if current.count(search) != 1:
            return (
                f"ERROR: search string must appear exactly once; found "
                f"{current.count(search)} occurrences."
            )
        child_code_holder[0] = current.replace(search, replace, 1)
        edit_accumulator.append((search, replace))
        return f"edit applied ({len(edit_accumulator)} total); new length={len(child_code_holder[0])}"

    async def run_tests(args):
        if ctx.evaluator is None:
            return "ERROR: run_tests not available (uses_evaluator=False)"
        scratch_id = ctx.scratch_id_factory()
        outcome = await ctx.evaluator.evaluate_isolated(
            args["code"], program_id=scratch_id, scratch=True
        )
        # Redact: return metrics + truncated artifact summary.
        art_summary = ""
        if outcome.artifacts:
            art_summary = json.dumps({k: str(v)[:200] for k, v in outcome.artifacts.items()})
            if len(art_summary) > 2000:
                art_summary = art_summary[:2000] + "...[truncated]"
        return json.dumps({
            "metrics": outcome.metrics,
            "artifacts_summary": art_summary,
        })

    async def submit_child(_args):
        # Terminal marker; worker detects submit_child via tool_name and handles
        # the code + changes_description extraction itself (avoids coupling here).
        return "OK"

    table = {
        "view_parent": view_parent,
        "view_program": view_program,
        "edit_file": edit_file,
        "run_tests": run_tests,
        "submit_child": submit_child,
    }
    return {n: table[n] for n in tool_names if n in table}


def build_tool_schemas(
    ctx: WorkerContext,
    tool_names: List[str],
) -> List[ToolSchema]:
    schemas = []
    builders = {
        "view_parent": view_parent_schema,
        "view_program": view_program_schema,
        "edit_file": edit_file_schema,
        "run_tests": run_tests_schema,
        "submit_child": submit_child_schema,
    }
    for name in tool_names:
        if name == "run_tests" and ctx.evaluator is None:
            continue
        if name in builders:
            schemas.append(builders[name]())
    return schemas
