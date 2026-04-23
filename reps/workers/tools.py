"""Shared tool implementations used by AnthropicToolRunnerWorker and
DSPyReActWorker. Each tool is a pair: (json_schema, async callable).

Tools take a `WorkerRequest` + `WorkerContext` closure for state. They return
strings (tool_result content) or structured dicts."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Tuple, Union

from reps.program_summarizer import format_summary_for_prompt
from reps.workers.base import WorkerContext, WorkerRequest


ToolSchema = Dict[str, Any]
ToolImpl = Callable[[Dict[str, Any]], Any]  # async-or-sync returning str/dict

# Per-artifact and overall caps for run_tests artifact summary. Sized so typical
# tracebacks survive unscathed while still bounding worst-case prompt bloat.
_ARTIFACT_PER_CAP = 2000
_ARTIFACT_TOTAL_CAP = 8000

# Float display precision for surfacing metrics to the model. 10 significant
# figures is enough to distinguish scores like 0.9999556474 from 0.9999,
# while still being human-readable.
_FLOAT_FMT = "{:.10g}"


def _format_number(v: Any) -> Any:
    """Render a float at 10-sig-fig precision; pass non-floats through."""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return _FLOAT_FMT.format(v)
    return v


def _format_metrics_precise(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-map `_format_number` across metric values. Non-float values
    (ints, strings, bools, None) pass through unchanged. Returns a new dict
    suitable for passing to `json.dumps`; floats are emitted as strings so
    json.dumps does not re-apply its default ~17-digit repr."""
    return {k: _format_number(v) for k, v in (metrics or {}).items()}


def submit_child_schema() -> ToolSchema:
    return {
        "name": "submit_child",
        "description": (
            "Submit the final child program. Call exactly once to end the "
            "iteration. `code` must be a complete program in the target language. "
            "The `code` argument REPLACES the child entirely. The edit_file "
            "in-flight buffer is NOT submitted automatically — you must read "
            "it via view_parent() after edits and pass it as `code`. "
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
        "description": (
            "Return the parent program's code, metrics, and notebook summary "
            "as a multi-section string with '## Code', '## Metrics', and "
            "'## Notebook' sections (same shape as view_program). When recent "
            "iteration history is available, a leading '## Recent score delta' "
            "section shows parent vs. most-recent-sibling score delta."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "allowed_callers": ["code_execution_20260120"],
    }


def view_program_schema() -> ToolSchema:
    return {
        "name": "view_program",
        "description": (
            "Return the source code, metrics, and notebook summary of an "
            "inspiration or top program by id. Returns a multi-section string "
            "with '## Code', '## Metrics', and '## Notebook' sections."
        ),
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
            "JSON string with metrics (at 10-sigfig precision) and a truncated "
            "artifacts summary. Intermediate artifacts are NOT persisted to the "
            "final child program's record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "allowed_callers": ["code_execution_20260120"],
    }


def mark_converged_schema() -> ToolSchema:
    return {
        "name": "mark_converged",
        "description": (
            "Call when the parent is at saturation (e.g. score matches top of "
            "archive, mutations only yield sub-1e-6 deltas) and no structural "
            "pivot is productive in this iteration. Terminates the iteration "
            "without producing a child. Include a 1-line `reason`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
        "allowed_callers": ["direct", "code_execution_20260120"],
    }


def _program_entry(p) -> Dict[str, Any]:
    """Extract the lookup entry (code/metrics/summary) for a Program object."""
    summary = None
    try:
        if p.metadata:
            summary = p.metadata.get("reps_annotations", {}).get("summary")
    except AttributeError:
        summary = None
    return {
        "code": p.code,
        "metrics": dict(p.metrics) if getattr(p, "metrics", None) else {},
        "summary": summary,
    }


def _format_view_program(entry: Union[str, Dict[str, Any]]) -> str:
    """Render a view_program lookup entry as a multi-section string.

    Accepts either a plain-string code (legacy) or a dict with keys
    `code`, `metrics`, `summary`. Always emits '## Code', '## Metrics',
    and '## Notebook' sections."""
    if isinstance(entry, str):
        code = entry
        metrics: Dict[str, Any] = {}
        summary = None
    else:
        code = entry.get("code", "") or ""
        metrics = entry.get("metrics") or {}
        summary = entry.get("summary")

    parts = ["## Code", code.rstrip(), "", "## Metrics"]
    if metrics:
        # Deterministic ordering: sort by key so the prompt is stable.
        # Use high-precision float formatting so scores like 0.9999556474
        # don't collapse into indistinguishable strings.
        parts.append(
            json.dumps(
                _format_metrics_precise(metrics),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        parts.append("(none)")
    parts.append("")
    if summary:
        nb = format_summary_for_prompt(summary, label="Notebook")
        parts.append(nb if nb else "## Notebook: (none)")
    else:
        parts.append("## Notebook: (none)")
    return "\n".join(parts)


def _edit_diff_window(new_code: str, replace: str, search_line_span: int) -> str:
    """Render a 3-before / [replacement] / 3-after preview around the edit.

    `new_code` is the mutated child; `replace` is the new text that was just
    inserted; `search_line_span` is the number of newlines in the original
    `search` substring (used to size the removed-line marker)."""
    # Locate the inserted replacement in the mutated code.
    idx = new_code.find(replace) if replace else -1
    lines = new_code.splitlines()
    if idx < 0 or not lines:
        return ""
    # Convert char idx -> (start_line, end_line) using newlines up to idx.
    start_line = new_code.count("\n", 0, idx)  # 0-based
    replace_lines = replace.splitlines() or [""]
    end_line = start_line + max(len(replace_lines) - 1, 0)  # 0-based inclusive

    before_start = max(0, start_line - 3)
    after_end = min(len(lines) - 1, end_line + 3)

    out: List[str] = []
    # Lines before the edit.
    for i in range(before_start, start_line):
        out.append(f"  {i + 1}: {lines[i]}")
    # Mark removed lines (the original search span), and added lines (the new replace span).
    # We don't have the pre-edit text here, so annotate by line count.
    for k in range(search_line_span):
        # Use placeholder line numbers pointing at the edit anchor.
        out.append(f"- {start_line + 1}: (removed)")
    for i, rline in enumerate(replace_lines):
        # Truncate each replacement display line to 200 chars.
        display = rline if len(rline) <= 200 else rline[:200] + "...[truncated]"
        out.append(f"+ {start_line + 1 + i}: {display}")
    # Lines after the edit.
    for i in range(end_line + 1, after_end + 1):
        out.append(f"  {i + 1}: {lines[i]}")
    return "\n".join(out)


def _buffer_digest(code: str) -> str:
    """Compute first-12-hex of SHA-256 over the in-flight buffer. Used so the
    model can verify it's reasoning about the current buffer state rather than
    a stale mental model of it."""
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()[:12]


def _buffer_tail(code: str, n_lines: int = 20) -> str:
    """Return the last `n_lines` lines of `code` as a single string."""
    lines = code.splitlines()
    tail = lines[-n_lines:] if len(lines) > n_lines else lines
    return "\n".join(tail)


def _truncate_artifact(key: str, value: Any) -> Tuple[str, bool]:
    """Truncate a single artifact to `_ARTIFACT_PER_CAP` chars.

    For artifacts containing 'Traceback' that exceed the cap, tail-truncate
    (keep LAST N chars, prefix '[truncated head] '). For all others,
    head-truncate (keep FIRST N chars, suffix ' [truncated tail]').
    Returns (rendered, was_truncated)."""
    s = str(value)
    if len(s) <= _ARTIFACT_PER_CAP:
        return s, False
    if "Traceback" in s:
        tail = s[-_ARTIFACT_PER_CAP:]
        return "[truncated head] " + tail, True
    head = s[:_ARTIFACT_PER_CAP]
    return head + " [truncated tail]", True


def _summarize_artifacts(artifacts: Dict[str, Any]) -> str:
    """Build the JSON-string artifacts_summary under per- and total-caps.

    Each artifact is truncated to `_ARTIFACT_PER_CAP` chars (traceback-aware).
    After per-artifact truncation, if the JSON-encoded total exceeds
    `_ARTIFACT_TOTAL_CAP`, progressively tail-truncate artifacts starting
    from the last-inserted key (the least-informative end of the dict)."""
    rendered: Dict[str, str] = {}
    for k, v in artifacts.items():
        text, _ = _truncate_artifact(k, v)
        rendered[k] = text

    encoded = json.dumps(rendered)
    if len(encoded) <= _ARTIFACT_TOTAL_CAP:
        return encoded

    # Progressively shrink least-informative (last) artifacts until under cap.
    keys = list(rendered.keys())
    # Try trimming from tail: halve repeatedly, then drop if still over.
    for k in reversed(keys):
        current = rendered[k]
        if len(current) <= 100:
            # Already tiny; dropping wouldn't help much — continue.
            continue
        # Shrink this artifact to a minimal placeholder.
        rendered[k] = current[:200] + " [truncated tail]"
        encoded = json.dumps(rendered)
        if len(encoded) <= _ARTIFACT_TOTAL_CAP:
            return encoded

    # Hard fallback: truncate the JSON blob itself.
    return encoded[: _ARTIFACT_TOTAL_CAP] + "...[truncated]"


def _recent_score_delta(request: WorkerRequest) -> str:
    """Compare parent.combined_score to the most-recent sibling's combined_score.

    Returns a one-line string suitable for a '## Recent score delta' section,
    or an empty string if the comparison isn't available (no recent iterations,
    no combined_score on either side)."""
    recent = getattr(request, "recent_iterations", None) or []
    if not recent:
        return ""
    sibling = recent[0]
    parent_metrics = getattr(request.parent, "metrics", None) or {}
    sibling_metrics = getattr(sibling, "metrics", None) or {}
    p_score = parent_metrics.get("combined_score")
    s_score = sibling_metrics.get("combined_score")
    if p_score is None or s_score is None:
        return ""
    try:
        delta = float(s_score) - float(p_score)
    except (TypeError, ValueError):
        return ""
    sign = "+" if delta >= 0 else ""
    return (
        f"parent ({_FLOAT_FMT.format(float(p_score))}) vs most_recent_sibling "
        f"({_FLOAT_FMT.format(float(s_score))}): delta={sign}{_FLOAT_FMT.format(delta)}"
    )


def build_tool_impls(
    request: WorkerRequest,
    ctx: WorkerContext,
    tool_names: List[str],
    edit_accumulator: List[Tuple[str, str]],
    child_code_holder: List[str],          # single-element list used as mutable ref
    lookup: Dict[str, Union[str, Dict[str, Any]]] = None,
) -> Dict[str, ToolImpl]:
    """Build a dict of {tool_name: async-callable} for the requested tool_names.

    `edit_accumulator` is the list that edit_file appends to; the worker reads
    it after submit_child to produce applied_edit.
    `child_code_holder[0]` is the current in-flight child code (starts as parent);
    edit_file mutates it, view_parent reads it.
    `lookup` maps program_id -> either a code string (legacy) or a dict with
    keys {code, metrics, summary}. When omitted, it's built from the request's
    parent / inspirations / top_programs / second_parent using `_program_entry`
    (includes metrics + summary).
    """
    if lookup is None:
        lookup = {request.parent.id: _program_entry(request.parent)}
        for p in request.inspirations:
            lookup[p.id] = _program_entry(p)
        for p in request.top_programs:
            lookup[p.id] = _program_entry(p)
        if request.second_parent is not None:
            lookup[request.second_parent.id] = _program_entry(request.second_parent)

    async def view_parent(_args):
        # Contract: returns a multi-section string describing the parent
        # program — code, metrics, and notebook summary — matching the
        # shape of view_program. A leading '## Recent score delta' section
        # is prepended when the most recent sibling iteration is available.
        delta = _recent_score_delta(request)
        body = _format_view_program(_program_entry(request.parent))
        if delta:
            return "## Recent score delta\n" + delta + "\n\n" + body
        return body

    async def view_program(args):
        # Contract: returns a multi-section string ('## Code', '## Metrics',
        # '## Notebook') for the requested program_id. Draws from the lookup
        # built from the WorkerRequest's parent/inspirations/top_programs.
        pid = args.get("program_id", "")
        entry = lookup.get(pid)
        if entry is None:
            return f"ERROR: program_id '{pid}' not available"
        return _format_view_program(entry)

    async def edit_file(args):
        # Contract: applies a single-occurrence SEARCH/REPLACE to the in-flight
        # child. On success, returns the applied-edit header plus a 3-before /
        # 3-after diff window around the replacement site, PLUS a trailing
        # buffer-tail section (last 20 lines + SHA-12 digest) so the model can
        # verify the current in-flight state without a fresh view_parent call.
        search = args["search"]
        replace = args["replace"]
        current = child_code_holder[0]
        if current.count(search) != 1:
            return (
                f"ERROR: search string must appear exactly once; found "
                f"{current.count(search)} occurrences."
            )
        new_code = current.replace(search, replace, 1)
        child_code_holder[0] = new_code
        edit_accumulator.append((search, replace))
        header = (
            f"edit applied ({len(edit_accumulator)} total); "
            f"new length={len(new_code)}"
        )
        search_line_span = max(search.count("\n"), 0) + 1 if search else 0
        window = _edit_diff_window(new_code, replace, search_line_span)
        sha12 = _buffer_digest(new_code)
        tail = _buffer_tail(new_code, 20)
        tail_section = (
            f"\n\n## Buffer tail (last 20 lines, SHA-12={sha12})\n{tail}"
        )
        if window:
            return header + "\n\n--- diff window ---\n" + window + tail_section
        return header + tail_section

    async def run_tests(args):
        # Contract: runs an isolated scratch evaluation and returns JSON with
        # metrics (at 10-sigfig precision) + truncated artifacts_summary
        # (per-artifact 2000 char cap, tail-truncate tracebacks, head-truncate
        # others, 8000 overall cap).
        if ctx.evaluator is None:
            return "ERROR: run_tests not available (uses_evaluator=False)"
        scratch_id = ctx.scratch_id_factory()
        outcome = await ctx.evaluator.evaluate_isolated(
            args["code"], program_id=scratch_id, scratch=True
        )
        art_summary = ""
        if outcome.artifacts:
            art_summary = _summarize_artifacts(outcome.artifacts)
        return json.dumps({
            "metrics": _format_metrics_precise(outcome.metrics),
            "artifacts_summary": art_summary,
        })

    async def submit_child(_args):
        # Contract: terminal marker. The worker detects submit_child via the
        # tool_name and extracts `code` / `changes_description` from the
        # tool_use payload itself. The worker is ALSO responsible for the
        # auto-reevaluation of the submitted code (see anthropic_tool_runner
        # / openai_tool_runner dispatch branches) — this in-band handler
        # only needs to return the string the worker will splice into the
        # tool_result content.
        return "OK"

    async def mark_converged(_args):
        # Contract: the worker's dispatch branch is responsible for setting
        # the converged_flag + reason on the WorkerResult — this in-band
        # handler only returns the acknowledgement string.
        return "acknowledged: converged"

    table = {
        "view_parent": view_parent,
        "view_program": view_program,
        "edit_file": edit_file,
        "run_tests": run_tests,
        "submit_child": submit_child,
        "mark_converged": mark_converged,
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
        "mark_converged": mark_converged_schema,
    }
    for name in tool_names:
        if name == "run_tests" and ctx.evaluator is None:
            continue
        if name in builders:
            schemas.append(builders[name]())
    return schemas
