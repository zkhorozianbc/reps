"""GEPA-style per-candidate trace-grounded reflection (Phase 3).

Given a parent Program with `feedback` (free-form ASI text) and
`per_instance_scores` (per-objective breakdown), prompt an LLM to produce
a *single concrete* mutation directive targeting the worst sub-objective.
The directive is cached on the parent and injected into the next mutation
prompt for that lineage.

This is intentionally narrow:
  - Pure async function — no global state, no I/O beyond the LLM call.
  - Skips the LLM call entirely when there's nothing useful to say
    (no feedback, feedback too short, no per-instance scores). The
    caller treats `None` as "no directive available; use the existing
    prompt unchanged".
  - Returns plain text — the directive is meant for direct injection
    into a prompt template, not parsed.

Sits ALONGSIDE the batch-level reflection (`reps/reflection_engine.py`),
not in place of it. Batch reflection sees aggregate patterns across
recent results; trace reflection sees one specific candidate's failure.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from reps.database import Program
from reps.llm.base import LLMInterface

logger = logging.getLogger(__name__)


_SYSTEM_MESSAGE = (
    "You are analyzing why an evolved program got a specific per-objective score "
    "breakdown. Propose ONE concrete next change, targeting the lowest-scoring "
    "sub-objective. Output 1-3 plain sentences. No preamble, apologies, "
    "headings, or code blocks. Be specific to the failure described — generic "
    "advice (\"try harder\", \"improve performance\") is not acceptable."
)


def _excerpt_code(code: str, max_chars: int) -> str:
    """Truncate `code` to `max_chars` while preserving the head and tail
    (where the interesting bits usually live)."""
    if len(code) <= max_chars:
        return code
    half = max(0, (max_chars - 32) // 2)  # leave room for the elision marker
    return code[:half] + "\n... [omitted middle] ...\n" + code[-half:]


def _format_score(metrics: Optional[dict]) -> str:
    """Render a Program's combined_score for the lineage block."""
    if not metrics:
        return "?"
    score = metrics.get("combined_score")
    if score is None:
        return "?"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def _build_lineage_block(ancestors: List[Program]) -> str:
    """Render a compact ancestral history (oldest first) for the prompt.

    Each line: `gen N | score X.XXXX | "<changes_description>"
                       [prior directive: "..."]`.
    Returns "" for an empty list so the caller can drop the section entirely.
    """
    if not ancestors:
        return ""

    lines = []
    for prog in ancestors:
        score = _format_score(prog.metrics)
        changes = (prog.changes_description or "").strip() or "(no description)"
        # Truncate long changes_description so one ancestor doesn't dominate.
        if len(changes) > 120:
            changes = changes[:117] + "..."
        line = f'  - gen {prog.generation} | score {score} | "{changes}"'
        if prog.mutation_directive:
            directive = prog.mutation_directive.strip().replace("\n", " ")
            if len(directive) > 100:
                directive = directive[:97] + "..."
            line += f'\n      [prior directive: "{directive}"]'
        lines.append(line)
    return "Recent ancestor lineage (oldest first):\n" + "\n".join(lines)


def _build_user_message(
    parent: Program,
    max_code_chars: int,
    ancestors: Optional[List[Program]] = None,
) -> str:
    scores_json = json.dumps(parent.per_instance_scores, indent=2, sort_keys=True)
    code_excerpt = _excerpt_code(parent.code or "", max_code_chars)

    lineage = ""
    if ancestors:
        lineage_block = _build_lineage_block(ancestors)
        if lineage_block:
            lineage = lineage_block + "\n\n"

    return (
        f"{lineage}"
        f"Per-objective scores (each in [0, 1] unless noted):\n{scores_json}\n\n"
        f"Diagnostic feedback from the evaluator:\n{parent.feedback}\n\n"
        f"Current program (excerpt):\n```\n{code_excerpt}\n```\n\n"
        f"What is the single most impactful change to attempt next? "
        f"Be specific to the failure described above."
    )


def should_generate_directive(
    parent: Program,
    *,
    min_feedback_length: int,
) -> bool:
    """Pure predicate — caller asks before paying for an LLM call.

    Skips when:
      - feedback is None or empty / under min_feedback_length
      - per_instance_scores is None (no granular signal to act on)
      - mutation_directive is already set (cached from a prior call)
    """
    if parent.mutation_directive:
        return False
    if not parent.feedback or len(parent.feedback.strip()) < min_feedback_length:
        return False
    if not parent.per_instance_scores:
        return False
    return True


async def generate_directive(
    parent: Program,
    llm: LLMInterface,
    *,
    min_feedback_length: int = 20,
    max_code_chars: int = 4000,
    ancestors: Optional[List[Program]] = None,
) -> Optional[str]:
    """Run one LLM call to produce a mutation_directive for `parent`.

    Returns None when the parent doesn't qualify (see `should_generate_directive`)
    or when the LLM call fails. The caller is responsible for caching the
    result on `parent.mutation_directive` if non-None.

    `ancestors` (Phase 5): optional ancestral chain of `parent`, oldest-first
    and NOT including `parent` itself. When provided, a compact lineage
    block (each ancestor's generation, score, changes_description, and
    cached mutation_directive) is prepended to the user message so the LLM
    can see what's been tried and how it scored. Pass [] or None to disable.

    Note: `ancestors` is NOT part of the cache key. The directive is
    cached on `parent.mutation_directive` after the first non-empty
    response, so re-sampling the same parent with a different lineage
    later in the run will not regenerate. This is intentional — the
    directive targets the parent's own sub-objective, not its ancestry.

    The LLM call uses `generate_with_context` with a fixed system message
    (cache-friendly) and a per-parent user message. No streaming, no tool use.
    """
    if not should_generate_directive(parent, min_feedback_length=min_feedback_length):
        return None

    user_message = _build_user_message(
        parent, max_code_chars=max_code_chars, ancestors=ancestors,
    )

    try:
        response = await llm.generate_with_context(
            system_message=_SYSTEM_MESSAGE,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:
        logger.warning(
            "trace_reflection: LLM call failed for parent %s: %s",
            parent.id[:8] if parent.id else "?", exc,
        )
        return None

    directive = (response or "").strip()
    if not directive:
        logger.debug(
            "trace_reflection: LLM returned empty directive for %s",
            parent.id[:8] if parent.id else "?",
        )
        return None
    return directive
