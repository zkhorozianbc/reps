"""Per-program reasoning summarizer — Sonnet 4.6 call that distills a
program's full trace into structured JSON insight for F8 Annotations.

Prompt architecture:
  system:  general summarizer role + strict output rules (generalizes to ANY
           benchmark — same across the whole REPS deployment)
  user:    optional task-specific instructions (per-benchmark guardrails)
           followed by the data to summarize (trace + code + metrics)

Benchmark authors should supply their task-specific instructions via
`config.reps.summarizer.task_instructions` (or pass directly to
`summarize_program(task_instructions=...)`) — this keeps the general
summarizer prompt stable and cache-friendly while letting each benchmark
correct for its own failure modes (e.g. "score=0 means overlap, not
broken validator").
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from reps.workers.trace_render import render_trace_from_dicts

logger = logging.getLogger(__name__)

_MAX_REASONING_CHARS = 12000  # truncate very long traces before sending

# General role + strict rules. Generalizes to any benchmark. Cache-stable.
_SYSTEM_PROMPT = """You are a technical analyst extracting actionable insights from code mutation attempts in an evolutionary search.

Your role is to read an agent's full reasoning + tool trace from one iteration of evolution — along with the code it submitted and its score — and produce a structured JSON summary that helps subsequent iterations learn from this attempt.

STRICT RULES:
1. Output ONLY a JSON object. No prose before or after. No markdown fences.
2. Do NOT speculate that the scoring / validator / evaluator system itself is broken. If a score is 0 or low, assume the code failed to meet the benchmark's constraints (invalid output, runtime error, constraint violation) — the evaluator is working as designed.
3. Be SPECIFIC. "The code failed" is useless; "the sort key returned None for empty inputs" is useful. Cite line-of-reasoning or concrete numbers when possible.
4. Pitfalls must be OBSERVED in the trace — bugs the agent actually hit, error messages it actually saw. Do not invent plausible-sounding bugs. If the trace shows no explicit failure, set pitfalls to [].
5. `next_directions` must be grounded: directions the parent CONSIDERED but did not pursue, or direct extensions of what demonstrably worked. Do not emit generic programming advice.
6. `avoid` must reference approaches the parent EXPLICITLY tried and found worse. Do not pre-emptively blacklist entire technique families based on one data point.
7. `key_insight` is one line capturing the most transferable lesson from this attempt. If nothing notable, use "none".

OUTPUT SCHEMA (exactly these keys, no others):
{
  "approach": "string, one line describing the mutation attempted",
  "pitfalls": ["up to 5 strings, each a specific observed failure"],
  "key_insight": "one line or \\"none\\"",
  "next_directions": ["up to 3 unexplored directions, grounded in the trace"],
  "avoid": ["up to 3 approaches the parent explicitly tried and found inferior"]
}"""


def _build_user_message(
    *,
    task_instructions: Optional[str],
    parent_score: float,
    child_score: float,
    improved: bool,
    code: str,
    reasoning: str,
) -> str:
    parts: list[str] = []
    if task_instructions:
        # Task-specific guardrails appear BEFORE the data so the model reads
        # them first when constructing its summary.
        parts.append("## Benchmark-specific guidance")
        parts.append(task_instructions.strip())
        parts.append("")
    parts.append("## Attempt data")
    parts.append(f"Parent score (combined_score): {parent_score:.6f}")
    parts.append(f"Child score (combined_score): {child_score:.6f}")
    parts.append(f"Improved: {improved}")
    parts.append("")
    parts.append("Child code (first 3000 chars):")
    parts.append("```python")
    parts.append(code[:3000])
    parts.append("```")
    parts.append("")
    parts.append("Agent's reasoning + tool trace:")
    parts.append(reasoning)
    return "\n".join(parts)


async def summarize_program(
    *,
    program_id: str,
    code: str,
    turns: List[Dict[str, Any]],
    parent_score: float,
    child_score: float,
    improved: bool,
    llm_ensemble,
    model_id: str = "claude-sonnet-4-6",
    task_instructions: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call Sonnet 4.6 to produce a structured per-program summary.

    Returns None on failure; callers should treat it as best-effort and
    continue without a summary.

    Args:
        task_instructions: Optional per-benchmark guardrails to append to
            the user message before the attempt data. Use this to inject
            correction guidance (e.g. "score=0 means overlap, not a broken
            validator") without touching the general system prompt.
    """
    if not turns:
        return None
    reasoning = render_trace_from_dicts(turns)
    if len(reasoning) > _MAX_REASONING_CHARS:
        head = reasoning[: _MAX_REASONING_CHARS // 2]
        tail = reasoning[-_MAX_REASONING_CHARS // 2 :]
        reasoning = f"{head}\n\n... [truncated {len(reasoning) - _MAX_REASONING_CHARS} chars] ...\n\n{tail}"

    user_text = _build_user_message(
        task_instructions=task_instructions,
        parent_score=parent_score,
        child_score=child_score,
        improved=improved,
        code=code,
        reasoning=reasoning,
    )

    try:
        resp_text = await llm_ensemble.generate_with_context(
            system_message=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            model=model_id,
            temperature=0.2,
        )
    except Exception as e:
        logger.warning(f"program summarizer call failed for {program_id[:8]}: {e}")
        return None

    # Strip common wrapper patterns (```json ... ```)
    s = (resp_text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        data = json.loads(s)
    except Exception as e:
        logger.warning(f"program summarizer JSON parse failed for {program_id[:8]}: {e}")
        return None

    out = {
        "approach": str(data.get("approach", ""))[:500],
        "pitfalls": [str(p)[:300] for p in (data.get("pitfalls") or [])[:5]],
        "key_insight": str(data.get("key_insight", ""))[:400],
        "next_directions": [str(p)[:300] for p in (data.get("next_directions") or [])[:3]],
        "avoid": [str(p)[:300] for p in (data.get("avoid") or [])[:3]],
    }
    return out


def format_summary_for_prompt(summary: Dict[str, Any], label: str = "Parent's notebook") -> str:
    """Render a summary dict into a compact block suitable for injection
    into the next iteration's user prompt."""
    if not summary:
        return ""
    lines = [f"## {label}"]
    if summary.get("approach"):
        lines.append(f"- Approach: {summary['approach']}")
    if summary.get("key_insight") and summary["key_insight"].lower() != "none":
        lines.append(f"- Key insight: {summary['key_insight']}")
    if summary.get("pitfalls"):
        lines.append("- Pitfalls hit:")
        lines.extend(f"  - {p}" for p in summary["pitfalls"])
    if summary.get("avoid"):
        lines.append("- Avoid:")
        lines.extend(f"  - {p}" for p in summary["avoid"])
    if summary.get("next_directions"):
        lines.append("- Unexplored directions:")
        lines.extend(f"  - {p}" for p in summary["next_directions"])
    return "\n".join(lines)
