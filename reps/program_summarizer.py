"""Per-program reasoning summarizer — Sonnet 4.6 call that distills a
program's full trace into structured JSON insight for F8 Annotations."""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from reps.workers.trace_render import render_trace_from_dicts

logger = logging.getLogger(__name__)

_MAX_REASONING_CHARS = 12000  # truncate very long traces before sending

_SYSTEM_PROMPT = (
    "You are a technical analyst extracting actionable insights from code "
    "mutation attempts in an evolutionary search. You output STRICT JSON only."
)

_USER_TEMPLATE = """Analyze this mutation attempt.

Parent score (combined_score): {parent_score:.6f}
Child score (combined_score): {child_score:.6f}
Improved: {improved}

Child code (first 3000 chars):
```python
{code_head}
```

Agent's reasoning + tool trace (reasoning blocks, tool calls, results):
{reasoning}

Output JSON with exactly these keys (no other fields, no prose):
- "approach": string, one line describing the mutation attempted
- "pitfalls": list of up to 5 strings, each a specific bug/failure/dead-end observed
- "key_insight": string, one line capturing what was learned (or "none" if nothing notable)
- "next_directions": list of up to 3 strings, unexplored directions worth trying
- "avoid": list of up to 3 strings, approaches to NOT retry (because they failed or are clearly inferior)

Respond with ONLY the JSON object, nothing else."""


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
) -> Optional[Dict[str, Any]]:
    """Call Sonnet 4.6 to produce a structured per-program summary.

    Returns None on failure; callers should treat it as best-effort and
    continue without a summary.
    """
    if not turns:
        return None
    reasoning = render_trace_from_dicts(turns)
    if len(reasoning) > _MAX_REASONING_CHARS:
        head = reasoning[: _MAX_REASONING_CHARS // 2]
        tail = reasoning[-_MAX_REASONING_CHARS // 2 :]
        reasoning = f"{head}\n\n... [truncated {len(reasoning) - _MAX_REASONING_CHARS} chars] ...\n\n{tail}"

    prompt = _USER_TEMPLATE.format(
        parent_score=parent_score,
        child_score=child_score,
        improved=improved,
        code_head=code[:3000],
        reasoning=reasoning,
    )

    try:
        resp_text = await llm_ensemble.generate_with_context(
            system_message=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            temperature=0.2,
        )
    except Exception as e:
        logger.warning(f"program summarizer call failed for {program_id[:8]}: {e}")
        return None

    # Strip common wrapper patterns (```json ... ```)
    s = (resp_text or "").strip()
    if s.startswith("```"):
        # strip ```json ... ```
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

    # Sanity-coerce the expected shape
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
