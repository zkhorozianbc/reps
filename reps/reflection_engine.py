"""
F1: Reflection Engine

A dedicated LLM pass that runs in the controller process after each batch of
iteration results. It produces structured analysis of what worked, what failed,
and what to try next. The reflection is injected into prompts via a {reflection}
template field.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Prompt template for the reflection LLM call
REFLECTION_PROMPT = """You are analyzing the results of an evolutionary code search.

## Current Best Program
```
{best_program}
```
Best score: {best_score}

## Top-K Candidates This Batch (improvements or near-improvements)
{top_k_section}

## Bottom-K Candidates This Batch (failures or regressions)
{bottom_k_section}

## Previous Reflection
{previous_reflection}

## Task
Analyze these results and produce a JSON object with exactly these fields:
- "working_patterns": list of strings describing what kinds of edits improved scores
- "failing_patterns": list of strings describing what kinds of edits hurt or stalled
- "hypotheses": list of strings with causal claims about why certain approaches work
- "suggested_directions": list of strings with concrete next steps to try

Be specific and actionable. Reference actual code patterns you see.
Respond with ONLY the JSON object, no other text.
"""


def _format_candidate(result, rank: int) -> str:
    """Format a single candidate for the reflection prompt."""
    lines = [f"### Candidate {rank}"]
    lines.append(f"Worker: {result.worker_type}, Score: {result.child_score:.6f}")
    if result.parent_score:
        delta = result.child_score - result.parent_score
        lines.append(f"Delta from parent: {delta:+.6f}")
    if result.diff:
        # Truncate long diffs
        diff_preview = result.diff[:1500]
        if len(result.diff) > 1500:
            diff_preview += "\n... (truncated)"
        lines.append(f"Diff:\n```\n{diff_preview}\n```")
    return "\n".join(lines)


class ReflectionEngine:
    """Post-batch reflection that runs in the controller process.

    Uses the openevolve/llm/ module directly (not through the process pool)
    to generate structured analysis of batch results.
    """

    def __init__(self, llm_ensemble, config: Dict[str, Any]):
        """
        Args:
            llm_ensemble: LLMEnsemble instance for making reflection LLM calls
            config: REPS reflection config dict with keys:
                - top_k (int): number of top candidates to analyze
                - bottom_k (int): number of bottom candidates to analyze
                - enabled (bool): whether reflection is active
                - model (Optional[str]): model id to use for reflection calls;
                  when None, the ensemble's default weighted sampling is used.
        """
        self.llm = llm_ensemble
        self.top_k = config.get("top_k", 3)
        self.bottom_k = config.get("bottom_k", 2)
        self.enabled = config.get("enabled", True)
        self.model_override = config.get("model")
        self._current_reflection: Dict[str, Any] = {}
        self._call_count = 0
        self._total_tokens = {"prompt_tokens": 0, "completion_tokens": 0}

    @property
    def current_reflection(self) -> Dict[str, Any]:
        return self._current_reflection

    @property
    def total_reflection_calls(self) -> int:
        return self._call_count

    @property
    def total_reflection_tokens(self) -> Dict[str, int]:
        return dict(self._total_tokens)

    async def reflect(
        self,
        batch_results: List,
        database,
        previous_reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze batch results and produce structured reflection.

        Args:
            batch_results: List of IterationResult objects from the completed batch
            database: ProgramDatabase for getting the current best program
            previous_reflection: The reflection from the previous batch

        Returns:
            Dict with keys: working_patterns, failing_patterns, hypotheses,
            suggested_directions
        """
        if not self.enabled:
            return previous_reflection or {}

        # Filter to results with valid scores
        scored = [r for r in batch_results if r.error is None and r.child_score is not None]
        if not scored:
            return previous_reflection or {}

        # Sort by child_score descending
        scored.sort(key=lambda r: r.child_score, reverse=True)

        top_k = scored[: self.top_k]
        bottom_k = scored[-self.bottom_k :] if len(scored) > self.bottom_k else []

        # Get current best program
        best_prog = database.get_best_program()
        best_code = best_prog.code[:3000] if best_prog else "(no best program yet)"
        best_score = 0.0
        if best_prog and best_prog.metrics:
            from reps.utils import safe_numeric_average
            best_score = best_prog.metrics.get(
                "combined_score", safe_numeric_average(best_prog.metrics)
            )

        # Build the prompt
        top_k_section = "\n\n".join(_format_candidate(r, i + 1) for i, r in enumerate(top_k))
        bottom_k_section = (
            "\n\n".join(_format_candidate(r, i + 1) for i, r in enumerate(bottom_k))
            if bottom_k
            else "(no bottom candidates)"
        )

        prev_str = json.dumps(previous_reflection, indent=2) if previous_reflection else "None"

        prompt = REFLECTION_PROMPT.format(
            best_program=best_code,
            best_score=f"{best_score:.6f}",
            top_k_section=top_k_section,
            bottom_k_section=bottom_k_section,
            previous_reflection=prev_str,
        )

        try:
            gen_kwargs: Dict[str, Any] = {}
            if self.model_override:
                gen_kwargs["model"] = self.model_override
            response = await self.llm.generate(prompt, **gen_kwargs)
            self._call_count += 1
            # Track reflection token costs
            usage = getattr(self.llm, "last_usage", {})
            self._total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)

            reflection = self._parse_reflection(response)
            self._current_reflection = reflection
            return reflection
        except Exception as e:
            logger.warning(f"Reflection LLM call failed: {e}")
            return previous_reflection or {}

    def _parse_reflection(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured reflection dict."""
        # Try to extract JSON from the response
        text = response.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse reflection response as JSON")
                    return self._fallback_reflection(response)
            else:
                return self._fallback_reflection(response)

        # Validate expected fields
        return {
            "working_patterns": data.get("working_patterns", []),
            "failing_patterns": data.get("failing_patterns", []),
            "hypotheses": data.get("hypotheses", []),
            "suggested_directions": data.get("suggested_directions", []),
        }

    def _fallback_reflection(self, raw: str) -> Dict[str, Any]:
        """Create a minimal reflection from unparseable response."""
        return {
            "working_patterns": [],
            "failing_patterns": [],
            "hypotheses": [f"(raw reflection) {raw[:500]}"],
            "suggested_directions": [],
        }

    def format_for_prompt(self, reflection: Optional[Dict[str, Any]] = None) -> str:
        """Format reflection dict as text suitable for injection into prompts."""
        r = reflection or self._current_reflection
        if not r:
            return ""

        lines = ["## Search Reflection (from previous batch analysis)"]

        if r.get("working_patterns"):
            lines.append("\n**What's working:**")
            for p in r["working_patterns"][:5]:
                lines.append(f"- {p}")

        if r.get("failing_patterns"):
            lines.append("\n**What's NOT working (avoid these):**")
            for p in r["failing_patterns"][:5]:
                lines.append(f"- {p}")

        if r.get("hypotheses"):
            lines.append("\n**Hypotheses:**")
            for h in r["hypotheses"][:3]:
                lines.append(f"- {h}")

        if r.get("suggested_directions"):
            lines.append("\n**Suggested directions:**")
            for d in r["suggested_directions"][:5]:
                lines.append(f"- {d}")

        return "\n".join(lines)
