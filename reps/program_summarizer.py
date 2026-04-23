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
import os
from typing import Any, Dict, List, Optional

from reps.workers.trace_render import render_trace_from_dicts

logger = logging.getLogger(__name__)

_MAX_REASONING_CHARS = 12000  # truncate very long traces before sending


def build_summarizer_llm(cfg) -> "SummarizerLLM":
    """Build a dedicated LLM client for the per-program summarizer.

    Takes a `REPSSummarizerConfig` (or any duck-typed object exposing the
    same attributes) and returns a small wrapper that knows how to call
    the right provider endpoint. Independent of the worker ensemble — so
    the summarizer works regardless of what models workers use.

    Fails LOUDLY at construction time if the provider/api_key cannot be
    resolved. This is intentional: a silent fallback here would hide
    real config mistakes for the rest of the run.
    """
    from reps.llm.provider_of import provider_of_model

    model_id = cfg.model_id
    provider = cfg.provider or provider_of_model(model_id)

    # Resolve api_key with a provider-appropriate env-var default.
    api_key = cfg.api_key
    if not api_key:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        raise ValueError(
            f"summarizer: provider={provider!r} but no api_key set "
            f"(neither reps.summarizer.api_key nor {env_var} in env)"
        )

    return SummarizerLLM(
        model_id=model_id,
        provider=provider,
        api_key=api_key,
        api_base=cfg.api_base,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        retries=cfg.retries,
        retry_delay=cfg.retry_delay,
    )


class SummarizerLLM:
    """Thin summarizer-only LLM wrapper.

    For anthropic models, delegates to `AnthropicLLM` (streaming + cache
    control etc.). For openai gpt-* / o*-* models, keeps the Responses
    API shortcut (which is what gpt-5.4-pro requires — chat.completions
    returns 404 for that model). Not a general-purpose client — the
    worker ensemble is for that.
    """

    def __init__(
        self,
        *,
        model_id: str,
        provider: str,
        api_key: str,
        api_base: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
        retries: int,
        retry_delay: int,
    ) -> None:
        self.model_id = model_id
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = self._build_client()

    def _build_client(self):
        """Lazily construct the provider-specific client.

        For anthropic we build an `AnthropicLLM` from the provider
        module (same retry/timeout behavior as worker calls). For
        openai we defer: `_openai_responses_call` creates the
        `AsyncOpenAI` inline since it uses the Responses API, not
        chat.completions.
        """
        if self.provider == "anthropic":
            from reps.config import LLMModelConfig
            from reps.llm.anthropic import AnthropicLLM

            cfg = LLMModelConfig(
                name=self.model_id,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                retries=self.retries,
                retry_delay=self.retry_delay,
                provider="anthropic",
            )
            return AnthropicLLM(cfg)
        if self.provider == "openai":
            # OpenAI path uses the Responses API — no persistent client
            # needed; `_openai_responses_call` constructs AsyncOpenAI per
            # call. We still validate the key is present here.
            return None
        raise ValueError(f"summarizer: unknown provider {self.provider!r}")

    async def call(self, *, system_prompt: str, user_text: str) -> str:
        """Invoke the summarizer. Returns the raw response text."""
        if self.provider == "anthropic":
            return await self._client.generate_with_context(
                system_message=system_prompt,
                messages=[{"role": "user", "content": user_text}],
                temperature=self.temperature,
            )
        if self.provider == "openai":
            return await _openai_responses_call(
                model_id=self.model_id.removeprefix("openai/"),
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_text=user_text,
            )
        raise ValueError(f"summarizer: unknown provider {self.provider!r}")


async def _openai_responses_call(
    *, model_id: str, api_key: str, system_prompt: str, user_text: str
) -> str:
    """Direct OpenAI Responses API call for gpt-* summarizer models.

    Uses `instructions=system_prompt` + `input=user_text` shape and returns
    `response.output_text`. Kept intentionally minimal — no streaming, no
    reasoning effort knob, no retries beyond what the SDK does internally.
    """
    from openai import AsyncOpenAI  # lazy import; only needed when routed here
    client = AsyncOpenAI(api_key=api_key)
    response = await client.responses.create(
        model=model_id,
        instructions=system_prompt,
        input=user_text,
    )
    return response.output_text or ""

# General role + strict rules. Generalizes to any benchmark. Cache-stable.
_SYSTEM_PROMPT = """You are a technical analyst extracting actionable insights from code mutation attempts in an evolutionary search.

Your role is to read an agent's full reasoning + tool trace from one iteration of evolution — along with the code it submitted and its score — and produce a structured JSON summary that helps subsequent iterations learn from this attempt.

STRICT RULES:
1. Output ONLY a JSON object. No prose before or after. No markdown fences.
2. Do NOT speculate that the scoring / validator / evaluator system itself is broken. If a score is 0 or low, assume the code failed to meet the benchmark's constraints (invalid output, runtime error, constraint violation) — the evaluator is working as designed.
3. Be SPECIFIC. "The code failed" is useless; "the sort key returned None for empty inputs" is useful. Cite line-of-reasoning or concrete numbers when possible.
4. Pitfalls must be OBSERVED in the trace — bugs the agent actually hit, error messages it actually saw. Do not invent plausible-sounding bugs. If the trace shows no explicit failure, set pitfalls to [].
5. `key_insight` is one line capturing the most transferable lesson from this attempt. If nothing notable, use "none".

OUTPUT SCHEMA (exactly these keys, no others):
{
  "approach": "string, one line describing the mutation attempted",
  "pitfalls": ["up to 5 strings, each a specific observed failure"],
  "key_insight": "one line or \\"none\\""
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
    summarizer_llm: "SummarizerLLM",
    task_instructions: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call the configured summarizer model to produce a structured per-program summary.

    Returns None on failure; callers should treat it as best-effort and
    continue without a summary. The summarizer uses its OWN LLM client
    (`summarizer_llm`) — not the worker ensemble — so its model is
    independent of what workers are configured with.

    Args:
        summarizer_llm: The dedicated summarizer LLM built at controller
            startup via `build_summarizer_llm`. Failing to construct this
            should surface at config-time, not here.
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
        resp_text = await summarizer_llm.call(
            system_prompt=_SYSTEM_PROMPT,
            user_text=user_text,
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
        "pitfalls": [str(p)[:500] for p in (data.get("pitfalls") or [])[:5]],
        "key_insight": str(data.get("key_insight", ""))[:800],
    }
    return out


def format_summary_for_prompt(summary: Dict[str, Any], label: str = "Parent's notebook") -> str:
    """Render a summary dict into a compact block suitable for injection
    into the next iteration's user prompt.

    Emits exactly the three fields in the schema: `approach`, `key_insight`
    (when present and not "none"), and `pitfalls` (only when non-empty).
    """
    if not summary:
        return ""
    lines = [f"## {label}"]
    if summary.get("approach"):
        lines.append(f"- Approach: {summary['approach']}")
    key_insight = summary.get("key_insight")
    if (
        isinstance(key_insight, str)
        and key_insight.strip()
        and key_insight.strip().lower() != "none"
    ):
        lines.append(f"- Key insight: {key_insight}")
    if summary.get("pitfalls"):
        lines.append("- Pitfalls hit:")
        lines.extend(f"  - {p}" for p in summary["pitfalls"])
    return "\n".join(lines)
