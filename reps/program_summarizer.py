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


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object extractor.

    Tries, in order:
      1. Parse the raw text.
      2. Strip a leading/trailing ```json … ``` fence.
      3. Find the first balanced {…} block and parse that.

    Returns the parsed dict, or None if nothing recognizable was found.
    Production-grade summarizer parsing: models occasionally prepend
    chatter ("Here's the JSON: …") or wrap output in fences even when
    told not to. Don't let that cost an iteration's annotation.
    """
    if not text:
        return None
    s = text.strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    if s.startswith("```"):
        s2 = s.strip("`")
        if s2.lower().startswith("json"):
            s2 = s2[4:].lstrip()
        if s2.endswith("```"):
            s2 = s2[:-3]
        s2 = s2.strip()
        try:
            obj = json.loads(s2)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


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
    _PROVIDER_ENV = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = _PROVIDER_ENV.get(provider, "ANTHROPIC_API_KEY")
    api_key = cfg.api_key or os.environ.get(env_var)
    if not api_key:
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
        if self.provider == "openrouter":
            from reps.config import LLMModelConfig
            from reps.llm.openai_compatible import OpenAICompatibleLLM

            cfg = LLMModelConfig(
                name=self.model_id,
                api_key=self.api_key,
                api_base=self.api_base or "https://openrouter.ai/api/v1",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                retries=self.retries,
                retry_delay=self.retry_delay,
                provider="openrouter",
            )
            return OpenAICompatibleLLM(cfg)
        raise ValueError(f"summarizer: unknown provider {self.provider!r}")

    async def call(self, *, system_prompt: str, user_text: str) -> str:
        """Invoke the summarizer. Returns the raw response text."""
        if self.provider in ("anthropic", "openrouter"):
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
    # The user message ends with an explicit final directive ("output the JSON
    # now") so the model has a clean instruction-shaped boundary to respond
    # to — not the trailing line of a code block, which some chat-tuned models
    # (especially Sonnet routed via OpenRouter) will happily continue instead
    # of summarizing. The trace and code are wrapped in <child_code> and
    # <trace> sentinels so the model can't confuse them with their own
    # response surface.
    parts: list[str] = []
    if task_instructions:
        parts.append("## Benchmark-specific guidance")
        parts.append(task_instructions.strip())
        parts.append("")
    parts.append("## Attempt data")
    parts.append(f"Parent score (combined_score): {parent_score:.6f}")
    parts.append(f"Child score (combined_score): {child_score:.6f}")
    parts.append(f"Improved: {improved}")
    parts.append("")
    parts.append("<child_code>")
    parts.append(code[:3000])
    parts.append("</child_code>")
    parts.append("")
    parts.append("<trace>")
    parts.append(reasoning)
    parts.append("</trace>")
    parts.append("")
    parts.append(
        "Now output the JSON summary described in the system prompt — "
        "a single JSON object with keys `approach`, `pitfalls`, "
        "`key_insight`. No prose before or after the JSON. No markdown."
    )
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

    data = _extract_json_object(resp_text or "")
    if data is None:
        # Log the first ~200 chars of what we got so failures are diagnosable
        # without re-running. INFO, not WARNING: this is a recoverable
        # best-effort annotation that the run continues without.
        preview = (resp_text or "")[:200].replace("\n", " ")
        logger.info(
            f"program summarizer produced no parseable JSON for "
            f"{program_id[:8]}; first 200 chars: {preview!r}"
        )
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
