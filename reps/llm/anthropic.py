"""
Native Anthropic API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic

from reps.llm.base import LLMInterface, call_with_retry

logger = logging.getLogger(__name__)

_logged_models: set = set()

# Reasoning models reject the temperature param.
REASONING_MODEL_PATTERNS = ("opus-4-7", "opus-4-8", "opus-4-9", "opus-5")


def _to_int(v: Any) -> int:
    return v if isinstance(v, int) else 0


class AnthropicLLM(LLMInterface):
    """LLM interface using the native Anthropic Python SDK"""

    # Per-class override of the module-level REASONING_MODEL_PATTERNS. The
    # LLM wrapper treats ALL current Claude models as temperature-accepting
    # (opus-4-7 accepts temperature despite being module-level-flagged as a
    # reasoning model — the module-level tuple is retained for
    # tool-runner consumers that use it to gate extended-thinking config).
    # Leaving this empty means `generate_with_context` always passes
    # `temperature`, restoring the REPS worker-diversity dial.
    REASONING_MODEL_PATTERNS: tuple = ()

    # Modern Claude models (4.6+) control thinking via the `effort` parameter
    # passed through `output_config`, not a manual `budget_tokens`. Accepted
    # values: "low", "medium", "high", "xhigh", "max". "xhigh" and "max" are
    # Opus-4.7-only per Anthropic docs; we pass through and let the API 400 on
    # mismatch (our non-retryable classifier surfaces it cleanly).
    VALID_EFFORT = {"low", "medium", "high", "xhigh", "max"}

    def __init__(self, model_cfg: Optional[dict] = None):
        raw_name = model_cfg.name
        self.model = raw_name.split("/", 1)[1] if "/" in raw_name else raw_name

        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = model_cfg.api_key
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves
        )

        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if self.model not in _logged_models:
            logger.info(f"Initialized AnthropicLLM with model: {self.model}")
            _logged_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        model_name = kwargs.get("model") or self.model
        is_reasoning = any(p in model_name.lower() for p in REASONING_MODEL_PATTERNS)

        params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Modern Claude 4.6+ controls thinking depth via the `effort` parameter
        # passed in `output_config`. Adaptive thinking (`thinking.type = adaptive`)
        # lets the model decide how deeply to think based on the effort level —
        # no manual budget_tokens. `budget_tokens` is rejected by Opus 4.7.
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort:
            if effort not in self.VALID_EFFORT:
                raise ValueError(
                    f"reasoning effort must be one of {sorted(self.VALID_EFFORT)}, got {effort!r}"
                )
            params["output_config"] = {"effort": effort}
            params["thinking"] = {"type": "adaptive"}

        if not is_reasoning:
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["temperature"] = temperature

        if system_message:
            # The system prompt is stable across iterations in this project, so
            # marking it with cache_control lets Anthropic's prompt caching kick
            # in and dramatically reduces token cost for repeated calls.
            params["system"] = [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return await call_with_retry(
            lambda: self._call_api(params),
            retries=kwargs.get("retries", self.retries),
            retry_delay=kwargs.get("retry_delay", self.retry_delay),
            timeout=kwargs.get("timeout", self.timeout),
        )

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Stream the Messages API, print each completed block live, return
        the accumulated answer text once the stream closes.

        Uses `client.messages.stream()` which hands us typed SSE events. We
        listen for `content_block_stop` events — each marks the end of a
        thinking or text block — and emit the whole block in one tidy
        printout. Final usage + reasoning are grabbed from `get_final_message`
        after the context exits.
        """
        loop = asyncio.get_running_loop()
        answer, reasoning, usage = await loop.run_in_executor(
            None, lambda: self._stream_and_collect(params)
        )

        if usage is not None:
            prompt_tokens = _to_int(getattr(usage, "input_tokens", 0))
            completion_tokens = _to_int(getattr(usage, "output_tokens", 0))
            cache_creation = _to_int(getattr(usage, "cache_creation_input_tokens", 0))
            cache_read = _to_int(getattr(usage, "cache_read_input_tokens", 0))
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }
            if cache_creation or cache_read:
                logger.info(
                    f"Cache: {cache_read} read, {cache_creation} created, {prompt_tokens} uncached"
                )
        else:
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self.last_reasoning = reasoning or None
        logger.debug(f"API parameters: {params}")
        return answer

    def _stream_and_collect(self, params: Dict[str, Any]):
        """Blocking: stream, print paragraphs as they complete, return
        (answer, reasoning, usage) at end.

        Uses the Anthropic SDK's synthesized `text` / `thinking` delta events
        to emit progress every `\\n\\n` (paragraph boundary) rather than
        waiting for the whole block to finish. For a long reasoning block,
        this means visible progress every few seconds instead of silent
        multi-minute waits.
        """
        from reps.llm.stream_print import emit_block, emit_status

        emit_status(params.get("model", "?"))
        answer_parts: List[str] = []
        reasoning_parts: List[str] = []

        mode: Optional[str] = None  # "answer" | "thinking"
        buf = ""

        def flush_paragraphs():
            nonlocal buf
            while "\n\n" in buf:
                para, _, buf = buf.partition("\n\n")
                if para.strip():
                    emit_block(mode, para)

        def flush_remaining():
            nonlocal buf
            if buf.strip():
                emit_block(mode, buf)
            buf = ""

        with self.client.messages.stream(**params) as stream:
            for event in stream:
                et = getattr(event, "type", None)

                if et == "thinking":
                    piece = getattr(event, "thinking", "") or ""
                    if piece:
                        if mode != "thinking":
                            flush_remaining()
                            mode = "thinking"
                        buf += piece
                        reasoning_parts.append(piece)
                        flush_paragraphs()

                elif et == "text":
                    piece = getattr(event, "text", "") or ""
                    if piece:
                        if mode != "answer":
                            flush_remaining()
                            mode = "answer"
                        buf += piece
                        answer_parts.append(piece)
                        flush_paragraphs()

                elif et == "content_block_stop":
                    # Block finished — flush remaining partial as the tail.
                    flush_remaining()
                    mode = None

            final_msg = stream.get_final_message()

        # Final flush for any trailing partial (usually already flushed).
        flush_remaining()

        usage = getattr(final_msg, "usage", None)
        answer = "".join(answer_parts)
        reasoning = "".join(reasoning_parts).strip() or None
        return answer, reasoning, usage
