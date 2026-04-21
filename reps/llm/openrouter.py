"""
OpenRouter / OpenAI-compatible API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import openai

from reps.llm.base import LLMInterface, call_with_retry

logger = logging.getLogger(__name__)

_logged_models: set = set()


class OpenRouterLLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs (including OpenRouter)"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Set up API client
        # OpenAI client requires max_retries to be int, not None
        max_retries = self.retries if self.retries is not None else 0
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=max_retries,
        )

        # Token usage from last API call (for cost tracking)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if self.model not in _logged_models:
            logger.info(f"Initialized OpenRouterLLM with model: {self.model}")
            _logged_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    # Reasoning models use max_completion_tokens and reject temperature/top_p.
    OPENAI_REASONING_MODEL_PREFIXES = (
        "o1-", "o1", "o3-", "o3", "o4-",
        "gpt-5-", "gpt-5",
        "gpt-oss-120b", "gpt-oss-20b",
    )
    GOOGLE_AI_STUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        # Only prepend the system role when we actually have a system message.
        # Some providers reject {"role": "system", "content": null/""}.
        if system_message:
            formatted_messages = [{"role": "system", "content": system_message}]
            formatted_messages.extend(messages)
        else:
            formatted_messages = list(messages)

        model_base = str(self.model).lower().split("/")[-1]
        is_reasoning = model_base.startswith(self.OPENAI_REASONING_MODEL_PREFIXES)
        is_openrouter = self.api_base and "openrouter.ai" in self.api_base

        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)

        if is_reasoning:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

        if reasoning_effort is not None:
            if is_openrouter:
                params["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
            else:
                params["reasoning_effort"] = reasoning_effort

        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == self.GOOGLE_AI_STUDIO_BASE:
                logger.warning("Google AI Studio does not support seed; reproducibility may be limited.")
            else:
                params["seed"] = seed

        return await call_with_retry(
            lambda: self._call_api(params),
            retries=kwargs.get("retries", self.retries),
            retry_delay=kwargs.get("retry_delay", self.retry_delay),
            timeout=kwargs.get("timeout", self.timeout),
        )

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Call the Chat Completions endpoint, streaming so we can live-log
        reasoning and content as they arrive. Full message is still returned
        to the caller as a single string once the stream completes.

        Output is written to stderr with an `[or pid=NNNN think]` or
        `[or pid=NNNN answer]` prefix, line-buffered so parallel workers
        interleave cleanly. The run.log file-handler does NOT get these
        chunks — we use `print(..., file=sys.stderr)` directly to keep the
        structured log readable.
        """
        stream_params = {
            **params,
            "stream": True,
            # Usage only arrives in the final chunk when this is set.
            "stream_options": {"include_usage": True},
        }
        loop = asyncio.get_running_loop()
        content, reasoning, usage = await loop.run_in_executor(
            None, lambda: self._stream_and_collect(stream_params)
        )

        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            ptd = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(ptd, "cached_tokens", 0) or 0 if ptd is not None else 0
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            }
            if cached:
                logger.info(f"Cache: {cached} read, {prompt_tokens - cached} uncached")
        else:
            self.last_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }

        self.last_reasoning = reasoning or None
        self.last_reasoning_details = None  # deltas only expose .reasoning strings

        logger.debug(f"API parameters: {params}")
        return content

    def _stream_and_collect(self, params: Dict[str, Any]):
        """Blocking: iterate the SSE stream, print each completed block live
        (not every chunk), return buffers.

        OpenRouter/OpenAI streams deltas under `delta.content` (answer) and
        `delta.reasoning` (thinking). There's no content_block_stop marker, so
        we treat a switch from one mode to the other (or stream end) as the
        end of a block and emit the whole block in one tidy printout.
        """
        from reps.llm.stream_print import emit_block, emit_status

        emit_status(params.get("model", "?"))
        stream = self.client.chat.completions.create(**params)
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage = None

        mode: Optional[str] = None   # "answer" | "thinking"
        buf: List[str] = []

        def flush():
            nonlocal buf
            if mode and buf:
                emit_block(mode, "".join(buf))
            buf = []

        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = u
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta

            r_piece = getattr(delta, "reasoning", None) or ""
            if r_piece:
                if mode != "thinking":
                    flush()
                    mode = "thinking"
                buf.append(r_piece)
                reasoning_parts.append(r_piece)

            c_piece = getattr(delta, "content", None) or ""
            if c_piece:
                if mode != "answer":
                    flush()
                    mode = "answer"
                buf.append(c_piece)
                content_parts.append(c_piece)

        flush()
        return "".join(content_parts), "".join(reasoning_parts), usage
