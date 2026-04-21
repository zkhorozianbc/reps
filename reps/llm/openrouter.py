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
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            # OpenRouter reports cache stats under prompt_tokens_details.cached_tokens.
            ptd = getattr(usage, "prompt_tokens_details", None)
            cached = 0
            if ptd is not None:
                cached = getattr(ptd, "cached_tokens", 0) or 0
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            }
            if cached:
                logger.info(
                    f"Cache: {cached} read, {prompt_tokens - cached} uncached"
                )
        else:
            self.last_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }

        message = response.choices[0].message
        content = message.content
        # Capture reasoning output if the model produced any (e.g. Opus 4.7 with
        # reasoning_effort set). OpenRouter surfaces it as `reasoning` (string)
        # and/or `reasoning_details` (list of blocks).
        reasoning = getattr(message, "reasoning", None)
        reasoning_details = getattr(message, "reasoning_details", None)
        self.last_reasoning = reasoning or None
        self.last_reasoning_details = reasoning_details or None

        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {content}")
        return content
