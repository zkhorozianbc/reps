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

        if not is_reasoning:
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["temperature"] = temperature

        if system_message:
            params["system"] = [{"type": "text", "text": system_message}]

        return await call_with_retry(
            lambda: self._call_api(params),
            retries=kwargs.get("retries", self.retries),
            retry_delay=kwargs.get("retry_delay", self.retry_delay),
            timeout=kwargs.get("timeout", self.timeout),
        )

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )

        usage = getattr(response, "usage", None)
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
                logger.debug(
                    f"Cache: {cache_read} read, {cache_creation} created, {prompt_tokens} uncached"
                )
        else:
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.content[0].text}")
        return response.content[0].text
