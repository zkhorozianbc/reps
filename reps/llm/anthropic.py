"""
Native Anthropic API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic

from reps.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """LLM interface using the native Anthropic Python SDK"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        # Strip provider prefix: "anthropic/claude-sonnet-4.6" -> "claude-sonnet-4.6"
        raw_name = model_cfg.name
        self.model = raw_name.split("/", 1)[1] if "/" in raw_name else raw_name

        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = model_cfg.api_key

        # Set up Anthropic client
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Token usage from last API call (for cost tracking)
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized AnthropicLLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Build parameters for the Anthropic API
        # Anthropic uses `system` as a top-level parameter, not in the messages list
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Anthropic API requires system as a list of content blocks
        if system_message:
            params["system"] = [{"type": "text", "text": system_message}]

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        if self.client is None:
            raise RuntimeError("Anthropic client is not initialized")

        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )

        # Capture token usage for cost tracking
        # Anthropic uses input_tokens/output_tokens; normalize to prompt_tokens/completion_tokens
        if hasattr(response, "usage") and response.usage is not None:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        else:
            self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.content[0].text}")
        return response.content[0].text
