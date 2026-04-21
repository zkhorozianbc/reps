"""
Base LLM interface and shared retry helper.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str: ...


async def call_with_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    retries: int,
    retry_delay: float,
    timeout: float,
) -> Any:
    """Call a coroutine factory with timeout + simple linear-delay retries.

    Reraises the last exception after `retries + 1` failed attempts.
    """
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout)
        except asyncio.TimeoutError:
            if attempt >= retries:
                logger.error(f"All {retries + 1} attempts failed with timeout")
                raise
            logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            if attempt >= retries:
                logger.error(f"All {retries + 1} attempts failed with error: {e}")
                raise
            logger.warning(f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying...")
            await asyncio.sleep(retry_delay)
