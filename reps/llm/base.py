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


_NON_RETRYABLE_STATUS = frozenset({
    400,  # Bad request (malformed payload)
    401,  # Unauthorized (bad key)
    402,  # Payment required / insufficient credits
    403,  # Forbidden
    404,  # Not found (bad model name)
    422,  # Unprocessable entity
})


def _is_non_retryable(exc: BaseException) -> bool:
    """Return True for HTTP errors that won't resolve by retrying."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in _NON_RETRYABLE_STATUS:
        return True
    # openai SDK raises APIStatusError subclasses with .status_code; catch
    # nested responses too.
    resp = getattr(exc, "response", None)
    if resp is not None:
        s = getattr(resp, "status_code", None)
        if s in _NON_RETRYABLE_STATUS:
            return True
    # Fallback: scan the message for a code hint.
    msg = str(exc)
    if "Error code: 402" in msg or "Error code: 401" in msg or "Error code: 400" in msg:
        return True
    return False


async def call_with_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    retries: int,
    retry_delay: float,
    timeout: float,
) -> Any:
    """Call a coroutine factory with timeout + simple linear-delay retries.

    Retries only for transient errors (timeouts, 5xx, network). Non-retryable
    errors (auth, billing, bad request) fail fast so we don't burn time + more
    billable calls on something that will never succeed.

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
            if _is_non_retryable(e):
                logger.error(f"Non-retryable error, failing immediately: {e}")
                raise
            if attempt >= retries:
                logger.error(f"All {retries + 1} attempts failed with error: {e}")
                raise
            logger.warning(f"Error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying...")
            await asyncio.sleep(retry_delay)
