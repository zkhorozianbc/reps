"""
Async utilities for REPS.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


def run_in_executor(f: Callable) -> Callable:
    """Decorator that runs a blocking function in the default executor."""

    @functools.wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


class TaskPool:
    """A simple bounded-concurrency task pool."""

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.tasks: List[asyncio.Task] = []

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:
        async with self.semaphore:
            return await coro(*args, **kwargs)

    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task

    async def wait_all(self) -> None:
        if self.tasks:
            await asyncio.gather(*self.tasks)

    async def cancel_all(self) -> None:
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
