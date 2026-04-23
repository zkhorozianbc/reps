"""Per-call program-id helper used by benchmark evaluators.

Under asyncio, multiple iterations run concurrently in one process. Each call
to Evaluator.evaluate_isolated sets the contextvar below; asyncio propagates
ContextVar state per-Task so concurrent calls see independent values.

Benchmarks that previously read os.environ["REPS_PROGRAM_ID"] should prefer
this helper. os.environ is no longer mutated by the evaluator in the asyncio
world — the env var is only set inside the subprocess child via Popen(env=...).
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

_program_id_var: ContextVar[Optional[str]] = ContextVar("reps_program_id", default=None)


def set_current_program_id(program_id: Optional[str]):
    """Set the current program_id for the calling asyncio Task. Returns a token
    the caller must pass to reset_current_program_id in a `finally` block."""
    return _program_id_var.set(program_id)


def reset_current_program_id(token) -> None:
    _program_id_var.reset(token)


def current_program_id() -> Optional[str]:
    """Return the current program_id for this Task. Falls back to
    os.environ['REPS_PROGRAM_ID'] for backwards compatibility inside
    subprocess children that inherit the env var."""
    pid = _program_id_var.get()
    if pid is not None:
        return pid
    return os.environ.get("REPS_PROGRAM_ID")
