"""Per-call helpers exposed to evolved code.

Under asyncio, multiple iterations run concurrently in one process. The
contextvars below propagate per-Task so concurrent evaluations see
independent values; executor threads inherit them via `context.run(...)`
in `reps.evaluator._run_evaluator_callable`.

- `current_program_id()` — the program id the current evaluation is
  scoring (used by benchmark evaluators to write artifacts under a unique
  key).
- `llm(prompt, **kwargs) -> str` — call the LLM the running
  `reps.Optimizer.optimize(...)` configured. Available to candidate
  programs so they can use an LLM at inference time, which is what makes
  prompt-tuning-style optimization possible: REPS evolves the *Python*
  freely (any text/tool-based mutation works), and the evolved code
  decides how to prompt, chain, parse, retry. The harness manages the
  client / credentials / token tracking — the candidate just calls
  `llm("...")`.
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Callable, Optional

_program_id_var: ContextVar[Optional[str]] = ContextVar("reps_program_id", default=None)
_current_llm_var: ContextVar[Optional[Callable[..., str]]] = ContextVar(
    "reps_current_llm", default=None
)


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


def set_current_llm(llm_callable: Callable[..., str]):
    """Set the LLM that `reps.runtime.llm()` calls inside this context. Returns
    a token; caller passes it to `reset_current_llm` in a `finally` block.

    The Optimizer sets this around the optimize() run so candidate programs
    can use an LLM at inference time. The contextvar makes concurrent
    optimize() calls (or nested ones) independent."""
    return _current_llm_var.set(llm_callable)


def reset_current_llm(token) -> None:
    _current_llm_var.reset(token)


def llm(prompt: str, **kwargs: Any) -> str:
    """Call the LLM the running REPS run has configured.

    The candidate (evolved) program calls this freely — once, in a loop,
    chained, with retries, with few-shot embedded, whatever the evolved
    Python decides. The harness owns the client, credentials, and token
    accounting; the candidate owns *what to ask*.

    Raises `RuntimeError` if no LLM is configured — e.g. when the candidate
    is being evaluated outside an active `reps.Optimizer.optimize(...)`
    scope. In that case, configure one explicitly with
    `reps.runtime.set_current_llm(...)` before evaluating.
    """
    fn = _current_llm_var.get()
    if fn is None:
        raise RuntimeError(
            "reps.runtime.llm: no LLM configured for this context. This "
            "callable is meant to run inside REPS' evolved code during "
            "`reps.Optimizer.optimize(...)`. Outside that scope, call "
            "`reps.runtime.set_current_llm(<callable>)` first."
        )
    return fn(prompt, **kwargs)
