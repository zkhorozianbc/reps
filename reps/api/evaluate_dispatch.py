"""Bridge user-supplied `evaluate(code: str)` callables into the path-based
contract `Evaluator` expects.

REPS internal `Evaluator._load_evaluation_function` imports an `evaluate`
function from a Python file and calls it with a temp file path. Users of
the public API give us a callable that accepts the program text directly
(`evaluate(code: str) -> float | dict | EvaluationResult`).

We bridge the two by writing a tiny shim `_reps_user_evaluator.py` to the
run's output directory. The shim reads a registry id from `os.environ`,
looks up the registered callable, and dispatches:

    1. Read the program text from the temp file.
    2. Call the user's `evaluate(code, **kwargs)` — `kwargs` populated via
       `inspect.signature` so legacy `evaluate(code, *, env=None,
       instances=None)` evaluators keep working.
    3. Coerce the return into a `dict` consumed by the rest of the
       pipeline:
         - `float`  -> {"combined_score": ..., "validity": 1.0}
         - `dict`   -> as-is
         - `EvaluationResult` -> EvaluationResult (passes through)

The registry is process-local — the asyncio runner stays in one process,
so this is safe. If REPS ever forks workers, the registry hands off via
the env-var registry id, but the callable would need to be re-registered
in the child (out of scope for v1, which is asyncio-only).
"""

from __future__ import annotations

import inspect
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from reps.evaluation_result import EvaluationResult


# Process-local registry. Keys are short hex ids; values are the user's
# evaluate callable. We keep this module-level (no global mutable state
# leaked outside the API surface) so the shim file can look it up by id.
_REGISTRY: Dict[str, Callable[..., Any]] = {}
_REGISTRY_LOCK = threading.Lock()
_REGISTRY_ENV_VAR = "REPS_USER_EVALUATOR_ID"


# Shim source written verbatim to <output_dir>/_reps_user_evaluator.py.
# It must be self-contained (only imports stdlib + reps.api.evaluate_dispatch).
_SHIM_SOURCE = '''"""Auto-generated shim — bridges the path-based Evaluator contract to
a user-supplied `evaluate(code: str)` callable. Do not edit by hand.

This file is regenerated each `reps.Optimizer.optimize()` call.
"""
import os

from reps.api.evaluate_dispatch import dispatch_user_evaluate


def evaluate(program_path, **kwargs):
    registry_id = os.environ["REPS_USER_EVALUATOR_ID"]
    return dispatch_user_evaluate(registry_id, program_path, **kwargs)
'''


def register_user_evaluate(fn: Callable[..., Any]) -> str:
    """Register a user evaluate callable; return a short id for the env var.

    The id is uuid4-derived (12 hex chars) — collisions across concurrent
    `optimize()` calls in the same process are vanishingly improbable.
    """
    rid = uuid.uuid4().hex[:12]
    with _REGISTRY_LOCK:
        _REGISTRY[rid] = fn
    return rid


def unregister_user_evaluate(rid: str) -> None:
    """Drop a callable from the registry. Safe to call with an unknown id."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(rid, None)


def write_shim(output_dir: Union[str, Path]) -> str:
    """Write the dispatch shim into `output_dir` and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shim_path = out / "_reps_user_evaluator.py"
    shim_path.write_text(_SHIM_SOURCE)
    return str(shim_path)


def _supported_kwargs(fn: Callable[..., Any]) -> set[str]:
    """Names of optional keyword params the user's `evaluate` accepts.

    The v1 contract is `Callable[[str], ...]`; we additionally pass `env`
    and `instances` keywords when the callable's signature names them, so
    legacy benchmark evaluators that accept these kwargs Just Work.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set()
    return {p.name for p in sig.parameters.values()}


def coerce_return(value: Any) -> Union[Dict[str, Any], EvaluationResult]:
    """Coerce a user evaluate return into the shape `Evaluator` expects.

      float / int  -> {"combined_score": float(value), "validity": 1.0}
      dict         -> as-is (Evaluator wraps it in EvaluationResult.from_dict)
      EvaluationResult -> passed through unchanged
      other        -> ValueError (caller logs + treats as eval failure)
    """
    if isinstance(value, EvaluationResult):
        return value
    if isinstance(value, dict):
        return value
    if isinstance(value, bool):
        # bool is a subclass of int — treat True/False as 1.0 / 0.0 to
        # match the obvious user intent.
        return {"combined_score": float(value), "validity": 1.0}
    if isinstance(value, (int, float)):
        return {"combined_score": float(value), "validity": 1.0}
    raise ValueError(
        "reps.Optimizer.optimize: `evaluate` must return float, dict, or "
        f"EvaluationResult; got {type(value).__name__}"
    )


def dispatch_user_evaluate(registry_id: str, program_path: str, **kwargs: Any) -> Any:
    """Called by the auto-generated shim.

    Reads program text, forwards only the kwargs the user's signature
    declares, and coerces the return.
    """
    with _REGISTRY_LOCK:
        fn = _REGISTRY.get(registry_id)
    if fn is None:
        raise RuntimeError(
            f"reps.api.evaluate_dispatch: no user evaluate callable registered "
            f"for id {registry_id!r}. Did the optimize() call exit before this "
            f"subprocess ran?"
        )

    code = Path(program_path).read_text()

    accepted = _supported_kwargs(fn)
    forwarded: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in accepted:
            forwarded[k] = v

    raw = fn(code, **forwarded)
    return coerce_return(raw)
