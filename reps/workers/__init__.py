"""reps.workers — Worker primitive and implementations."""
import warnings

from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    Worker,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)

# Importing impls here triggers @register decorators.
from reps.workers import single_call  # noqa: F401
from reps.workers import anthropic_tool_runner  # noqa: F401
from reps.workers import openai_tool_runner  # noqa: F401

# dspy_react is gated behind the optional [dspy] extra. If dspy is not
# installed, skip registration silently — users requesting `impl: dspy_react`
# in their YAML will get a clear ImportError at use-time from
# reps.controller's lazy `make_dspy_lm` import.
try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The 'prefix' argument in InputField/OutputField is deprecated.*",
            category=DeprecationWarning,
            module=r"dspy\..*",
        )
        from reps.workers import dspy_react  # noqa: F401
except ImportError:
    pass
