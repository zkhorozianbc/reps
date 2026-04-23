"""reps.workers — Worker primitive and implementations."""
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
from reps.workers import dspy_react  # noqa: F401
