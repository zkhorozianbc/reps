"""Public Python API surface for REPS.

Re-exports the v1 user-facing classes (`Model`, `ModelKwargs`, `Optimizer`,
`OptimizationResult`, `EvaluationResult`). Internals stay in their existing
modules; this package is a thin facade.
"""

from reps.api.model import Model, ModelKwargs
from reps.api.optimizer import Optimizer
from reps.api.result import OptimizationResult
from reps.evaluation_result import EvaluationResult
from reps import interpret

__all__ = [
    "Model",
    "ModelKwargs",
    "Optimizer",
    "OptimizationResult",
    "EvaluationResult",
    "interpret",
]
