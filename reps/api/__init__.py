"""Public Python API surface for REPS.

Re-exports the v1 user-facing classes. Internals stay in their existing
modules; this package is a thin facade.
"""

from reps.api.example import Example, Prediction
from reps.api.model import Model, ModelKwargs
from reps.api.objective import LLMJudge, Objective
from reps.api.optimizer import Optimizer
from reps.api.result import OptimizationResult
from reps.evaluation_result import EvaluationResult

__all__ = [
    "Example",
    "Prediction",
    "Objective",
    "LLMJudge",
    "Model",
    "ModelKwargs",
    "Optimizer",
    "OptimizationResult",
    "EvaluationResult",
]
