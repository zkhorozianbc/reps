"""Public Python API surface for REPS.

Re-exports the v1 user-facing classes (`LM`, `REPS`, `OptimizationResult`,
`EvaluationResult`). Internals stay in their existing modules; this package
is a thin facade.
"""

from reps.api.lm import LM

__all__ = ["LM"]

