"""Internal re-exports for users who imported from the previous flat surface.

Before the v1 Python API landed, REPS only had a flat namespace:
`reps.ReflectionEngine`, `reps.WorkerPool`, etc. Power users built on
those directly. The v1 API package narrows `reps.*` to the public
facade (`LM`, `REPS`, `OptimizationResult`, `EvaluationResult`); this
sub-package keeps the old names reachable for back-compat.

`reps.internal.*` is *not* part of the v1 stable API — it's the escape
hatch for advanced users until they migrate. Symbols here may move,
rename, or change shape between releases without deprecation warnings.

Today's contents track `reps/__init__.py`'s prior export list.
"""

from reps.iteration_config import IterationConfig, IterationResult
from reps.reflection_engine import ReflectionEngine
from reps.worker_pool import WorkerPool
from reps.convergence_monitor import ConvergenceMonitor, ConvergenceAction
from reps.contract_selector import ContractSelector, Contract
from reps.sota_controller import SOTAController, SearchRegime
from reps.metrics_logger import MetricsLogger

__all__ = [
    "IterationConfig",
    "IterationResult",
    "ReflectionEngine",
    "WorkerPool",
    "ConvergenceMonitor",
    "ConvergenceAction",
    "ContractSelector",
    "Contract",
    "SOTAController",
    "SearchRegime",
    "MetricsLogger",
]
