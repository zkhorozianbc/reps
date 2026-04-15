"""
REPS: Recursive Evolutionary Program Search

Extension modules for OpenEvolve that add:
- F1: Reflection Engine (post-batch self-reflection)
- F2: ε-Revisitation (revisit underexplored parents)
- F3: Worker Type Diversity (exploiter/explorer/crossover)
- F4: Convergence Monitor (edit entropy + strategy divergence)
- F5: Intelligence Contracts (Thompson-sampling model selection)
- F6: SOTA-Distance Steering (gap-aware search modulation)
- F7: Compute Signature Tracking
- F8: Enriched Program Annotations
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
