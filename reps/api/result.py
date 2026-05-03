"""`OptimizationResult` — the v1 return shape from `reps.REPS.optimize()`.

Plain dataclass; no `save`/`load`/`history`/`as_callable` (those defer
to v1.5 — see docs/python_api_spec.md "Deferred to v1.5+").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OptimizationResult:
    """Outcome of one `reps.REPS.optimize()` call.

    Fields mirror the spec exactly — best_* fields come from the database's
    tracked best Program; iterations_run is the number of completed
    iterations (excluding the seed eval); total_metric_calls counts all
    evaluator invocations during the run; total_tokens aggregates
    prompt + completion tokens across worker LLM calls.
    """
    best_code: str
    best_score: float
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_per_instance_scores: Optional[Dict[str, float]] = None
    best_feedback: Optional[str] = None
    iterations_run: int = 0
    total_metric_calls: int = 0
    total_tokens: Dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0})
    output_dir: Optional[str] = None
