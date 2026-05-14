"""`reps.Objective` and `reps.LLMJudge` — the objective layer.

An `Objective` compiles a `(entrypoint, train_set, metric)` triple into the
existing evaluator contract: `Objective.evaluate(code: str) -> EvaluationResult`.
`Optimizer.optimize(objective=...)` registers `objective.evaluate` through the
same dispatch shim it uses for raw `evaluate=` callables.

See docs/objective_api_spec.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from reps.api.example import Example, Prediction, as_prediction
from reps.evaluation_result import EvaluationResult

MetricCallable = Callable[..., Any]

_DIRECTIONS = ("maximize", "minimize")
_NOT_IN_V1 = {"semantic_f1"}


# --- aggregation helpers ----------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rmse_aggregate(squared_errors: Sequence[float]) -> float:
    return math.sqrt(_mean(squared_errors))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- built-in per-example metrics -------------------------------------------


def _accuracy(example: Example, pred: Prediction, trace: Any = None) -> float:
    return 1.0 if pred.get("answer") == example.get("answer") else 0.0


def _exact_match(example: Example, pred: Prediction, trace: Any = None) -> float:
    # Case-sensitive exact match for string answers.
    return 1.0 if str(pred.get("answer")) == str(example.get("answer")) else 0.0


def _abs_error(example: Example, pred: Prediction, trace: Any = None) -> float:
    return abs(float(pred.get("answer")) - float(example.get("answer")))


def _squared_error(example: Example, pred: Prediction, trace: Any = None) -> float:
    diff = float(pred.get("answer")) - float(example.get("answer"))
    return diff * diff


@dataclass(frozen=True)
class _BuiltinMetric:
    per_example: MetricCallable
    aggregate: Callable[[Sequence[float]], float]
    direction: str


_BUILTIN_METRICS: dict[str, _BuiltinMetric] = {
    "accuracy": _BuiltinMetric(_accuracy, _mean, "maximize"),
    "exact_match": _BuiltinMetric(_exact_match, _mean, "maximize"),
    "mae": _BuiltinMetric(_abs_error, _mean, "minimize"),
    "mse": _BuiltinMetric(_squared_error, _mean, "minimize"),
    "rmse": _BuiltinMetric(_squared_error, _rmse_aggregate, "minimize"),
}


def _resolve_metric(
    metric: "str | MetricCallable", direction: str
) -> "tuple[str, MetricCallable, Callable[[Sequence[float]], float]]":
    """Resolve `metric` into `(metric_name, per_example_fn, aggregate_fn)`.

    Built-in metric names are validated against `direction`; custom callables
    inherit `direction` from the calling classmethod with no validation.
    """
    if isinstance(metric, str):
        if metric in _NOT_IN_V1:
            raise ValueError(
                f"reps.Objective: metric {metric!r} is named in the spec but "
                f"not implemented in v1 of the objective layer. Pass a custom "
                f"metric callable `metric(example, pred, trace=None) -> float`."
            )
        builtin = _BUILTIN_METRICS.get(metric)
        if builtin is None:
            raise ValueError(
                f"reps.Objective: unknown metric {metric!r}. Built-in metrics: "
                f"{sorted(_BUILTIN_METRICS)}. Or pass a callable "
                f"`metric(example, pred, trace=None) -> float`."
            )
        if builtin.direction != direction:
            raise ValueError(
                f"reps.Objective: built-in metric {metric!r} is a "
                f"{builtin.direction} metric — use "
                f"`reps.Objective.{builtin.direction}(...)`."
            )
        return metric, builtin.per_example, builtin.aggregate
    if callable(metric):
        return getattr(metric, "__name__", "metric"), metric, _mean
    raise TypeError(
        f"reps.Objective: `metric` must be a built-in metric name (str) or a "
        f"callable, got {type(metric).__name__}"
    )
