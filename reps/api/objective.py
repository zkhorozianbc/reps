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


class Objective:
    """Deterministic objective layer for `reps.Optimizer.optimize(objective=...)`.

    Build one with `Objective.minimize(...)` or `Objective.maximize(...)`:

        objective = reps.Objective.minimize(
            entrypoint="predict",
            train_set=[reps.Example(x=-4, answer=30).with_inputs("x")],
            metric="mae",
        )

    `failure_score` is the per-example metric value (in the metric's natural
    space) assigned when an example's entrypoint call, prediction wrap, or
    metric call raises — and to every example when the candidate program
    fails to load. The `0.0` default suits maximize objectives; minimize
    objectives should pass a large positive value so crashing candidates are
    penalized rather than rewarded.
    """

    def __init__(
        self,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        direction: str,
        metric_name: str,
        per_example_fn: "MetricCallable | None",
        aggregate_fn: Callable[[Sequence[float]], float],
        failure_score: float = 0.0,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"reps.Objective: direction must be one of {_DIRECTIONS}, got "
                f"{direction!r}"
            )
        if not entrypoint or not isinstance(entrypoint, str):
            raise ValueError(
                f"reps.Objective: `entrypoint` must be a non-empty function "
                f"name, got {entrypoint!r}"
            )
        if not train_set:
            raise ValueError("reps.Objective: `train_set` must be non-empty.")

        coerced: list[Example] = []
        for i, row in enumerate(train_set):
            ex = row if isinstance(row, Example) else Example(row)
            if not ex.input_keys:
                raise ValueError(
                    f"reps.Objective: train_set[{i}] has no input keys. Call "
                    f"`.with_inputs(...)` on each reps.Example — REPS does not "
                    f"infer input fields."
                )
            coerced.append(ex)

        ids = [str(ex["id"]) for ex in coerced if "id" in ex]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "reps.Objective: train_set has duplicate `id` fields; "
                "per-instance score keys must be unique."
            )

        self.entrypoint = entrypoint
        self.direction = direction
        self.failure_score = float(failure_score)
        self.train_set: list[Example] = coerced
        self.metric_name = metric_name
        self._per_example_fn = per_example_fn
        self._aggregate_fn = aggregate_fn

    @classmethod
    def maximize(
        cls,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        metric: "str | MetricCallable",
        failure_score: float = 0.0,
    ) -> "Objective":
        name, per_ex, agg = _resolve_metric(metric, "maximize")
        return cls(
            entrypoint=entrypoint,
            train_set=train_set,
            direction="maximize",
            metric_name=name,
            per_example_fn=per_ex,
            aggregate_fn=agg,
            failure_score=failure_score,
        )

    @classmethod
    def minimize(
        cls,
        *,
        entrypoint: str,
        train_set: "Sequence[Example | Mapping[str, Any]]",
        metric: "str | MetricCallable",
        failure_score: float = 0.0,
    ) -> "Objective":
        name, per_ex, agg = _resolve_metric(metric, "minimize")
        return cls(
            entrypoint=entrypoint,
            train_set=train_set,
            direction="minimize",
            metric_name=name,
            per_example_fn=per_ex,
            aggregate_fn=agg,
            failure_score=failure_score,
        )
