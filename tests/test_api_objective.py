"""Unit tests for `reps.Objective` and `reps.LLMJudge`."""

from __future__ import annotations

import math

import pytest

import reps
from reps.api.example import Example, Prediction
from reps.api.objective import _BUILTIN_METRICS, _resolve_metric


# --- built-in metric registry ----------------------------------------------


def test_builtin_metric_names_and_directions():
    assert _BUILTIN_METRICS["accuracy"].direction == "maximize"
    assert _BUILTIN_METRICS["exact_match"].direction == "maximize"
    assert _BUILTIN_METRICS["mae"].direction == "minimize"
    assert _BUILTIN_METRICS["mse"].direction == "minimize"
    assert _BUILTIN_METRICS["rmse"].direction == "minimize"


def test_accuracy_per_example():
    fn = _BUILTIN_METRICS["accuracy"].per_example
    assert fn(Example(answer="pos"), Prediction(answer="pos")) == 1.0
    assert fn(Example(answer="pos"), Prediction(answer="neg")) == 0.0


def test_mae_per_example_and_aggregate():
    m = _BUILTIN_METRICS["mae"]
    assert m.per_example(Example(answer=2.0), Prediction(answer=5.0)) == 3.0
    assert m.aggregate([1.0, 3.0]) == 2.0


def test_mse_per_example_and_aggregate():
    m = _BUILTIN_METRICS["mse"]
    assert m.per_example(Example(answer=2.0), Prediction(answer=5.0)) == 9.0
    assert m.aggregate([1.0, 9.0]) == 5.0


def test_rmse_aggregate_is_sqrt_of_mean_squared_error():
    m = _BUILTIN_METRICS["rmse"]
    # per_example returns squared error; aggregate takes sqrt(mean).
    assert m.per_example(Example(answer=0.0), Prediction(answer=3.0)) == 9.0
    assert m.aggregate([9.0, 16.0]) == math.sqrt(12.5)


# --- _resolve_metric --------------------------------------------------------


def test_resolve_builtin_metric_matching_direction():
    name, per_ex, agg = _resolve_metric("mae", "minimize")
    assert name == "mae"
    assert per_ex is _BUILTIN_METRICS["mae"].per_example
    assert agg is _BUILTIN_METRICS["mae"].aggregate


def test_resolve_builtin_metric_direction_mismatch_raises():
    with pytest.raises(ValueError, match="minimize metric"):
        _resolve_metric("mae", "maximize")
    with pytest.raises(ValueError, match="maximize metric"):
        _resolve_metric("accuracy", "minimize")


def test_resolve_unknown_metric_name_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        _resolve_metric("f1_score", "maximize")


def test_resolve_semantic_f1_not_in_v1():
    with pytest.raises(ValueError, match="not implemented in v1"):
        _resolve_metric("semantic_f1", "maximize")


def test_resolve_custom_callable():
    def close_enough(example, pred, trace=None):
        return abs(float(pred.answer) - float(example.answer)) <= 0.1

    name, per_ex, agg = _resolve_metric(close_enough, "maximize")
    assert name == "close_enough"
    assert per_ex is close_enough


def test_resolve_rejects_non_str_non_callable():
    with pytest.raises(TypeError, match="must be a built-in metric name"):
        _resolve_metric(123, "maximize")  # type: ignore[arg-type]
