"""The `evaluate=` escape hatch: a raw callable returning `EvaluationResult`.

Most users should reach for `reps.Objective` (see `basic_optimizer.py`). This
example shows the power-user path: a hand-written `evaluate(code)` callable
that emits a headline metric plus per-test-case scores and feedback, so
Pareto/mixed selection can preserve candidates with different strengths.
"""

from __future__ import annotations

import math
import os

import reps


MODEL = os.environ.get("REPS_MODEL", "anthropic/claude-sonnet-4-6")
TEST_CASES = [-4.0, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0]

SEED = """
def predict(x):
    return x
"""


def target(x: float) -> float:
    return x * x - 3.0 * x + 2.0


def evaluate(code: str) -> reps.EvaluationResult:
    namespace: dict[str, object] = {}
    try:
        exec(code, namespace)
        predict = namespace["predict"]
    except Exception as exc:
        return reps.EvaluationResult(
            metrics={"combined_score": 0.0},
            per_instance_scores={f"x={x:g}": 0.0 for x in TEST_CASES},
            feedback=f"Program failed to load: {exc}",
        )

    errors: dict[str, float] = {}
    for x in TEST_CASES:
        key = f"x={x:g}"
        try:
            prediction = float(predict(x))
            if not math.isfinite(prediction):
                raise ValueError("prediction was not finite")
            errors[key] = abs(prediction - target(x))
        except Exception as exc:
            return reps.EvaluationResult(
                metrics={"combined_score": 0.0},
                per_instance_scores={f"x={case:g}": 0.0 for case in TEST_CASES},
                feedback=f"Prediction failed for {key}: {exc}",
            )

    mean_error = sum(errors.values()) / len(errors)
    per_instance_scores = {
        key: 1.0 / (1.0 + error)
        for key, error in errors.items()
    }
    worst_case = max(errors, key=errors.get)

    return reps.EvaluationResult(
        metrics={"combined_score": 1.0 / (1.0 + mean_error)},
        per_instance_scores=per_instance_scores,
        feedback=f"Worst case is {worst_case} with absolute error {errors[worst_case]:.3f}.",
    )


def main() -> None:
    result = reps.Optimizer(
        model=MODEL,
        max_iterations=20,
        num_islands=3,
        selection_strategy="mixed",
        pareto_fraction=0.4,
        trace_reflection=True,
        lineage_depth=2,
    ).optimize(initial=SEED, evaluate=evaluate)

    print(f"best_score={result.best_score:.3f}")
    print(result.best_code)


if __name__ == "__main__":
    main()
