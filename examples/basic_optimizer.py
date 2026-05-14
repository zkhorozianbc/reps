"""Minimal REPS Python API example.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/basic_optimizer.py

A `reps.Objective` runs each candidate's `predict` entrypoint against the
train set and scores it with `mae` — no hand-written evaluator needed.
"""

from __future__ import annotations

import os

import reps

MODEL = os.environ.get("REPS_MODEL", "anthropic/claude-sonnet-4-6")

SEED = """
def predict(x):
    return x
"""

# The target relationship is predict(x) = x*x - 3x + 2.
OBJECTIVE = reps.Objective.minimize(
    entrypoint="predict",
    train_set=[
        reps.Example(x=-4, answer=30).with_inputs("x"),
        reps.Example(x=0, answer=2).with_inputs("x"),
        reps.Example(x=3, answer=2).with_inputs("x"),
        reps.Example(x=5, answer=12).with_inputs("x"),
    ],
    metric="mae",
)


def main() -> None:
    result = reps.Optimizer(
        model=MODEL,
        max_iterations=10,
        num_islands=2,
    ).optimize(initial=SEED, objective=OBJECTIVE)

    print(f"best_score={result.best_score:.3f}")
    print(f"best_mae={result.best_metrics.get('mae', float('nan')):.3f}")
    print(result.best_code)


if __name__ == "__main__":
    main()
