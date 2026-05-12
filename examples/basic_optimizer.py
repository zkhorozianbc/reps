"""Minimal REPS Python API example.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python examples/basic_optimizer.py

The evaluator executes generated code for clarity. In production, run
untrusted candidates in an isolated process or sandbox.
"""

from __future__ import annotations

import os

import reps


MODEL = os.environ.get("REPS_MODEL", "anthropic/claude-sonnet-4-6")
TARGET = 42.0

SEED = """
def solve():
    return 1
"""


def evaluate(code: str) -> float:
    namespace: dict[str, object] = {}
    try:
        exec(code, namespace)
        value = namespace["solve"]()
        return -abs(TARGET - float(value))
    except Exception as exc:
        # Candidate failures are useful search signal, so return a clear
        # penalty rather than crashing the whole run.
        return -1_000_000.0 - len(str(exc))


def main() -> None:
    result = reps.Optimizer(
        model=MODEL,
        max_iterations=10,
        num_islands=2,
    ).optimize(initial=SEED, evaluate=evaluate)

    print(f"best_score={result.best_score:.3f}")
    print(result.best_code)


if __name__ == "__main__":
    main()
