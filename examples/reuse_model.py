"""Share one `reps.Model` across optimizer runs."""

from __future__ import annotations

import os

import reps

from basic_optimizer import SEED, evaluate


MODEL = os.environ.get("REPS_MODEL", "anthropic/claude-sonnet-4-6")


def main() -> None:
    model = reps.Model(MODEL, temperature=0.7, max_tokens=4096)

    short_run = reps.Optimizer(
        model=model,
        max_iterations=5,
        num_islands=2,
    )
    deeper_run = reps.Optimizer(
        model=model,
        max_iterations=15,
        num_islands=3,
        merge=True,
    )

    first = short_run.optimize(initial=SEED, evaluate=evaluate)
    second = deeper_run.optimize(initial=first.best_code, evaluate=evaluate)

    print(f"first_score={first.best_score:.3f}")
    print(f"second_score={second.best_score:.3f}")
    print(second.best_code)


if __name__ == "__main__":
    main()
