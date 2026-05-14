"""LLM-as-judge objective for subjective outputs.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...      # mutation model
    export OPENAI_API_KEY=sk-...             # judge model
    uv run python examples/llm_judge.py

Prefer a deterministic `reps.Objective` when you can — an LLM judge is a
useful but imperfect metric. Reach for it when outputs are long-form or
subjective enough that exact-match / numeric metrics don't apply.
"""

from __future__ import annotations

import os

import reps

MODEL = os.environ.get("REPS_MODEL", "anthropic/claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("REPS_JUDGE_MODEL", "openai/gpt-5.1-mini")

SEED = """
def answer(question):
    return ""
"""

OBJECTIVE = reps.LLMJudge(
    entrypoint="answer",
    train_set=[
        reps.Example(
            question="What does REPS optimize?",
            answer="Whole Python programs.",
        ).with_inputs("question"),
        reps.Example(
            question="Name one REPS search feature.",
            answer="Pareto selection, trace reflection, or system-aware merge.",
        ).with_inputs("question"),
    ],
    rubric="Score whether the answer is factually correct and concise.",
    model=JUDGE_MODEL,
)


def main() -> None:
    result = reps.Optimizer(
        model=MODEL,
        max_iterations=10,
        num_islands=2,
    ).optimize(initial=SEED, objective=OBJECTIVE)

    print(f"best_score={result.best_score:.3f}")
    print(result.best_code)


if __name__ == "__main__":
    main()
