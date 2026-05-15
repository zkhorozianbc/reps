"""Unit tests for `reps.PromptObjective`.

The artifact REPS optimizes is a PROMPT TEMPLATE STRING (not Python code).
At evaluation:
  1. fill {field} placeholders with example.inputs()
  2. call the configured LLM
  3. wrap the response as Prediction(answer=...)  (or via parse=)
  4. score with the metric

These tests use a fake `model` callable so no API key / network is needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import reps
from reps.api.example import Example
from reps.api.objective import Objective, PromptObjective


def _ex_set():
    return [
        Example(question="2+2?", answer="4").with_inputs("question"),
        Example(question="3+5?", answer="8").with_inputs("question"),
    ]


# --- construction -----------------------------------------------------------


def test_prompt_objective_is_objective_subclass():
    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=lambda prompt: "4",
    )
    assert isinstance(obj, Objective)
    assert obj.direction == "maximize"
    assert obj.metric_name == "exact_match"


def test_prompt_objective_minimize_classmethod():
    obj = PromptObjective.minimize(
        train_set=[
            Example(x="1", answer=10.0).with_inputs("x"),
        ],
        metric="mae",
        model=lambda prompt: "5",
    )
    assert obj.direction == "minimize"
    assert obj.metric_name == "mae"


def test_prompt_objective_rejects_built_in_direction_mismatch():
    with pytest.raises(ValueError, match="maximize metric"):
        PromptObjective.minimize(
            train_set=_ex_set(),
            metric="accuracy",
            model=lambda p: "x",
        )


# --- evaluate (fake LLM) ----------------------------------------------------


def test_evaluate_fills_placeholders_and_calls_llm():
    seen = []

    def fake_llm(prompt: str) -> str:
        seen.append(prompt)
        # Echo the question's answer for the demo cases:
        if "2+2" in prompt:
            return "4"
        if "3+5" in prompt:
            return "8"
        return "?"

    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=fake_llm,
    )
    template = "Q: {question}\nA:"
    result = obj.evaluate(template)
    assert result.metrics["combined_score"] == 1.0
    assert result.metrics["exact_match"] == 1.0
    assert result.metrics["validity"] == 1.0
    assert result.per_instance_scores == {"train/0": 1.0, "train/1": 1.0}
    # Both prompts were the template with the example's inputs substituted.
    # Per-example LLM calls run concurrently, so completion order is
    # non-deterministic — assert the set of calls, not their sequence.
    assert sorted(seen) == sorted(["Q: 2+2?\nA:", "Q: 3+5?\nA:"])


def test_evaluate_runs_parse_before_scoring():
    """`parse` maps raw LLM output to the value passed into Prediction(answer=...)."""
    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=lambda prompt: "The answer is 4." if "2+2" in prompt else "The answer is 8.",
        parse=lambda out: out.rsplit(" ", 1)[-1].rstrip("."),
    )
    result = obj.evaluate("Q: {question}\nA:")
    assert result.metrics["exact_match"] == 1.0


def test_evaluate_caches_repeated_calls():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "4" if "2+2" in prompt else "8"

    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=fake_llm,
    )
    template = "Q: {question}\nA:"
    obj.evaluate(template)
    obj.evaluate(template)
    # Same template, same examples, same model -> all cached on second pass.
    assert len(calls) == 2


def test_evaluate_template_format_failure_uses_failure_score():
    """A `{missing_field}` placeholder gets caught per-example, not crashed."""
    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=lambda p: "ignored",
    )
    result = obj.evaluate("Bad template {does_not_exist}")
    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["validity"] == 0.0
    assert "template" in result.feedback


def test_evaluate_llm_failure_uses_failure_score():
    def boom(prompt: str) -> str:
        raise RuntimeError("rate limit")

    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=boom,
    )
    result = obj.evaluate("Q: {question}\nA:")
    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["validity"] == 0.0
    assert "LLM call failed" in result.feedback


def test_evaluate_feedback_includes_per_example_detail():
    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=lambda p: "4" if "2+2" in p else "wrong",
    )
    result = obj.evaluate("Q: {question}\nA:")
    fb = result.feedback
    assert "per-example results:" in fb
    assert "expected '4'" in fb
    assert "expected '8'" in fb
    # The LLM output should appear in the per-example line.
    assert "'4'" in fb
    assert "'wrong'" in fb


def test_prompt_objective_string_model_builds_reps_model_lazily(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    obj = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model="openrouter/anthropic/claude-sonnet-4.6",
    )
    # Lazy: Model not built at construction; no key required yet.
    assert obj._llm_callable is None


# --- top-level re-export ----------------------------------------------------


def test_prompt_objective_is_reexported_at_top_level():
    assert reps.PromptObjective is PromptObjective


# --- end-to-end via Optimizer (mocked run_reps) -----------------------------


def test_optimize_accepts_prompt_objective(monkeypatch, tmp_path):
    """`PromptObjective` is an `Objective`, so the existing `objective=` path
    of `Optimizer.optimize` accepts it without changes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("reps.llm.anthropic.anthropic.Anthropic"):
        lm = reps.Model("anthropic/claude-sonnet-4.6")

    objective = PromptObjective.maximize(
        train_set=_ex_set(),
        metric="exact_match",
        model=lambda prompt: "4" if "2+2" in prompt else "8",
    )

    seen = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_reps_user_evaluator", evaluator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seen["result"] = module.evaluate(initial_program)
        # Seed a minimal database so _collect_result has something.
        from pathlib import Path
        import json

        out = Path(output_dir)
        (out / "programs").mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps({
            "best_program_id": "best",
            "islands": [[] for _ in range(config.database.num_islands)],
            "island_best_programs": [None] * config.database.num_islands,
            "archive": [], "last_iteration": 1, "current_island": 0,
            "island_generations": [0] * config.database.num_islands,
            "last_migration_generation": 0, "feature_stats": {},
        }))
        (out / "programs" / "best.json").write_text(json.dumps({
            "id": "best", "code": "Q: {question}\nA:", "language": "python",
            "metrics": {"combined_score": 1.0, "exact_match": 1.0, "validity": 1.0},
            "iteration_found": 1, "metadata": {},
        }))

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = reps.Optimizer(
            model=lm, max_iterations=1, output_dir=str(tmp_path / "run")
        ).optimize(initial="Q: {question}\nA:", objective=objective)

    # The dispatch shim routed the prompt template through PromptObjective.evaluate.
    assert seen["result"].metrics["exact_match"] == 1.0
    assert isinstance(result, reps.OptimizationResult)
