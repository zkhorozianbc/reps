"""Integration tests for the objective API.

These exercise the objective layer through the *real* harness machinery —
the dispatch shim, `write_shim`, and a real `reps.evaluator.Evaluator` — not
mocks. The LLM/controller (`run_reps`) is still mocked: a real evolutionary
run needs API credits and is out of scope for the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import reps
from reps.api.optimizer import Optimizer
from reps.evaluator import Evaluator


def _make_lm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("reps.llm.anthropic.anthropic.Anthropic"):
        return reps.Model("anthropic/claude-sonnet-4.6")


def _seed_database(run_dir: str, *, num_islands: int) -> None:
    out = Path(run_dir)
    (out / "programs").mkdir(parents=True, exist_ok=True)
    metadata = {
        "best_program_id": "best",
        "islands": [[] for _ in range(num_islands)],
        "island_best_programs": [None] * num_islands,
        "archive": [],
        "last_iteration": 1,
        "current_island": 0,
        "island_generations": [0] * num_islands,
        "last_migration_generation": 0,
        "feature_stats": {},
    }
    (out / "metadata.json").write_text(json.dumps(metadata))
    best = {
        "id": "best",
        "code": "def predict(x):\n    return x * x - 3 * x + 2\n",
        "language": "python",
        "metrics": {"combined_score": 0.0, "mae": 0.0, "validity": 1.0},
        "iteration_found": 1,
        "metadata": {},
    }
    (out / "programs" / "best.json").write_text(json.dumps(best))


# --- Objective.evaluate end-to-end (no harness) -----------------------------


def test_objective_minimize_full_evaluation_result():
    """The deterministic objective produces a complete EvaluationResult."""
    objective = reps.Objective.minimize(
        entrypoint="predict",
        train_set=[
            reps.Example(x=-4, answer=30).with_inputs("x"),
            reps.Example(x=0, answer=2).with_inputs("x"),
            reps.Example(x=3, answer=2).with_inputs("x"),
        ],
        metric="mae",
    )
    perfect = objective.evaluate("def predict(x):\n    return x * x - 3 * x + 2\n")
    assert perfect.metrics["mae"] == 0.0
    assert perfect.metrics["combined_score"] == 0.0
    assert perfect.metrics["validity"] == 1.0

    identity = objective.evaluate("def predict(x):\n    return x\n")
    # errors: |-4-30|=34, |0-2|=2, |3-2|=1 -> mae = 37/3
    assert identity.metrics["mae"] == pytest.approx(37 / 3)
    assert identity.metrics["combined_score"] == pytest.approx(-37 / 3)
    assert identity.metrics["combined_score"] < perfect.metrics["combined_score"]


# --- Objective through the real Evaluator + dispatch shim -------------------


def test_objective_round_trips_through_real_evaluator(monkeypatch, tmp_path):
    """Objective -> optimize() -> dispatch shim -> write_shim -> real
    Evaluator.evaluate_isolated -> dispatch_user_evaluate -> objective.evaluate."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(model=lm, max_iterations=1, output_dir=str(output_dir))

    objective = reps.Objective.minimize(
        entrypoint="predict",
        train_set=[
            reps.Example(x=0, answer=2).with_inputs("x"),
            reps.Example(x=3, answer=2).with_inputs("x"),
        ],
        metric="mae",
    )
    seen = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        config.evaluator.cascade_evaluation = False
        config.evaluator.max_retries = 0
        real_evaluator = Evaluator(config.evaluator, evaluator)
        outcome = await real_evaluator.evaluate_isolated(
            Path(initial_program).read_text(), program_id="cand-1"
        )
        seen["metrics"] = outcome.metrics
        seen["per_instance_scores"] = outcome.per_instance_scores
        seen["feedback"] = outcome.feedback
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("def predict(x):\n    return x\n", objective=objective)

    # Seed predict(x)=x -> mae over [(0,2),(3,2)] = 1.5; combined_score = -1.5.
    assert seen["metrics"]["mae"] == 1.5
    assert seen["metrics"]["combined_score"] == -1.5
    assert seen["per_instance_scores"] == {"train/0": -2.0, "train/1": -1.0}
    assert "per-example results:" in seen["feedback"]
    assert "predict(x=0) -> 0" in seen["feedback"]
    # And the OptimizationResult comes back from the seeded DB.
    assert isinstance(result, reps.OptimizationResult)
    assert result.best_metrics["mae"] == 0.0


def test_llm_judge_round_trips_through_real_evaluator(monkeypatch, tmp_path):
    """LLMJudge with a fake judge callable, driven through the real Evaluator."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(model=lm, max_iterations=1, output_dir=str(output_dir))

    objective = reps.LLMJudge(
        entrypoint="answer",
        train_set=[
            reps.Example(question="q", answer="a").with_inputs("question"),
        ],
        rubric="Score whether the answer is correct.",
        model=lambda prompt: '{"score": 0.75, "rationale": "close"}',
    )
    seen = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        config.evaluator.cascade_evaluation = False
        config.evaluator.max_retries = 0
        real_evaluator = Evaluator(config.evaluator, evaluator)
        outcome = await real_evaluator.evaluate_isolated(
            Path(initial_program).read_text(), program_id="cand-1"
        )
        seen["metrics"] = outcome.metrics
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        opt.optimize("def answer(question):\n    return 'a'\n", objective=objective)

    assert seen["metrics"]["judge_score"] == 0.75
    assert seen["metrics"]["combined_score"] == 0.75
