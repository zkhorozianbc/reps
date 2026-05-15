"""Integration test: the objective API works end-to-end with DSPy datasets.

`reps.Example` accepts a `dspy.Example` directly — both expose the dict-like
`keys()` + `__getitem__` protocol — so a row from a DSPy built-in dataset
(https://dspy.ai/deep-dive/data-handling/built-in-datasets/) drops straight
into a `reps.Objective`.

- The `Colors` tests run offline whenever `dspy` is installed.
- The `HotPotQA` test additionally needs the HuggingFace `datasets` package
  and network access, so it is gated with `importorskip` + the `integration`
  marker.

The whole file is skipped when `dspy` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import reps
from reps.api.optimizer import Optimizer
from reps.evaluator import Evaluator

dspy = pytest.importorskip("dspy")


# ---------------------------------------------------------------------------
# helpers (mirror tests/test_api_objective_integration.py)
# ---------------------------------------------------------------------------


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
        "code": "def normalize(color):\n    return color.upper()\n",
        "language": "python",
        "metrics": {"combined_score": 1.0, "exact_match": 1.0, "validity": 1.0},
        "iteration_found": 1,
        "metadata": {},
    }
    (out / "programs" / "best.json").write_text(json.dumps(best))


# ---------------------------------------------------------------------------
# dspy.Example <-> reps.Example interop
# ---------------------------------------------------------------------------


def test_reps_example_accepts_dspy_example_directly():
    """A `dspy.Example` is dict-like; `reps.Example` consumes it natively."""
    dspy_ex = dspy.Example(
        question="What is REPS?", answer="A program search harness."
    ).with_inputs("question")

    reps_ex = reps.Example(dspy_ex).with_inputs("question")
    assert reps_ex.to_dict() == {
        "question": "What is REPS?",
        "answer": "A program search harness.",
    }
    assert reps_ex.inputs().to_dict() == {"question": "What is REPS?"}
    assert reps_ex.labels().to_dict() == {"answer": "A program search harness."}


# ---------------------------------------------------------------------------
# dspy.datasets.Colors — a built-in dataset, runs offline
# ---------------------------------------------------------------------------


def _colors_train_set(n: int):
    """Load the real `dspy.datasets.Colors` built-in and turn the first `n`
    rows into a `reps` train set. Task: uppercase the color name.

    `reps.Example(row, answer=...)` feeds the dspy.Example straight in as the
    base record and adds the derived label."""
    from dspy.datasets import Colors

    colors = Colors(sort_by_suffix=True)
    return [
        reps.Example(row, answer=row["color"].upper()).with_inputs("color")
        for row in colors.train[:n]
    ]


def test_objective_scores_dspy_colors_dataset():
    """Build a reps.Objective from real dspy.datasets.Colors rows and confirm
    it discriminates a wrong candidate from a correct one."""
    train = _colors_train_set(8)
    assert len(train) == 8
    # Rows hold real dspy.datasets.Colors data, converted to reps.Example.
    # (reps.Example copies what dspy.Example.keys() exposes — the real fields,
    # not dspy's dspy_uuid/dspy_split bookkeeping.)
    assert all(isinstance(ex["color"], str) and ex["color"] for ex in train)
    assert train[0]["answer"] == train[0]["color"].upper()
    assert train[0].input_keys == frozenset({"color"})

    objective = reps.Objective.maximize(
        entrypoint="normalize",
        train_set=train,
        metric="exact_match",
    )

    wrong = objective.evaluate("def normalize(color):\n    return color\n")
    correct = objective.evaluate("def normalize(color):\n    return color.upper()\n")

    assert correct.metrics["exact_match"] == 1.0
    assert correct.metrics["combined_score"] == 1.0
    assert correct.metrics["validity"] == 1.0
    assert all(v == 1.0 for v in correct.per_instance_scores.values())
    assert wrong.metrics["combined_score"] < correct.metrics["combined_score"]


def test_dspy_colors_objective_through_optimize(monkeypatch, tmp_path):
    """The DSPy-sourced objective wires through optimize() -> dispatch shim
    -> a real Evaluator end-to-end (LLM/controller mocked)."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(model=lm, max_iterations=1, output_dir=str(output_dir))

    objective = reps.Objective.maximize(
        entrypoint="normalize",
        train_set=_colors_train_set(6),
        metric="exact_match",
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
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize(
            "def normalize(color):\n    return color.upper()\n", objective=objective
        )

    assert seen["metrics"]["exact_match"] == 1.0
    assert seen["metrics"]["combined_score"] == 1.0
    assert len(seen["per_instance_scores"]) == 6
    assert isinstance(result, reps.OptimizationResult)


# ---------------------------------------------------------------------------
# dspy.datasets.HotPotQA — needs the HuggingFace `datasets` package + network
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_objective_scores_dspy_hotpotqa_dataset():
    """Load the real dspy.datasets.HotPotQA built-in (question -> answer) and
    score candidates with `exact_match`. Gated: needs `datasets` + network."""
    pytest.importorskip("datasets")
    from dspy.datasets import HotPotQA

    ds = HotPotQA(
        train_seed=1, train_size=5, eval_seed=2023, dev_size=0, test_size=0
    )
    # HotPotQA rows already carry `question` + `answer`; drop them straight in.
    train = [reps.Example(row).with_inputs("question") for row in ds.train]
    assert len(train) == 5

    objective = reps.Objective.maximize(
        entrypoint="answer",
        train_set=train,
        metric="exact_match",
    )

    # A candidate that returns the gold answer per question scores 1.0 — this
    # proves the scoring pipeline against real HotPotQA data, not a solver.
    gold = {row["question"]: row["answer"] for row in ds.train}
    perfect_code = (
        f"GOLD = {gold!r}\n"
        "def answer(question):\n"
        "    return GOLD.get(question, '')\n"
    )
    perfect = objective.evaluate(perfect_code)
    assert perfect.metrics["exact_match"] == 1.0
    assert perfect.metrics["combined_score"] == 1.0

    blank = objective.evaluate("def answer(question):\n    return ''\n")
    assert blank.metrics["combined_score"] < perfect.metrics["combined_score"]
