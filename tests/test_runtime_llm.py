"""Tests for `reps.runtime.llm` — the LLM callable exposed to evolved code.

The general REPS-ethos pattern: REPS evolves arbitrary Python; evolved code
calls `reps.runtime.llm(...)` at inference time when it wants the LLM to do
reasoning. The harness owns the client + credentials + token accounting; the
candidate owns *what* to ask. This is strictly more general than a
prompt-template-with-placeholders abstraction.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import reps
from reps.api.example import Example
from reps.runtime import llm, reset_current_llm, set_current_llm


# --- standalone contextvar behavior ----------------------------------------


def test_llm_raises_without_configured_context():
    with pytest.raises(RuntimeError, match="no LLM configured"):
        llm("hello")


def test_llm_dispatches_to_configured_callable():
    calls = []

    def fake_llm(prompt: str, **kwargs):
        calls.append((prompt, kwargs))
        return "ok: " + prompt

    token = set_current_llm(fake_llm)
    try:
        out = llm("hello")
        assert out == "ok: hello"
        out2 = llm("ping", temperature=0.0)
        assert out2 == "ok: ping"
    finally:
        reset_current_llm(token)

    assert calls == [("hello", {}), ("ping", {"temperature": 0.0})]
    # Reset restored the previous state.
    with pytest.raises(RuntimeError):
        llm("hello")


def test_nested_set_current_llm_unwinds_correctly():
    def outer(p, **k):
        return "outer:" + p

    def inner(p, **k):
        return "inner:" + p

    t1 = set_current_llm(outer)
    try:
        assert llm("x") == "outer:x"
        t2 = set_current_llm(inner)
        try:
            assert llm("y") == "inner:y"
        finally:
            reset_current_llm(t2)
        assert llm("z") == "outer:z"
    finally:
        reset_current_llm(t1)
    with pytest.raises(RuntimeError):
        llm("a")


# --- end-to-end through Objective ------------------------------------------


def test_objective_evolved_code_can_use_runtime_llm():
    """An evolved Python program that imports `reps.runtime.llm` and uses it
    at inference time scores correctly through `Objective.evaluate`."""

    def fake_llm(prompt: str, **kwargs):
        # Pretend to answer math questions: extract the trailing number.
        # "What is 2+2?" -> "4"; "What is 3+5?" -> "8".
        if "2+2" in prompt:
            return "4"
        if "3+5" in prompt:
            return "8"
        return "?"

    seed = (
        "from reps.runtime import llm\n"
        "\n"
        "def solve(question):\n"
        "    return llm(f'Compute and answer with just the number: {question}')\n"
    )

    objective = reps.Objective.maximize(
        entrypoint="solve",
        train_set=[
            Example(question="2+2?", answer="4").with_inputs("question"),
            Example(question="3+5?", answer="8").with_inputs("question"),
        ],
        metric="exact_match",
    )

    token = set_current_llm(fake_llm)
    try:
        result = objective.evaluate(seed)
    finally:
        reset_current_llm(token)

    assert result.metrics["combined_score"] == 1.0
    assert result.metrics["exact_match"] == 1.0
    assert result.per_instance_scores == {"train/0": 1.0, "train/1": 1.0}


def test_objective_llm_failure_is_caught_per_example():
    """If the LLM raises on a candidate's call, that example gets
    `failure_score` and Objective's per-example failure path kicks in."""

    def boom(prompt, **kwargs):
        raise RuntimeError("rate limit")

    seed = (
        "from reps.runtime import llm\n"
        "def solve(question):\n"
        "    return llm(question)\n"
    )

    objective = reps.Objective.maximize(
        entrypoint="solve",
        train_set=[
            Example(question="hi", answer="ok").with_inputs("question"),
        ],
        metric="exact_match",
    )

    token = set_current_llm(boom)
    try:
        result = objective.evaluate(seed)
    finally:
        reset_current_llm(token)

    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["validity"] == 0.0
    assert "failures:" in result.feedback


# --- end-to-end through Optimizer.optimize ---------------------------------


def test_optimizer_sets_and_resets_llm_context(monkeypatch, tmp_path):
    """`Optimizer.optimize` makes `reps.runtime.llm` resolve to the Optimizer's
    configured Model during the run, and clears it afterwards."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("reps.llm.anthropic.anthropic.Anthropic"):
        lm = reps.Model("anthropic/claude-sonnet-4.6")

    seen = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        # While the run is active, the contextvar should resolve to `lm`.
        from reps.runtime import _current_llm_var

        seen["llm_during_run"] = _current_llm_var.get()
        # Seed a minimal DB so _collect_result is happy.
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
            "id": "best", "code": "def solve(q): return ''", "language": "python",
            "metrics": {"combined_score": 0.0}, "iteration_found": 1, "metadata": {},
        }))

    with patch("reps.runner.run_reps", new=fake_run_reps):
        reps.Optimizer(
            model=lm, max_iterations=1, output_dir=str(tmp_path / "run")
        ).optimize("seed", lambda c: 0.0)

    # During the run, the contextvar was the Optimizer's Model.
    assert seen["llm_during_run"] is lm
    # After optimize() returns, the contextvar is back to its default (None).
    from reps.runtime import _current_llm_var

    assert _current_llm_var.get() is None


# --- top-level access via reps.runtime --------------------------------------


def test_runtime_llm_accessible_from_top_level_module():
    """Sanity: evolved code can reach the helper via `reps.runtime.llm`."""
    import reps.runtime as runtime

    assert runtime.llm is llm
    assert runtime.set_current_llm is set_current_llm
    assert runtime.reset_current_llm is reset_current_llm
