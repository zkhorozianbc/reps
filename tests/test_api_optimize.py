"""End-to-end tests for `reps.REPS.optimize()` (Phase B of the v1 API).

The full controller path is mocked at `reps.runner.run_reps` — we don't
need to spin up an LLM client to verify that:
  - constructor kwargs map onto the right Config fields,
  - the user's `evaluate` callable round-trips through the dispatch shim
    (registry + signature introspection + return-shape coercion),
  - `from_config` accepts a fully-formed Config,
  - `_collect_result` produces a sensible OptimizationResult from the
    saved database.

Edge-case coverage (provider failures, mid-run shutdown, etc.) is left
to a follow-up test agent.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import reps
from reps.api.evaluate_dispatch import (
    coerce_return,
    dispatch_user_evaluate,
    register_user_evaluate,
    unregister_user_evaluate,
    write_shim,
)
from reps.api.optimizer import REPS, _StubLM
from reps.api.result import OptimizationResult
from reps.config import Config
from reps.evaluation_result import EvaluationResult


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


def test_top_level_exports():
    assert reps.REPS is REPS
    assert reps.OptimizationResult is OptimizationResult
    assert reps.EvaluationResult is EvaluationResult


# ---------------------------------------------------------------------------
# `coerce_return` — return-shape coercion
# ---------------------------------------------------------------------------


def test_coerce_float():
    out = coerce_return(0.42)
    assert out == {"combined_score": 0.42, "validity": 1.0}


def test_coerce_int():
    out = coerce_return(7)
    assert out == {"combined_score": 7.0, "validity": 1.0}


def test_coerce_bool():
    assert coerce_return(True) == {"combined_score": 1.0, "validity": 1.0}
    assert coerce_return(False) == {"combined_score": 0.0, "validity": 1.0}


def test_coerce_dict_passthrough():
    payload = {"combined_score": 0.5, "extra": 1}
    out = coerce_return(payload)
    assert out is payload


def test_coerce_evaluation_result_passthrough():
    r = EvaluationResult(metrics={"combined_score": 0.5})
    assert coerce_return(r) is r


def test_coerce_unknown_type_raises():
    with pytest.raises(ValueError, match="must return float, dict, or EvaluationResult"):
        coerce_return("not a score")


# ---------------------------------------------------------------------------
# Dispatch shim — registry + signature introspection
# ---------------------------------------------------------------------------


def test_dispatch_shim_round_trip(tmp_path):
    code_path = tmp_path / "prog.py"
    code_path.write_text("def f(): return 42")

    captured = {}

    def evaluate(code: str) -> float:
        captured["code"] = code
        return 0.9

    rid = register_user_evaluate(evaluate)
    try:
        out = dispatch_user_evaluate(rid, str(code_path))
    finally:
        unregister_user_evaluate(rid)

    assert captured["code"] == "def f(): return 42"
    assert out == {"combined_score": 0.9, "validity": 1.0}


def test_dispatch_shim_forwards_known_kwargs(tmp_path):
    code_path = tmp_path / "prog.py"
    code_path.write_text("# program")

    seen = {}

    def evaluate(code: str, *, env=None, instances=None):
        seen["env"] = env
        seen["instances"] = instances
        return {"combined_score": 1.0}

    rid = register_user_evaluate(evaluate)
    try:
        dispatch_user_evaluate(
            rid, str(code_path), env={"X": "1"}, instances=["a", "b"]
        )
    finally:
        unregister_user_evaluate(rid)

    assert seen["env"] == {"X": "1"}
    assert seen["instances"] == ["a", "b"]


def test_dispatch_shim_drops_unsupported_kwargs(tmp_path):
    code_path = tmp_path / "prog.py"
    code_path.write_text("# program")

    def evaluate(code: str) -> float:  # no env, no instances
        return 0.5

    rid = register_user_evaluate(evaluate)
    try:
        # `env` and `instances` should be silently dropped.
        out = dispatch_user_evaluate(
            rid, str(code_path), env={"X": "1"}, instances=["a"]
        )
    finally:
        unregister_user_evaluate(rid)
    assert out == {"combined_score": 0.5, "validity": 1.0}


def test_dispatch_shim_unknown_id_raises():
    with pytest.raises(RuntimeError, match="no user evaluate callable"):
        dispatch_user_evaluate("doesnotexist", "/tmp/anything")


def test_write_shim_creates_dispatcher(tmp_path):
    shim_path = write_shim(tmp_path)
    assert Path(shim_path).exists()
    contents = Path(shim_path).read_text()
    assert "dispatch_user_evaluate" in contents
    assert "REPS_USER_EVALUATOR_ID" in contents


# ---------------------------------------------------------------------------
# Constructor — kwarg → Config mapping
# ---------------------------------------------------------------------------


def _make_lm(monkeypatch, model="anthropic/claude-sonnet-4.6"):
    """Build a real LM with a mocked SDK client + dummy api key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("reps.llm.anthropic.anthropic.Anthropic"):
        return reps.LM(model)


def test_constructor_defaults_match_spec(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm)
    assert opt.max_iterations == 100
    assert opt.selection_strategy == "map_elites"
    assert opt.pareto_fraction == 0.0
    assert opt.trace_reflection_enabled is False
    assert opt.lineage_depth == 3
    assert opt.merge_enabled is False
    assert opt.minibatch_size is None
    assert opt.num_islands == 5
    assert opt.output_dir is None


def test_build_config_maps_kwargs(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(
        lm=lm,
        max_iterations=42,
        selection_strategy="mixed",
        pareto_fraction=0.3,
        trace_reflection=True,
        lineage_depth=7,
        merge=True,
        minibatch_size=2,
        num_islands=4,
    )
    cfg = opt._build_config(seed=123)
    assert cfg.max_iterations == 42
    assert cfg.database.selection_strategy == "mixed"
    assert cfg.database.pareto_fraction == 0.3
    assert cfg.database.num_islands == 4
    assert cfg.reps.trace_reflection.enabled is True
    assert cfg.reps.trace_reflection.lineage_depth == 7
    assert cfg.reps.merge.enabled is True
    # Master switch flips on when any GEPA feature is enabled.
    assert cfg.reps.enabled is True
    assert cfg.evaluator.minibatch_size == 2
    assert cfg.random_seed == 123
    assert cfg.database.random_seed == 123


def test_build_config_master_switch_off_when_no_gepa_features(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.reps.enabled is False


def test_build_config_anthropic_provider_routes_to_anthropic(monkeypatch):
    lm = _make_lm(monkeypatch, model="anthropic/claude-sonnet-4.6")
    opt = REPS(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.provider == "anthropic"
    assert cfg.llm.models[0].name == "claude-sonnet-4.6"


def test_build_config_openrouter_provider_routes_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("reps.llm.openai_compatible.openai.OpenAI"):
        lm = reps.LM("openrouter/google/gemini-2.5-flash")
    opt = REPS(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.provider == "openrouter"
    assert cfg.llm.models[0].name == "google/gemini-2.5-flash"


def test_constructor_validates_kwargs(monkeypatch):
    lm = _make_lm(monkeypatch)
    with pytest.raises(ValueError, match="max_iterations"):
        REPS(lm=lm, max_iterations=0)
    with pytest.raises(ValueError, match="selection_strategy"):
        REPS(lm=lm, selection_strategy="random")
    with pytest.raises(ValueError, match="num_islands"):
        REPS(lm=lm, num_islands=0)
    with pytest.raises(TypeError, match="reps.LM"):
        REPS(lm="not an LM")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# `from_config` escape hatch
# ---------------------------------------------------------------------------


def test_from_config_round_trips():
    cfg = Config()
    cfg.max_iterations = 17
    cfg.database.num_islands = 8
    cfg.database.selection_strategy = "pareto"
    cfg.reps.merge.enabled = True
    opt = REPS.from_config(cfg)
    assert opt.max_iterations == 17
    assert opt.num_islands == 8
    assert opt.selection_strategy == "pareto"
    assert opt.merge_enabled is True
    # _build_config returns the original cfg when from_config was used.
    assert opt._build_config(seed=None) is cfg


def test_from_config_rejects_non_config():
    with pytest.raises(TypeError, match="reps.config.Config"):
        REPS.from_config("not a Config")  # type: ignore[arg-type]


def test_from_config_stamps_seed():
    cfg = Config()
    opt = REPS.from_config(cfg)
    out_cfg = opt._build_config(seed=99)
    assert out_cfg.random_seed == 99
    assert out_cfg.database.random_seed == 99


# ---------------------------------------------------------------------------
# `optimize()` — sync wrapping
# ---------------------------------------------------------------------------


def test_optimize_rejects_running_loop(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm)

    async def run():
        with pytest.raises(RuntimeError, match="async context"):
            opt.optimize("seed", lambda code: 1.0)

    asyncio.run(run())


def test_optimize_rejects_non_string_initial(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm)
    with pytest.raises(TypeError, match="program text"):
        opt.optimize(b"bytes", lambda c: 1.0)  # type: ignore[arg-type]


def test_optimize_rejects_non_callable_evaluate(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm)
    with pytest.raises(TypeError, match="must be callable"):
        opt.optimize("def f(): pass", "not a callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end with `run_reps` mocked
# ---------------------------------------------------------------------------


def _seed_database(run_dir: str, *, num_islands: int) -> None:
    """Write a minimal saved database into `run_dir` so _collect_result
    can load it back. Mirrors the on-disk layout produced by
    ProgramDatabase.save."""
    out = Path(run_dir)
    (out / "programs").mkdir(parents=True, exist_ok=True)
    metadata = {
        "best_program_id": "best",
        "islands": [[] for _ in range(num_islands)],
        "island_best_programs": [None] * num_islands,
        "archive": [],
        "last_iteration": 3,
        "current_island": 0,
        "island_generations": [0] * num_islands,
        "last_migration_generation": 0,
        "feature_stats": {},
    }
    (out / "metadata.json").write_text(json.dumps(metadata))

    # Two programs: the seed and one improved child.
    seed = {
        "id": "initial",
        "code": "def f(): return 0",
        "language": "python",
        "metrics": {"combined_score": 0.1, "validity": 1.0},
        "iteration_found": 0,
        "metadata": {},
    }
    child = {
        "id": "best",
        "code": "def f(): return 1",
        "language": "python",
        "parent_id": "initial",
        "metrics": {"combined_score": 0.9, "validity": 1.0},
        "iteration_found": 3,
        "per_instance_scores": {"a": 0.9, "b": 1.0},
        "feedback": "much better",
        "metadata": {"reps_meta": {"tokens_in": 100, "tokens_out": 25}},
    }
    (out / "programs" / "initial.json").write_text(json.dumps(seed))
    (out / "programs" / "best.json").write_text(json.dumps(child))


def test_optimize_happy_path_returns_optimization_result(monkeypatch, tmp_path):
    """Mock `run_reps` to write a saved DB, then verify result aggregation."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = REPS(lm=lm, max_iterations=3, num_islands=2, output_dir=str(output_dir))

    captured = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        captured["config"] = config
        captured["initial_program"] = initial_program
        captured["evaluator"] = evaluator
        captured["output_dir"] = output_dir
        # Verify the seed file + shim were written before run_reps was invoked.
        assert Path(initial_program).exists()
        assert Path(evaluator).exists()
        assert Path(evaluator).name == "_reps_user_evaluator.py"
        # Verify the registry env var is live during the "run".
        assert os.environ.get("REPS_USER_EVALUATOR_ID")
        _seed_database(output_dir, num_islands=config.database.num_islands)

    def evaluate(code: str) -> float:
        return 0.42

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("def f(): return 0", evaluate, seed=7)

    assert isinstance(result, OptimizationResult)
    assert result.best_code == "def f(): return 1"
    assert result.best_score == 0.9
    assert result.best_metrics == {"combined_score": 0.9, "validity": 1.0}
    assert result.best_per_instance_scores == {"a": 0.9, "b": 1.0}
    assert result.best_feedback == "much better"
    assert result.iterations_run == 3
    assert result.total_metric_calls == 2  # seed + 1 child
    assert result.total_tokens == {"in": 100, "out": 25}
    assert result.output_dir == str(output_dir.resolve())

    # Config wiring sanity: the captured Config has the right kwargs.
    cfg = captured["config"]
    assert cfg.max_iterations == 3
    assert cfg.database.num_islands == 2
    assert cfg.random_seed == 7

    # Initial code was written verbatim to the run dir.
    assert Path(captured["initial_program"]).read_text() == "def f(): return 0"

    # Registry env var is cleared after optimize() returns.
    assert os.environ.get("REPS_USER_EVALUATOR_ID") is None


def test_optimize_uses_tempdir_when_output_dir_is_none(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = REPS(lm=lm, max_iterations=1)

    seen_output: list = []

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        seen_output.append(output_dir)
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    # No output_dir on the result when persisting wasn't requested.
    assert result.output_dir is None
    # Tempdir cleaned up after optimize() returns.
    assert not Path(seen_output[0]).exists()


def test_optimize_runs_user_evaluate_via_shim(monkeypatch, tmp_path):
    """The dispatch path: write_shim + register + dispatch_user_evaluate
    should round-trip the user's `evaluate` callable when the shim's
    `evaluate(program_path)` function is called."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = REPS(lm=lm, max_iterations=1, output_dir=str(output_dir))

    received_codes: list = []

    def evaluate(code: str) -> float:
        received_codes.append(code)
        return 0.7

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        # Drive the shim the same way Evaluator would.
        import importlib.util
        spec = importlib.util.spec_from_file_location("_reps_user_evaluator", evaluator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = module.evaluate(initial_program)
        # The shim should coerce the float through coerce_return.
        assert out == {"combined_score": 0.7, "validity": 1.0}
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        opt.optimize("seed_code", evaluate)

    assert received_codes == ["seed_code"]


# ---------------------------------------------------------------------------
# `_StubLM` (used by from_config)
# ---------------------------------------------------------------------------


def test_stub_lm_repr_does_not_crash():
    cfg = Config()
    cfg.llm.models = [
        type(cfg.llm.models)([])  # placeholder
    ] if False else []
    # Non-empty path: build the stub directly.
    from reps.config import LLMModelConfig
    stub = _StubLM(LLMModelConfig(name="x/y", provider="anthropic"))
    assert stub.model == "x/y"
    assert stub.provider == "anthropic"
    repr(stub)  # no crash


# ---------------------------------------------------------------------------
# `reps.internal` — back-compat re-exports
# ---------------------------------------------------------------------------


def test_reps_internal_reexports_old_symbols():
    import reps.internal as internal
    # Sanity: a representative subset.
    assert hasattr(internal, "ReflectionEngine")
    assert hasattr(internal, "WorkerPool")
    assert hasattr(internal, "ConvergenceMonitor")
    assert hasattr(internal, "SOTAController")
