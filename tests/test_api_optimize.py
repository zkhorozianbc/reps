"""End-to-end tests for `reps.Optimizer.optimize()` (Phase B of the v1 API).

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
from reps.api.optimizer import Optimizer, _StubLM
from reps.api.result import OptimizationResult
from reps.config import Config
from reps.evaluation_result import EvaluationResult


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


def test_top_level_exports():
    assert reps.Optimizer is Optimizer
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
    opt = Optimizer(lm=lm)
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
    opt = Optimizer(
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
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.reps.enabled is False


def test_build_config_anthropic_provider_routes_to_anthropic(monkeypatch):
    lm = _make_lm(monkeypatch, model="anthropic/claude-sonnet-4.6")
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.provider == "anthropic"
    assert cfg.llm.models[0].name == "claude-sonnet-4.6"


def test_build_config_openrouter_provider_routes_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("reps.llm.openai_compatible.openai.OpenAI"):
        lm = reps.LM("openrouter/google/gemini-2.5-flash")
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.provider == "openrouter"
    assert cfg.llm.models[0].name == "google/gemini-2.5-flash"


def test_constructor_validates_kwargs(monkeypatch):
    lm = _make_lm(monkeypatch)
    with pytest.raises(ValueError, match="max_iterations"):
        Optimizer(lm=lm, max_iterations=0)
    with pytest.raises(ValueError, match="selection_strategy"):
        Optimizer(lm=lm, selection_strategy="random")
    with pytest.raises(ValueError, match="num_islands"):
        Optimizer(lm=lm, num_islands=0)
    with pytest.raises(TypeError, match="reps.LM"):
        Optimizer(lm="not an LM")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# `from_config` escape hatch
# ---------------------------------------------------------------------------


def test_from_config_round_trips():
    cfg = Config()
    cfg.max_iterations = 17
    cfg.database.num_islands = 8
    cfg.database.selection_strategy = "pareto"
    cfg.reps.merge.enabled = True
    opt = Optimizer.from_config(cfg)
    assert opt.max_iterations == 17
    assert opt.num_islands == 8
    assert opt.selection_strategy == "pareto"
    assert opt.merge_enabled is True
    # _build_config returns the original cfg when from_config was used.
    assert opt._build_config(seed=None) is cfg


def test_from_config_rejects_non_config():
    with pytest.raises(TypeError, match="reps.config.Config"):
        Optimizer.from_config("not a Config")  # type: ignore[arg-type]


def test_from_config_stamps_seed():
    cfg = Config()
    opt = Optimizer.from_config(cfg)
    out_cfg = opt._build_config(seed=99)
    assert out_cfg.random_seed == 99
    assert out_cfg.database.random_seed == 99


# ---------------------------------------------------------------------------
# `optimize()` — sync wrapping
# ---------------------------------------------------------------------------


def test_optimize_rejects_running_loop(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)

    async def run():
        with pytest.raises(RuntimeError, match="async context"):
            opt.optimize("seed", lambda code: 1.0)

    asyncio.run(run())


def test_optimize_rejects_non_string_initial(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    with pytest.raises(TypeError, match="program text"):
        opt.optimize(b"bytes", lambda c: 1.0)  # type: ignore[arg-type]


def test_optimize_rejects_non_callable_evaluate(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
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
    opt = Optimizer(lm=lm, max_iterations=3, num_islands=2, output_dir=str(output_dir))

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
    opt = Optimizer(lm=lm, max_iterations=1)

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
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

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


# ===========================================================================
# ADVERSARIAL TESTS
# ===========================================================================


# ---------------------------------------------------------------------------
# coerce_return — additional failure modes
# ---------------------------------------------------------------------------


def test_coerce_none_raises():
    """`None` is a common evaluator bug (forgetting `return`); fail with a
    clear ValueError, not a confusing AttributeError downstream."""
    with pytest.raises(ValueError, match="must return float, dict, or EvaluationResult"):
        coerce_return(None)


def test_coerce_list_raises():
    """A list (e.g. per-instance scores accidentally returned bare) is
    not a valid return shape — must fail explicitly."""
    with pytest.raises(ValueError, match="must return float, dict, or EvaluationResult"):
        coerce_return([0.1, 0.5, 0.9])


def test_coerce_tuple_raises():
    with pytest.raises(ValueError, match="must return float, dict, or EvaluationResult"):
        coerce_return((0.5, "feedback"))


def test_coerce_dict_with_only_combined_score():
    """Spec: `dict` may include only `combined_score` (other fields
    optional). Pass-through, no normalization."""
    payload = {"combined_score": 0.5}
    out = coerce_return(payload)
    assert out is payload  # identity passthrough — no copy


def test_coerce_dict_with_per_instance_and_feedback_preserved():
    """Coercion must not strip optional rich-dict fields."""
    payload = {
        "combined_score": 0.7,
        "validity": 1.0,
        "per_instance_scores": {"a": 0.6, "b": 0.8},
        "feedback": "good",
    }
    out = coerce_return(payload)
    assert out is payload
    assert out["per_instance_scores"] == {"a": 0.6, "b": 0.8}
    assert out["feedback"] == "good"


def test_coerce_evaluation_result_with_per_instance_preserved():
    """EvaluationResult passthrough must preserve per_instance_scores +
    feedback so the Pareto / trace-reflection paths still see them."""
    r = EvaluationResult(
        metrics={"combined_score": 0.5, "validity": 1.0},
        per_instance_scores={"x": 0.4, "y": 0.6},
        feedback="trace",
    )
    out = coerce_return(r)
    assert out is r
    assert out.per_instance_scores == {"x": 0.4, "y": 0.6}
    assert out.feedback == "trace"


def test_coerce_negative_float():
    """Negative scores are valid (e.g. loss minimization)."""
    out = coerce_return(-3.14)
    assert out == {"combined_score": -3.14, "validity": 1.0}


# ---------------------------------------------------------------------------
# dispatch — variant signatures (the three the spec lists)
# ---------------------------------------------------------------------------


def test_dispatch_variant_code_only(tmp_path):
    """Variant 1: `evaluate(code)` — no extra kwargs forwarded."""
    code_path = tmp_path / "p.py"
    code_path.write_text("seed")

    received_kwargs = {}

    def evaluate(code):
        received_kwargs["called_with"] = "code-only"
        return 0.5

    rid = register_user_evaluate(evaluate)
    try:
        out = dispatch_user_evaluate(rid, str(code_path), env={"a": 1})
    finally:
        unregister_user_evaluate(rid)
    # No env/instances accepted by the user signature; both dropped.
    assert received_kwargs["called_with"] == "code-only"
    assert out == {"combined_score": 0.5, "validity": 1.0}


def test_dispatch_variant_code_with_env(tmp_path):
    """Variant 2: `evaluate(code, *, env=None)` — only `env` forwarded."""
    code_path = tmp_path / "p.py"
    code_path.write_text("seed")

    captured = {}

    def evaluate(code, *, env=None):
        captured["env"] = env
        captured["has_instances_kwarg"] = False
        return 0.6

    rid = register_user_evaluate(evaluate)
    try:
        # Pass both; only `env` should be forwarded since `instances` is
        # not in the signature.
        dispatch_user_evaluate(
            rid, str(code_path), env={"X": 1}, instances=["a"]
        )
    finally:
        unregister_user_evaluate(rid)
    assert captured["env"] == {"X": 1}


def test_dispatch_variant_code_env_instances(tmp_path):
    """Variant 3: `evaluate(code, *, env=None, instances=None)` — both
    forwarded."""
    code_path = tmp_path / "p.py"
    code_path.write_text("seed")

    captured = {}

    def evaluate(code, *, env=None, instances=None):
        captured["env"] = env
        captured["instances"] = instances
        return 0.7

    rid = register_user_evaluate(evaluate)
    try:
        dispatch_user_evaluate(
            rid, str(code_path), env={"E": "v"}, instances=["i1", "i2"]
        )
    finally:
        unregister_user_evaluate(rid)
    assert captured["env"] == {"E": "v"}
    assert captured["instances"] == ["i1", "i2"]


# ---------------------------------------------------------------------------
# dispatch — error propagation
# ---------------------------------------------------------------------------


def test_dispatch_propagates_user_exception(tmp_path):
    """If the user's evaluate raises, the dispatch must re-raise (not
    swallow). The harness's evaluator-error handling further up the
    stack converts this into an eval failure."""
    code_path = tmp_path / "p.py"
    code_path.write_text("seed")

    class BoomError(RuntimeError):
        pass

    def evaluate(code):
        raise BoomError("user blew up")

    rid = register_user_evaluate(evaluate)
    try:
        with pytest.raises(BoomError, match="user blew up"):
            dispatch_user_evaluate(rid, str(code_path))
    finally:
        unregister_user_evaluate(rid)


def test_dispatch_user_returning_none_raises_value_error(tmp_path):
    """User forgot `return` — coerce_return must surface a clear
    ValueError rather than letting `None.metrics` AttributeError later."""
    code_path = tmp_path / "p.py"
    code_path.write_text("seed")

    def evaluate(code):
        pass  # implicit None

    rid = register_user_evaluate(evaluate)
    try:
        with pytest.raises(ValueError, match="must return float"):
            dispatch_user_evaluate(rid, str(code_path))
    finally:
        unregister_user_evaluate(rid)


# ---------------------------------------------------------------------------
# Registry — concurrency / cleanup invariants
# ---------------------------------------------------------------------------


def test_concurrent_registrations_get_distinct_ids():
    """Two `optimize()` calls in the same process must get distinct
    registry ids — no cross-contamination of evaluate callables."""
    def fn_a(code): return 0.1
    def fn_b(code): return 0.9

    id_a = register_user_evaluate(fn_a)
    id_b = register_user_evaluate(fn_b)
    try:
        assert id_a != id_b
        # Each id resolves to its own function, not the other's.
        from reps.api.evaluate_dispatch import _REGISTRY
        assert _REGISTRY[id_a] is fn_a
        assert _REGISTRY[id_b] is fn_b
    finally:
        unregister_user_evaluate(id_a)
        unregister_user_evaluate(id_b)


def test_unregister_unknown_id_is_silent():
    """Idempotent cleanup — unregistering an unknown id must not raise."""
    unregister_user_evaluate("never-registered-12345")  # no exception


def test_registry_env_var_does_not_collide_with_existing_reps_env_vars():
    """The shim env var name must not collide with existing REPS_* env
    vars (REPS_PROGRAM_ID, REPS_RUN_DIR, REPS_ATR_MSG_DEBUG)."""
    from reps.api.evaluate_dispatch import _REGISTRY_ENV_VAR
    forbidden = {"REPS_PROGRAM_ID", "REPS_RUN_DIR", "REPS_ATR_MSG_DEBUG"}
    assert _REGISTRY_ENV_VAR not in forbidden


def test_optimize_cleans_up_registry_and_shim_on_success(monkeypatch, tmp_path):
    """After a successful optimize() run, the registry id must be gone
    AND the shim file in the run dir is allowed to remain (it's part of
    the persisted run dir for a persisted output_dir, but the registry
    id itself must not leak)."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    captured_id = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        captured_id["env"] = os.environ.get("REPS_USER_EVALUATOR_ID")
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        opt.optimize("seed", lambda c: 0.5)

    # Env var cleared.
    assert os.environ.get("REPS_USER_EVALUATOR_ID") is None
    # Registry no longer holds that id.
    rid = captured_id["env"]
    assert rid is not None
    from reps.api.evaluate_dispatch import _REGISTRY
    assert rid not in _REGISTRY


def test_optimize_cleans_up_registry_on_exception(monkeypatch, tmp_path):
    """Cleanup must happen even when run_reps raises — finally: block
    invariant."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    captured_id = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        captured_id["env"] = os.environ.get("REPS_USER_EVALUATOR_ID")
        raise RuntimeError("simulated controller crash")

    with patch("reps.runner.run_reps", new=fake_run_reps):
        with pytest.raises(RuntimeError, match="simulated controller crash"):
            opt.optimize("seed", lambda c: 0.5)

    # Env var cleared even on failure.
    assert os.environ.get("REPS_USER_EVALUATOR_ID") is None
    # Registry no longer holds that id.
    rid = captured_id["env"]
    assert rid is not None
    from reps.api.evaluate_dispatch import _REGISTRY
    assert rid not in _REGISTRY


def test_optimize_restores_pre_existing_env_var(monkeypatch, tmp_path):
    """If the env var was already set when optimize() was called
    (nested run, weird env), the prior value must be restored on exit
    — not blindly popped."""
    monkeypatch.setenv("REPS_USER_EVALUATOR_ID", "preexisting-value")

    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        # During the run, the env var should carry the new id, not the
        # preexisting one.
        assert os.environ.get("REPS_USER_EVALUATOR_ID") != "preexisting-value"
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        opt.optimize("seed", lambda c: 0.5)

    assert os.environ.get("REPS_USER_EVALUATOR_ID") == "preexisting-value"


# ---------------------------------------------------------------------------
# Constructor — kwarg → Config field mapping (per-field unit checks)
# ---------------------------------------------------------------------------


def test_build_config_selection_strategy_pareto(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm, selection_strategy="pareto")
    cfg = opt._build_config(seed=None)
    assert cfg.database.selection_strategy == "pareto"


def test_build_config_pareto_fraction_default_is_zero(monkeypatch):
    """Spec promises `pareto_fraction=0.0` default."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.database.pareto_fraction == 0.0


def test_build_config_master_switch_only_trace_reflection(monkeypatch):
    """Setting only `trace_reflection=True` (without merge) flips
    cfg.reps.enabled."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm, trace_reflection=True)
    cfg = opt._build_config(seed=None)
    assert cfg.reps.enabled is True
    assert cfg.reps.trace_reflection.enabled is True
    assert cfg.reps.merge.enabled is False


def test_build_config_master_switch_only_merge(monkeypatch):
    """Setting only `merge=True` (without trace_reflection) flips
    cfg.reps.enabled."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm, merge=True)
    cfg = opt._build_config(seed=None)
    assert cfg.reps.enabled is True
    assert cfg.reps.merge.enabled is True
    assert cfg.reps.trace_reflection.enabled is False


def test_build_config_summarizer_disabled_by_default(monkeypatch):
    """Spec note: summarizer is opt-in via from_config to avoid silent
    use of the user's API key for a second LLM."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.reps.summarizer.enabled is False


def test_build_config_evaluator_models_independent_from_models(monkeypatch):
    """`_clone_model_cfg` exists specifically so post-init mutations on
    cfg.llm.models[0] don't bleed into evaluator_models[0]."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert len(cfg.llm.models) == 1
    assert len(cfg.llm.evaluator_models) == 1
    # Different python objects, identical values.
    assert cfg.llm.models[0] is not cfg.llm.evaluator_models[0]
    # Mutating one does not affect the other.
    cfg.llm.models[0].temperature = 9.9
    assert cfg.llm.evaluator_models[0].temperature != 9.9


def test_build_config_minibatch_default_is_none(monkeypatch):
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    assert cfg.evaluator.minibatch_size is None


def test_build_config_no_seed_does_not_overwrite_default(monkeypatch):
    """When `seed=None`, _build_config must leave random_seed alone
    (Config has its own default of 42)."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm)
    cfg = opt._build_config(seed=None)
    # Config defaults random_seed to 42; we don't clobber it to None.
    assert cfg.random_seed is not None


# ---------------------------------------------------------------------------
# Constructor — validation edges
# ---------------------------------------------------------------------------


def test_constructor_rejects_negative_max_iterations(monkeypatch):
    lm = _make_lm(monkeypatch)
    with pytest.raises(ValueError, match="max_iterations"):
        Optimizer(lm=lm, max_iterations=-5)


def test_constructor_rejects_zero_islands(monkeypatch):
    lm = _make_lm(monkeypatch)
    with pytest.raises(ValueError, match="num_islands"):
        Optimizer(lm=lm, num_islands=0)


def test_constructor_rejects_lm_being_none():
    with pytest.raises(TypeError, match="reps.LM"):
        Optimizer(lm=None)  # type: ignore[arg-type]


def test_constructor_rejects_lm_being_wrong_type():
    """Passing an LLMModelConfig directly (not an LM) must fail with a
    helpful error pointing at `reps.LM`."""
    from reps.config import LLMModelConfig
    with pytest.raises(TypeError, match="reps.LM"):
        Optimizer(lm=LLMModelConfig(name="x"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_config — partial Config + round-trip semantics
# ---------------------------------------------------------------------------


def test_from_config_partial_config_uses_defaults():
    """A bare `Config()` (only dataclass defaults) must construct a
    valid Optimizer — users shouldn't need to know which fields are
    required."""
    cfg = Config()  # no overrides
    opt = Optimizer.from_config(cfg)
    # Defaults from Config flow through.
    assert opt.max_iterations == cfg.max_iterations
    assert opt.num_islands == cfg.database.num_islands
    assert opt.selection_strategy == cfg.database.selection_strategy
    # _build_config returns the original cfg verbatim.
    assert opt._build_config(seed=None) is cfg


def test_from_config_preserves_all_kwarg_mapped_fields():
    """Round-trip every field the simple constructor maps through
    from_config — any field rename in Config will surface here."""
    cfg = Config()
    cfg.max_iterations = 31
    cfg.database.selection_strategy = "mixed"
    cfg.database.pareto_fraction = 0.42
    cfg.database.num_islands = 9
    cfg.reps.trace_reflection.enabled = True
    cfg.reps.trace_reflection.lineage_depth = 5
    cfg.reps.merge.enabled = True
    cfg.evaluator.minibatch_size = 7
    cfg.output = "/some/path"

    opt = Optimizer.from_config(cfg)
    assert opt.max_iterations == 31
    assert opt.selection_strategy == "mixed"
    assert opt.pareto_fraction == 0.42
    assert opt.num_islands == 9
    assert opt.trace_reflection_enabled is True
    assert opt.lineage_depth == 5
    assert opt.merge_enabled is True
    assert opt.minibatch_size == 7
    assert opt.output_dir == "/some/path"


def test_from_config_does_not_force_summarizer_off():
    """The simple constructor disables summarizer by default; from_config
    is the escape hatch — summarizer setting in cfg must be honored."""
    cfg = Config()
    cfg.reps.summarizer.enabled = True
    opt = Optimizer.from_config(cfg)
    out_cfg = opt._build_config(seed=None)
    # _build_config passes cfg through unchanged when from_config was used.
    assert out_cfg.reps.summarizer.enabled is True


def test_from_config_with_no_models_uses_none_lm():
    """Edge case: `Config.llm.models` is `[]` by default — `from_config`
    sets `instance.lm = None` rather than crashing."""
    cfg = Config()
    # Explicitly empty
    cfg.llm.models = []
    opt = Optimizer.from_config(cfg)
    assert opt.lm is None


def test_from_config_with_models_builds_stub_lm():
    """Round-trip — when cfg.llm.models has an entry, from_config builds
    a _StubLM so introspection works."""
    cfg = Config()
    from reps.config import LLMModelConfig
    cfg.llm.models = [LLMModelConfig(name="claude-x", provider="anthropic")]
    opt = Optimizer.from_config(cfg)
    assert opt.lm is not None
    assert opt.lm.model == "claude-x"
    assert opt.lm.provider == "anthropic"


def test_from_config_rejects_dict():
    """`from_config_dict` is deferred to v1.5; a raw dict must error."""
    with pytest.raises(TypeError, match="reps.config.Config"):
        Optimizer.from_config({"max_iterations": 5})  # type: ignore[arg-type]


def test_from_config_rejects_none():
    with pytest.raises(TypeError, match="reps.config.Config"):
        Optimizer.from_config(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# optimize() — sync wrapper safety from a fresh thread
# ---------------------------------------------------------------------------


def test_optimize_works_outside_async_context(monkeypatch, tmp_path):
    """Sanity: from a vanilla synchronous context (no running loop),
    optimize() must succeed."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)
    assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# OptimizationResult — empty database edge case
# ---------------------------------------------------------------------------


def test_collect_result_handles_empty_database(monkeypatch, tmp_path):
    """If `run_reps` finishes without producing programs (rare —
    controller died early?) the collector must return an
    OptimizationResult with sensible defaults, not crash on
    `best is None`."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        # Empty DB — just metadata, no programs/.
        out = Path(output_dir)
        (out / "programs").mkdir(parents=True, exist_ok=True)
        metadata = {
            "best_program_id": None,
            "islands": [[] for _ in range(config.database.num_islands)],
            "island_best_programs": [None] * config.database.num_islands,
            "archive": [],
            "last_iteration": 0,
            "current_island": 0,
            "island_generations": [0] * config.database.num_islands,
            "last_migration_generation": 0,
            "feature_stats": {},
        }
        (out / "metadata.json").write_text(json.dumps(metadata))

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    assert isinstance(result, OptimizationResult)
    assert result.best_code == ""
    assert result.best_score == 0.0
    assert result.output_dir == str(output_dir.resolve())


def test_collect_result_total_metric_calls_counts_all_programs(monkeypatch, tmp_path):
    """`total_metric_calls` is set to `len(db.programs)` — verify with
    a 3-program DB."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        out = Path(output_dir)
        (out / "programs").mkdir(parents=True, exist_ok=True)
        metadata = {
            "best_program_id": "p2",
            "islands": [[] for _ in range(config.database.num_islands)],
            "island_best_programs": [None] * config.database.num_islands,
            "archive": [],
            "last_iteration": 5,
            "current_island": 0,
            "island_generations": [0] * config.database.num_islands,
            "last_migration_generation": 0,
            "feature_stats": {},
        }
        (out / "metadata.json").write_text(json.dumps(metadata))
        for i, score in enumerate([0.1, 0.5, 0.9]):
            prog = {
                "id": f"p{i}",
                "code": f"# program {i}",
                "language": "python",
                "metrics": {"combined_score": score, "validity": 1.0},
                "iteration_found": i * 2,
                "metadata": {},
            }
            (out / "programs" / f"p{i}.json").write_text(json.dumps(prog))

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    assert result.total_metric_calls == 3
    assert result.iterations_run == 4  # max iteration_found among non-seeds (i=2 -> 4)


def test_collect_result_aggregates_tokens_across_programs(monkeypatch, tmp_path):
    """`total_tokens` sums tokens_in / tokens_out across all programs'
    `reps_meta` dicts."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "run"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        out = Path(output_dir)
        (out / "programs").mkdir(parents=True, exist_ok=True)
        metadata = {
            "best_program_id": "p1",
            "islands": [[] for _ in range(config.database.num_islands)],
            "island_best_programs": [None] * config.database.num_islands,
            "archive": [],
            "last_iteration": 3,
            "current_island": 0,
            "island_generations": [0] * config.database.num_islands,
            "last_migration_generation": 0,
            "feature_stats": {},
        }
        (out / "metadata.json").write_text(json.dumps(metadata))
        # Three programs with varying token metadata.
        progs = [
            ("p0", {}, 1, 0.1),  # no reps_meta — contributes 0.
            ("p1", {"reps_meta": {"tokens_in": 100, "tokens_out": 50}}, 2, 0.5),
            ("p2", {"reps_meta": {"tokens_in": 200, "tokens_out": 75}}, 3, 0.9),
        ]
        for pid, md, it, score in progs:
            prog = {
                "id": pid,
                "code": "# x",
                "language": "python",
                "metrics": {"combined_score": score, "validity": 1.0},
                "iteration_found": it,
                "metadata": md,
            }
            (out / "programs" / f"{pid}.json").write_text(json.dumps(prog))

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    assert result.total_tokens == {"in": 300, "out": 125}


# ---------------------------------------------------------------------------
# OptimizationResult — output_dir resolution semantics
# ---------------------------------------------------------------------------


def test_output_dir_none_yields_none_on_result(monkeypatch, tmp_path):
    """Spec: when `output_dir=None`, the result's `output_dir` is
    `None` (the run used a tempdir internally — not exposed)."""
    lm = _make_lm(monkeypatch)
    opt = Optimizer(lm=lm, max_iterations=1)  # output_dir=None

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    assert result.output_dir is None


def test_output_dir_set_yields_resolved_absolute_path_on_result(monkeypatch, tmp_path):
    """When `output_dir` is set, the resolved absolute path appears on
    the result (so users can re-load programs after the call returns)."""
    lm = _make_lm(monkeypatch)
    output_dir = tmp_path / "myrun"
    opt = Optimizer(lm=lm, max_iterations=1, output_dir=str(output_dir))

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir):
        _seed_database(output_dir, num_islands=config.database.num_islands)

    with patch("reps.runner.run_reps", new=fake_run_reps):
        result = opt.optimize("seed", lambda c: 0.5)

    # Resolved path (absolute, no symlinks) — survives across cwd changes.
    assert result.output_dir == str(output_dir.resolve())
    assert Path(result.output_dir).is_absolute()


# ---------------------------------------------------------------------------
# `reps.internal` back-compat — full re-export check + symbol identity
# ---------------------------------------------------------------------------


def test_reps_internal_full_symbol_list():
    """The full set of symbols promised by reps/internal/__init__.py is
    importable. Catches accidentally-dropped re-exports."""
    from reps.internal import (
        ReflectionEngine,
        WorkerPool,
        ConvergenceMonitor,
        ConvergenceAction,
        ContractSelector,
        Contract,
        SOTAController,
        SearchRegime,
        MetricsLogger,
        IterationConfig,
        IterationResult,
    )
    # Sanity: each is a class/dataclass (not None)
    for sym in (
        ReflectionEngine, WorkerPool, ConvergenceMonitor, ConvergenceAction,
        ContractSelector, Contract, SOTAController, SearchRegime,
        MetricsLogger, IterationConfig, IterationResult,
    ):
        assert sym is not None


def test_reps_internal_symbols_match_top_level_for_back_compat():
    """`from reps import ReflectionEngine` and `from reps.internal import
    ReflectionEngine` must point at the same object — that's the
    back-compat contract."""
    import reps as top_level
    import reps.internal as internal
    for name in (
        "ReflectionEngine", "WorkerPool", "ConvergenceMonitor",
        "ConvergenceAction", "ContractSelector", "Contract",
        "SOTAController", "SearchRegime", "MetricsLogger",
        "IterationConfig", "IterationResult",
    ):
        assert getattr(top_level, name) is getattr(internal, name), (
            f"{name} re-export drifted between reps and reps.internal"
        )
