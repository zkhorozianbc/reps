"""Unit tests for `reps.interpret` and the Evaluator-level integration.

The interpret module is pure — these tests don't touch the LLM, the
controller, or any optimizer state. The Evaluator-level test exercises the
single integration seam: the `interpret` callable should override
`combined_score` after `_apply_interpretation` runs.
"""

from __future__ import annotations

import math

import pytest

from reps import interpret as I
from reps.evaluation_result import EvaluationResult


# ---------------------------------------------------------------------------
# combined() — default / legacy-preserving
# ---------------------------------------------------------------------------


def test_combined_uses_metrics_combined_score_when_present():
    fn = I.combined()
    assert fn({"a": 0.0, "b": 1.0}, {"combined_score": 0.5}) == 0.5


def test_combined_falls_back_to_mean_when_combined_score_absent():
    fn = I.combined()
    assert fn({"a": 0.4, "b": 0.6}, {}) == pytest.approx(0.5)


def test_combined_falls_back_when_combined_score_nonfinite():
    fn = I.combined()
    assert fn({"a": 0.4, "b": 0.6}, {"combined_score": float("nan")}) == pytest.approx(0.5)
    assert fn({"a": 0.4, "b": 0.6}, {"combined_score": float("inf")}) == pytest.approx(0.5)


def test_combined_custom_fallback_used_when_combined_score_absent():
    fn = I.combined(fallback=I.worst())
    assert fn({"a": 0.1, "b": 0.9}, {}) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# mean / worst / best
# ---------------------------------------------------------------------------


def test_mean_simple():
    assert I.mean()({"a": 1.0, "b": 2.0, "c": 3.0}, {}) == pytest.approx(2.0)


def test_mean_drops_nan_and_inf():
    fn = I.mean()
    # NaN and inf get filtered; mean of remaining {1.0, 3.0} = 2.0.
    assert fn({"a": 1.0, "b": float("nan"), "c": 3.0, "d": float("inf")}, {}) == pytest.approx(2.0)


def test_mean_falls_back_when_per_instance_empty():
    fn = I.mean()
    assert fn(None, {"combined_score": 0.42}) == 0.42
    assert fn({}, {"combined_score": 0.42}) == 0.42
    assert fn(None, {}) == 0.0


def test_worst_returns_min():
    assert I.worst()({"a": 0.9, "b": 0.1, "c": 0.5}, {}) == pytest.approx(0.1)


def test_best_returns_max():
    assert I.best()({"a": 0.9, "b": 0.1, "c": 0.5}, {}) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------------


def test_quantile_endpoints_match_min_and_max():
    scores = {"a": 0.0, "b": 0.5, "c": 1.0}
    assert I.quantile(0.0)(scores, {}) == pytest.approx(0.0)
    assert I.quantile(1.0)(scores, {}) == pytest.approx(1.0)


def test_quantile_median_three_points():
    assert I.quantile(0.5)({"a": 0.0, "b": 0.5, "c": 1.0}, {}) == pytest.approx(0.5)


def test_quantile_interpolates_between_ranks():
    # Two points: q=0.5 → halfway between them.
    assert I.quantile(0.5)({"a": 0.0, "b": 1.0}, {}) == pytest.approx(0.5)
    # Four points (0,1,2,3): q=0.25 → linear interp between rank 0 (=0) and rank 1 (=1) at frac=0.75.
    assert I.quantile(0.25)({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}, {}) == pytest.approx(0.75)


def test_quantile_rejects_out_of_range():
    with pytest.raises(ValueError):
        I.quantile(-0.1)
    with pytest.raises(ValueError):
        I.quantile(1.5)


# ---------------------------------------------------------------------------
# cvar
# ---------------------------------------------------------------------------


def test_cvar_full_alpha_equals_mean():
    scores = {"a": 0.1, "b": 0.5, "c": 0.9}
    assert I.cvar(1.0)(scores, {}) == pytest.approx(0.5)


def test_cvar_takes_worst_fraction():
    # 10 instances, alpha=0.2 → worst 2: {0.0, 0.1}, mean = 0.05.
    scores = {f"i{i}": i * 0.1 for i in range(10)}
    assert I.cvar(0.2)(scores, {}) == pytest.approx(0.05)


def test_cvar_ceils_to_at_least_one_instance():
    # alpha=0.01 over 5 instances → ceil(0.05)=1 → just the worst single value.
    scores = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.4, "e": 0.7}
    assert I.cvar(0.01)(scores, {}) == pytest.approx(0.1)


def test_cvar_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        I.cvar(0.0)
    with pytest.raises(ValueError):
        I.cvar(1.5)


# ---------------------------------------------------------------------------
# weighted
# ---------------------------------------------------------------------------


def test_weighted_basic():
    fn = I.weighted({"a": 1.0, "b": 3.0})
    # (1*1 + 3*0.0) / (1+3) = 0.25
    assert fn({"a": 1.0, "b": 0.0}, {}) == pytest.approx(0.25)


def test_weighted_skips_keys_absent_in_scores():
    fn = I.weighted({"a": 1.0, "missing": 5.0})
    # Only "a" overlaps; result == score["a"].
    assert fn({"a": 0.7}, {}) == pytest.approx(0.7)


def test_weighted_falls_back_when_no_overlap():
    fn = I.weighted({"x": 1.0})
    assert fn({"a": 1.0}, {"combined_score": 0.42}) == 0.42


def test_weighted_rejects_negative_weights():
    with pytest.raises(ValueError):
        I.weighted({"a": -1.0})


# ---------------------------------------------------------------------------
# pass_rate
# ---------------------------------------------------------------------------


def test_pass_rate_threshold_inclusive():
    fn = I.pass_rate(0.5)
    # 0.5 passes (>=); 0.4 fails; 0.7 passes. 2/3.
    assert fn({"a": 0.5, "b": 0.4, "c": 0.7}, {}) == pytest.approx(2 / 3)


def test_pass_rate_all_fail():
    assert I.pass_rate(1.0)({"a": 0.0, "b": 0.5}, {}) == 0.0


def test_pass_rate_all_pass():
    assert I.pass_rate(0.0)({"a": 0.0, "b": 0.5}, {}) == 1.0


# ---------------------------------------------------------------------------
# Empty / fallback behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [I.mean(), I.worst(), I.best(), I.quantile(0.5), I.cvar(0.5), I.pass_rate(0.5)],
)
def test_all_interpretations_fall_back_on_empty_per_instance(fn):
    assert fn(None, {"combined_score": 0.42}) == 0.42
    assert fn({}, {"combined_score": 0.42}) == 0.42
    assert fn(None, {}) == 0.0


# ---------------------------------------------------------------------------
# Evaluator integration: the only place where interpret() is wired into the
# harness. Test by constructing an Evaluator with a stub config and bypassing
# the file-loading path.
# ---------------------------------------------------------------------------


class _StubEvaluatorConfig:
    """Minimal stand-in for EvaluatorConfig with the fields Evaluator reads."""
    parallel_evaluations = 1
    cascade_evaluation = False
    max_retries = 0
    timeout = 60
    use_llm_feedback = False
    llm_feedback_weight = 0.0


def _make_evaluator(interpret_fn):
    """Build an Evaluator that skips file loading — we only test the
    `_apply_interpretation` path, not the subprocess pipeline."""
    from reps.evaluator import Evaluator

    e = Evaluator.__new__(Evaluator)
    e.config = _StubEvaluatorConfig()
    e.evaluation_file = "<stub>"
    e.program_suffix = ".py"
    e.llm_ensemble = None
    e.prompt_sampler = None
    e.database = None
    e._pending_artifacts = {}
    e.interpret = interpret_fn
    return e


def test_evaluator_interpret_none_is_identity():
    e = _make_evaluator(None)
    er = EvaluationResult(metrics={"combined_score": 0.7}, per_instance_scores={"a": 0.1, "b": 0.9})
    out = e._apply_interpretation(er)
    assert out == {"combined_score": 0.7}


def test_evaluator_interpret_overrides_combined_score():
    e = _make_evaluator(I.worst())
    er = EvaluationResult(
        metrics={"combined_score": 0.7},
        per_instance_scores={"a": 0.1, "b": 0.9},
    )
    out = e._apply_interpretation(er)
    assert out["combined_score"] == pytest.approx(0.1)
    assert out["combined_score_raw"] == pytest.approx(0.7)
    assert "interpreted_score" not in out  # dead key removed


def test_evaluator_interpret_fills_combined_score_when_missing():
    e = _make_evaluator(I.mean())
    er = EvaluationResult(metrics={}, per_instance_scores={"a": 0.4, "b": 0.6})
    out = e._apply_interpretation(er)
    assert out["combined_score"] == pytest.approx(0.5)
    assert "combined_score_raw" not in out  # nothing to preserve
    assert "interpreted_score" not in out


def test_evaluator_interpret_failure_falls_back_silently():
    def boom(_per_instance, _metrics):
        raise RuntimeError("synthetic")

    e = _make_evaluator(boom)
    er = EvaluationResult(
        metrics={"combined_score": 0.5},
        per_instance_scores={"a": 0.1},
    )
    # Must not raise; metrics returned unchanged.
    out = e._apply_interpretation(er)
    assert out == {"combined_score": 0.5}


def test_evaluator_interpret_propagates_to_outcome_metrics():
    """End-to-end: per_instance_scores survives, metrics carries the override."""
    e = _make_evaluator(I.cvar(0.5))
    er = EvaluationResult(
        metrics={"combined_score": 0.99},
        per_instance_scores={"i1": 0.0, "i2": 0.5, "i3": 1.0, "i4": 1.0},
    )
    out = e._apply_interpretation(er)
    # Worst 50% of 4 = 2 instances: {0.0, 0.5}, mean = 0.25.
    assert out["combined_score"] == pytest.approx(0.25)
    assert out["combined_score_raw"] == pytest.approx(0.99)


def test_evaluator_interpret_does_not_mutate_caller_dict():
    """Regression: `from_dict` aliases the user's dict; interpretation must
    not write into it.
    """
    e = _make_evaluator(I.worst())
    user_metrics = {"combined_score": 0.7}
    er = EvaluationResult(metrics=user_metrics, per_instance_scores={"a": 0.1, "b": 0.9})
    e._apply_interpretation(er)
    # The caller's original dict must be unchanged.
    assert user_metrics == {"combined_score": 0.7}
    # But the EvaluationResult should now point at the interpreted dict.
    assert er.metrics["combined_score"] == pytest.approx(0.1)
    assert er.metrics is not user_metrics


def test_pass_rate_rejects_nonfinite_threshold():
    with pytest.raises(ValueError):
        I.pass_rate(float("nan"))
    with pytest.raises(ValueError):
        I.pass_rate(float("inf"))


# ---------------------------------------------------------------------------
# from_spec — YAML/CLI bridge
# ---------------------------------------------------------------------------


def test_from_spec_parameterless():
    fn = I.from_spec("mean")
    assert fn({"a": 0.0, "b": 1.0}, {}) == pytest.approx(0.5)


def test_from_spec_empty_parens():
    fn = I.from_spec("worst()")
    assert fn({"a": 0.1, "b": 0.9}, {}) == pytest.approx(0.1)


def test_from_spec_with_args():
    fn = I.from_spec("cvar(0.5)")
    # Worst 50% of {0.0, 0.5, 1.0, 1.0} → ceil(2) = 2 → mean({0, 0.5}) = 0.25.
    assert fn({"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.0}, {}) == pytest.approx(0.25)


def test_from_spec_tolerates_whitespace():
    fn = I.from_spec("  quantile( 0.5 )  ")
    assert fn({"a": 0.0, "b": 0.5, "c": 1.0}, {}) == pytest.approx(0.5)


def test_from_spec_unknown_name():
    with pytest.raises(ValueError, match="unknown interpret name"):
        I.from_spec("bogus")


def test_from_spec_missing_close_paren():
    with pytest.raises(ValueError, match="closing paren"):
        I.from_spec("cvar(0.5")


def test_from_spec_non_numeric_arg():
    with pytest.raises(ValueError, match="numeric"):
        I.from_spec("cvar(low)")


# ---------------------------------------------------------------------------
# Optimizer.from_config — interpret kwarg + YAML field plumbing
# ---------------------------------------------------------------------------


def test_optimizer_from_config_kwarg_wins_over_yaml():
    from reps.api.optimizer import Optimizer
    from reps.config import Config

    cfg = Config()
    cfg.evaluator.interpret = "mean"
    explicit = I.worst()
    opt = Optimizer.from_config(cfg, interpret=explicit)
    assert opt.interpret is explicit


def test_optimizer_from_config_parses_yaml_spec_when_kwarg_absent():
    from reps.api.optimizer import Optimizer
    from reps.config import Config

    cfg = Config()
    cfg.evaluator.interpret = "cvar(0.5)"
    opt = Optimizer.from_config(cfg)
    # Round-trip: feed the resolved callable a known distribution.
    assert opt.interpret is not None
    assert opt.interpret({"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.0}, {}) == pytest.approx(0.25)


def test_optimizer_from_config_no_interpret_means_none():
    from reps.api.optimizer import Optimizer
    from reps.config import Config

    cfg = Config()
    opt = Optimizer.from_config(cfg)
    assert opt.interpret is None


# ---------------------------------------------------------------------------
# End-to-end: Optimizer(interpret=...) → run_reps → controller.evaluator
# ---------------------------------------------------------------------------


def test_optimizer_threads_interpret_to_controller_evaluator():
    """Verify the full plumbing: an interpret callable passed to `Optimizer`
    reaches the per-iteration `Evaluator` constructed inside the controller.
    Mocks `run_reps` and inspects what it was called with.
    """
    import asyncio
    from unittest.mock import patch

    from reps.api.optimizer import Optimizer

    captured: dict = {}

    async def fake_run_reps(*, config, initial_program, evaluator, output_dir, interpret=None):
        captured["interpret"] = interpret
        # Exercise the production path that builds the controller's Evaluator
        # so a regression in run_reps wiring would surface here too.
        from reps.controller import ProcessParallelController
        from reps.database import ProgramDatabase

        db = ProgramDatabase(config.database)
        controller = ProcessParallelController(
            config=config,
            evaluation_file=evaluator,
            database=db,
            output_dir=output_dir,
            interpret=interpret,
        )
        controller.start()
        captured["controller_evaluator_interpret"] = controller.evaluator.interpret
        controller.stop()

    chosen = I.worst()
    opt = Optimizer(model="anthropic/claude-sonnet-4.6", api_key="x", interpret=chosen)
    with patch("reps.runner.run_reps", new=fake_run_reps):
        try:
            opt.optimize("seed_code", lambda c: 0.5)
        except Exception:
            # `_collect_result` will fail because no DB was saved — fine, we
            # only care that run_reps was reached with the right kwarg.
            pass
    assert captured["interpret"] is chosen
    assert captured["controller_evaluator_interpret"] is chosen


def test_evaluator_config_interpret_yaml_field_round_trips():
    """The CLI codepath relies on `cfg.evaluator.interpret` being a string
    that `from_spec` can parse. Round-trip through the dataclass to ensure
    nothing in Config init clobbers or transforms the field.
    """
    from reps.config import Config

    cfg = Config()
    assert cfg.evaluator.interpret is None  # default
    cfg.evaluator.interpret = "cvar(0.5)"
    fn = I.from_spec(cfg.evaluator.interpret)
    assert fn({"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.0}, {}) == pytest.approx(0.25)
