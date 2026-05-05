"""Phase 6.2 — evaluator wiring tests for `Evaluator.evaluate_with_promotion`.

Mock-LLM-free: each test stands up a tiny in-memory benchmark file with
the signatures the wiring contract expects, instantiates an `Evaluator`,
and drives the promotion path end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from reps.config import EvaluatorConfig
from reps.evaluator import Evaluator


def _write_bench(
    tmp_path: Path,
    *,
    minibatch_score: float = 0.4,
    full_score: float = 0.9,
    registry_kind: str = "constant",  # "constant" | "function" | "missing"
    accepts_instances: bool = True,
) -> Path:
    """Write a tiny benchmark module with controllable fidelity behavior.

    The benchmark records each `evaluate(...)` call into a sidecar JSON
    file so tests can inspect call ordering / instances without brittle
    monkey-patching of the imported module.
    """
    log = tmp_path / "calls.json"
    log.write_text("[]")

    if registry_kind == "constant":
        registry_block = "INSTANCES = ['t0', 't1', 't2', 't3']\n"
    elif registry_kind == "function":
        registry_block = (
            "def list_instances():\n"
            "    return ['t0', 't1', 't2', 't3']\n"
        )
    else:
        registry_block = ""  # missing on purpose

    if accepts_instances:
        sig = "def evaluate(program_path, env=None, instances=None):\n"
        inst_expr = "list(instances) if instances is not None else None"
    else:
        sig = "def evaluate(program_path, env=None):\n"
        inst_expr = "None"

    body = (
        "    calls = json.loads(_LOG.read_text())\n"
        f"    inst = {inst_expr}\n"
        "    calls.append({'instances': inst})\n"
        "    _LOG.write_text(json.dumps(calls))\n"
        "    score = _MB_SCORE if inst is not None else _FULL_SCORE\n"
        "    return {'combined_score': score, 'validity': 1.0}\n"
    )

    header = (
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"{registry_block}"
        "\n"
        f"_LOG = Path({str(log)!r})\n"
        f"_MB_SCORE = {minibatch_score}\n"
        f"_FULL_SCORE = {full_score}\n"
        "\n"
    )

    bench = tmp_path / "bench.py"
    bench.write_text(header + sig + body)
    return bench


def _read_calls(tmp_path: Path) -> list:
    import json
    return json.loads((tmp_path / "calls.json").read_text())


def _make_evaluator(
    bench: Path,
    *,
    cascade: bool = False,
    minibatch_size=None,
    threshold: float = 0.5,
    strategy: str = "fixed_subset",
) -> Evaluator:
    cfg = EvaluatorConfig(
        cascade_evaluation=cascade,
        timeout=10,
        parallel_evaluations=1,
        max_retries=0,
        minibatch_size=minibatch_size,
        minibatch_promotion_threshold=threshold,
        minibatch_strategy=strategy,
    )
    return Evaluator(cfg, str(bench))


class TestMutualExclusionAtConfigLoad:
    def test_minibatch_with_cascade_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            EvaluatorConfig(cascade_evaluation=True, minibatch_size=4)

    def test_minibatch_without_cascade_is_fine(self):
        cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_size=4)
        assert cfg.minibatch_size == 4

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="minibatch_strategy"):
            EvaluatorConfig(cascade_evaluation=False, minibatch_strategy="bogus")


class TestPromotionPath:
    def test_minibatch_below_threshold_skips_full_eval(self, tmp_path: Path):
        bench = _write_bench(tmp_path, minibatch_score=0.2, full_score=0.9)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["fidelity"] == "minibatch"
        assert outcome.metrics["combined_score"] == 0.2
        # Only the minibatch call should have happened.
        calls = _read_calls(tmp_path)
        assert len(calls) == 1
        assert calls[0]["instances"] is not None
        assert len(calls[0]["instances"]) == 2

    def test_minibatch_at_or_above_threshold_runs_full_eval(self, tmp_path: Path):
        bench = _write_bench(tmp_path, minibatch_score=0.7, full_score=0.95)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["fidelity"] == "full"
        # Full result wins — combined_score reflects the highest-fidelity eval.
        assert outcome.metrics["combined_score"] == 0.95
        calls = _read_calls(tmp_path)
        assert len(calls) == 2
        assert calls[0]["instances"] is not None  # minibatch first
        assert calls[1]["instances"] is None  # full eval second

    def test_minibatch_size_none_is_byte_identical(self, tmp_path: Path):
        # When minibatch_size is None, evaluate_with_promotion must defer
        # to the legacy direct path — exactly one call, instances=None.
        bench = _write_bench(tmp_path, minibatch_score=0.2, full_score=0.85)
        ev = _make_evaluator(bench, minibatch_size=None)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=5)
        )
        assert outcome.metrics["combined_score"] == 0.85
        # `fidelity` should still be tagged so archive policy has consistent input.
        assert outcome.metrics["fidelity"] == "full"
        calls = _read_calls(tmp_path)
        assert len(calls) == 1
        assert calls[0]["instances"] is None

    def test_random_strategy_passes_subset(self, tmp_path: Path):
        bench = _write_bench(tmp_path, minibatch_score=0.7, full_score=0.95)
        ev = _make_evaluator(
            bench, minibatch_size=2, threshold=0.5, strategy="random"
        )
        asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=3)
        )
        calls = _read_calls(tmp_path)
        # Minibatch + full
        assert len(calls) == 2
        assert calls[0]["instances"] is not None
        assert len(calls[0]["instances"]) == 2

    def test_list_instances_function_works(self, tmp_path: Path):
        bench = _write_bench(
            tmp_path, minibatch_score=0.2, full_score=0.95, registry_kind="function"
        )
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["fidelity"] == "minibatch"

    def test_missing_registry_raises(self, tmp_path: Path):
        bench = _write_bench(
            tmp_path, minibatch_score=0.2, full_score=0.95, registry_kind="missing"
        )
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        with pytest.raises(ValueError, match="INSTANCES"):
            asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="x", iteration=0
                )
            )

    def test_legacy_evaluator_without_instances_falls_back(self, tmp_path: Path, caplog):
        # Benchmark whose `evaluate(program_path, env=None)` does NOT accept
        # `instances`. Must not raise; must fall back to a single full eval
        # and emit a one-time warning.
        bench = _write_bench(
            tmp_path,
            minibatch_score=0.2,
            full_score=0.95,
            registry_kind="constant",
            accepts_instances=False,
        )
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        with caplog.at_level("WARNING"):
            outcome1 = asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="x", iteration=0
                )
            )
            outcome2 = asyncio.run(
                ev.evaluate_with_promotion(
                    "def f(): return 1\n", program_id="y", iteration=1
                )
            )
        assert outcome1.metrics["fidelity"] == "full"
        assert outcome2.metrics["fidelity"] == "full"
        # Both calls were full evals (instances=None).
        calls = _read_calls(tmp_path)
        assert len(calls) == 2
        assert all(c["instances"] is None for c in calls)
        # Warning emitted once, not twice.
        warnings = [
            r for r in caplog.records if "instances" in r.getMessage().lower()
        ]
        assert len(warnings) == 1

    def test_subset_at_or_above_registry_size_takes_full_path(self, tmp_path: Path):
        # When minibatch_size >= len(INSTANCES), select_instances returns the
        # full set; running a separate "minibatch" pass would be wasted work,
        # so the wiring goes straight to full eval.
        bench = _write_bench(tmp_path, minibatch_score=0.2, full_score=0.95)
        ev = _make_evaluator(bench, minibatch_size=10, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["fidelity"] == "full"
        calls = _read_calls(tmp_path)
        assert len(calls) == 1
        assert calls[0]["instances"] is None


class TestCombinedScoreFidelityRecord:
    def test_promoted_combined_score_reflects_full_eval(self, tmp_path: Path):
        bench = _write_bench(tmp_path, minibatch_score=0.55, full_score=0.92)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["combined_score"] == 0.92
        assert outcome.metrics["fidelity"] == "full"

    def test_rejected_combined_score_reflects_minibatch_eval(self, tmp_path: Path):
        bench = _write_bench(tmp_path, minibatch_score=0.10, full_score=0.92)
        ev = _make_evaluator(bench, minibatch_size=2, threshold=0.5)
        outcome = asyncio.run(
            ev.evaluate_with_promotion("def f(): return 1\n", program_id="x", iteration=0)
        )
        assert outcome.metrics["combined_score"] == 0.10
        assert outcome.metrics["fidelity"] == "minibatch"
