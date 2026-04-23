"""Tests for REPS feature modules F1-F8."""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from reps.iteration_config import IterationConfig, IterationResult
from reps.worker_pool import WorkerPool
from reps.convergence_monitor import ConvergenceMonitor, ConvergenceAction, classify_edit
from reps.contract_selector import ContractSelector, Contract
from reps.sota_controller import SOTAController, SearchRegime
from reps.reflection_engine import ReflectionEngine
from reps.metrics_logger import MetricsLogger


# ---------------------------------------------------------------------------
# F3: WorkerPool
# ---------------------------------------------------------------------------


class TestWorkerPool:
    """Tests for the WorkerPool (F3: Worker Type Diversity)."""

    def test_initialization_with_config(self):
        config = {
            "types": ["exploiter", "explorer", "crossover"],
            "initial_allocation": {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15},
        }
        pool = WorkerPool(config)
        assert set(pool.allocation.keys()) == {"exploiter", "explorer", "crossover"}
        assert len(pool.yield_tracker) == 3

    def test_initialization_defaults(self):
        pool = WorkerPool({})
        assert "exploiter" in pool.allocation
        assert "explorer" in pool.allocation
        assert "crossover" in pool.allocation

    def test_build_iteration_config_returns_iteration_config(self):
        pool = WorkerPool({"types": ["exploiter"], "initial_allocation": {"exploiter": 1.0}})
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={"reflection": "test"})
        assert isinstance(config, IterationConfig)
        assert config.worker_type == "exploiter"

    def test_build_iteration_config_override_type(self):
        pool = WorkerPool({"types": ["exploiter", "explorer"]})
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={}, override_type="explorer")
        assert config.worker_type == "explorer"

    def test_build_iteration_config_temperature_from_worker(self):
        pool = WorkerPool({
            "types": ["exploiter"],
            "initial_allocation": {"exploiter": 1.0},
            "exploiter_temperature": 0.42,
        })
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={}, override_type="exploiter")
        assert config.temperature == 0.42

    def test_record_result_and_get_yield_rate(self):
        pool = WorkerPool({"types": ["exploiter"]})
        pool.record_result("exploiter", True)
        pool.record_result("exploiter", False)
        pool.record_result("exploiter", True)
        assert pool.get_yield_rate("exploiter") == pytest.approx(2 / 3)

    def test_get_yield_rate_empty(self):
        pool = WorkerPool({"types": ["exploiter"]})
        assert pool.get_yield_rate("exploiter") == 0.0

    def test_get_yield_rate_unknown_worker(self):
        pool = WorkerPool({"types": ["exploiter"]})
        assert pool.get_yield_rate("nonexistent") == 0.0

    def test_allocation_normalization_sums_to_one(self):
        config = {
            "types": ["exploiter", "explorer", "crossover"],
            "initial_allocation": {"exploiter": 3.0, "explorer": 2.0, "crossover": 5.0},
        }
        pool = WorkerPool(config)
        total = sum(pool.allocation.values())
        assert total == pytest.approx(1.0)

    def test_set_allocation_normalizes(self):
        pool = WorkerPool({"types": ["exploiter", "explorer"]})
        pool.set_allocation({"exploiter": 4.0, "explorer": 6.0})
        assert pool.allocation["exploiter"] == pytest.approx(0.4)
        assert pool.allocation["explorer"] == pytest.approx(0.6)

    def test_boost_explorer(self):
        pool = WorkerPool({
            "types": ["exploiter", "explorer"],
            "initial_allocation": {"exploiter": 0.7, "explorer": 0.3},
        })
        old_explorer = pool.allocation["explorer"]
        pool.boost_explorer(0.2)
        # After boosting and renormalization, explorer share should be higher
        assert pool.allocation["explorer"] > old_explorer


# ---------------------------------------------------------------------------
# F4: ConvergenceMonitor
# ---------------------------------------------------------------------------


class TestClassifyEdit:
    """Tests for the classify_edit helper function."""

    def test_empty_string(self):
        assert classify_edit("") == "empty"

    def test_none_like_empty(self):
        # Empty string is falsy
        assert classify_edit("") == "empty"

    def test_function_content(self):
        result = classify_edit("def foo():\n    return 42")
        assert "function" in result

    def test_class_content(self):
        result = classify_edit("class MyClass:\n    pass")
        assert "class" in result

    def test_import_content(self):
        result = classify_edit("import os\nimport sys\nfrom pathlib import Path")
        assert "import" in result

    def test_loop_content(self):
        result = classify_edit("for i in range(100):\n    total += i")
        assert "loop" in result

    def test_size_prefix_tiny(self):
        # Under 50 chars
        result = classify_edit("x += 1")
        assert result.startswith("tiny_")

    def test_size_prefix_small(self):
        # 50-199 chars
        code = "def foo():\n" + "    x = 1\n" * 5
        assert len(code) >= 50
        result = classify_edit(code)
        assert result.startswith("small_")

    def test_size_prefix_large(self):
        # >= 1000 chars
        code = "def foo():\n" + "    x = 1\n" * 200
        assert len(code) >= 1000
        result = classify_edit(code)
        assert result.startswith("large_")


class TestConvergenceMonitor:
    """Tests for the ConvergenceMonitor (F4)."""

    def _make_result(self, diff="", worker_type="exploiter", improved=False):
        return IterationResult(
            diff=diff,
            worker_name=worker_type,
            improved=improved,
            error=None,
        )

    def test_returns_none_when_insufficient_data(self):
        monitor = ConvergenceMonitor({"enabled": True})
        # Fewer than 10 results
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = monitor.update(results)
        assert action == ConvergenceAction.NONE

    def test_returns_none_when_disabled(self):
        monitor = ConvergenceMonitor({"enabled": False})
        results = [self._make_result(diff=f"x = {i}") for i in range(20)]
        action = monitor.update(results)
        assert action == ConvergenceAction.NONE

    def test_high_entropy_returns_none(self):
        """Diverse edits should produce high entropy and NONE action."""
        monitor = ConvergenceMonitor({
            "enabled": True,
            "entropy_threshold_mild": 0.5,
        })
        # Feed diverse edits across multiple types
        diverse_diffs = [
            "def foo(): pass",          # function
            "class Bar: pass",          # class
            "import os",                # import
            "for i in range(10): pass", # loop
            "if x > 0: y = 1",         # conditional
            "return x + y",            # return
            "x += 1",                  # arithmetic
            "def baz(): return 0",     # function
            "class Qux: pass",         # class
            "while True: break",       # loop
            "from sys import exit",    # import
            "x = y + z * 2",          # arithmetic
        ]
        results = [
            self._make_result(diff=d, worker_type=wt)
            for d, wt in zip(diverse_diffs, ["exploiter", "explorer"] * 6)
        ]
        action = monitor.update(results)
        assert action == ConvergenceAction.NONE

    def test_low_entropy_triggers_escalation(self):
        """Identical edits should produce low entropy and trigger escalation."""
        monitor = ConvergenceMonitor({
            "enabled": True,
            "entropy_threshold_mild": 0.999,  # Very high threshold to guarantee trigger
            "entropy_threshold_moderate": 0.998,
            "entropy_threshold_severe": 0.997,
        })
        # All identical edits => minimal entropy
        results = [self._make_result(diff="x = 1", worker_type="exploiter") for _ in range(15)]
        action = monitor.update(results)
        # With all identical edits, entropy should be 0 (one category => normalized to 0)
        # which is below severe threshold
        assert action in (
            ConvergenceAction.MILD_BOOST,
            ConvergenceAction.MODERATE_DIVERSIFY,
            ConvergenceAction.SEVERE_RESTART,
        )


# ---------------------------------------------------------------------------
# F5: ContractSelector
# ---------------------------------------------------------------------------


class TestContractSelector:
    """Tests for the ContractSelector (F5: Intelligence Contracts)."""

    def test_initialization_arm_count(self):
        config = {
            "models": ["gpt-4", "claude-3"],
            "temperatures": [0.3, 0.7, 1.0],
            "enabled": True,
        }
        selector = ContractSelector(config)
        assert len(selector.arms) == 6  # 2 models x 3 temperatures

    def test_initialization_no_models(self):
        selector = ContractSelector({"models": [], "enabled": True})
        assert len(selector.arms) == 0

    def test_select_returns_contract(self):
        config = {
            "models": ["gpt-4"],
            "temperatures": [0.7],
            "enabled": True,
            "random_seed": 42,
        }
        selector = ContractSelector(config)
        contract = selector.select()
        assert isinstance(contract, Contract)
        assert contract.model_id == "gpt-4"
        assert contract.temperature == 0.7

    def test_select_returns_none_when_disabled(self):
        config = {
            "models": ["gpt-4"],
            "temperatures": [0.7],
            "enabled": False,
        }
        selector = ContractSelector(config)
        assert selector.select() is None

    def test_select_returns_none_when_no_arms(self):
        config = {"models": [], "enabled": True}
        selector = ContractSelector(config)
        assert selector.select() is None

    def test_update_changes_posteriors(self):
        config = {
            "models": ["gpt-4"],
            "temperatures": [0.7],
            "enabled": True,
        }
        selector = ContractSelector(config)
        arm = ("gpt-4", 0.7)
        old_alpha = selector.posteriors[arm]["alpha"]
        old_beta = selector.posteriors[arm]["beta"]

        selector.update("gpt-4", 0.7, success=True)
        assert selector.posteriors[arm]["alpha"] == old_alpha + 1.0
        assert selector.posteriors[arm]["beta"] == old_beta  # unchanged

        selector.update("gpt-4", 0.7, success=False)
        assert selector.posteriors[arm]["beta"] == old_beta + 1.0

    def test_update_noop_when_disabled(self):
        config = {
            "models": ["gpt-4"],
            "temperatures": [0.7],
            "enabled": False,
        }
        selector = ContractSelector(config)
        # Should not raise
        selector.update("gpt-4", 0.7, success=True)

    def test_posteriors_summary(self):
        config = {
            "models": ["model-a"],
            "temperatures": [0.5],
            "enabled": True,
        }
        selector = ContractSelector(config)
        selector.update("model-a", 0.5, success=True)
        summary = selector.get_posteriors_summary()
        assert "model-a@0.5" in summary
        assert summary["model-a@0.5"]["alpha"] == 2.0  # 1 initial + 1 success


# ---------------------------------------------------------------------------
# F6: SOTAController
# ---------------------------------------------------------------------------


class TestSOTAController:
    """Tests for the SOTAController (F6: SOTA-Distance Steering)."""

    def test_far_regime_when_gap_large(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        regime = ctrl.get_regime(current_best=50.0)  # 50% gap
        assert regime == SearchRegime.FAR

    def test_mid_regime(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        regime = ctrl.get_regime(current_best=80.0)  # 20% gap
        assert regime == SearchRegime.MID

    def test_near_regime_when_gap_small(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        regime = ctrl.get_regime(current_best=95.0)  # 5% gap
        assert regime == SearchRegime.NEAR

    def test_polishing_regime_when_gap_tiny(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        regime = ctrl.get_regime(current_best=99.0)  # 1% gap
        assert regime == SearchRegime.POLISHING

    def test_format_for_prompt_includes_target_and_gap(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        ctrl.get_regime(current_best=90.0)  # 10% gap
        text = ctrl.format_for_prompt()
        assert "100.0" in text
        assert "10.00%" in text

    def test_format_for_prompt_empty_when_disabled(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": False})
        assert ctrl.format_for_prompt() == ""

    def test_format_for_prompt_empty_when_no_target(self):
        ctrl = SOTAController({"enabled": True})
        assert ctrl.format_for_prompt() == ""

    def test_format_for_prompt_empty_when_no_gap_history(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        # No get_regime call yet, so no gap history
        assert ctrl.format_for_prompt() == ""

    def test_disabled_returns_mid(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": False})
        regime = ctrl.get_regime(current_best=50.0)
        assert regime == SearchRegime.MID

    def test_no_target_returns_mid(self):
        ctrl = SOTAController({"enabled": True})
        regime = ctrl.get_regime(current_best=50.0)
        assert regime == SearchRegime.MID

    def test_gap_history_tracked(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        ctrl.get_regime(current_best=70.0)
        ctrl.get_regime(current_best=85.0)
        assert len(ctrl.gap_history) == 2
        assert ctrl.current_gap == pytest.approx(0.15)

    def test_prompt_injection_per_regime(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        far_text = ctrl.get_prompt_injection(SearchRegime.FAR)
        near_text = ctrl.get_prompt_injection(SearchRegime.NEAR)
        assert "fundamentally different" in far_text.lower() or "radical" in far_text.lower()
        assert "surgical" in near_text.lower() or "precise" in near_text.lower()

    def test_modulate_worker_allocation(self):
        ctrl = SOTAController({"target_score": 100.0, "enabled": True})
        alloc = ctrl.modulate_worker_allocation(SearchRegime.FAR)
        assert alloc["explorer"] > alloc["exploiter"]  # FAR => more exploration
        alloc_near = ctrl.modulate_worker_allocation(SearchRegime.NEAR)
        assert alloc_near["exploiter"] > alloc_near["explorer"]  # NEAR => more exploitation


# ---------------------------------------------------------------------------
# F1: ReflectionEngine
# ---------------------------------------------------------------------------


class TestReflectionEngine:
    """Tests for the ReflectionEngine (F1)."""

    def _make_engine(self, enabled=True):
        llm = MagicMock()
        config = {"top_k": 3, "bottom_k": 2, "enabled": enabled}
        return ReflectionEngine(llm, config)

    def test_format_for_prompt_empty_when_no_reflection(self):
        engine = self._make_engine()
        assert engine.format_for_prompt() == ""

    def test_format_for_prompt_empty_with_empty_dict(self):
        engine = self._make_engine()
        assert engine.format_for_prompt({}) == ""

    def test_format_for_prompt_includes_working_patterns(self):
        engine = self._make_engine()
        reflection = {
            "working_patterns": ["Using binary search improves speed"],
            "failing_patterns": [],
            "hypotheses": [],
            "suggested_directions": [],
        }
        text = engine.format_for_prompt(reflection)
        assert "working" in text.lower()
        assert "binary search" in text.lower()

    def test_format_for_prompt_includes_failing_patterns(self):
        engine = self._make_engine()
        reflection = {
            "working_patterns": [],
            "failing_patterns": ["Brute force is too slow"],
            "hypotheses": [],
            "suggested_directions": [],
        }
        text = engine.format_for_prompt(reflection)
        assert "not working" in text.lower()
        assert "brute force" in text.lower()

    def test_format_for_prompt_includes_all_sections(self):
        engine = self._make_engine()
        reflection = {
            "working_patterns": ["Pattern A"],
            "failing_patterns": ["Pattern B"],
            "hypotheses": ["Hypothesis C"],
            "suggested_directions": ["Direction D"],
        }
        text = engine.format_for_prompt(reflection)
        assert "Pattern A" in text
        assert "Pattern B" in text
        assert "Hypothesis C" in text
        assert "Direction D" in text

    def test_parse_reflection_valid_json(self):
        engine = self._make_engine()
        raw = json.dumps({
            "working_patterns": ["fast sorting"],
            "failing_patterns": ["slow loops"],
            "hypotheses": ["caching helps"],
            "suggested_directions": ["try memoization"],
        })
        result = engine._parse_reflection(raw)
        assert result["working_patterns"] == ["fast sorting"]
        assert result["failing_patterns"] == ["slow loops"]
        assert result["hypotheses"] == ["caching helps"]
        assert result["suggested_directions"] == ["try memoization"]

    def test_parse_reflection_markdown_code_block(self):
        engine = self._make_engine()
        raw = '```json\n{"working_patterns": ["x"], "failing_patterns": [], "hypotheses": [], "suggested_directions": []}\n```'
        result = engine._parse_reflection(raw)
        assert result["working_patterns"] == ["x"]

    def test_parse_reflection_missing_fields(self):
        engine = self._make_engine()
        raw = json.dumps({"working_patterns": ["only this"]})
        result = engine._parse_reflection(raw)
        assert result["working_patterns"] == ["only this"]
        assert result["failing_patterns"] == []
        assert result["hypotheses"] == []
        assert result["suggested_directions"] == []

    def test_parse_reflection_invalid_json_fallback(self):
        engine = self._make_engine()
        result = engine._parse_reflection("this is not json at all")
        # Should use fallback
        assert "working_patterns" in result
        assert "failing_patterns" in result
        assert len(result["hypotheses"]) > 0  # fallback puts raw text here

    def test_parse_reflection_json_embedded_in_text(self):
        engine = self._make_engine()
        raw = 'Here is my analysis:\n{"working_patterns": ["embedded"], "failing_patterns": [], "hypotheses": [], "suggested_directions": []}\nEnd.'
        result = engine._parse_reflection(raw)
        assert result["working_patterns"] == ["embedded"]


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    """Tests for the MetricsLogger."""

    def test_initialization_creates_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            metrics_dir = os.path.join(tmpdir, "metrics")
            assert os.path.isdir(metrics_dir)
            assert os.path.exists(os.path.join(metrics_dir, "score_trajectory.csv"))
            assert os.path.exists(os.path.join(metrics_dir, "worker_yield.csv"))
            assert os.path.exists(os.path.join(metrics_dir, "diversity.csv"))
            assert os.path.exists(os.path.join(metrics_dir, "cost.csv"))

    def test_csv_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            score_csv = os.path.join(tmpdir, "metrics", "score_trajectory.csv")
            with open(score_csv) as f:
                reader = csv.reader(f)
                headers = next(reader)
            assert "batch" in headers
            assert "best_score" in headers
            assert "num_improvements" in headers

    def test_log_batch_writes_to_score_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            # Create mock results
            r1 = IterationResult(
                child_score=0.85,
                worker_name="exploiter",
                improved=True,
                error=None,
                tokens_in=100,
                tokens_out=50,
                wall_clock_seconds=1.5,
            )
            r2 = IterationResult(
                child_score=0.72,
                worker_name="explorer",
                improved=False,
                error=None,
                tokens_in=200,
                tokens_out=80,
                wall_clock_seconds=2.0,
            )
            ml.log_batch(batch_number=1, batch_results=[r1, r2])

            score_csv = os.path.join(tmpdir, "metrics", "score_trajectory.csv")
            with open(score_csv) as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Header + 1 data row
            assert len(rows) == 2
            assert rows[1][0] == "1"  # batch number

    def test_log_batch_writes_worker_yield(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            r1 = IterationResult(
                child_score=0.9,
                worker_name="exploiter",
                improved=True,
                error=None,
                tokens_in=100,
                tokens_out=50,
                wall_clock_seconds=1.0,
            )
            ml.log_batch(batch_number=1, batch_results=[r1])

            worker_csv = os.path.join(tmpdir, "metrics", "worker_yield.csv")
            with open(worker_csv) as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert len(rows) == 2  # header + 1 worker type row
            # Check the worker type column
            assert rows[1][1] == "exploiter"

    def test_log_reflection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            reflection = {"working_patterns": ["test pattern"]}
            ml.log_reflection(batch_number=1, reflection=reflection, reflection_calls=1)

            jsonl_path = os.path.join(tmpdir, "metrics", "reflection_log.jsonl")
            with open(jsonl_path) as f:
                entry = json.loads(f.readline())
            assert entry["batch"] == 1
            assert entry["reflection"]["working_patterns"] == ["test pattern"]
