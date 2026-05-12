"""Tests for REPS feature modules F1-F8."""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from reps.config import REPSWorkersConfig
from reps.iteration_config import IterationConfig, IterationResult
from reps.worker_pool import WorkerPool
from reps.workers.base import WorkerConfig
from reps.convergence_monitor import ConvergenceMonitor, ConvergenceAction, classify_edit
from reps.contract_selector import ContractSelector, Contract
from reps.sota_controller import SOTAController, SearchRegime
from reps.reflection_engine import ReflectionEngine
from reps.metrics_logger import MetricsLogger


# ---------------------------------------------------------------------------
# F3: WorkerPool
# ---------------------------------------------------------------------------


def _make_workers_config(
    names=("exploiter", "explorer", "crossover"),
    weights=None,
    temperatures=None,
    generation_modes=None,
    model_id="",
):
    """Build a REPSWorkersConfig from simple lists so tests stay terse."""
    if weights is None:
        weights = {}
    if temperatures is None:
        temperatures = {}
    if generation_modes is None:
        generation_modes = {}

    configs = []
    for name in names:
        configs.append(
            WorkerConfig(
                name=name,
                impl="single_call",
                role=name,
                model_id=model_id,
                temperature=temperatures.get(name, 0.7),
                generation_mode=generation_modes.get(name, "diff"),
                weight=weights.get(name, 1.0),
                owns_temperature=True,
            )
        )
    return REPSWorkersConfig(types=configs)


class TestWorkerPool:
    """Tests for the WorkerPool (F3: Worker Type Diversity)."""

    def test_initialization_with_config(self):
        cfg = _make_workers_config(
            names=("exploiter", "explorer", "crossover"),
            weights={"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15},
        )
        pool = WorkerPool(cfg)
        assert set(pool.allocation.keys()) == {"exploiter", "explorer", "crossover"}
        assert len(pool.yield_tracker) == 3

    def test_initialization_empty_raises(self):
        """Empty workers list now raises — no silent fall-back to presets."""
        with pytest.raises(ValueError, match="non-empty"):
            WorkerPool(REPSWorkersConfig(types=[]))

    def test_build_iteration_config_returns_iteration_config(self):
        pool = WorkerPool(
            _make_workers_config(names=("exploiter",), weights={"exploiter": 1.0})
        )
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={"reflection": "test"})
        assert isinstance(config, IterationConfig)
        assert config.worker_type == "exploiter"

    def test_build_iteration_config_override_type(self):
        pool = WorkerPool(_make_workers_config(names=("exploiter", "explorer")))
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={}, override_type="explorer")
        assert config.worker_type == "explorer"

    def test_build_iteration_config_temperature_from_worker(self):
        pool = WorkerPool(
            _make_workers_config(
                names=("exploiter",),
                weights={"exploiter": 1.0},
                temperatures={"exploiter": 0.42},
            )
        )
        db = MagicMock()
        config = pool.build_iteration_config(db, prompt_extras={}, override_type="exploiter")
        assert config.temperature == 0.42

    def test_record_result_and_get_yield_rate(self):
        pool = WorkerPool(_make_workers_config(names=("exploiter",)))
        pool.record_result("exploiter", True)
        pool.record_result("exploiter", False)
        pool.record_result("exploiter", True)
        assert pool.get_yield_rate("exploiter") == pytest.approx(2 / 3)

    def test_get_yield_rate_empty(self):
        pool = WorkerPool(_make_workers_config(names=("exploiter",)))
        assert pool.get_yield_rate("exploiter") == 0.0

    def test_get_yield_rate_unknown_worker(self):
        pool = WorkerPool(_make_workers_config(names=("exploiter",)))
        assert pool.get_yield_rate("nonexistent") == 0.0

    def test_allocation_normalization_sums_to_one(self):
        cfg = _make_workers_config(
            names=("exploiter", "explorer", "crossover"),
            weights={"exploiter": 3.0, "explorer": 2.0, "crossover": 5.0},
        )
        pool = WorkerPool(cfg)
        total = sum(pool.allocation.values())
        assert total == pytest.approx(1.0)

    def test_set_allocation_normalizes(self):
        pool = WorkerPool(_make_workers_config(names=("exploiter", "explorer")))
        pool.set_allocation({"exploiter": 4.0, "explorer": 6.0})
        assert pool.allocation["exploiter"] == pytest.approx(0.4)
        assert pool.allocation["explorer"] == pytest.approx(0.6)

    def test_boost_explorer(self):
        pool = WorkerPool(
            _make_workers_config(
                names=("exploiter", "explorer"),
                weights={"exploiter": 0.7, "explorer": 0.3},
            )
        )
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


class _FakeDatabase:
    """Minimal database stand-in for ConvergenceMonitor tests.

    Exposes the surface the monitor reads: `island_feature_maps` (list of
    dicts) plus a `config.feature_dimensions` / `feature_bins` so capacity
    can be computed.
    """

    class _Cfg:
        def __init__(self, dims, bins):
            self.feature_dimensions = dims
            self.feature_bins = bins

    class _Program:
        def __init__(self, score):
            self.metrics = {"combined_score": score} if score is not None else {}

    def __init__(
        self,
        *,
        niche_sequence,
        score_sequence=None,
        num_islands=1,
        dims=("a", "b"),
        bins=10,
    ):
        self._niche_seq = list(niche_sequence)
        self._score_seq = list(score_sequence) if score_sequence is not None else None
        self._step = 0
        self._num_islands = num_islands
        self.config = self._Cfg(list(dims), bins)
        self.feature_bins = bins
        self.island_feature_maps = [{} for _ in range(num_islands)]
        self._best_program = None

    def advance(self):
        total = self._niche_seq[min(self._step, len(self._niche_seq) - 1)]
        self.island_feature_maps = [{} for _ in range(self._num_islands)]
        for i in range(total):
            self.island_feature_maps[i % self._num_islands][f"k{i}"] = f"p{i}"

        if self._score_seq is not None:
            idx = min(self._step, len(self._score_seq) - 1)
            self._best_program = self._Program(self._score_seq[idx])
        self._step += 1

    def get_best_program(self):
        return self._best_program


class TestConvergenceMonitor:
    """Tests for the ConvergenceMonitor (F4)."""

    def _make_result(self, diff="", worker_type="exploiter", improved=False):
        return IterationResult(
            diff=diff,
            worker_name=worker_type,
            improved=improved,
            error=None,
        )

    def _drive(self, monitor, db, results_per_batch, n_batches):
        """Run n batches, returning the action from the final one."""
        last = ConvergenceAction.NONE
        for _ in range(n_batches):
            db.advance()
            last = monitor.update(results_per_batch, database=db)
        return last

    def test_returns_none_when_insufficient_data(self):
        monitor = ConvergenceMonitor({"enabled": True, "window_size": 8})
        db = _FakeDatabase(niche_sequence=[1, 2, 3])
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        # Need window_size+1 batches before any action fires.
        action = self._drive(monitor, db, results, n_batches=3)
        assert action == ConvergenceAction.NONE

    def test_returns_none_when_disabled(self):
        monitor = ConvergenceMonitor({"enabled": False})
        db = _FakeDatabase(niche_sequence=[1] * 20)
        results = [self._make_result(diff=f"x = {i}") for i in range(20)]
        action = self._drive(monitor, db, results, n_batches=20)
        assert action == ConvergenceAction.NONE

    def test_returns_none_without_database(self):
        """Without a database the action driver has no signal to read."""
        monitor = ConvergenceMonitor({"enabled": True, "window_size": 4})
        results = [self._make_result(diff="x = 1") for _ in range(3)]
        for _ in range(10):
            assert monitor.update(results) == ConvergenceAction.NONE

    def test_healthy_growth_returns_none(self):
        """Niche map filling steadily means search is exploring; no action."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_mild": 0.2,
        })
        # 3 children per batch, 3 new niches per batch -> normalized growth 1.0
        db = _FakeDatabase(niche_sequence=[3, 6, 9, 12, 15, 18])
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.NONE

    def test_stalled_growth_triggers_severe(self):
        """Niche map stops growing -> normalized growth 0 -> severe action."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_mild": 0.5,
            "niche_growth_threshold_moderate": 0.3,
            "niche_growth_threshold_severe": 0.15,
        })
        # Niche map sits at 20 for the whole window -> growth = 0
        db = _FakeDatabase(niche_sequence=[20] * 8)
        results = [self._make_result(diff="x = 1") for _ in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.SEVERE_RESTART

    def test_partial_growth_triggers_mild(self):
        """Growth above severe but below mild -> mild action."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_mild": 0.5,
            "niche_growth_threshold_moderate": 0.3,
            "niche_growth_threshold_severe": 0.15,
        })
        # 3 children/batch, gain 1 niche/batch -> normalized 0.33
        # That lands in [moderate=0.3, mild=0.5) -> MILD_BOOST
        db = _FakeDatabase(niche_sequence=[10, 11, 12, 13, 14, 15])
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.MILD_BOOST

    def test_saturated_map_returns_none(self):
        """A nearly-full map is success, not collapse; skip escalation."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_severe": 0.99,  # would otherwise always fire
            "saturation_threshold": 0.5,
        })
        # 1 island * 10**2 bins = 100-cell capacity. Hold at 80 (saturation
        # 0.80 >= threshold 0.5) -> no escalation despite zero growth.
        db = _FakeDatabase(niche_sequence=[80] * 8, num_islands=1, dims=("a", "b"), bins=10)
        results = [self._make_result(diff="x = 1") for _ in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.NONE

    def test_legacy_entropy_threshold_keys_still_accepted(self):
        """Existing YAMLs use `entropy_threshold_*` — keep them working."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "entropy_threshold_severe": 0.5,
            "entropy_threshold_moderate": 0.4,
            "entropy_threshold_mild": 0.3,
        })
        assert monitor.thresholds == {"mild": 0.3, "moderate": 0.4, "severe": 0.5}

    def test_score_plateau_triggers_action(self):
        """Niches growing healthily but best-score flat -> escalate on plateau.

        Replicates the Sonnet/circle_packing failure mode: search keeps
        filling new MAP-Elites cells, but with same-fitness programs, so
        niche-growth says "healthy" while score is stuck.
        """
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_severe": 0.0,  # don't fire on niche
            "niche_growth_threshold_moderate": 0.0,
            "niche_growth_threshold_mild": 0.0,
            "score_plateau_threshold_severe": 0.0,
            "score_plateau_threshold_moderate": 0.001,
            "score_plateau_threshold_mild": 0.01,
        })
        # Niches grow steadily (1/batch * 3 children = 1.0 normalized) but
        # best_score sits at 0.93 the whole window.
        db = _FakeDatabase(
            niche_sequence=[3, 6, 9, 12, 15, 18],
            score_sequence=[0.93, 0.93, 0.93, 0.93, 0.93, 0.93],
        )
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.SEVERE_RESTART

    def test_score_improving_returns_none(self):
        """Score climbing across the window -> no plateau action."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            "niche_growth_threshold_mild": 0.0,  # never fires
            "score_plateau_threshold_mild": 0.01,
        })
        db = _FakeDatabase(
            niche_sequence=[3, 6, 9, 12, 15, 18],
            score_sequence=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        )
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.NONE

    def test_most_severe_action_wins(self):
        """When both signals fire, the higher-severity action is returned."""
        monitor = ConvergenceMonitor({
            "enabled": True, "window_size": 4,
            # Niche fires MILD (low growth)
            "niche_growth_threshold_mild": 0.5,
            "niche_growth_threshold_moderate": 0.3,
            "niche_growth_threshold_severe": 0.15,
            # Score fires SEVERE (zero improvement)
            "score_plateau_threshold_severe": 0.0,
            "score_plateau_threshold_moderate": 0.001,
            "score_plateau_threshold_mild": 0.01,
        })
        db = _FakeDatabase(
            niche_sequence=[10, 11, 12, 13, 14, 15],  # mild growth
            score_sequence=[0.5] * 6,                  # severe plateau
        )
        results = [self._make_result(diff=f"x = {i}") for i in range(3)]
        action = self._drive(monitor, db, results, n_batches=6)
        assert action == ConvergenceAction.SEVERE_RESTART


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


# ---------------------------------------------------------------------------
# Run-health tracker (annotation success + convergence-action recovery)
# ---------------------------------------------------------------------------


class TestRunHealthTracking:
    """Tests for MetricsLogger.write_health + per-run health counters.

    Covers Gap 1 (annotation success rate) and Gap 2 (action recovery)
    described in the task. Counters are exercised directly the same way the
    controller would call them; we don't spin up the full controller.
    """

    def test_annotation_counters_persisted_to_health_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            ml.record_annotation_attempt(success=True)
            ml.record_annotation_attempt(success=True)
            ml.record_annotation_attempt(success=False)
            health = ml.write_health()
            assert health["annotations"]["attempts"] == 3
            assert health["annotations"]["successes"] == 2
            assert abs(health["annotations"]["success_rate"] - 2 / 3) < 1e-9
            health_path = os.path.join(tmpdir, "metrics", "health.json")
            assert os.path.exists(health_path)
            with open(health_path) as f:
                on_disk = json.load(f)
            assert on_disk["annotations"]["attempts"] == 3

    def test_low_annotation_rate_emits_warning(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            for _ in range(10):
                ml.record_annotation_attempt(success=False)
            ml.record_annotation_attempt(success=True)
            with caplog.at_level("WARNING", logger="reps.metrics_logger"):
                ml.write_health()
            assert any(
                "summarizer success rate" in r.message and r.levelname == "WARNING"
                for r in caplog.records
            ), caplog.text

    def test_healthy_annotation_rate_does_not_warn(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            for _ in range(10):
                ml.record_annotation_attempt(success=True)
            with caplog.at_level("WARNING", logger="reps.metrics_logger"):
                ml.write_health()
            assert not any(
                "summarizer success rate" in r.message for r in caplog.records
            )

    def test_action_recovery_via_score_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            # Fire SEVERE_RESTART at batch 5 with best=0.5, niche=10
            ml.record_action_fired("SEVERE_RESTART", 5, best_score=0.5, niche_occupancy=10)
            # Next batch: best score climbed → recovered.
            ml.observe_post_action(6, best_score=0.6, niche_occupancy=10)
            health = ml.write_health()
            assert health["convergence_actions"]["fired_per_level"] == {"SEVERE_RESTART": 1}
            assert health["convergence_actions"]["recovered_per_level"] == {"SEVERE_RESTART": 1}
            assert health["convergence_actions"]["recovery_rate"] == 1.0

    def test_action_recovery_via_niche_growth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            ml.record_action_fired("MILD_BOOST", 5, best_score=0.5, niche_occupancy=10)
            # Score flat, niche jumps from 10 -> 13 (30% growth, > 20% threshold).
            ml.observe_post_action(6, best_score=0.5, niche_occupancy=13)
            health = ml.write_health()
            assert health["convergence_actions"]["recovered_per_level"] == {"MILD_BOOST": 1}

    def test_action_no_recovery_after_lookahead(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            ml.record_action_fired("SEVERE_RESTART", 5, best_score=0.5, niche_occupancy=10)
            # Two flat batches → exceeds lookahead, never recovers.
            ml.observe_post_action(6, best_score=0.5, niche_occupancy=10)
            ml.observe_post_action(7, best_score=0.5, niche_occupancy=10)
            health = ml.write_health()
            assert health["convergence_actions"]["recovered_per_level"] == {}
            assert health["convergence_actions"]["recovery_rate"] == 0.0

    def test_repeated_unrecovered_actions_emit_warning(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            # Fire 3 actions, none recover (flat score + flat niches).
            for batch in range(1, 4):
                ml.record_action_fired(
                    "SEVERE_RESTART", batch, best_score=0.5, niche_occupancy=10,
                )
                ml.observe_post_action(batch + 1, best_score=0.5, niche_occupancy=10)
                ml.observe_post_action(batch + 2, best_score=0.5, niche_occupancy=10)
            with caplog.at_level("WARNING", logger="reps.metrics_logger"):
                ml.write_health()
            assert any(
                "adaptive escalation appears ineffective" in r.message
                and r.levelname == "WARNING"
                for r in caplog.records
            ), caplog.text

    def test_healthy_actions_do_not_warn(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            # Fire 3 actions, all recover via score improvement.
            for batch in range(1, 4):
                ml.record_action_fired(
                    "MILD_BOOST", batch, best_score=0.5 + batch * 0.01, niche_occupancy=10,
                )
                ml.observe_post_action(batch + 1, best_score=0.5 + batch * 0.01 + 0.05, niche_occupancy=10)
            with caplog.at_level("WARNING", logger="reps.metrics_logger"):
                ml.write_health()
            assert not any(
                "adaptive escalation appears ineffective" in r.message
                for r in caplog.records
            )

    def test_no_warnings_on_quiet_run(self, caplog):
        """Empty run (no annotations attempted, no actions fired) — silent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ml = MetricsLogger(tmpdir)
            with caplog.at_level("WARNING", logger="reps.metrics_logger"):
                health = ml.write_health()
            assert health["annotations"]["success_rate"] is None
            assert health["convergence_actions"]["recovery_rate"] is None
            assert not any(
                "summarizer success rate" in r.message
                or "adaptive escalation" in r.message
                for r in caplog.records
            )
