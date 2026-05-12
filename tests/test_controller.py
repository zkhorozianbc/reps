"""
Tests for the async controller (asyncio-only; no process fork).
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

# Set dummy API key for testing
os.environ["OPENAI_API_KEY"] = "test"

from reps.config import Config, LLMModelConfig
from reps.database import Program, ProgramDatabase
from reps.controller import ProcessParallelController, SerializableResult
from reps.workers.base import WorkerConfig


def _default_reps_workers():
    """Minimal WorkerConfig list that keeps REPS-enabled paths runnable.

    Mirrors the three-worker preset exploiter/explorer/crossover that
    was formerly provided by the deleted ``legacy_default_configs`` shim.
    """
    return [
        WorkerConfig(
            name="exploiter",
            impl="single_call",
            role="exploiter",
            model_id="",
            temperature=0.3,
            generation_mode="diff",
            weight=0.7,
            owns_temperature=True,
        ),
        WorkerConfig(
            name="explorer",
            impl="single_call",
            role="explorer",
            model_id="",
            temperature=1.0,
            generation_mode="full",
            weight=0.15,
            owns_temperature=True,
        ),
        WorkerConfig(
            name="crossover",
            impl="single_call",
            role="crossover",
            model_id="",
            temperature=0.7,
            generation_mode="full",
            weight=0.15,
            owns_temperature=True,
        ),
    ]


class TestProcessParallel(unittest.TestCase):
    """Tests for async controller (class name kept for back-compat)."""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

        self.config = Config()
        self.config.max_iterations = 10
        self.config.evaluator.parallel_evaluations = 2
        self.config.evaluator.timeout = 10
        self.config.database.num_islands = 2
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        self.eval_content = """
def evaluate(program_path):
    return {"score": 0.5, "performance": 0.6}
"""
        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write(self.eval_content)

        self.database = ProgramDatabase(self.config.database)

        for i in range(3):
            program = Program(
                id=f"test_{i}",
                code=f"def func_{i}(): return {i}",
                language="python",
                metrics={"score": 0.5 + i * 0.1, "performance": 0.4 + i * 0.1},
                iteration_found=0,
            )
            self.database.add(program)

    def tearDown(self):
        """Clean up test environment"""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_controller_initialization(self):
        """Test that controller initializes correctly"""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        self.assertEqual(controller.num_workers, 2)
        # Shared singletons and concurrency primitives are lazy (built in start()).
        self.assertIsNone(controller.evaluator)
        self.assertIsNone(controller.llm_ensemble)
        self.assertIsNone(controller._iter_semaphore)
        self.assertIsNone(controller._shutdown)

    def test_controller_with_reps_does_not_create_default_output_dir(self):
        """Direct controller construction should not create a fallback output dir."""
        self.config.reps.enabled = True
        self.config.reps.workers.types = _default_reps_workers()

        old_cwd = os.getcwd()
        isolated_cwd = tempfile.mkdtemp()
        try:
            os.chdir(isolated_cwd)
            controller = ProcessParallelController(self.config, self.eval_file, self.database)
            self.assertIsNone(controller.output_dir)
            self.assertIsNone(controller._reps_metrics)
            self.assertFalse(os.path.exists("openevolve_output"))
        finally:
            os.chdir(old_cwd)
            import shutil

            shutil.rmtree(isolated_cwd, ignore_errors=True)

    def test_controller_start_stop(self):
        """Starting the controller wires shared singletons; stop sets shutdown."""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)

        controller.start()
        self.assertIsNotNone(controller.llm_ensemble)
        self.assertIsNotNone(controller.evaluator)
        self.assertIsNotNone(controller.prompt_sampler)
        self.assertIsNotNone(controller._iter_semaphore)
        self.assertIsNotNone(controller._shutdown)
        self.assertFalse(controller._shutdown.is_set())

        controller.stop()
        self.assertTrue(controller._shutdown.is_set())

    def test_run_evolution_basic(self):
        """Basic evolution run with _run_iteration mocked."""

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.start()

            stub_result = SerializableResult(
                child_program_dict={
                    "id": "child_1",
                    "code": "def evolved(): return 1",
                    "language": "python",
                    "parent_id": "test_0",
                    "generation": 1,
                    "metrics": {"score": 0.7, "performance": 0.8},
                    "iteration_found": 1,
                    "metadata": {"changes": "test", "island": 0},
                },
                parent_id="test_0",
                iteration_time=0.1,
                iteration=1,
            )

            with patch.object(
                controller, "_run_iteration", new=AsyncMock(return_value=stub_result)
            ) as m:
                await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )
                self.assertGreaterEqual(m.await_count, 1)

            # Verify program was added to database
            self.assertIn("child_1", self.database.programs)
            child = self.database.get("child_1")
            self.assertEqual(child.metrics["score"], 0.7)

        asyncio.run(run_test())

    def test_request_shutdown(self):
        """Graceful shutdown sets the asyncio event."""
        controller = ProcessParallelController(self.config, self.eval_file, self.database)
        controller.start()
        controller.request_shutdown()
        self.assertTrue(controller._shutdown.is_set())

    def test_serializable_result(self):
        """Test SerializableResult dataclass"""
        result = SerializableResult(
            child_program_dict={"id": "test", "code": "pass"},
            parent_id="parent",
            iteration_time=1.5,
            iteration=10,
            error=None,
        )

        self.assertEqual(result.child_program_dict["id"], "test")
        self.assertEqual(result.parent_id, "parent")
        self.assertEqual(result.iteration_time, 1.5)
        self.assertEqual(result.iteration, 10)
        self.assertIsNone(result.error)

        error_result = SerializableResult(error="Test error", iteration=5)
        self.assertEqual(error_result.error, "Test error")
        self.assertIsNone(error_result.child_program_dict)

    def test_sota_steering_uses_same_raw_metric_for_prompt_and_reallocation(self):
        """F6 should use the same score basis across both steering callsites."""
        self.config.reps.enabled = True
        self.config.reps.workers.types = _default_reps_workers()
        self.config.reps.sota.enabled = True
        self.config.reps.sota.target_score = 2.635
        self.config.reps.sota.target_metric = "sum_radii"
        self.config.reps.convergence.enabled = False
        self.config.reps.contracts.enabled = False
        self.config.reps.reflection.enabled = False

        database = ProgramDatabase(self.config.database)
        database.add(
            Program(
                id="best_raw",
                code="def solve(): pass",
                language="python",
                metrics={
                    "sum_radii": 2.5,
                    "combined_score": 0.95,
                },
                iteration_found=0,
            )
        )

        controller = ProcessParallelController(self.config, self.eval_file, database)
        self.assertIsNone(controller._reps_metrics)

        extras = controller._reps_build_prompt_extras()
        expected_gap = (2.635 - 2.5) / 2.635
        self.assertAlmostEqual(controller._reps_sota.current_gap, expected_gap)
        self.assertIn("5.12%", extras["sota_injection"])

        async def run_batch():
            await controller._reps_process_batch(
                [SerializableResult(iteration=1, error="synthetic failure")]
            )

        asyncio.run(run_batch())

        self.assertAlmostEqual(controller._reps_sota.current_gap, expected_gap)
        self.assertEqual(controller._reps_sota.current_regime.name, "NEAR")
        self.assertAlmostEqual(controller._reps_worker_pool.allocation["exploiter"], 0.7)
        self.assertAlmostEqual(controller._reps_worker_pool.allocation["explorer"], 0.15)
        self.assertAlmostEqual(controller._reps_worker_pool.allocation["crossover"], 0.15)


class TestNoCombinedScoreWarning(unittest.TestCase):
    """
    The "no 'combined_score'" WARNING must distinguish a misconfigured
    evaluator (no combined_score in an otherwise normal metrics dict) from
    a timeout/error sentinel result (already warned upstream by the evaluator
    or by the iteration error handler).
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.config = Config()
        self.config.max_iterations = 1
        self.config.evaluator.parallel_evaluations = 1
        self.config.evaluator.timeout = 10
        self.config.evaluator.cascade_evaluation = False
        self.config.database.num_islands = 1
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write("def evaluate(p):\n    return {}\n")

        self.database = ProgramDatabase(self.config.database)
        self.database.add(
            Program(
                id="seed",
                code="def f(): pass",
                language="python",
                metrics={"combined_score": 0.5},
                iteration_found=0,
            )
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run_one_iter_with_metrics(self, metrics):
        """Drive a single controller iteration whose child has the given metrics
        and reps_meta with a non-zero token count (the warning is gated behind
        the reps_meta token block). Returns the captured log records."""

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.start()

            stub_result = SerializableResult(
                child_program_dict={
                    "id": "child_x",
                    "code": "def evolved(): pass",
                    "language": "python",
                    "parent_id": "seed",
                    "generation": 1,
                    "metrics": metrics,
                    "iteration_found": 1,
                    "metadata": {"changes": "test", "island": 0},
                },
                parent_id="seed",
                iteration_time=0.1,
                iteration=1,
                reps_meta={"tokens_in": 1, "tokens_out": 1},
            )

            with patch.object(
                controller, "_run_iteration", new=AsyncMock(return_value=stub_result)
            ):
                await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )
            return controller

        import logging

        handler_records = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                handler_records.append(record)

        controller_logger = logging.getLogger("reps.controller")
        list_handler = _ListHandler(level=logging.WARNING)
        controller_logger.addHandler(list_handler)
        prev_level = controller_logger.level
        controller_logger.setLevel(logging.DEBUG)
        try:
            controller = asyncio.run(run_test())
        finally:
            controller_logger.removeHandler(list_handler)
            controller_logger.setLevel(prev_level)

        return controller, handler_records

    def _has_combined_score_warning(self, records):
        return any(
            r.levelname == "WARNING" and "No 'combined_score' metric" in r.getMessage()
            for r in records
        )

    def test_warning_suppressed_for_timeout_shaped_metrics(self):
        """Timeout sentinel ({error: 0.0, timeout: True}) must not trigger
        the misleading misconfiguration warning."""
        controller, records = self._run_one_iter_with_metrics(
            {"error": 0.0, "timeout": True}
        )
        self.assertFalse(
            self._has_combined_score_warning(records),
            "no-combined_score WARNING should be suppressed for timeout sentinel",
        )
        self.assertFalse(controller._warned_about_combined_score)

    def test_warning_fires_for_misconfigured_evaluator_metrics(self):
        """A normal-shaped metrics dict that genuinely lacks combined_score
        (operator misconfiguration) MUST still trigger the warning."""
        controller, records = self._run_one_iter_with_metrics(
            {"score": 0.5, "performance": 0.6}
        )
        self.assertTrue(
            self._has_combined_score_warning(records),
            "no-combined_score WARNING should fire for operator misconfiguration",
        )
        self.assertTrue(controller._warned_about_combined_score)


class TestControllerHardFailures(unittest.TestCase):
    """Harness/persistence failures must not be treated as candidate failures."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.config = Config()
        self.config.max_iterations = 1
        self.config.evaluator.parallel_evaluations = 1
        self.config.evaluator.timeout = 10
        self.config.evaluator.cascade_evaluation = False
        self.config.database.num_islands = 1
        self.config.database.in_memory = True
        self.config.checkpoint_interval = 5

        self.eval_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.eval_file, "w") as f:
            f.write("def evaluate(p):\n    return {'combined_score': 0.5}\n")

        self.database = ProgramDatabase(self.config.database)
        self.database.add(
            Program(
                id="seed",
                code="def f(): pass",
                language="python",
                metrics={"combined_score": 0.5},
                iteration_found=0,
            )
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _child_result(self, metrics=None):
        return SerializableResult(
            child_program_dict={
                "id": "child_hard_failure",
                "code": "def evolved(): pass",
                "language": "python",
                "parent_id": "seed",
                "generation": 1,
                "metrics": metrics or {"combined_score": 0.6},
                "iteration_found": 1,
                "metadata": {"changes": "test", "island": 0},
            },
            parent_id="seed",
            iteration_time=0.1,
            iteration=1,
        )

    def test_invalid_feature_dimension_raises_from_result_processing(self):
        """MAP-Elites feature config errors are harness errors, not LLM noise."""

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.start()
            self.config.database.feature_dimensions = ["missing_metric"]

            with patch.object(
                controller,
                "_run_iteration",
                new=AsyncMock(return_value=self._child_result()),
            ):
                await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )

        with self.assertRaisesRegex(ValueError, "Feature dimension 'missing_metric'"):
            asyncio.run(run_test())

    def test_per_iteration_save_failure_raises(self):
        """A failed child JSON/trace save must stop the run loudly."""

        async def run_test():
            controller = ProcessParallelController(
                self.config,
                self.eval_file,
                self.database,
                output_dir=self.test_dir,
            )
            controller.start()

            with patch.object(
                controller,
                "_run_iteration",
                new=AsyncMock(return_value=self._child_result()),
            ), patch.object(
                self.database,
                "_save_program",
                side_effect=RuntimeError("disk full"),
            ):
                await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )

        with self.assertRaisesRegex(RuntimeError, "disk full"):
            asyncio.run(run_test())

    def test_failed_seed_metrics_raise_before_spawning_iterations(self):
        """Seed evaluator failure sentinels should stop before any LLM work."""
        self.database.programs["seed"].metrics = {"error": 0.0, "timeout": True}

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.start()
            run_iteration = AsyncMock(return_value=self._child_result())
            with patch.object(controller, "_run_iteration", new=run_iteration):
                with self.assertRaisesRegex(RuntimeError, "seed.*failed evaluation"):
                    await controller.run_evolution(
                        start_iteration=1, max_iterations=1, target_score=None
                    )
            self.assertEqual(run_iteration.await_count, 0)

        asyncio.run(run_test())

    def test_candidate_error_result_remains_soft(self):
        """Explicit candidate/worker failure results still count as soft failures."""

        async def run_test():
            controller = ProcessParallelController(
                self.config, self.eval_file, self.database
            )
            controller.start()

            with patch.object(
                controller,
                "_run_iteration",
                new=AsyncMock(
                    return_value=SerializableResult(error="candidate failed", iteration=1)
                ),
            ):
                best = await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )
            self.assertEqual(best.id, "seed")
            self.assertNotIn("child_hard_failure", self.database.programs)

        asyncio.run(run_test())

    def test_database_rejection_skips_artifacts_and_persistence(self):
        """Rejected children must not be persisted after ProgramDatabase.add."""

        async def run_test():
            controller = ProcessParallelController(
                self.config,
                self.eval_file,
                self.database,
                output_dir=self.test_dir,
            )
            controller.start()

            with patch.object(
                controller,
                "_run_iteration",
                new=AsyncMock(
                    return_value=self._child_result({"combined_score": 0.6})
                ),
            ), patch.object(
                self.database,
                "add",
                return_value=False,
            ) as add, patch.object(
                self.database,
                "store_artifacts",
            ) as store_artifacts, patch.object(
                self.database,
                "_save_program",
            ) as save_program:
                best = await controller.run_evolution(
                    start_iteration=1, max_iterations=1, target_score=None
                )

            add.assert_called_once()
            store_artifacts.assert_not_called()
            save_program.assert_not_called()
            self.assertEqual(best.id, "seed")

        asyncio.run(run_test())


class TestControllerWorkerPoolWiring(unittest.TestCase):
    def test_reps_worker_pool_receives_config_random_seed(self):
        config = Config()
        config.random_seed = 123
        config.reps.enabled = True
        config.llm.models = [LLMModelConfig(name="local-model", provider="local")]
        config.llm.evaluator_models = list(config.llm.models)
        config.reps.workers.types = [
            WorkerConfig(name="exploit", impl="single_call", role="exploiter"),
        ]
        database = ProgramDatabase(config.database)

        with patch("reps.worker_pool.WorkerPool") as worker_pool_cls:
            ProcessParallelController(config, __file__, database)

        worker_pool_cls.assert_called_once()
        self.assertEqual(worker_pool_cls.call_args.kwargs["random_seed"], 123)


if __name__ == "__main__":
    unittest.main()
