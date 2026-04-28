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

from reps.config import Config
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


if __name__ == "__main__":
    unittest.main()
