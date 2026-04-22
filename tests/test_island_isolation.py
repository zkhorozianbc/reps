"""
Tests for worker-to-island pinning under the async controller.

These tests assert the invariants that matter post-refactor:
- _pick_iteration_inputs threads the requested island through target_island.
- database.sample_from_island is called with the intended island id.
- The initial batch is distributed across islands.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from reps.config import Config
from reps.controller import ProcessParallelController, SerializableResult
from reps.database import Program, ProgramDatabase


class TestIslandIsolation(unittest.TestCase):
    """Test that island targeting is preserved through the async dispatch path."""

    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.config.database.num_islands = 3
        self.config.evaluator.parallel_evaluations = 6  # 2 workers per island
        self.config.database.population_size = 30

        self.database = ProgramDatabase(self.config.database)

        # Real eval file so controller.start() can construct an Evaluator.
        self.test_dir = tempfile.mkdtemp()
        self.evaluation_file = os.path.join(self.test_dir, "evaluator.py")
        with open(self.evaluation_file, "w") as f:
            f.write(
                "def evaluate(program_path):\n"
                "    return {'score': 0.5, 'performance': 0.6}\n"
            )

        # Seed each island so sample_from_island has something to return.
        for i in range(9):
            program = Program(
                id=f"test_prog_{i}",
                code=f"# Test program {i}",
                metrics={"combined_score": 0.5},
            )
            island_id = i % 3
            program.metadata["island"] = island_id
            self.database.add(program)
            self.database.islands[island_id].add(program.id)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pick_iteration_inputs_uses_requested_island(self):
        """_pick_iteration_inputs sets target_island on the IterationConfig
        and calls sample_from_island with that id."""
        controller = ProcessParallelController(
            self.config, self.evaluation_file, self.database
        )

        sampled_islands = []
        original_sample = self.database.sample_from_island

        def tracking_sample(island_id, num_inspirations=None):
            sampled_islands.append(island_id)
            return original_sample(island_id, num_inspirations=num_inspirations)

        with patch.object(
            self.database, "sample_from_island", side_effect=tracking_sample
        ):
            for want_island in (0, 1, 2, 0):
                parent_id, inspiration_ids, cfg = controller._pick_iteration_inputs(
                    iteration=10, island_id=want_island, reps_iter_config=None
                )
                self.assertEqual(cfg.target_island, want_island)
                self.assertIn(parent_id, self.database.programs)

        self.assertEqual(sampled_islands, [0, 1, 2, 0])

    def test_database_current_island_not_mutated_by_sampling(self):
        """sample_from_island must not mutate current_island."""
        controller = ProcessParallelController(
            self.config, self.evaluation_file, self.database
        )
        self.database.current_island = 1
        original_island = self.database.current_island

        controller._pick_iteration_inputs(
            iteration=100, island_id=2, reps_iter_config=None
        )

        self.assertEqual(self.database.current_island, original_island)

    def test_island_distribution_in_initial_batch(self):
        """run_evolution's initial batch is distributed round-robin across islands."""
        controller = ProcessParallelController(
            self.config, self.evaluation_file, self.database
        )

        submitted_islands = []

        original_pick = controller._pick_iteration_inputs

        def tracking_pick(iteration, island_id, reps_iter_config):
            submitted_islands.append(island_id)
            return original_pick(iteration, island_id, reps_iter_config)

        async def run_test():
            controller.start()
            try:
                with patch.object(
                    controller, "_pick_iteration_inputs", side_effect=tracking_pick
                ):
                    with patch.object(
                        controller,
                        "_run_iteration",
                        new=AsyncMock(
                            return_value=SerializableResult(
                                error="stub", iteration=0
                            )
                        ),
                    ):
                        await controller.run_evolution(1, 6)
            finally:
                controller.stop()

        asyncio.run(run_test())

        island_counts = {0: 0, 1: 0, 2: 0}
        for island_id in submitted_islands:
            island_counts[island_id] += 1

        for count in island_counts.values():
            self.assertGreater(count, 0)


class TestIslandMigration(unittest.TestCase):
    """Test that migration still works with island pinning"""

    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.config.database.num_islands = 3
        self.config.database.migration_interval = 10
        self.config.database.migration_rate = 0.1
        self.database = ProgramDatabase(self.config.database)

    def test_migration_preserves_island_structure(self):
        """Test that migration works correctly with pinned workers"""
        for i in range(30):
            program = Program(
                id=f"prog_{i}", code=f"# Program {i}", metrics={"combined_score": i * 0.1}
            )
            island_id = i % 3
            program.metadata["island"] = island_id

            self.database.programs[program.id] = program
            self.database.islands[island_id].add(program.id)

        island_sizes_before = [len(island) for island in self.database.islands]
        original_program_count = len(self.database.programs)

        self.assertEqual(sum(island_sizes_before), 30)
        self.assertEqual(original_program_count, 30)

        self.database.migrate_programs()

        island_sizes_after = [len(island) for island in self.database.islands]
        total_programs_after = len(self.database.programs)

        for size in island_sizes_after:
            self.assertGreater(size, 0)

        self.assertGreater(total_programs_after, original_program_count)
        self.assertGreater(sum(island_sizes_after), sum(island_sizes_before))

        migrant_count = 0
        for program in self.database.programs.values():
            if program.metadata.get("migrant", False):
                migrant_count += 1
                self.assertNotIn(
                    "_migrant_",
                    program.id,
                    "New implementation should not create _migrant suffix programs",
                )

        self.assertGreater(migrant_count, 0)

        migrant_suffix_count = sum(
            1 for p in self.database.programs.values() if "_migrant_" in p.id
        )
        self.assertEqual(
            migrant_suffix_count,
            0,
            "No programs should have _migrant_ suffixes with new implementation",
        )


if __name__ == "__main__":
    unittest.main()
