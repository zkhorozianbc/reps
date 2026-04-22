"""
F3: Worker Type Diversity

Manages structurally different worker types that operate on programs in
fundamentally different ways:
- Exploiter: Small, targeted diffs. Low temperature.
- Explorer: Full rewrites within EVOLVE-BLOCK regions. High temperature.
- Crossover: Merges two parents from different MAP-Elites niches.

The WorkerPool wraps OpenEvolve's llm/ module — it decides WHICH call to make
and encodes that into IterationConfig objects. It does not make LLM calls directly.
"""

import logging
import random as _random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from reps.iteration_config import IterationConfig

logger = logging.getLogger(__name__)

# Module-level seeded RNG — set via WorkerPool.seed()
_rng = _random.Random()

# Default worker configurations
DEFAULT_WORKER_CONFIGS = {
    "exploiter": {
        "temperature": 0.3,
        "generation_mode": "diff",
        "description": "Small targeted diffs, low temperature",
    },
    "explorer": {
        "temperature": 1.0,
        "generation_mode": "full",
        "description": "Full rewrites within EVOLVE-BLOCKs, high temperature",
    },
    "crossover": {
        "temperature": 0.7,
        "generation_mode": "full",
        "description": "Merges two parents from different niches",
    },
}


class WorkerPool:
    """Manages worker type selection and IterationConfig building.

    Lives in the controller process. Does not make LLM calls directly.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: REPS workers config dict with keys:
                - types: list of worker type names
                - initial_allocation: dict of type -> weight
                - exploiter_temperature: float
                - explorer_temperature: float
        """
        types = config.get("types", ["exploiter", "explorer", "crossover"])
        alloc = config.get(
            "initial_allocation",
            {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15},
        )

        self.allocation = {t: alloc.get(t, 1.0 / len(types)) for t in types}
        self._normalize_allocation()

        self.yield_tracker: Dict[str, deque] = {
            t: deque(maxlen=100) for t in self.allocation
        }

        # Worker configs with overridable temperatures
        self.worker_configs = {}
        for t in types:
            base = DEFAULT_WORKER_CONFIGS.get(t, {"temperature": 0.7, "generation_mode": "diff"})
            self.worker_configs[t] = dict(base)

        # Apply config overrides
        if "exploiter_temperature" in config:
            self.worker_configs["exploiter"]["temperature"] = config["exploiter_temperature"]
        if "explorer_temperature" in config:
            self.worker_configs["explorer"]["temperature"] = config["explorer_temperature"]

        # Forced explorer counter (for convergence monitor)
        self._force_explorer_batches = 0

        # Seed RNG from config if provided
        seed = config.get("random_seed")
        if seed is not None:
            _rng.seed(seed)

        logger.info(f"WorkerPool initialized: {self.allocation}")

    def _normalize_allocation(self):
        total = sum(self.allocation.values())
        if total > 0:
            self.allocation = {k: v / total for k, v in self.allocation.items()}

    def build_iteration_config(
        self,
        database,
        prompt_extras: Dict[str, str],
        override_type: Optional[str] = None,
        target_island: Optional[int] = None,
    ) -> IterationConfig:
        """Build an IterationConfig for one iteration dispatch.

        Args:
            database: ProgramDatabase for sampling crossover parents
            prompt_extras: Dict with reflection, SOTA injection, dead-end warnings
            override_type: Force a specific worker type
            target_island: Target island for this iteration
        """
        worker_type = override_type or self._sample_worker_type()
        wconf = self.worker_configs.get(worker_type, {"temperature": 0.7, "generation_mode": "diff"})

        config = IterationConfig(
            parent_id=None,
            worker_name=worker_type,
            model_id=None,
            temperature=wconf["temperature"],
            prompt_extras=dict(prompt_extras),
            second_parent_id=None,
            is_revisitation=False,
            generation_mode=wconf["generation_mode"],
            target_island=target_island,
        )

        # Crossover needs a second parent from a distant niche
        if worker_type == "crossover" and database is not None:
            second = self._sample_distant_parent(database, target_island)
            if second is not None:
                config.second_parent_id = second

        return config

    def record_result(self, worker_type: str, improved: bool):
        """Track yield per worker type for allocation rebalancing."""
        if worker_type in self.yield_tracker:
            self.yield_tracker[worker_type].append(1.0 if improved else 0.0)

    def get_yield_rate(self, worker_type: str) -> float:
        """Get the recent yield rate for a worker type."""
        tracker = self.yield_tracker.get(worker_type)
        if not tracker:
            return 0.0
        return sum(tracker) / max(1, len(tracker))

    def _sample_worker_type(self) -> str:
        """Sample a worker type based on current allocation weights."""
        if self._force_explorer_batches > 0:
            self._force_explorer_batches -= 1
            # 50% explorer, distribute rest proportionally
            if _rng.random() < 0.5 and "explorer" in self.allocation:
                return "explorer"

        types = list(self.allocation.keys())
        weights = [self.allocation[t] for t in types]
        return _rng.choices(types, weights=weights, k=1)[0]

    def _sample_distant_parent(self, database, target_island: Optional[int]) -> Optional[str]:
        """Sample a parent from a different island or distant niche for crossover."""
        try:
            # Try to get a program from a different island
            num_islands = len(database.islands)
            if num_islands <= 1:
                return None

            # Pick a random different island
            current = target_island if target_island is not None else database.current_island
            other_islands = [i for i in range(num_islands) if i != current]
            if not other_islands:
                return None

            other_island = _rng.choice(other_islands)
            island_pids = list(database.islands[other_island])
            if not island_pids:
                return None

            return _rng.choice(island_pids)
        except Exception as e:
            logger.debug(f"Could not sample distant parent: {e}")
            return None

    # --- Convergence monitor actions ---

    def set_allocation(self, new_allocation: Dict[str, float]):
        """Set worker allocation directly (used by SOTA controller)."""
        self.allocation = dict(new_allocation)
        self._normalize_allocation()
        logger.info(f"Worker allocation set to: {self.allocation}")

    def boost_explorer(self, amount: float):
        """Increase explorer allocation by a relative amount."""
        if "explorer" in self.allocation:
            self.allocation["explorer"] += amount
            self._normalize_allocation()
            logger.info(f"Boosted explorer: {self.allocation}")

    def bump_temperatures(self, delta: float):
        """Increase all worker temperatures by delta."""
        for wt in self.worker_configs:
            self.worker_configs[wt]["temperature"] = min(
                2.0, self.worker_configs[wt]["temperature"] + delta
            )
        logger.info(f"Bumped temperatures by {delta}")

    def force_explorer_majority(self, num_batches: int):
        """Force explorer to be majority for the next N batches."""
        self._force_explorer_batches = num_batches
        logger.info(f"Forcing explorer majority for {num_batches} batches")

    def force_model_switch(self):
        """Signal that models should be switched (handled by contract selector)."""
        logger.info("Model switch requested by convergence monitor")

    def get_alternative_worker_type(self, original_type: str) -> str:
        """Get a different worker type for revisitation."""
        types = [t for t in self.allocation if t != original_type]
        if not types:
            return original_type
        return _rng.choice(types)
