"""WorkerPool: sample among named WorkerConfig entries weighted by cfg.weight.

Crossover is a role (cfg.role == "crossover"), not a name. Any config with
role=="crossover" participates in the second-parent sampling path.
"""
from __future__ import annotations

import logging
import math
import random as _random
from collections import deque
from typing import Dict, List, Optional

from reps.iteration_config import IterationConfig
from reps.workers.base import WorkerConfig

logger = logging.getLogger(__name__)


class WorkerPool:
    def __init__(
        self,
        workers_config,
        *,
        default_model_id: str = "",
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            workers_config: A ``REPSWorkersConfig`` dataclass. Its ``types``
                field must be a non-empty ``List[WorkerConfig]``.
            default_model_id: Reserved for future use; no longer consumed
                during pool construction.
        """
        typed_types: List[WorkerConfig] = list(getattr(workers_config, "types", []) or [])
        if random_seed is None:
            random_seed = getattr(workers_config, "random_seed", None)

        if not typed_types:
            raise ValueError(
                "WorkerPool requires workers_config.types to be a non-empty "
                "List[WorkerConfig]. Declare workers under reps.workers.types "
                "in your YAML config."
            )

        names = [c.name for c in typed_types]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate worker name(s): {', '.join(duplicates)}")

        for cfg in typed_types:
            if not isinstance(cfg.weight, (int, float)) or not math.isfinite(cfg.weight) or cfg.weight <= 0:
                raise ValueError(
                    f"Worker {cfg.name!r} weight must be positive finite; got {cfg.weight!r}"
                )

        self._rng = _random.Random(random_seed)
        self._configs: Dict[str, WorkerConfig] = {c.name: c for c in typed_types}
        total = sum(c.weight for c in typed_types)
        self._weights: Dict[str, float] = {c.name: c.weight / total for c in typed_types}
        self.yield_tracker: Dict[str, deque] = {
            c.name: deque(maxlen=100) for c in typed_types
        }
        self._force_explorer_batches = 0

        logger.info(
            "WorkerPool initialized: %s",
            {n: round(w, 3) for n, w in self._weights.items()},
        )

    # ------------------------------------------------------------------
    # Backward-compat alias: old code reads pool.allocation
    # ------------------------------------------------------------------

    @property
    def allocation(self) -> Dict[str, float]:
        """Alias for _weights (backward compat with old callers and tests)."""
        return self._weights

    @allocation.setter
    def allocation(self, value: Dict[str, float]):
        self._weights = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_worker_config(self, name: str) -> WorkerConfig:
        """Return the WorkerConfig for the given name (replaces Task 10 shim)."""
        return self._configs[name]

    def build_iteration_config(
        self,
        database,
        prompt_extras: Dict[str, str],
        override_name: Optional[str] = None,
        target_island: Optional[int] = None,
        # Backward-compat alias
        override_type: Optional[str] = None,
    ) -> IterationConfig:
        """Build an IterationConfig for one iteration dispatch.

        Args:
            database: ProgramDatabase for sampling crossover parents.
            prompt_extras: Dict with reflection, SOTA injection, dead-end warnings.
            override_name: Force a specific named worker config.
            target_island: Target island for this iteration.
            override_type: Deprecated alias for override_name.
        """
        name = override_name or override_type or self._sample()
        cfg = self._configs[name]

        second_parent_id = None
        if cfg.role == "crossover" and database is not None:
            second_parent_id = self._sample_distant_parent(database, target_island)

        return IterationConfig(
            parent_id=None,
            worker_name=cfg.name,
            model_id=cfg.model_id if cfg.owns_model else None,
            temperature=cfg.temperature if cfg.owns_temperature else None,
            prompt_extras=dict(prompt_extras),
            second_parent_id=second_parent_id,
            is_revisitation=False,
            generation_mode=cfg.generation_mode,
            target_island=target_island,
        )

    def record_result(self, name: str, improved: bool):
        """Track yield per worker name for allocation rebalancing."""
        if name in self.yield_tracker:
            self.yield_tracker[name].append(1.0 if improved else 0.0)

    def get_yield_rate(self, name: str) -> float:
        """Get the recent yield rate for a worker name."""
        t = self.yield_tracker.get(name)
        if not t:
            return 0.0
        return sum(t) / max(1, len(t))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self) -> str:
        """Sample a worker name based on current weight distribution."""
        names = list(self._weights)
        weights = [self._weights[n] for n in names]
        if self._force_explorer_batches > 0:
            self._force_explorer_batches -= 1
            explorers = [n for n in names if self._configs[n].role == "explorer"]
            if explorers and self._rng.random() < 0.5:
                return self._rng.choice(explorers)
        return self._rng.choices(names, weights=weights, k=1)[0]

    def _sample_distant_parent(
        self, database, target_island: Optional[int]
    ) -> Optional[str]:
        """Sample a parent from a different island or distant niche for crossover."""
        try:
            num_islands = len(database.islands)
            if num_islands <= 1:
                return None
            current = (
                target_island if target_island is not None else database.current_island
            )
            other = [i for i in range(num_islands) if i != current]
            if not other:
                return None
            pick_island = self._rng.choice(other)
            pids = list(database.islands[pick_island])
            if not pids:
                return None
            return self._rng.choice(pids)
        except Exception as e:
            logger.debug(f"distant parent sample failed: {e}")
            return None

    # ------------------------------------------------------------------
    # SOTA / Convergence Monitor adjustments
    # ------------------------------------------------------------------

    def set_allocation(self, new_allocation: Dict[str, float]):
        """Set worker allocation directly (used by SOTA controller)."""
        for name, weight in new_allocation.items():
            if name in self._weights and (
                not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    f"Worker {name!r} allocation weight must be positive finite; got {weight!r}"
                )
        total = sum(new_allocation.get(name, 0.0) for name in self._weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Worker allocation must include at least one positive finite weight")
        for name, weight in new_allocation.items():
            if name in self._weights:
                self._weights[name] = weight / total
        logger.info(f"Worker allocation set to: {self._weights}")

    def boost_explorer(self, amount: float):
        """Increase explorer allocation by a relative amount."""
        explorers = [n for n in self._weights if self._configs[n].role == "explorer"]
        if not explorers:
            return
        share = amount / len(explorers)
        for n in explorers:
            self._weights[n] += share
        # renormalize
        total = sum(self._weights.values()) or 1.0
        for n in self._weights:
            self._weights[n] /= total
        logger.info(f"Boosted explorer: {self._weights}")

    def bump_temperatures(self, delta: float):
        """Increase all worker temperatures by delta."""
        for cfg in self._configs.values():
            if cfg.temperature is not None:
                cfg.temperature = min(2.0, cfg.temperature + delta)
        logger.info(f"Bumped temperatures by {delta}")

    def force_explorer_majority(self, num_batches: int):
        """Force explorer to be majority for the next N batches."""
        self._force_explorer_batches = num_batches
        logger.info(f"Forcing explorer majority for {num_batches} batches")

    def get_alternative_worker_name(self, original: str) -> str:
        """Get a different worker name for revisitation."""
        names = [n for n in self._configs if n != original]
        return self._rng.choice(names) if names else original

    # Backward-compat alias
    def get_alternative_worker_type(self, original_type: str) -> str:
        """Deprecated alias for get_alternative_worker_name."""
        return self.get_alternative_worker_name(original_type)
