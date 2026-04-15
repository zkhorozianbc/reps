"""
F5: Intelligence Contracts

Instead of fixed model weights, each LLM call has a learnable contract tuple:
(model_id, temperature). A Thompson-sampling bandit learns which configs
produce the best yield for different program states.
"""

import logging
import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Contract:
    """A selected contract: model + temperature to use for an iteration."""
    model_id: str
    temperature: float


class ContractSelector:
    """Thompson-sampling bandit over (model_id, temperature) arms.

    Each arm maintains a Beta(alpha, beta) posterior. The bandit samples
    from posteriors and selects the arm with the highest sample.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: REPS contracts config dict with keys:
                - models: list of model name strings
                - temperatures: list of temperature floats
                - enabled: bool
        """
        self.enabled = config.get("enabled", True)
        self.arms = self._build_arms(config)
        # Beta(alpha, beta) posterior per arm
        self.posteriors: Dict[Tuple[str, float], Dict[str, float]] = {
            arm: {"alpha": 1.0, "beta": 1.0} for arm in self.arms
        }
        seed = config.get("random_seed")
        self._rng = random.Random(seed)

        if self.arms:
            logger.info(f"ContractSelector initialized with {len(self.arms)} arms: {self.arms}")

    def select(self, context: Optional[Dict[str, Any]] = None) -> Optional[Contract]:
        """Thompson sampling: sample from each posterior, pick highest.

        Args:
            context: Optional context dict (for future context-dependent bandits)
                Keys: current_best, sota_gap, batch_number, parent_complexity,
                      worker_type, num_failed_attempts

        Returns:
            Contract with selected model_id and temperature, or None if disabled
        """
        if not self.enabled or not self.arms:
            return None

        samples = {}
        for arm, params in self.posteriors.items():
            # Sample from Beta distribution
            sample = self._rng.betavariate(
                max(0.01, params["alpha"]),
                max(0.01, params["beta"]),
            )
            samples[arm] = sample

        best_arm = max(samples, key=samples.get)
        return Contract(model_id=best_arm[0], temperature=best_arm[1])

    def update(self, model_id: str, temperature: float, success: bool):
        """Update posterior with observation.

        Args:
            model_id: Model that was used
            temperature: Temperature that was used
            success: Whether the iteration produced a score improvement
        """
        if not self.enabled:
            return

        arm = (model_id, temperature)
        if arm not in self.posteriors:
            # Find closest arm
            arm = self._find_closest_arm(model_id, temperature)
            if arm is None:
                return

        if success:
            self.posteriors[arm]["alpha"] += 1.0
        else:
            self.posteriors[arm]["beta"] += 1.0

    def _find_closest_arm(self, model_id: str, temperature: float) -> Optional[Tuple[str, float]]:
        """Find the closest arm to the given model/temp combination."""
        # First try exact model match with closest temperature
        model_arms = [a for a in self.arms if a[0] == model_id]
        if model_arms:
            return min(model_arms, key=lambda a: abs(a[1] - temperature))

        # Fall back to closest overall
        if self.arms:
            return min(self.arms, key=lambda a: abs(a[1] - temperature))

        return None

    def _build_arms(self, config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Build the set of (model_id, temperature) arms."""
        models = config.get("models", [])
        temperatures = config.get("temperatures", [0.3, 0.7, 1.0])

        if not models:
            return []

        return list(product(models, temperatures))

    def get_posteriors_summary(self) -> Dict[str, Dict[str, float]]:
        """Return posteriors for logging/visualization."""
        return {
            f"{arm[0]}@{arm[1]}": {
                "alpha": params["alpha"],
                "beta": params["beta"],
                "mean": params["alpha"] / (params["alpha"] + params["beta"]),
                "trials": params["alpha"] + params["beta"] - 2,  # subtract initial priors
            }
            for arm, params in self.posteriors.items()
        }
