"""
F6: SOTA-Distance Steering

For benchmark problems with a known state-of-the-art score, the system uses
the gap between current best and target as a continuous control signal that
modulates search behavior.
"""

import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SearchRegime(Enum):
    """Search regime based on gap to SOTA."""
    FAR = auto()        # >30% gap
    MID = auto()        # 10-30% gap
    NEAR = auto()       # 2-10% gap
    POLISHING = auto()  # <2% gap


# Prompt injections per regime
REGIME_INJECTIONS = {
    SearchRegime.FAR: (
        "You are far from the best known result. "
        "Try fundamentally different algorithmic approaches. "
        "Radical restructuring is encouraged."
    ),
    SearchRegime.MID: (
        "The current approach has merit but needs structural improvements. "
        "Look for algorithmic inefficiencies and consider hybrid strategies."
    ),
    SearchRegime.NEAR: (
        "You are close to the best known result. "
        "Focus on surgical parameter tuning and micro-optimizations. "
        "Small, precise changes are more likely to help than large rewrites."
    ),
    SearchRegime.POLISHING: (
        "You are within 2% of the best known result. "
        "Only make changes you are highly confident will improve the score. "
        "Verify numerical precision and edge cases."
    ),
}

# Worker allocations per regime
REGIME_ALLOCATIONS = {
    SearchRegime.FAR:       {"exploiter": 0.3, "explorer": 0.5, "crossover": 0.2},
    SearchRegime.MID:       {"exploiter": 0.5, "explorer": 0.3, "crossover": 0.2},
    SearchRegime.NEAR:      {"exploiter": 0.7, "explorer": 0.15, "crossover": 0.15},
    SearchRegime.POLISHING: {"exploiter": 0.85, "explorer": 0.05, "crossover": 0.10},
}


class SOTAController:
    """Gap-aware search modulation.

    Runs in the controller process. Modulates worker allocation and prompt
    injection based on the gap between current best and known SOTA.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: REPS SOTA config dict with keys:
                - target_score: float, the known SOTA or theoretical bound
                - enabled: bool
        """
        self.enabled = config.get("enabled", True)
        self.target = config.get("target_score", None)
        self.gap_history: List[float] = []
        self._current_regime = SearchRegime.FAR

        if self.target is not None:
            logger.info(f"SOTAController targeting score: {self.target}")

    @property
    def current_regime(self) -> SearchRegime:
        return self._current_regime

    @property
    def current_gap(self) -> Optional[float]:
        """Current gap as a fraction (0.0 = at SOTA, 1.0 = 100% away)."""
        if self.gap_history:
            return self.gap_history[-1]
        return None

    def get_regime(self, current_best: float) -> SearchRegime:
        """Determine search regime based on gap to SOTA.

        Args:
            current_best: Current best score achieved

        Returns:
            SearchRegime enum value
        """
        if not self.enabled or self.target is None:
            return SearchRegime.MID

        if self.target == 0:
            gap = abs(current_best) if current_best != 0 else 1.0
        else:
            gap = (self.target - current_best) / abs(self.target)

        # Clamp gap to [0, 1]
        gap = max(0.0, min(1.0, gap))
        self.gap_history.append(gap)

        if gap > 0.30:
            self._current_regime = SearchRegime.FAR
        elif gap > 0.10:
            self._current_regime = SearchRegime.MID
        elif gap > 0.02:
            self._current_regime = SearchRegime.NEAR
        else:
            self._current_regime = SearchRegime.POLISHING

        logger.debug(
            f"SOTA gap: {gap:.4f}, regime: {self._current_regime.name}"
        )
        return self._current_regime

    def get_prompt_injection(self, regime: Optional[SearchRegime] = None) -> str:
        """Get the prompt injection text for the current or given regime."""
        r = regime or self._current_regime
        return REGIME_INJECTIONS.get(r, "")

    def modulate_worker_allocation(self, regime: Optional[SearchRegime] = None) -> Dict[str, float]:
        """Get the worker allocation for the current or given regime."""
        r = regime or self._current_regime
        return dict(REGIME_ALLOCATIONS.get(r, REGIME_ALLOCATIONS[SearchRegime.MID]))

    def format_for_prompt(self) -> str:
        """Format SOTA info as text suitable for injection into prompts."""
        if not self.enabled or self.target is None:
            return ""

        gap = self.current_gap
        if gap is None:
            return ""

        injection = self.get_prompt_injection()
        lines = [
            f"## SOTA Guidance (target: {self.target}, gap: {gap:.2%})",
            injection,
        ]
        return "\n".join(lines)
