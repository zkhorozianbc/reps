"""
F4: Convergence Monitor

Detects when evolutionary search is collapsing — when different workers and
lineages converge to produce the same kinds of edits. Runs in the controller
process at REPS batch boundaries.

Two metrics tracked over a sliding window:
- Edit entropy: Shannon entropy over edit type distribution
- Strategy divergence: KL divergence between worker type output distributions
"""

import logging
from collections import Counter, deque
from enum import Enum, auto
from math import log2
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConvergenceAction(Enum):
    """Escalation levels for convergence detection."""
    NONE = auto()
    MILD_BOOST = auto()       # Increase Explorer allocation, bump temperatures
    MODERATE_DIVERSIFY = auto()  # Force Explorer majority, inject distant restarts
    SEVERE_RESTART = auto()    # Switch model, double ε-revisitation, force old crossover


def classify_edit(diff: str) -> str:
    """Classify a diff into an edit type category.

    Uses a simple heuristic based on diff content patterns.
    More sophisticated approaches (AST-based) can be swapped in later.
    """
    if not diff:
        return "empty"

    diff_lower = diff.lower()
    length = len(diff)

    # Classify by structural patterns
    if length < 50:
        size = "tiny"
    elif length < 200:
        size = "small"
    elif length < 1000:
        size = "medium"
    else:
        size = "large"

    # Classify by content
    if "def " in diff or "function " in diff:
        content = "function"
    elif "class " in diff:
        content = "class"
    elif "import " in diff or "from " in diff:
        content = "import"
    elif "for " in diff or "while " in diff:
        content = "loop"
    elif "if " in diff or "else" in diff:
        content = "conditional"
    elif any(op in diff for op in ["+=", "-=", "*=", "/=", "=", "+"]):
        content = "arithmetic"
    elif "return " in diff:
        content = "return"
    else:
        content = "other"

    return f"{size}_{content}"


class ConvergenceMonitor:
    """Monitors edit diversity and detects convergence collapse.

    Runs in the controller process after each batch returns from the process pool.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: REPS convergence config dict with keys:
                - window_size: number of batches in sliding window
                - entropy_threshold_mild: float
                - entropy_threshold_moderate: float
                - entropy_threshold_severe: float
                - enabled: bool
        """
        self.enabled = config.get("enabled", True)
        self.window_size = config.get("window_size", 20)
        self.edit_history: deque = deque(maxlen=self.window_size * 50)
        self.thresholds = {
            "mild": config.get("entropy_threshold_mild", 0.5),
            "moderate": config.get("entropy_threshold_moderate", 0.3),
            "severe": config.get("entropy_threshold_severe", 0.15),
        }
        self._last_entropy = 0.0
        self._last_divergence = 0.0

    @property
    def last_entropy(self) -> float:
        return self._last_entropy

    @property
    def last_divergence(self) -> float:
        return self._last_divergence

    def update(self, batch_results: List) -> ConvergenceAction:
        """Called after each batch returns from process pool.

        Args:
            batch_results: List of IterationResult objects

        Returns:
            ConvergenceAction indicating what corrective action to take
        """
        if not self.enabled:
            return ConvergenceAction.NONE

        # Record edits from this batch
        for result in batch_results:
            if result.error is None:
                self.edit_history.append({
                    "diff": result.diff,
                    "worker_type": result.worker_type,
                    "improved": result.improved,
                })

        if len(self.edit_history) < 10:
            return ConvergenceAction.NONE

        self._last_entropy = self._compute_edit_entropy()
        self._last_divergence = self._compute_strategy_divergence()

        if self._last_entropy < self.thresholds["severe"]:
            logger.warning(
                f"SEVERE convergence collapse detected: entropy={self._last_entropy:.4f}"
            )
            return ConvergenceAction.SEVERE_RESTART
        elif self._last_entropy < self.thresholds["moderate"]:
            logger.warning(
                f"Moderate convergence detected: entropy={self._last_entropy:.4f}"
            )
            return ConvergenceAction.MODERATE_DIVERSIFY
        elif self._last_entropy < self.thresholds["mild"]:
            logger.info(
                f"Mild convergence detected: entropy={self._last_entropy:.4f}"
            )
            return ConvergenceAction.MILD_BOOST

        return ConvergenceAction.NONE

    def _compute_edit_entropy(self) -> float:
        """Shannon entropy over edit type distribution across the window."""
        edit_types = [classify_edit(e["diff"]) for e in self.edit_history]
        counts = Counter(edit_types)
        total = sum(counts.values())
        if total == 0:
            return 0.0

        probs = [c / total for c in counts.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)

        # Normalize by max possible entropy (log2 of number of categories)
        max_entropy = log2(max(len(counts), 1)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_strategy_divergence(self) -> float:
        """KL divergence between output distributions of different worker types.

        Measures whether different workers are producing the same kinds of edits.
        Low divergence = workers are converging on the same strategies.
        """
        # Group edits by worker type
        worker_edits: Dict[str, List[str]] = {}
        for e in self.edit_history:
            wt = e["worker_type"]
            if wt not in worker_edits:
                worker_edits[wt] = []
            worker_edits[wt].append(classify_edit(e["diff"]))

        if len(worker_edits) < 2:
            return 0.0

        # Get all edit categories
        all_categories = set()
        for edits in worker_edits.values():
            all_categories.update(edits)

        if not all_categories:
            return 0.0

        # Compute distribution per worker type
        distributions: Dict[str, Dict[str, float]] = {}
        for wt, edits in worker_edits.items():
            counts = Counter(edits)
            total = sum(counts.values())
            distributions[wt] = {
                cat: (counts.get(cat, 0) + 0.01) / (total + 0.01 * len(all_categories))
                for cat in all_categories
            }

        # Average pairwise KL divergence
        worker_types = list(distributions.keys())
        total_kl = 0.0
        pairs = 0
        for i in range(len(worker_types)):
            for j in range(i + 1, len(worker_types)):
                p = distributions[worker_types[i]]
                q = distributions[worker_types[j]]
                kl = sum(
                    p[cat] * log2(p[cat] / q[cat])
                    for cat in all_categories
                    if p[cat] > 0 and q[cat] > 0
                )
                total_kl += kl
                pairs += 1

        return total_kl / max(1, pairs)
