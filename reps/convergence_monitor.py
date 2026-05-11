"""
F4: Convergence Monitor

Detects when evolutionary search is stuck — when the MAP-Elites feature map
stops gaining new niches over a sliding window of batches. Runs in the
controller at REPS batch boundaries.

The action-driving signal is **niche-occupancy growth** (new MAP-Elites cells
filled per batch), normalized by batch size. This works mode-agnostically:
both full-mode workers (which produce whole programs) and diff-mode workers
(which produce edits) populate the same feature map, so the signal is
faithful to behavioral diversity regardless of how children are generated.

Two legacy diff-text signals — edit entropy (Shannon over edit categories)
and strategy divergence (KL between worker output distributions) — are still
computed and exposed via `last_entropy` / `last_divergence` so the diversity
CSV continues to capture them. They are informational only; they no longer
drive escalation actions, because both depend on `classify_edit(diff)` which
collapses to a single category when the diff is a full program rewrite.
"""

import logging
from collections import Counter, deque
from enum import Enum, auto
from math import log2
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConvergenceAction(Enum):
    """Escalation levels for convergence detection."""
    NONE = auto()
    MILD_BOOST = auto()       # Increase Explorer allocation, bump temperatures
    MODERATE_DIVERSIFY = auto()  # Force Explorer majority, inject distant restarts
    SEVERE_RESTART = auto()    # Switch model, double ε-revisitation, force old crossover


def classify_edit(diff: str) -> str:
    """Classify a diff into an edit type category.

    Used by the informational entropy / divergence signals. Note: for
    full-mode workers the controller stores the entire new program in
    `result.diff`, which collapses every result into the same category
    (e.g. `large_function`). That's why the action driver no longer uses
    these signals — see the module docstring.
    """
    if not diff:
        return "empty"

    length = len(diff)
    if length < 50:
        size = "tiny"
    elif length < 200:
        size = "small"
    elif length < 1000:
        size = "medium"
    else:
        size = "large"

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


def _feature_map_capacity(database) -> int:
    """Total addressable cells across all islands.

    Returns 0 if the database doesn't expose enough structure to compute
    capacity; callers should treat that as "saturation check unavailable"
    rather than "fully saturated."
    """
    fmaps = getattr(database, "island_feature_maps", None)
    if not fmaps:
        return 0
    cfg = getattr(database, "config", None)
    dims = getattr(cfg, "feature_dimensions", None) if cfg else None
    bins = getattr(database, "feature_bins", None)
    if not dims or not bins:
        return 0
    return len(fmaps) * (bins ** len(dims))


class ConvergenceMonitor:
    """Monitors MAP-Elites niche-occupancy growth and detects search stalls.

    Runs in the controller process after each batch returns from the process
    pool. The legacy `entropy_threshold_*` config keys are accepted as
    aliases for `niche_growth_threshold_*` — both live in [0, 1] and have
    the same operator semantic ("escalate when the diversity signal drops
    below this fraction"), so existing YAMLs keep working without changes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: REPS convergence config dict with keys:
                - enabled: bool
                - window_size: int — batches in the growth window
                - niche_growth_threshold_{mild,moderate,severe}: float in [0,1]
                - entropy_threshold_{mild,moderate,severe}: legacy alias
                - saturation_threshold: float in [0,1] — when occupied/capacity
                  exceeds this, skip stall detection (a full map means the
                  search succeeded, not collapsed).
        """
        self.enabled = config.get("enabled", True)
        self.window_size = config.get("window_size", 8)

        # +1 so we can compare current vs window_size batches ago
        self.niche_history: deque = deque(maxlen=self.window_size + 1)
        # Edit history for informational entropy / divergence signals only.
        self.edit_history: deque = deque(maxlen=self.window_size * 50)

        self.thresholds = {
            level: config.get(
                f"niche_growth_threshold_{level}",
                config.get(f"entropy_threshold_{level}", default),
            )
            for level, default in (("mild", 0.5), ("moderate", 0.3), ("severe", 0.15))
        }
        self.saturation_threshold = config.get("saturation_threshold", 0.8)

        self._last_entropy = 0.0
        self._last_divergence = 0.0
        self._last_niche_growth: Optional[float] = None
        self._last_saturation = 0.0

    @property
    def last_entropy(self) -> float:
        return self._last_entropy

    @property
    def last_divergence(self) -> float:
        return self._last_divergence

    @property
    def last_niche_growth(self) -> Optional[float]:
        return self._last_niche_growth

    @property
    def last_saturation(self) -> float:
        return self._last_saturation

    def update(self, batch_results: List, database=None) -> ConvergenceAction:
        """Called after each batch returns from the process pool.

        Args:
            batch_results: List of IterationResult objects
            database: ProgramDatabase. When None, falls back to NONE (no
                action) — the action driver requires the niche map.

        Returns:
            ConvergenceAction indicating what corrective action to take
        """
        if not self.enabled:
            return ConvergenceAction.NONE

        # Update informational edit signals (drive CSV columns, not actions).
        for r in batch_results:
            if r.error is None:
                self.edit_history.append({
                    "diff": r.diff,
                    "worker_type": r.worker_type,
                    "improved": r.improved,
                })
        if self.edit_history:
            self._last_entropy = self._compute_edit_entropy()
            self._last_divergence = self._compute_strategy_divergence()

        if database is None:
            return ConvergenceAction.NONE

        occupied = sum(
            len(fmap) for fmap in getattr(database, "island_feature_maps", [])
        )
        capacity = _feature_map_capacity(database)
        self._last_saturation = occupied / capacity if capacity else 0.0
        self.niche_history.append(occupied)

        # Wait for a full window before computing growth.
        if len(self.niche_history) <= self.window_size:
            return ConvergenceAction.NONE

        # A nearly-full map is success, not collapse — growth necessarily
        # slows when there's nowhere left to go.
        if capacity and self._last_saturation >= self.saturation_threshold:
            return ConvergenceAction.NONE

        # Normalize growth-per-batch by the per-batch ceiling: at most one
        # new niche per non-error child. A normalized growth of 1.0 means
        # every child landed in an unoccupied cell over the whole window;
        # 0.0 means none did.
        batch_size = max(1, sum(1 for r in batch_results if r.error is None))
        growth_per_batch = (self.niche_history[-1] - self.niche_history[0]) / self.window_size
        normalized = growth_per_batch / batch_size
        self._last_niche_growth = normalized

        if normalized < self.thresholds["severe"]:
            logger.info(
                f"Severe stall: niche growth={normalized:.3f}/batch "
                f"(saturation={self._last_saturation:.1%}) "
                f"-> triggering SEVERE_RESTART"
            )
            return ConvergenceAction.SEVERE_RESTART
        if normalized < self.thresholds["moderate"]:
            logger.info(
                f"Moderate stall: niche growth={normalized:.3f}/batch "
                f"(saturation={self._last_saturation:.1%}) "
                f"-> triggering MODERATE_DIVERSIFY"
            )
            return ConvergenceAction.MODERATE_DIVERSIFY
        if normalized < self.thresholds["mild"]:
            logger.info(
                f"Mild stall: niche growth={normalized:.3f}/batch "
                f"(saturation={self._last_saturation:.1%}) "
                f"-> triggering MILD_BOOST"
            )
            return ConvergenceAction.MILD_BOOST

        return ConvergenceAction.NONE

    def _compute_edit_entropy(self) -> float:
        """Shannon entropy over edit type distribution across the window.

        Informational only — see module docstring.
        """
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

        Informational only — see module docstring.
        """
        worker_edits: Dict[str, List[str]] = {}
        for e in self.edit_history:
            wt = e["worker_type"]
            if wt not in worker_edits:
                worker_edits[wt] = []
            worker_edits[wt].append(classify_edit(e["diff"]))

        if len(worker_edits) < 2:
            return 0.0

        all_categories = set()
        for edits in worker_edits.values():
            all_categories.update(edits)
        if not all_categories:
            return 0.0

        distributions: Dict[str, Dict[str, float]] = {}
        for wt, edits in worker_edits.items():
            counts = Counter(edits)
            total = sum(counts.values())
            distributions[wt] = {
                cat: (counts.get(cat, 0) + 0.01) / (total + 0.01 * len(all_categories))
                for cat in all_categories
            }

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
