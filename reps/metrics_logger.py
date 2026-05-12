"""
Metrics logging for REPS ablation studies.

Writes CSV files at batch boundaries for:
- Score trajectory
- Worker yield rates
- Diversity metrics (edit entropy, strategy divergence, niche occupancy)
- Cost tracking (tokens, wall-clock time)
- Reflection logs (JSON)
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Logs REPS metrics to CSV files at batch boundaries."""

    # Health-tracking thresholds (class attributes so they're not magic
    # numbers buried in write_health). Tweak here, not at call sites.
    ANNOTATION_SUCCESS_WARN_THRESHOLD = 0.5
    ACTION_RECOVERY_WARN_MIN_FIRES = 3
    ACTION_RECOVERY_WARN_RATE = 0.0  # warn if 0% of fired actions recovered
    ACTION_RECOVERY_LOOKAHEAD_BATCHES = 2
    # Niche-occupancy growth ratio counted as a "recovery" signal — matches
    # the convergence monitor's mild threshold spirit (any healthy growth).
    ACTION_RECOVERY_NICHE_GROWTH = 0.2

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_csvs()
        self._batch_count = 0
        # --- run-health counters ---
        self._annotation_attempts = 0
        self._annotation_successes = 0
        # Per-action-level fire counts and recovery counts.
        self._action_fired: Dict[str, int] = {}
        self._action_recovered: Dict[str, int] = {}
        # Pending fires awaiting a recovery decision in the next N batches.
        # Each entry: (action_level, fired_at_batch, best_score_at_fire,
        # niche_occupancy_at_fire).
        self._pending_action_outcomes: List[Dict[str, Any]] = []
        logger.info(f"MetricsLogger writing to {self.output_dir}")

    def _init_csvs(self):
        """Initialize CSV files with headers."""
        self._score_file = self.output_dir / "score_trajectory.csv"
        self._worker_file = self.output_dir / "worker_yield.csv"
        self._diversity_file = self.output_dir / "diversity.csv"
        self._cost_file = self.output_dir / "cost.csv"
        self._reflection_file = self.output_dir / "reflection_log.jsonl"

        if not self._score_file.exists():
            self._write_header(
                self._score_file,
                ["batch", "best_score", "mean_score", "worst_score", "num_improvements", "timestamp"],
            )
        if not self._worker_file.exists():
            self._write_header(
                self._worker_file,
                ["batch", "worker_type", "num_candidates", "num_improvements", "yield_rate"],
            )
        if not self._diversity_file.exists():
            self._write_header(
                self._diversity_file,
                ["batch", "edit_entropy", "strategy_divergence", "niche_occupancy", "unique_edit_types"],
            )
        if not self._cost_file.exists():
            self._write_header(
                self._cost_file,
                ["batch", "model", "tokens_in", "tokens_out", "wall_clock_seconds"],
            )

    def _write_header(self, path: Path, headers: List[str]):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _append_row(self, path: Path, row: List[Any]):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_batch(
        self,
        batch_number: int,
        batch_results: List,
        database=None,
        edit_entropy: float = 0.0,
        strategy_divergence: float = 0.0,
    ):
        """Called in controller after each batch of iterations returns.

        Args:
            batch_number: Current batch number
            batch_results: List of IterationResult objects from the batch
            database: ProgramDatabase for niche occupancy
            edit_entropy: From convergence monitor
            strategy_divergence: From convergence monitor
        """
        self._batch_count = batch_number

        # Score trajectory
        scores = []
        num_improvements = 0
        for r in batch_results:
            if r.child_score is not None and r.error is None:
                scores.append(r.child_score)
                if r.improved:
                    num_improvements += 1

        if scores:
            best_db_score = 0.0
            if database and database.best_program_id:
                best_prog = database.get_best_program()
                if best_prog and best_prog.metrics:
                    from reps.utils import safe_numeric_average
                    best_db_score = best_prog.metrics.get(
                        "combined_score", safe_numeric_average(best_prog.metrics)
                    )
            self._append_row(
                self._score_file,
                [
                    batch_number,
                    f"{best_db_score:.6f}",
                    f"{sum(scores) / len(scores):.6f}",
                    f"{min(scores):.6f}",
                    num_improvements,
                    f"{time.time():.2f}",
                ],
            )

        # Worker yield
        worker_stats: Dict[str, Dict[str, int]] = {}
        for r in batch_results:
            wt = r.worker_type
            if wt not in worker_stats:
                worker_stats[wt] = {"candidates": 0, "improvements": 0}
            worker_stats[wt]["candidates"] += 1
            if r.improved:
                worker_stats[wt]["improvements"] += 1

        for wt, stats in worker_stats.items():
            yield_rate = stats["improvements"] / max(1, stats["candidates"])
            self._append_row(
                self._worker_file,
                [batch_number, wt, stats["candidates"], stats["improvements"], f"{yield_rate:.4f}"],
            )

        # Diversity
        niche_occupancy = 0
        if database:
            niche_occupancy = self._compute_niche_occupancy(database)
        unique_types = len(worker_stats)
        self._append_row(
            self._diversity_file,
            [
                batch_number,
                f"{edit_entropy:.4f}",
                f"{strategy_divergence:.4f}",
                niche_occupancy,
                unique_types,
            ],
        )

        # Cost
        model_costs: Dict[str, Dict[str, float]] = {}
        for r in batch_results:
            mid = r.model_id or "default"
            if mid not in model_costs:
                model_costs[mid] = {"tokens_in": 0, "tokens_out": 0, "wall_clock": 0.0}
            model_costs[mid]["tokens_in"] += r.tokens_in
            model_costs[mid]["tokens_out"] += r.tokens_out
            model_costs[mid]["wall_clock"] += r.wall_clock_seconds

        for mid, costs in model_costs.items():
            self._append_row(
                self._cost_file,
                [
                    batch_number,
                    mid,
                    costs["tokens_in"],
                    costs["tokens_out"],
                    f"{costs['wall_clock']:.2f}",
                ],
            )

    def log_reflection(
        self,
        batch_number: int,
        reflection: Dict[str, Any],
        reflection_calls: int = 0,
        reflection_tokens: Optional[Dict[str, int]] = None,
    ):
        """Append a reflection JSON entry with cost tracking."""
        entry = {
            "batch": batch_number,
            "reflection": reflection,
            "total_reflection_llm_calls": reflection_calls,
            "total_reflection_tokens": reflection_tokens or {},
        }
        with open(self._reflection_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _compute_niche_occupancy(self, database) -> int:
        """Count occupied niches across all islands."""
        total = 0
        if hasattr(database, "island_feature_maps"):
            for fmap in database.island_feature_maps:
                total += len(fmap)
        return total

    # ------------------------------------------------------------------ #
    # Run-health tracking (F8 annotations + convergence-action recovery) #
    # ------------------------------------------------------------------ #

    def record_annotation_attempt(self, success: bool) -> None:
        """Increment annotation attempt counters. Called from controller."""
        self._annotation_attempts += 1
        if success:
            self._annotation_successes += 1

    def record_action_fired(
        self,
        action_level: str,
        batch_number: int,
        best_score: Optional[float],
        niche_occupancy: int,
    ) -> None:
        """Record that a convergence action fired this batch.

        Recovery is decided later, when subsequent batches are observed via
        observe_post_action(). Healthy actions (NONE) shouldn't be recorded.
        """
        self._action_fired[action_level] = self._action_fired.get(action_level, 0) + 1
        self._pending_action_outcomes.append({
            "level": action_level,
            "fired_at": batch_number,
            "best_score": best_score if best_score is not None else 0.0,
            "niche_occupancy": niche_occupancy,
        })

    def observe_post_action(
        self,
        batch_number: int,
        best_score: Optional[float],
        niche_occupancy: int,
    ) -> None:
        """Settle pending action recoveries up to ACTION_RECOVERY_LOOKAHEAD."""
        if not self._pending_action_outcomes:
            return
        cur_score = best_score if best_score is not None else 0.0
        still_pending = []
        for entry in self._pending_action_outcomes:
            elapsed = batch_number - entry["fired_at"]
            if elapsed < 1:
                still_pending.append(entry)
                continue
            score_improved = cur_score > entry["best_score"]
            base_niche = max(1, entry["niche_occupancy"])
            niche_growth = (niche_occupancy - entry["niche_occupancy"]) / base_niche
            recovered = score_improved or niche_growth >= self.ACTION_RECOVERY_NICHE_GROWTH
            if recovered:
                self._action_recovered[entry["level"]] = (
                    self._action_recovered.get(entry["level"], 0) + 1
                )
            elif elapsed < self.ACTION_RECOVERY_LOOKAHEAD_BATCHES:
                # Still within the lookahead window, give it another batch.
                still_pending.append(entry)
        self._pending_action_outcomes = still_pending

    def write_health(self) -> Dict[str, Any]:
        """Write health.json summarising annotation + action outcomes.

        Emits WARNINGs (once each) when thresholds are violated. Returns the
        written dict for tests / programmatic use.
        """
        ann_attempts = self._annotation_attempts
        ann_successes = self._annotation_successes
        ann_rate = (ann_successes / ann_attempts) if ann_attempts else None

        total_fired = sum(self._action_fired.values())
        total_recovered = sum(self._action_recovered.values())
        recovery_rate = (total_recovered / total_fired) if total_fired else None

        health = {
            "annotations": {
                "attempts": ann_attempts,
                "successes": ann_successes,
                "success_rate": ann_rate,
                "warn_threshold": self.ANNOTATION_SUCCESS_WARN_THRESHOLD,
            },
            "convergence_actions": {
                "fired_per_level": dict(self._action_fired),
                "recovered_per_level": dict(self._action_recovered),
                "total_fired": total_fired,
                "total_recovered": total_recovered,
                "recovery_rate": recovery_rate,
                "warn_min_fires": self.ACTION_RECOVERY_WARN_MIN_FIRES,
                "warn_recovery_rate": self.ACTION_RECOVERY_WARN_RATE,
            },
        }

        if ann_attempts > 0 and ann_rate is not None and ann_rate < self.ANNOTATION_SUCCESS_WARN_THRESHOLD:
            logger.warning(
                "F8 per-iteration summarizer success rate is %.1f%% "
                "(%d/%d) — below %.0f%% threshold; annotations are "
                "effectively disabled this run. Check summarizer model/JSON "
                "parsing.",
                ann_rate * 100, ann_successes, ann_attempts,
                self.ANNOTATION_SUCCESS_WARN_THRESHOLD * 100,
            )

        if (
            total_fired >= self.ACTION_RECOVERY_WARN_MIN_FIRES
            and recovery_rate is not None
            and recovery_rate <= self.ACTION_RECOVERY_WARN_RATE
        ):
            logger.warning(
                "Convergence actions fired %d times but %d recovered "
                "(rate=%.0f%%) — adaptive escalation appears ineffective. "
                "Per-level fires=%s.",
                total_fired, total_recovered,
                (recovery_rate or 0.0) * 100, dict(self._action_fired),
            )

        health_path = self.output_dir / "health.json"
        with open(health_path, "w") as f:
            json.dump(health, f, indent=2, sort_keys=True)
        return health
