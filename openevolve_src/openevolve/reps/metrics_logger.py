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

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_csvs()
        self._batch_count = 0
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
                    from openevolve.utils.metrics_utils import safe_numeric_average
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
