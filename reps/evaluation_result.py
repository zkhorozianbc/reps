"""
Evaluation result structures for REPS (extracted from OpenEvolve)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts

    This maintains backward compatibility with the existing dict[str, float] contract
    while adding a side-channel for arbitrary artifacts (text or binary data).

    IMPORTANT: For custom MAP-Elites features, metrics values must be raw continuous
    scores (e.g., actual counts, percentages, continuous measurements), NOT pre-computed
    bin indices. The database handles all binning internally using min-max scaling.

    Examples:
        Correct: {"combined_score": 0.85, "prompt_length": 1247, "execution_time": 0.234}
        Wrong:   {"combined_score": 0.85, "prompt_length": 7, "execution_time": 3}

    Optional fields (Phase 1, GEPA-style ASI):
        per_instance_scores: per-test/per-example scalar scores keyed by instance id.
            Used by the Pareto sampler and trace-grounded reflection. None means the
            evaluator does not expose per-instance breakdown.
        feedback: free-form textual diagnostic intended for the reflection LLM
            (errors, profiler output, intermediate reasoning). None means no feedback
            available.
    """

    metrics: Dict[str, float]  # mandatory - existing contract
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)  # optional side-channel
    per_instance_scores: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "EvaluationResult":
        """Auto-wrap dict returns. Pre-Phase-1.1 callers passed flat dicts where
        every key is a metric. Phase 1.1+ allows top-level `per_instance_scores`
        and `feedback` keys; when present they're peeled out into the
        corresponding ASI fields and excluded from `metrics`. The input dict is
        not mutated (callers may keep using their own copy)."""
        if not isinstance(data, dict):
            return cls(metrics=data)  # let dataclass raise if it's truly bad

        per_instance_scores = data.get("per_instance_scores")
        feedback = data.get("feedback")
        if per_instance_scores is None and feedback is None:
            return cls(metrics=data)

        metrics = {
            k: v for k, v in data.items()
            if k not in ("per_instance_scores", "feedback")
        }
        return cls(
            metrics=metrics,
            per_instance_scores=per_instance_scores,
            feedback=feedback,
        )

    def to_dict(self) -> Dict[str, float]:
        """Backward compatibility - return just metrics"""
        return self.metrics

    def has_artifacts(self) -> bool:
        """Check if this result contains any artifacts"""
        return bool(self.artifacts)

    def get_artifact_keys(self) -> list:
        """Get list of artifact keys"""
        return list(self.artifacts.keys())

    def get_artifact_size(self, key: str) -> int:
        """Get size of a specific artifact in bytes"""
        if key not in self.artifacts:
            return 0

        value = self.artifacts[key]
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        else:
            return len(str(value).encode("utf-8"))

    def get_total_artifact_size(self) -> int:
        """Get total size of all artifacts in bytes"""
        return sum(self.get_artifact_size(key) for key in self.artifacts.keys())
