"""Evaluator for the custom benchmark example.

The CLI calls `evaluate(program_path)` with the candidate program saved on
disk. Return a dict with `combined_score` for selection, plus optional
per-instance scores and feedback for richer REPS features.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType


TEST_CASES = [-4.0, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0]


def target(x: float) -> float:
    return x * x - 3.0 * x + 2.0


def _load_candidate(program_path: str) -> ModuleType:
    path = Path(program_path).resolve()
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failure(feedback: str) -> dict:
    return {
        "combined_score": 0.0,
        "per_instance_scores": {f"x={x:g}": 0.0 for x in TEST_CASES},
        "feedback": feedback,
    }


def evaluate(program_path: str) -> dict:
    try:
        candidate = _load_candidate(program_path)
        predict = candidate.predict
    except Exception as exc:
        return _failure(f"Program failed to load: {exc}")

    errors: dict[str, float] = {}
    for x in TEST_CASES:
        key = f"x={x:g}"
        try:
            prediction = float(predict(x))
            if not math.isfinite(prediction):
                raise ValueError("prediction was not finite")
            errors[key] = abs(prediction - target(x))
        except Exception as exc:
            return _failure(f"Prediction failed for {key}: {exc}")

    mean_error = sum(errors.values()) / len(errors)
    per_instance_scores = {
        key: 1.0 / (1.0 + error)
        for key, error in errors.items()
    }
    worst_case = max(errors, key=errors.get)

    return {
        "combined_score": 1.0 / (1.0 + mean_error),
        "mean_error": mean_error,
        "per_instance_scores": per_instance_scores,
        "feedback": f"Worst case is {worst_case} with absolute error {errors[worst_case]:.3f}.",
    }
