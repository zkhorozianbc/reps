"""Default preset WorkerConfig entries used by the YAML shim when a legacy
config (no `reps.workers.types` list) is loaded."""
from __future__ import annotations

from reps.workers.base import WorkerConfig


def legacy_default_configs(
    model_id: str,
    exploiter_temperature: float = 0.3,
    explorer_temperature: float = 1.0,
    allocation: dict | None = None,
) -> list[WorkerConfig]:
    alloc = allocation or {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15}
    # Legacy presets own their temperature (no ContractSelector axis override needed).
    return [
        WorkerConfig(
            name="exploiter",
            impl="single_call",
            role="exploiter",
            model_id=model_id,
            temperature=exploiter_temperature,
            generation_mode="diff",
            weight=alloc.get("exploiter", 1.0),
            owns_temperature=True,
        ),
        WorkerConfig(
            name="explorer",
            impl="single_call",
            role="explorer",
            model_id=model_id,
            temperature=explorer_temperature,
            generation_mode="full",
            weight=alloc.get("explorer", 1.0),
            owns_temperature=True,
        ),
        WorkerConfig(
            name="crossover",
            impl="single_call",
            role="crossover",
            model_id=model_id,
            temperature=0.7,
            generation_mode="full",
            weight=alloc.get("crossover", 1.0),
            owns_temperature=True,
        ),
    ]
