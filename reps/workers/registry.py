"""Registry mapping impl names to Worker classes. Each worker module decorates
its class with @register(impl_name) to self-register on import."""
from __future__ import annotations

from typing import Dict, Type

from reps.workers.base import Worker, WorkerConfig

_IMPLS: Dict[str, Type[Worker]] = {}


def register(impl_name: str):
    def deco(cls: Type[Worker]) -> Type[Worker]:
        if impl_name in _IMPLS:
            raise ValueError(f"Worker impl '{impl_name}' already registered")
        _IMPLS[impl_name] = cls
        return cls
    return deco


def build_worker(cfg: WorkerConfig) -> Worker:
    try:
        cls = _IMPLS[cfg.impl]
    except KeyError:
        known = ", ".join(sorted(_IMPLS)) or "(none registered)"
        raise ValueError(
            f"Unknown worker impl '{cfg.impl}' for config '{cfg.name}'. Known: {known}"
        ) from None
    return cls.from_config(cfg)


def known_impls() -> list[str]:
    return sorted(_IMPLS)
