"""`reps.Optimizer` — the v1 optimizer entry point.

Constructor takes a small, deliberately tight set of GEPA-style knobs
(see docs/python_api_spec.md). Internally builds a `Config`, writes the
user's seed code + a dispatch shim to the output directory, and runs the
existing `reps.runner.run_reps` async pipeline.

Power users with needs that exceed the constructor surface use
`Optimizer.from_config(cfg)` to hand in a fully-formed `Config`.

Sync wrapping: `optimize()` calls `asyncio.run` on the internal async
runner. Calling from inside a running event loop raises a clear error —
`aoptimize()` lands in v1.5.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Union

try:
    from typing import Unpack  # Python 3.11+
except ImportError:  # pragma: no cover - we require 3.12+, but be defensive
    from typing_extensions import Unpack  # type: ignore[assignment]

from reps.api.evaluate_dispatch import (
    register_user_evaluate,
    unregister_user_evaluate,
    write_shim,
)
from reps.api.model import Model, ModelKwargs
from reps.api.result import OptimizationResult
from reps.config import Config, LLMConfig, LLMModelConfig
from reps.database import ProgramDatabase

logger = logging.getLogger(__name__)


class Optimizer:
    """The optimizer.

    Construct with a `reps.Model` (or a model-name string) and the
    optimization knobs; call `optimize(initial, evaluate)` to run. Returns
    an `OptimizationResult`.

    See docs/python_api_spec.md, section "v1 surface", for the contract.
    """

    def __init__(
        self,
        *,
        # Model — accept either a built Model or a model-name string. When a
        # string is passed, api_key + ModelKwargs are forwarded to a fresh
        # Model(...). When a Model is passed, those must be set on the Model
        # itself; passing them alongside raises.
        model: Union[Model, str],
        api_key: Optional[str] = None,
        # Search budget
        max_iterations: int = 100,
        # GEPA-style features (Phases 1-5)
        selection_strategy: str = "map_elites",
        pareto_fraction: float = 0.0,
        trace_reflection: bool = False,
        lineage_depth: int = 3,
        merge: bool = False,
        minibatch_size: Optional[int] = None,
        # Population
        num_islands: int = 5,
        # Output
        output_dir: Optional[str] = None,
        # Inline Model construction (only used when `model` is a string).
        # Typed via Unpack[ModelKwargs] so type checkers catch typos —
        # runtime users without type checking should be aware that
        # unrecognized kwargs flow to the SDK constructor.
        **model_kwargs: Unpack[ModelKwargs],
    ) -> None:
        if isinstance(model, str):
            self.model: Model = Model(model, api_key=api_key, **model_kwargs)
        elif isinstance(model, Model):
            if api_key is not None or model_kwargs:
                raise ValueError(
                    "reps.Optimizer: when `model` is a reps.Model instance, "
                    "do not also pass `api_key` or other Model kwargs — set "
                    "those on the Model directly. Got "
                    f"api_key={api_key!r}, model_kwargs={list(model_kwargs)!r}."
                )
            self.model = model
        else:
            raise TypeError(
                f"reps.Optimizer: `model` must be a reps.Model instance or a "
                f"model-name string, got {type(model).__name__}"
            )
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if selection_strategy not in {"map_elites", "pareto", "mixed"}:
            raise ValueError(
                f"selection_strategy must be one of map_elites|pareto|mixed, "
                f"got {selection_strategy!r}"
            )
        if num_islands < 1:
            raise ValueError(f"num_islands must be >= 1, got {num_islands}")

        self.max_iterations = max_iterations
        self.selection_strategy = selection_strategy
        self.pareto_fraction = pareto_fraction
        self.trace_reflection_enabled = trace_reflection
        self.lineage_depth = lineage_depth
        self.merge_enabled = merge
        self.minibatch_size = minibatch_size
        self.num_islands = num_islands
        self.output_dir = output_dir

        self._config: Optional[Config] = None  # built lazily in _build_config

    # ---------------------------------------------------------------- #
    # Public — escape hatch for power users
    # ---------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: Config) -> "Optimizer":
        """Construct a `Optimizer` from a fully-formed internal `Config`.

        Bypasses the simple constructor's kwarg → Config mapping for users
        who need knobs the constructor doesn't expose (population_size,
        explicit worker pools, etc.). The resulting `Optimizer` runs `cfg`
        verbatim — provider/api_key/api_base must be set in `cfg.llm`.
        """
        if not isinstance(cfg, Config):
            raise TypeError(
                f"Optimizer.from_config: expected reps.config.Config, "
                f"got {type(cfg).__name__}"
            )
        # Build a stub Model from the first model in cfg so the optimize()
        # path has something to reference. The Model client itself is never
        # used downstream — runner.run_reps reads cfg.llm directly — so
        # we tolerate a "missing key" by leaving it unset.
        primary = cfg.llm.models[0] if cfg.llm.models else None
        instance = cls.__new__(cls)
        instance.model = _StubModel(primary) if primary else None  # type: ignore[assignment]
        instance.max_iterations = cfg.max_iterations
        instance.selection_strategy = cfg.database.selection_strategy
        instance.pareto_fraction = cfg.database.pareto_fraction
        instance.trace_reflection_enabled = cfg.reps.trace_reflection.enabled
        instance.lineage_depth = cfg.reps.trace_reflection.lineage_depth
        instance.merge_enabled = cfg.reps.merge.enabled
        instance.minibatch_size = getattr(cfg.evaluator, "minibatch_size", None)
        instance.num_islands = cfg.database.num_islands
        instance.output_dir = cfg.output
        instance._config = cfg
        return instance

    # ---------------------------------------------------------------- #
    # Public — entry point
    # ---------------------------------------------------------------- #

    def optimize(
        self,
        initial: str,
        evaluate: Callable[..., Any],
        *,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """Run the evolutionary search and return the best artifact found.

        Args:
            initial: seed program text (NOT a file path).
            evaluate: a `(code: str) -> float | dict | EvaluationResult`
                callable. Optional `env` and `instances` keywords are
                forwarded automatically when present in the signature.
            seed: optional deterministic seed.
        """
        if not isinstance(initial, str):
            raise TypeError(
                f"reps.Optimizer.optimize: `initial` must be the program text "
                f"(str), got {type(initial).__name__}. To load from a "
                f"file, pass `open(path).read()`."
            )
        if not callable(evaluate):
            raise TypeError(
                f"reps.Optimizer.optimize: `evaluate` must be callable, got "
                f"{type(evaluate).__name__}"
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aoptimize_internal(initial, evaluate, seed=seed))
        raise RuntimeError(
            "reps.Optimizer.optimize() cannot be called from an async context. "
            "Use aoptimize() (v1.5) or asyncio.run() yourself on a new loop."
        )

    # ---------------------------------------------------------------- #
    # Internal — async pipeline
    # ---------------------------------------------------------------- #

    async def _aoptimize_internal(
        self,
        initial: str,
        evaluate: Callable[..., Any],
        *,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        # Late import to avoid circular dep at module-import time
        # (runner imports controller imports config imports api.model).
        from reps.runner import run_reps

        # Resolve output directory. None ⇒ tempdir. Persisted runs use the
        # caller's path verbatim (no auto-versioning here — the spec says
        # the CLI does that, not the API).
        cleanup_tempdir: Optional[tempfile.TemporaryDirectory] = None
        if self.output_dir is None:
            cleanup_tempdir = tempfile.TemporaryDirectory(prefix="reps_")
            run_dir = cleanup_tempdir.name
            persisted_output: Optional[str] = None
        else:
            run_dir = str(Path(self.output_dir).resolve())
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            persisted_output = run_dir

        # Write seed code + the per-run dispatch shim to the run dir.
        initial_path = Path(run_dir) / "initial_program.py"
        initial_path.write_text(initial)

        registry_id = register_user_evaluate(evaluate)
        evaluator_path = write_shim(run_dir, registry_id=registry_id)
        cfg = self._build_config(seed=seed)

        try:
            await run_reps(
                config=cfg,
                initial_program=str(initial_path),
                evaluator=evaluator_path,
                output_dir=run_dir,
            )
            return self._collect_result(
                run_dir, cfg=cfg, persisted_output=persisted_output
            )
        finally:
            unregister_user_evaluate(registry_id)
            if cleanup_tempdir is not None:
                cleanup_tempdir.cleanup()

    # ---------------------------------------------------------------- #
    # Config construction
    # ---------------------------------------------------------------- #

    def _build_config(self, *, seed: Optional[int]) -> Config:
        """Build a `Config` from the constructor kwargs.

        When `from_config` was used, a ready-made `Config` is already
        stored on `self._config` — we honor it and only stamp on the
        deterministic seed.
        """
        if self._config is not None:
            cfg = self._config
            if seed is not None:
                cfg.random_seed = seed
                cfg.database.random_seed = seed
            return cfg

        cfg = Config()
        cfg.max_iterations = self.max_iterations
        cfg.database.selection_strategy = self.selection_strategy
        cfg.database.pareto_fraction = self.pareto_fraction
        cfg.database.num_islands = self.num_islands
        cfg.reps.trace_reflection.enabled = self.trace_reflection_enabled
        cfg.reps.trace_reflection.lineage_depth = self.lineage_depth
        cfg.reps.merge.enabled = self.merge_enabled
        cfg.evaluator.minibatch_size = self.minibatch_size

        # Master switch — Phase 3/4 features won't fire without it.
        if self.trace_reflection_enabled or self.merge_enabled:
            cfg.reps.enabled = True

        # LLM: drop the LM into models[0]. Provider routing matches the LM.
        model_cfg = self.model._to_model_config()
        cfg.llm = LLMConfig(
            api_base=model_cfg.api_base,
            api_key=model_cfg.api_key,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_tokens=model_cfg.max_tokens,
            timeout=model_cfg.timeout,
            retries=model_cfg.retries,
            retry_delay=model_cfg.retry_delay,
            reasoning_effort=model_cfg.reasoning_effort,
        )
        # Bypass the LLMConfig.__post_init__ shorthand wiring — set models
        # directly. evaluator_models follow the same single-LM ensemble
        # for v1.
        cfg.llm.models = [model_cfg]
        cfg.llm.evaluator_models = [_clone_model_cfg(model_cfg)]

        # Top-level provider so runner.run_reps's "anthropic" path stamps
        # provider= on each model.
        cfg.provider = "anthropic" if self.model.provider == "anthropic" else "openrouter"

        if cfg.reps.enabled and not cfg.reps.workers.types:
            cfg.reps.workers.types = [_default_api_worker_config(model_cfg.name)]

        # Disable the per-program summarizer by default — it would try to
        # build its own anthropic LLM out of the user's lm and most users
        # haven't asked for it. Power users opt in via from_config.
        cfg.reps.summarizer.enabled = False

        if seed is not None:
            cfg.random_seed = seed
            cfg.database.random_seed = seed

        return cfg

    # ---------------------------------------------------------------- #
    # Result construction
    # ---------------------------------------------------------------- #

    def _collect_result(
        self, run_dir: str, *, cfg: Config, persisted_output: Optional[str]
    ) -> OptimizationResult:
        """Read back the saved DB and assemble an `OptimizationResult`.

        `runner.run_reps` calls `db.save(output_dir)` in its `finally:`
        block, so by the time we reach this method the JSON files are
        on disk. We re-load via `ProgramDatabase.load` to avoid keeping
        a reference to the controller's database.
        """
        db = ProgramDatabase(cfg.database)
        db.load(run_dir)
        best = db.get_best_program()

        if best is None:
            # No programs evaluated — empty result.
            return OptimizationResult(
                best_code="",
                best_score=0.0,
                output_dir=persisted_output,
            )

        score = float(best.metrics.get("combined_score", 0.0))
        # Aggregate token usage from each program's metadata (set by the
        # controller's reps_meta dict). Programs that pre-date Phase A or
        # come from non-LLM paths simply contribute 0.
        tokens_in = 0
        tokens_out = 0
        for prog in db.programs.values():
            md = (prog.metadata or {}).get("reps_meta") or {}
            tokens_in += int(md.get("tokens_in", 0) or 0)
            tokens_out += int(md.get("tokens_out", 0) or 0)

        # iterations_run = max iteration_found across non-seed programs, else 0.
        non_seed_iterations = [
            p.iteration_found for p in db.programs.values()
            if p.iteration_found and p.iteration_found > 0
        ]
        iterations_run = max(non_seed_iterations) if non_seed_iterations else 0

        return OptimizationResult(
            best_code=best.code,
            best_score=score,
            best_metrics=dict(best.metrics) if best.metrics else {},
            best_per_instance_scores=(
                dict(best.per_instance_scores) if best.per_instance_scores else None
            ),
            best_feedback=best.feedback,
            iterations_run=iterations_run,
            total_metric_calls=db.metric_call_count or len(db.programs),
            total_tokens={"in": tokens_in, "out": tokens_out},
            output_dir=persisted_output,
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _clone_model_cfg(src: LLMModelConfig) -> LLMModelConfig:
    """Independent copy used for evaluator_models so post-init mutations
    on one don't bleed into the other."""
    return LLMModelConfig(
        name=src.name,
        api_base=src.api_base,
        api_key=src.api_key,
        weight=src.weight,
        system_message=src.system_message,
        temperature=src.temperature,
        top_p=src.top_p,
        max_tokens=src.max_tokens,
        timeout=src.timeout,
        retries=src.retries,
        retry_delay=src.retry_delay,
        random_seed=src.random_seed,
        reasoning_effort=src.reasoning_effort,
        provider=src.provider,
    )


def _default_api_worker_config(model_id: str):
    """Minimal worker preset for public API REPS knobs.

    This mirrors the controller's non-REPS fallback worker so enabling
    trace reflection or merge does not require users to know the internal
    worker YAML surface.
    """
    from reps.workers.base import WorkerConfig

    return WorkerConfig(
        name="exploiter",
        impl="single_call",
        role="exploiter",
        model_id=model_id,
        temperature=0.7,
        generation_mode="diff",
        weight=1.0,
    )


class _StubModel:
    """Minimal `lm`-like stub used when `from_config` constructs an Optimizer.

    `_aoptimize_internal` doesn't actually need to call the LM directly
    — the controller pulls models out of `Config.llm.models` — but other
    parts of the API (e.g. `__repr__`) still touch `self.model`. The stub
    just exposes the same `provider` / `model` attributes as `reps.LM`
    so introspection works without dragging in real credentials.
    """

    def __init__(self, src: LLMModelConfig) -> None:
        self.model = src.name
        self.provider = src.provider or "unknown"

    def __repr__(self) -> str:
        return f"_StubModel(model={self.model!r}, provider={self.provider!r})"
