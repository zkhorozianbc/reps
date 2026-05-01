"""
Async controller for parallel evolution (asyncio-only; no process fork).

Extended with REPS (Recursive Evolutionary Program Search) features:
- F1: Reflection Engine (post-batch self-reflection)
- F2: ε-Revisitation (revisit underexplored parents)
- F3: Worker Type Diversity (exploiter/explorer/crossover)
- F4: Convergence Monitor (edit entropy + strategy divergence)
- F5: Intelligence Contracts (Thompson-sampling model selection)
- F6: SOTA-Distance Steering
- F7: Compute Signature Tracking
- F8: Enriched Program Annotations
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from reps.config import Config
from reps.database import Program, ProgramDatabase
from reps.utils import safe_numeric_average

if TYPE_CHECKING:
    from reps.iteration_config import IterationConfig
    from reps.workers.base import WorkerConfig

logger = logging.getLogger(__name__)


def _classify_outcome(metrics: Dict[str, Any], child_score: float, parent_score: float) -> str:
    """Produce a short verdict tag for the iteration log.

    Honors evaluators that emit strict_pass/tolerant_pass (a useful pattern for
    distinguishing genuine crashes from strict-vs-tolerant edge cases). Generic
    otherwise.
    """
    if not metrics:
        return "NO_METRICS"
    if metrics.get("error"):
        return "EVAL_ERROR"
    validity = metrics.get("validity")
    strict = metrics.get("strict_pass")
    tolerant = metrics.get("tolerant_pass")
    if validity == 0.0 and tolerant == 1.0 and strict == 0.0:
        return "STRICT_FAIL (tolerant OK)"
    if validity == 0.0:
        return "INVALID"
    if child_score > parent_score:
        return "IMPROVED"
    if child_score < parent_score:
        return "REGRESSED"
    return "UNCHANGED"


@dataclass
class SerializableResult:
    """Result of one async iteration.

    The name stays from the old ProcessPool era for compatibility; nothing
    needs to cross a pickle boundary anymore, but downstream REPS modules
    still read its fields by name.
    """

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None
    target_island: Optional[int] = None  # Island where child should be placed

    # REPS metadata (turns, worker_name, tokens, diff, etc.)
    reps_meta: Optional[Dict[str, Any]] = None

    # Convergence signalling — set when the worker invoked `mark_converged`.
    # The controller short-circuits: no child is persisted, but the event is
    # logged to metrics/convergence_events.jsonl and counts as a completed
    # iteration for budget accounting. child_program_dict will be None.
    converged: bool = False
    converged_reason: Optional[str] = None
    worker_name: Optional[str] = None


# _run_iteration_worker removed — see ProcessParallelController._run_iteration.
# _worker_init / _lazy_init_worker_components removed — components are built
# once in ProcessParallelController.start() and shared by all async tasks.
# _create_database_snapshot removed — tasks read self.database directly.


def _derive_prompt_from_turns(turns) -> Dict[str, str]:
    """Reconstruct {'system': ..., 'user': ...} from the first system/user turns."""
    system_text = ""
    user_text = ""
    for t in turns:
        if t.role == "system" and not system_text:
            system_text = "\n".join(b.text or "" for b in t.blocks if b.type == "text")
        if t.role == "user" and not user_text:
            user_text = "\n".join(b.text or "" for b in t.blocks if b.type == "text")
        if system_text and user_text:
            break
    return {"system": system_text, "user": user_text}


def _derive_llm_response_from_turns(turns) -> str:
    """Concatenate text blocks from the last assistant turn (skip thinking/tool_use)."""
    for t in reversed(turns):
        if t.role == "assistant":
            return "\n".join(b.text or "" for b in t.blocks if b.type == "text")
    return ""


def _turn_to_dict(t) -> Dict[str, Any]:
    """Dataclass-to-dict, preserving signature bytes and all provider_extras."""
    return asdict(t)


def _strip_notebook(program: Optional[Program]) -> Optional[Program]:
    """Return a shallow-copied Program whose metadata lacks
    ``reps_annotations.summary`` — so the sampler's ``_extract_notebook``
    renders to empty and the worker's parent-notebook prepend is skipped.

    The live DB program is NOT mutated — we rebuild the dataclass with a
    fresh metadata dict so the worker sees a sanitized view while other
    callers (summarizer, metrics logging) keep the real summary.
    """
    if program is None:
        return None
    if not program.metadata:
        return program
    ann = program.metadata.get("reps_annotations")
    if not isinstance(ann, dict) or "summary" not in ann:
        # Nothing to strip — avoid an unnecessary copy.
        return program
    new_metadata = dict(program.metadata)
    new_ann = dict(ann)
    new_ann.pop("summary", None)
    new_metadata["reps_annotations"] = new_ann
    # Rebuild via dataclasses.replace so we keep all other Program fields
    # (code, metrics, artifacts, etc.) exactly as they were.
    from dataclasses import replace as _dc_replace
    return _dc_replace(program, metadata=new_metadata)


class ProcessParallelController:
    """Async controller for evolution (single process, single event loop).

    Name kept for backward compatibility with callers (runner.py). Under the
    hood this is asyncio-only: an asyncio.Semaphore bounds concurrency, and
    asyncio.Tasks spawn iterations that share one LLMEnsemble / Evaluator /
    PromptSampler with the controller.

    Extended with REPS features that run at batch boundaries in the event loop.
    """

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        file_suffix: str = ".py",
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.file_suffix = file_suffix
        self.output_dir = output_dir
        self._reps_metrics = None

        # Shared singletons — populated by start()
        self.llm_ensemble = None
        self.evaluator = None
        self.prompt_sampler = None
        self.worker_pool = None

        self._iter_semaphore: Optional[asyncio.Semaphore] = None
        self._shutdown: Optional[asyncio.Event] = None
        self._llm_factory = None
        self._dspy_lm_factory = None

        self.early_stopping_triggered = False
        self._cumulative_tokens = {"in": 0, "out": 0}
        self._warned_about_combined_score = False

        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        # --- REPS initialization ---
        self._reps_enabled = config.reps.enabled
        self._reps_batch_size = config.reps.batch_size
        self._reps_current_reflection: Optional[Dict[str, Any]] = None
        self._reps_epsilon = config.reps.revisitation.epsilon_start
        self._reps_batch_count = 0
        self._reps_batch_results: List = []
        self._reps_rng = random.Random(config.random_seed)

        # --- Plateau tracking (item #2 from the 3-agent analysis) ------
        # Rolling history of best combined_score per completed iteration so
        # _detect_plateau() can tell whether progress has stalled. When it
        # has, _pick_iteration_inputs temporarily tilts the sampler toward
        # exploration for a few iterations (see _plateau_boost_remaining).
        # The original config.database.exploration_ratio /
        # exploitation_ratio are preserved and restored after each sample
        # call so a mid-run checkpoint restart does not inherit bias.
        self._plateau_window: int = 5
        self._plateau_boost_iters: int = 3
        self._plateau_best_history: List[float] = []
        self._plateau_boost_remaining: int = 0
        # --- Convergence-event log (item #1) ---------------------------
        # Populated the first time a WorkerResult with converged=True is
        # handled. File lives under self.output_dir/metrics/.
        self._convergence_events_path: Optional[str] = None

        if self._reps_enabled:
            from reps.worker_pool import WorkerPool
            from reps.convergence_monitor import ConvergenceMonitor
            from reps.contract_selector import ContractSelector
            from reps.sota_controller import SOTAController
            from reps.metrics_logger import MetricsLogger

            reps = config.reps

            primary_model = config.llm.models[0].name if config.llm.models else ""
            self._reps_worker_pool = WorkerPool(
                reps.workers,
                default_model_id=primary_model,
            )
            self._reps_convergence = ConvergenceMonitor(asdict(reps.convergence))
            contracts_cfg = asdict(reps.contracts)
            contracts_cfg["random_seed"] = config.random_seed
            self._reps_contracts = ContractSelector(contracts_cfg)
            self._reps_sota = SOTAController(asdict(reps.sota))

            if self.output_dir:
                self._reps_metrics = MetricsLogger(self.output_dir)
            else:
                logger.info("REPS metrics logging disabled: no output_dir provided")

            self._reps_reflection = None
            self._reps_reflection_config = asdict(reps.reflection)

            # Dedicated summarizer LLM (independent of worker ensemble).
            # Fail LOUDLY now if the summarizer is enabled but its model
            # can't be constructed — better than silently disabling F8
            # annotations for the whole run.
            self._reps_summarizer_llm = None
            summarizer_cfg = getattr(reps, "summarizer", None)
            if summarizer_cfg is not None and summarizer_cfg.enabled:
                from reps.program_summarizer import build_summarizer_llm
                self._reps_summarizer_llm = build_summarizer_llm(summarizer_cfg)
                logger.info(
                    f"Summarizer LLM: model={summarizer_cfg.model_id} "
                    f"provider={self._reps_summarizer_llm.provider}"
                )

            worker_names = list(self._reps_worker_pool._configs.keys())
            logger.info(
                f"REPS enabled: batch_size={self._reps_batch_size}, "
                f"workers={worker_names}, "
                f"ε={reps.revisitation.epsilon_start}"
            )
        else:
            logger.info("REPS features disabled")

        logger.info(f"Initialized async controller with {self.num_workers} workers")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Build shared singletons + concurrency primitives.

        Safe to call outside a running event loop: asyncio.Semaphore and
        asyncio.Event don't require a loop at construction time in 3.10+.
        """
        from reps.evaluator import Evaluator
        from reps.llm.ensemble import LLMEnsemble
        from reps.prompt_sampler import PromptSampler
        from reps.worker_pool import WorkerPool

        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.prompt_sampler = PromptSampler(self.config.prompt)

        evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        evaluator_prompt_sampler.set_templates("evaluator_system_message")
        self.evaluator = Evaluator(
            self.config.evaluator,
            self.evaluation_file,
            LLMEnsemble(self.config.llm.evaluator_models),
            evaluator_prompt_sampler,
            database=self.database,
            suffix=self.file_suffix,
        )

        max_conc = (
            self.config.evaluator.max_concurrent_iterations
            if self.config.evaluator.max_concurrent_iterations is not None
            else self.config.evaluator.parallel_evaluations
        )
        max_conc = max(1, int(max_conc))
        self._iter_semaphore = asyncio.Semaphore(max_conc)
        self._shutdown = asyncio.Event()

        def llm_factory(model_id):
            # Trivial path: hand out the shared ensemble. Specific-model
            # overrides go via WorkerRequest.model_id into generate_with_context.
            return self.llm_ensemble

        self._llm_factory = llm_factory

        def dspy_lm_factory(wc):
            from reps.workers.dspy_react import make_dspy_lm
            return make_dspy_lm(self.config, wc)

        self._dspy_lm_factory = dspy_lm_factory

        # Reuse the existing REPS WorkerPool if enabled; otherwise build a minimal one.
        if self._reps_enabled:
            self.worker_pool = self._reps_worker_pool
        else:
            # Minimal pool so _run_iteration has something to consult.
            from reps.config import REPSWorkersConfig
            from reps.workers.base import WorkerConfig

            primary_model = self.config.llm.models[0].name if self.config.llm.models else ""
            minimal_cfg = REPSWorkersConfig(
                types=[
                    WorkerConfig(
                        name="exploiter",
                        impl="single_call",
                        role="exploiter",
                        model_id=primary_model,
                        temperature=0.7,
                        generation_mode="diff",
                        weight=1.0,
                    )
                ]
            )
            self.worker_pool = WorkerPool(
                minimal_cfg,
                default_model_id=primary_model,
            )

        logger.info(
            "AsyncController started (asyncio-only, max_concurrent_iterations=%d)",
            max_conc,
        )

    def stop(self) -> None:
        """Signal shutdown. Running tasks see self._shutdown.is_set()."""
        if self._shutdown is not None:
            self._shutdown.set()
        logger.info("AsyncController stopping")

    def request_shutdown(self) -> None:
        logger.info("Graceful shutdown requested...")
        self.stop()

    # ------------------------------------------------------------------ #
    # One iteration (async, no process fork)
    # ------------------------------------------------------------------ #

    async def _run_iteration(
        self,
        iteration: int,
        parent_id: str,
        inspiration_ids: List[str],
        iteration_config: "IterationConfig",
    ) -> "SerializableResult":
        """One iteration: build WorkerRequest, dispatch Worker, evaluate child."""
        from reps.workers.base import WorkerContext, WorkerRequest
        from reps.workers.registry import build_worker

        final_child_id = str(uuid.uuid4())

        if parent_id not in self.database.programs:
            return SerializableResult(
                error=f"parent_id {parent_id!r} not in database",
                iteration=iteration,
            )

        parent = self.database.programs[parent_id]
        inspirations = [
            self.database.programs[pid]
            for pid in inspiration_ids
            if pid in self.database.programs
        ]

        # Phase 3.2: per-candidate trace-grounded reflection. Generates + caches
        # mutation_directive on the parent (no-op when disabled or parent lacks
        # ASI signal). Resulting block is injected via prompt_extras below.
        trace_directive_block = await self._maybe_generate_trace_directive(parent)
        if trace_directive_block:
            iteration_config.prompt_extras["trace_directive"] = trace_directive_block

        parent_island = parent.metadata.get("island", self.database.current_island)
        island_pids = list(self.database.islands[parent_island])
        island_programs = [
            self.database.programs[p]
            for p in island_pids
            if p in self.database.programs
        ]
        island_programs.sort(
            key=lambda p: p.metrics.get(
                "combined_score", safe_numeric_average(p.metrics)
            ),
            reverse=True,
        )
        top_k = (
            self.config.prompt.num_top_programs
            + self.config.prompt.num_diverse_programs
        )
        _programs_for_prompt = island_programs[:top_k]
        best_programs_only = island_programs[: self.config.prompt.num_top_programs]

        second_parent = None
        if (
            iteration_config.second_parent_id
            and iteration_config.second_parent_id in self.database.programs
        ):
            second_parent = self.database.programs[iteration_config.second_parent_id]

        # Collect the last 5 completed programs across the whole database
        # (newest-first), irrespective of island. Program.timestamp is set at
        # dataclass construction; fall back to iteration_found if equal.
        all_programs = list(self.database.programs.values())
        all_programs.sort(
            key=lambda p: (getattr(p, "timestamp", 0.0), getattr(p, "iteration_found", 0)),
            reverse=True,
        )
        recent_iterations = all_programs[:5]

        # Load parent's eval artifacts (stdout/stderr) so the worker sees
        # the actual output of the previous run. Fall back to None on any
        # failure — artifacts are best-effort.
        parent_artifacts = None
        try:
            parent_artifacts = self.database.get_artifacts(parent.id) or None
        except Exception as e:
            logger.debug(f"Failed to load parent artifacts for {parent.id[:8]}: {e}")

        # Plateau-mode notebook hide: when the last K=3 iterations have not
        # improved best_score, suppress per-program notebooks from the
        # worker prompt. The siblings block (short-id + one-liner) still
        # renders because it survives summary absence (it just falls back
        # to the program's score). `view_program` remains available so the
        # worker can pull any archive entry on demand.
        hide_archive = (
            self._reps_enabled and self._reps_archive_hidden()
        )
        if hide_archive:
            logger.info(
                "Archive notebooks hidden for iteration %d (plateau detected)",
                iteration,
            )
            parent_for_worker = _strip_notebook(parent)
            inspirations_for_worker = [_strip_notebook(p) for p in inspirations]
            best_programs_for_worker = [_strip_notebook(p) for p in best_programs_only]
            second_parent_for_worker = _strip_notebook(second_parent)
            recent_iters_for_worker = [_strip_notebook(p) for p in recent_iterations]
        else:
            parent_for_worker = parent
            inspirations_for_worker = inspirations
            best_programs_for_worker = best_programs_only
            second_parent_for_worker = second_parent
            recent_iters_for_worker = recent_iterations

        request = WorkerRequest(
            parent=parent_for_worker,
            inspirations=inspirations_for_worker,
            top_programs=best_programs_for_worker,
            second_parent=second_parent_for_worker,
            iteration=iteration,
            language=self.config.language,
            feature_dimensions=self.database.config.feature_dimensions,
            generation_mode=iteration_config.generation_mode,
            prompt_extras=dict(iteration_config.prompt_extras),
            temperature=iteration_config.temperature,
            model_id=iteration_config.model_id,
            recent_iterations=recent_iters_for_worker,
            parent_artifacts=parent_artifacts,
        )

        worker_cfg = self.worker_pool.get_worker_config(iteration_config.worker_name)
        ctx = WorkerContext(
            prompt_sampler=self.prompt_sampler,
            llm_factory=self._llm_factory,
            dspy_lm_factory=self._dspy_lm_factory,
            evaluator=self.evaluator if worker_cfg.uses_evaluator else None,
            scratch_id_factory=lambda: f"scratch-{uuid.uuid4().hex[:12]}",
            final_child_id=final_child_id,
            config=self.config,
            iteration_config=iteration_config,
        )

        worker = build_worker(worker_cfg)

        t_start = time.time()
        try:
            result = await worker.run(request, ctx)
        except Exception as e:
            logger.exception(f"Worker.run raised for iteration {iteration}")
            return SerializableResult(
                error=f"Worker raised: {e}",
                iteration=iteration,
                iteration_time=time.time() - t_start,
            )
        iteration_time = time.time() - t_start

        if result.error is not None:
            return SerializableResult(
                error=str(result.error),
                iteration=iteration,
                iteration_time=iteration_time,
            )

        # Worker signalled mark_converged: no child was produced. Surface the
        # signal up through SerializableResult so the main loop can log the
        # event, skip database.add, and advance the iteration counter.
        if getattr(result, "converged", False):
            return SerializableResult(
                converged=True,
                converged_reason=result.converged_reason,
                worker_name=iteration_config.worker_name,
                parent_id=parent.id,
                iteration=iteration,
                iteration_time=iteration_time,
                target_island=iteration_config.target_island,
                reps_meta={
                    "worker_name": iteration_config.worker_name,
                    "is_revisitation": iteration_config.is_revisitation,
                    "model_id": iteration_config.model_id,
                    "temperature": iteration_config.temperature,
                    "parent_score": (
                        parent.metrics.get(
                            "combined_score", safe_numeric_average(parent.metrics)
                        )
                        if parent.metrics
                        else 0.0
                    ),
                    "tokens_in": result.usage.get(
                        "prompt_tokens", result.usage.get("input_tokens", 0)
                    ),
                    "tokens_out": result.usage.get(
                        "completion_tokens", result.usage.get("output_tokens", 0)
                    ),
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "converged": True,
                    "converged_reason": result.converged_reason,
                },
            )

        if len(result.child_code) > self.config.max_code_length:
            return SerializableResult(
                error=(
                    f"Generated code exceeds maximum length "
                    f"({len(result.child_code)} > {self.config.max_code_length})"
                ),
                iteration=iteration,
                iteration_time=iteration_time,
            )

        # Final (scored) evaluation — artifacts of this call land in the DB.
        try:
            outcome = await self.evaluator.evaluate_isolated(
                result.child_code, program_id=final_child_id
            )
        except Exception as e:
            logger.exception(f"Evaluator failed for iteration {iteration}")
            return SerializableResult(
                error=f"Evaluator failed: {e}",
                iteration=iteration,
                iteration_time=iteration_time,
            )

        child_metadata = {
            "changes": result.changes_summary or "",
            "parent_metrics": parent.metrics,
            "island": parent_island,
            "reps_worker_name": iteration_config.worker_name,
            "reps_is_revisitation": iteration_config.is_revisitation,
        }
        if iteration_config.model_id:
            child_metadata["reps_model_id"] = iteration_config.model_id
        child_metadata["turns"] = [_turn_to_dict(t) for t in result.turns]

        parent_score = (
            parent.metrics.get("combined_score", safe_numeric_average(parent.metrics))
            if parent.metrics
            else 0.0
        )
        child_score = outcome.metrics.get(
            "combined_score", safe_numeric_average(outcome.metrics)
        )

        # Per-iteration summarization: run BEFORE the child enters the DB so
        # descendants that start concurrently see the summary. Uses a
        # dedicated summarizer LLM (built at controller startup) — NOT the
        # worker ensemble — so the summarizer model is independent of what
        # workers use. Best-effort: failures don't block the child.
        summarizer_cfg = getattr(self.config.reps, "summarizer", None)
        if (
            self._reps_enabled
            and self._reps_summarizer_llm is not None
            and (summarizer_cfg is None or summarizer_cfg.enabled)
        ):
            try:
                from reps.program_summarizer import summarize_program

                turns_dicts = [_turn_to_dict(t) for t in result.turns]
                summary = await summarize_program(
                    program_id=final_child_id,
                    code=result.child_code,
                    turns=turns_dicts,
                    parent_score=parent_score,
                    child_score=child_score,
                    improved=child_score > parent_score,
                    summarizer_llm=self._reps_summarizer_llm,
                    task_instructions=(
                        summarizer_cfg.task_instructions if summarizer_cfg else None
                    ),
                )
                if summary:
                    ann = child_metadata.setdefault("reps_annotations", {})
                    ann["summary"] = summary
            except Exception as e:
                logger.warning(
                    "per-iteration summarizer failed for %s: %s", final_child_id[:8], e
                )

        child = Program(
            id=final_child_id,
            code=result.child_code,
            changes_description=result.changes_description,
            language=self.config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=outcome.metrics,
            per_instance_scores=outcome.per_instance_scores,
            feedback=outcome.feedback,
            iteration_found=iteration,
            metadata=child_metadata,
        )

        prompt_dict = _derive_prompt_from_turns(result.turns)
        llm_response_str = _derive_llm_response_from_turns(result.turns)

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt_dict,
            llm_response=llm_response_str,
            artifacts=outcome.artifacts,
            iteration=iteration,
            target_island=iteration_config.target_island,
            reps_meta={
                "worker_name": iteration_config.worker_name,
                "is_revisitation": iteration_config.is_revisitation,
                "model_id": iteration_config.model_id,
                "temperature": iteration_config.temperature,
                "parent_score": parent_score,
                "diff": result.applied_edit,
                "turns": [_turn_to_dict(t) for t in result.turns],
                "tokens_in": result.usage.get(
                    "prompt_tokens", result.usage.get("input_tokens", 0)
                ),
                "tokens_out": result.usage.get(
                    "completion_tokens", result.usage.get("output_tokens", 0)
                ),
                "wall_clock_seconds": result.wall_clock_seconds,
            },
        )

    async def _spawn_iteration(
        self,
        iteration: int,
        parent_id: str,
        inspiration_ids: List[str],
        iteration_config: "IterationConfig",
    ) -> "SerializableResult":
        """Semaphore-bounded wrapper that shields siblings from one failure."""
        assert self._iter_semaphore is not None, "start() must be called first"
        async with self._iter_semaphore:
            try:
                return await self._run_iteration(
                    iteration=iteration,
                    parent_id=parent_id,
                    inspiration_ids=inspiration_ids,
                    iteration_config=iteration_config,
                )
            except Exception as e:
                logger.exception(f"iteration {iteration} failed in spawn")
                return SerializableResult(error=str(e), iteration=iteration)

    async def _maybe_generate_trace_directive(self, parent: Program) -> str:
        """Per-candidate trace-grounded reflection (Phase 3.2).

        Returns prompt-ready text (header + directive) when:
          - reps.trace_reflection.enabled is True
          - the parent qualifies (per `should_generate_directive`)
          - the LLM call returned a non-empty directive

        Otherwise returns "". Caches the directive on `parent.mutation_directive`
        so a re-sampled parent doesn't pay for the LLM call again.

        Best-effort: any failure returns "" — the iteration proceeds with the
        parent's existing prompt unchanged.
        """
        if not self._reps_enabled:
            return ""
        cfg = self.config.reps.trace_reflection
        if not cfg.enabled:
            return ""

        from reps.trace_reflection import generate_directive, should_generate_directive

        if not should_generate_directive(parent, min_feedback_length=cfg.min_feedback_length):
            return ""

        try:
            directive = await generate_directive(
                parent,
                self.llm_ensemble,
                min_feedback_length=cfg.min_feedback_length,
                max_code_chars=cfg.max_code_chars,
            )
        except Exception as e:
            logger.warning(
                "trace_reflection: unexpected error for parent %s: %s",
                parent.id[:8] if parent.id else "?", e,
            )
            return ""

        if not directive:
            return ""

        # Cache on the live Program so a re-sampled parent doesn't repeat the
        # call. The new directive is also written to the child's metadata
        # downstream so it's persisted with the program.
        parent.mutation_directive = directive

        return (
            "## Suggested next change (from trace reflection)\n"
            f"{directive}"
        )

    def _should_sample_pareto(self) -> bool:
        """Decide whether the next parent pick should come from the Pareto
        frontier rather than MAP-Elites/exploration sampling.

        - "map_elites" (default): always False — status quo behavior.
        - "pareto":               always True.
        - "mixed":                True with probability `pareto_fraction`.
        """
        strategy = self.database.config.selection_strategy
        if strategy == "pareto":
            return True
        if strategy == "mixed":
            frac = max(0.0, min(1.0, self.database.config.pareto_fraction))
            return random.random() < frac
        return False

    def _pick_iteration_inputs(
        self, iteration: int, island_id: int, reps_iter_config: Optional["IterationConfig"]
    ):
        """Sample a parent + inspirations for one dispatch.

        Returns (parent_id, inspiration_ids, iteration_config_with_target_island).
        Raises on DB errors (caller logs + skips).
        """
        from reps.iteration_config import IterationConfig

        target_island = island_id if island_id is not None else self.database.current_island

        if reps_iter_config is None:
            reps_iter_config = IterationConfig(target_island=target_island)
        else:
            if reps_iter_config.target_island is None:
                reps_iter_config.target_island = target_island

        forced_parent_id = reps_iter_config.parent_id if reps_iter_config else None

        # Plateau rebalance (item #2): when best-score has stalled for
        # `_plateau_window` iters, temporarily tilt sampling toward
        # exploration for `_plateau_boost_iters` sample calls. We mutate
        # self.database.config in-place JUST for the sample_from_island call
        # and restore afterwards so a checkpoint save does not persist the
        # bias and a restart does not inherit it.
        boosting = False
        orig_exploration = None
        orig_exploitation = None
        if self._plateau_boost_remaining > 0:
            boosting = True
            orig_exploration = self.database.config.exploration_ratio
            orig_exploitation = self.database.config.exploitation_ratio
            self.database.config.exploration_ratio = min(
                1.0, orig_exploration + 0.2
            )
            self.database.config.exploitation_ratio = max(
                0.0, orig_exploitation - 0.2
            )
            self._plateau_boost_remaining -= 1

        try:
            if forced_parent_id and forced_parent_id in self.database.programs:
                parent = self.database.programs[forced_parent_id]
                _, inspirations = self.database.sample_from_island(
                    island_id=target_island,
                    num_inspirations=self.config.prompt.num_top_programs,
                )
            elif self._should_sample_pareto():
                parent, inspirations = self.database.sample_pareto_from_island(
                    island_id=target_island,
                    num_inspirations=self.config.prompt.num_top_programs,
                    instance_keys=self.database.config.pareto_instance_keys,
                )
            else:
                parent, inspirations = self.database.sample_from_island(
                    island_id=target_island,
                    num_inspirations=self.config.prompt.num_top_programs,
                )
        finally:
            if boosting:
                self.database.config.exploration_ratio = orig_exploration
                self.database.config.exploitation_ratio = orig_exploitation

        return parent.id, [p.id for p in inspirations], reps_iter_config

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution on the current event loop.

        Concurrency is bounded by self._iter_semaphore. Each iteration runs
        as an asyncio.Task via _spawn_iteration. Completed tasks are drained
        with asyncio.wait(..., FIRST_COMPLETED). REPS batch processing runs
        at batch boundaries with self.database directly (no snapshot).
        """
        from reps.iteration_config import IterationConfig

        if self.evaluator is None or self.llm_ensemble is None:
            raise RuntimeError("Controller not started — call .start() first")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting async evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        pending: Dict[int, asyncio.Task] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        reps_prompt_extras = (
            self._reps_build_prompt_extras() if self._reps_enabled else {}
        )
        if reps_prompt_extras:
            logger.info(
                f"REPS prompt extras keys: {list(reps_prompt_extras.keys())}"
            )

        def _build_iter_config(island_id: int) -> "IterationConfig":
            if self._reps_enabled:
                cfg = self._reps_build_iteration_config(
                    island_id, reps_prompt_extras, max_iterations
                )
                if cfg is not None:
                    return cfg
            return IterationConfig(target_island=island_id)

        def _spawn(iteration: int, island_id: int) -> Optional[asyncio.Task]:
            try:
                iteration_config = _build_iter_config(island_id)
                parent_id, inspiration_ids, iteration_config = self._pick_iteration_inputs(
                    iteration, island_id, iteration_config
                )
            except Exception as e:
                logger.error(f"Error preparing iteration {iteration}: {e}")
                return None
            return asyncio.create_task(
                self._spawn_iteration(
                    iteration=iteration,
                    parent_id=parent_id,
                    inspiration_ids=inspiration_ids,
                    iteration_config=iteration_config,
                )
            )

        # Initial batch — round-robin across islands.
        batch_per_island = (
            max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        )
        current_iteration = start_iteration
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    task = _spawn(current_iteration, island_id)
                    if task is not None:
                        pending[current_iteration] = task
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0
        reps_batch_accumulator: List = []

        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            if self.config.early_stopping_patience < 0:
                logger.info(
                    f"Early stopping patience is set to a negative value, running event-based early-stopping, "
                    f"Early stop when metric '{self.config.early_stopping_metric}' reaches {self.config.convergence_threshold}"
                )
            else:
                logger.info(
                    f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                    f"threshold={self.config.convergence_threshold}, "
                    f"metric={self.config.early_stopping_metric}"
                )
        else:
            logger.info("Early stopping disabled")

        # Iteration drain loop.
        while (
            pending
            and completed_iterations < max_iterations
            and not self._shutdown.is_set()
        ):
            timeout_seconds = self.config.evaluator.timeout + 30
            done_tasks, _pending_set = await asyncio.wait(
                list(pending.values()),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout_seconds,
            )

            if not done_tasks:
                # Timeout elapsed with nothing completing — log once and continue.
                logger.warning(
                    f"No iteration completed within {timeout_seconds}s; continuing to wait."
                )
                continue

            for task in done_tasks:
                # Map task → iteration key.
                completed_iteration = None
                for it_key, t in list(pending.items()):
                    if t is task:
                        completed_iteration = it_key
                        break
                if completed_iteration is None:
                    continue

                del pending[completed_iteration]

                result: Optional[SerializableResult] = None
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    logger.info(
                        f"Iteration {completed_iteration} cancelled; continuing."
                    )
                    completed_iterations += 1
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing result from iteration {completed_iteration}: {e}"
                    )
                    completed_iterations += 1
                    continue

                try:
                    if result is None:
                        pass
                    elif result.converged:
                        # Worker called mark_converged: persist a telemetry row
                        # and treat the iteration as successfully completed
                        # (for budget accounting) without adding a child to
                        # the database. Summarization is skipped — there is
                        # nothing to summarize.
                        self._record_convergence_event(
                            iteration=completed_iteration,
                            parent_id=result.parent_id,
                            reason=result.converged_reason,
                            worker_name=result.worker_name,
                        )
                        logger.info(
                            "Iteration %d: worker marked converged (reason: %s) "
                            "— no child added",
                            completed_iteration,
                            result.converged_reason or "(unspecified)",
                        )
                    elif result.error:
                        logger.warning(
                            f"Iteration {completed_iteration} error: {result.error}"
                        )
                    elif result.child_program_dict:
                        child_program = Program(**result.child_program_dict)

                        self.database.add(
                            child_program,
                            iteration=completed_iteration,
                            target_island=result.target_island,
                        )

                        if result.artifacts:
                            self.database.store_artifacts(
                                child_program.id, result.artifacts
                            )

                        # Persist this child's JSON + trace sidecar immediately
                        # so a killed or crashed run doesn't lose completed work.
                        # Checkpoint / final save_database still happen; this
                        # just ensures no per-iter state is in-memory-only.
                        try:
                            self.database._save_program(
                                child_program, base_path=self.output_dir
                            )
                        except Exception as persist_exc:
                            logger.warning(
                                "per-iteration persist failed for %s: %s",
                                child_program.id[:8], persist_exc,
                            )

                        if result.prompt:
                            self.database.log_prompt(
                                template_key=(
                                    "full_rewrite_user"
                                    if not self.config.diff_based_evolution
                                    else "diff_user"
                                ),
                                program_id=child_program.id,
                                prompt=result.prompt,
                                responses=(
                                    [result.llm_response] if result.llm_response else []
                                ),
                            )

                        island_id = child_program.metadata.get(
                            "island", self.database.current_island
                        )
                        self.database.increment_island_generation(island_idx=island_id)

                        if self.database.should_migrate():
                            logger.info(
                                f"Performing migration at iteration {completed_iteration}"
                            )
                            self.database.migrate_programs()
                            self.database.log_island_status()

                        logger.info(
                            f"Iteration {completed_iteration}: "
                            f"Program {child_program.id} "
                            f"(parent: {result.parent_id}) "
                            f"completed in {result.iteration_time:.2f}s"
                        )

                        if child_program.metrics:
                            metrics_str = ", ".join(
                                [
                                    f"{k}={v}"
                                    for k, v in child_program.metrics.items()
                                ]
                            )
                            logger.info(f"Metrics: {metrics_str}")

                        if result.reps_meta:
                            t_in = result.reps_meta.get("tokens_in", 0)
                            t_out = result.reps_meta.get("tokens_out", 0)
                            if t_in or t_out:
                                self._cumulative_tokens["in"] += t_in
                                self._cumulative_tokens["out"] += t_out
                                logger.info(
                                    f"Tokens: in={t_in}, out={t_out}, "
                                    f"cumulative_in={self._cumulative_tokens['in']}, "
                                    f"cumulative_out={self._cumulative_tokens['out']}"
                                )

                            if (
                                child_program.metrics
                                and "combined_score" not in child_program.metrics
                                and not self._warned_about_combined_score
                            ):
                                avg_score = safe_numeric_average(
                                    child_program.metrics
                                )
                                logger.warning(
                                    f"No 'combined_score' metric found in evaluation results. "
                                    f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                    f"Return a 'combined_score' from your evaluator for better results."
                                )
                                self._warned_about_combined_score = True

                        if self.database.best_program_id == child_program.id:
                            logger.info(
                                f"New best at iteration {completed_iteration}: {child_program.id}"
                            )

                        if (
                            completed_iteration > 0
                            and completed_iteration % self.config.checkpoint_interval == 0
                        ):
                            logger.info(
                                f"Checkpoint interval reached at iteration {completed_iteration}"
                            )
                            self.database.log_island_status()
                            if checkpoint_callback:
                                checkpoint_callback(completed_iteration)

                        if target_score is not None and child_program.metrics:
                            if (
                                "combined_score" in child_program.metrics
                                and child_program.metrics["combined_score"]
                                >= target_score
                            ):
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                break

                        if early_stopping_enabled and child_program.metrics:
                            current_score = None
                            if (
                                self.config.early_stopping_metric
                                in child_program.metrics
                            ):
                                current_score = child_program.metrics[
                                    self.config.early_stopping_metric
                                ]
                            elif self.config.early_stopping_metric == "combined_score":
                                current_score = safe_numeric_average(
                                    child_program.metrics
                                )
                            else:
                                logger.warning(
                                    f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                                )
                                current_score = safe_numeric_average(
                                    child_program.metrics
                                )

                            if current_score is not None and isinstance(
                                current_score, (int, float)
                            ):
                                if self.config.early_stopping_patience > 0:
                                    improvement = current_score - best_score
                                    if (
                                        improvement
                                        >= self.config.convergence_threshold
                                    ):
                                        best_score = current_score
                                        iterations_without_improvement = 0
                                        logger.debug(
                                            f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                        )
                                    else:
                                        iterations_without_improvement += 1
                                        logger.debug(
                                            f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                        )

                                    if (
                                        iterations_without_improvement
                                        >= self.config.early_stopping_patience
                                    ):
                                        self.early_stopping_triggered = True
                                        logger.info(
                                            f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                            f"No improvement for {iterations_without_improvement} iterations "
                                            f"(best score: {best_score:.4f})"
                                        )
                                        break
                                else:
                                    if (
                                        current_score
                                        == self.config.convergence_threshold
                                    ):
                                        best_score = current_score
                                        logger.info(
                                            f"🛑 Early stopping (event-based) triggered at iteration {completed_iteration}: "
                                            f"Task successfully solved with score {best_score:.4f}."
                                        )
                                        self.early_stopping_triggered = True
                                        break
                except Exception as e:
                    logger.error(
                        f"Error processing result from iteration {completed_iteration}: {e}"
                    )

                completed_iterations += 1

                # Plateau tracking (item #2): append the current best
                # combined_score to the rolling history and, if the window
                # is full and no progress was made, arm the exploration
                # boost for the next few sample calls. Armed state is
                # cleared each time a new best arrives.
                self._update_plateau_history()
                if (
                    self._plateau_boost_remaining == 0
                    and self._detect_plateau()
                ):
                    self._plateau_boost_remaining = self._plateau_boost_iters
                    logger.info(
                        "Plateau detected at iteration %d (best has not "
                        "improved across last %d iterations) — boosting "
                        "exploration for the next %d sample calls",
                        completed_iteration,
                        self._plateau_window,
                        self._plateau_boost_iters,
                    )

                if self._reps_enabled and result is not None:
                    reps_batch_accumulator.append(result)
                    if len(reps_batch_accumulator) >= self._reps_batch_size:
                        await self._reps_process_batch(reps_batch_accumulator)
                        self._reps_update_epsilon(
                            completed_iterations, max_iterations
                        )
                        reps_prompt_extras = self._reps_build_prompt_extras()
                        reps_batch_accumulator = []

                # Remove from island-pending tracking.
                for island_id, iteration_list in island_pending.items():
                    if completed_iteration in iteration_list:
                        iteration_list.remove(completed_iteration)
                        break

                # Dispatch one new iteration on this island if room remains.
                for island_id in range(self.num_islands):
                    if (
                        len(island_pending[island_id]) < batch_per_island
                        and next_iteration < total_iterations
                        and not self._shutdown.is_set()
                    ):
                        task = _spawn(next_iteration, island_id)
                        if task is not None:
                            pending[next_iteration] = task
                            island_pending[island_id].append(next_iteration)
                            next_iteration += 1
                            break

            # End of done_tasks loop; check for early-stopping flags from inside.
            if self.early_stopping_triggered:
                break
            if target_score is not None:
                best = self.database.get_best_program()
                if (
                    best
                    and best.metrics
                    and "combined_score" in best.metrics
                    and best.metrics["combined_score"] >= target_score
                ):
                    break

        # Final REPS batch flush.
        if self._reps_enabled and reps_batch_accumulator:
            await self._reps_process_batch(reps_batch_accumulator)
            reps_batch_accumulator = []

        # Shutdown drain: cancel any still-pending tasks.
        if pending:
            for t in pending.values():
                t.cancel()
            await asyncio.gather(*pending.values(), return_exceptions=True)
            pending.clear()

        if self.early_stopping_triggered:
            logger.info("Evolution completed: early stopping triggered by convergence")
        elif self._shutdown.is_set():
            logger.info("Evolution completed: shutdown requested")
        else:
            logger.info("Evolution completed: maximum iterations reached")

        return self.database.get_best_program()

    # ------------------------------------------------------------------ #
    # REPS controller-side methods (unchanged plumbing, snapshot-free)
    # ------------------------------------------------------------------ #

    def _reps_init_reflection_engine(self, llm_ensemble):
        """Lazily initialize the reflection engine with an LLM ensemble."""
        if self._reps_reflection is None and self._reps_enabled:
            from reps.reflection_engine import ReflectionEngine
            self._reps_reflection = ReflectionEngine(
                llm_ensemble, self._reps_reflection_config
            )

    def _reps_get_sota_score(self, program: Optional[Program]) -> Optional[float]:
        """Return the metric F6 should compare against the configured target score.

        Prefer the configured target_metric when available, since `combined_score`
        may be a normalized surrogate used for selection rather than the metric the
        target is expressed in.
        """
        if program is None or not program.metrics:
            return None

        metrics = program.metrics
        target_metric = self.config.reps.sota.target_metric
        if target_metric:
            raw = metrics.get(target_metric)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)

        combined_score = metrics.get("combined_score")
        if isinstance(combined_score, (int, float)) and not isinstance(combined_score, bool):
            return float(combined_score)

        return safe_numeric_average(metrics)

    def _reps_build_prompt_extras(self) -> Dict[str, str]:
        """Build the prompt extras dict from all REPS modules."""
        extras = {}

        if not self._reps_enabled:
            return extras

        # F1: Reflection
        if self._reps_reflection and self._reps_current_reflection:
            extras["reflection"] = self._reps_reflection.format_for_prompt(
                self._reps_current_reflection
            )

        # F6: SOTA injection
        if self._reps_sota.enabled and self._reps_sota.target is not None:
            best = self.database.get_best_program()
            best_score = self._reps_get_sota_score(best)
            if best_score is not None:
                self._reps_sota.get_regime(best_score)
                extras["sota_injection"] = self._reps_sota.format_for_prompt()

        # F8: Dead-end warnings
        extras["dead_end_warnings"] = self._reps_build_dead_end_warnings()

        return extras

    # ------------------------------------------------------------------ #
    # Helpers introduced by the 4-team parallel fix (items #1, #2, #3, #4)
    # ------------------------------------------------------------------ #

    def _record_convergence_event(
        self,
        *,
        iteration: int,
        parent_id: Optional[str],
        reason: Optional[str],
        worker_name: Optional[str],
    ) -> None:
        """Append a single convergence event to
        ``<output_dir>/metrics/convergence_events.jsonl``.

        Best-effort: any I/O failure is logged as a warning but does not
        propagate, because the controller must continue advancing the
        iteration counter (a dropped telemetry row is far less bad than a
        dropped budget tick).
        """
        import json
        import os

        try:
            if self._convergence_events_path is None:
                metrics_dir = os.path.join(self.output_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                self._convergence_events_path = os.path.join(
                    metrics_dir, "convergence_events.jsonl"
                )
            row = {
                "iteration": iteration,
                "parent_id": parent_id,
                "reason": reason,
                "timestamp": time.time(),
                "worker_type": worker_name,
            }
            with open(self._convergence_events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            logger.warning(
                "Failed to persist convergence event for iteration %d: %s",
                iteration,
                e,
            )

    def _update_plateau_history(self) -> None:
        """Append the current best combined_score to the plateau-history
        ring. Called once per completed iteration (after the child, if any,
        has been added to the database).
        """
        best = self.database.get_best_program()
        if best is None or not best.metrics:
            score = float("-inf")
        else:
            score = best.metrics.get(
                "combined_score", safe_numeric_average(best.metrics)
            )
            if not isinstance(score, (int, float)):
                score = float("-inf")
        self._plateau_best_history.append(float(score))
        # Keep it bounded; we only ever inspect the last plateau_window+1.
        keep = self._plateau_window + 1
        if len(self._plateau_best_history) > keep:
            self._plateau_best_history = self._plateau_best_history[-keep:]

    def _detect_plateau(self) -> bool:
        """Return True when the best combined_score has not strictly
        improved across the last `self._plateau_window` completed
        iterations. We need at least window+1 history entries — the
        reference score (before the window) plus `window` samples in the
        window. A constant score qualifies as a plateau (no strict
        improvement).
        """
        window = self._plateau_window
        if window <= 0:
            return False
        if len(self._plateau_best_history) < window + 1:
            return False
        reference = self._plateau_best_history[-(window + 1)]
        recent = self._plateau_best_history[-window:]
        # Plateau <=> no strictly-greater value in `recent` vs. reference.
        return all(r <= reference for r in recent)

    def _reps_archive_hidden(self) -> bool:
        """Return True when the last K=3 completed iterations' best_score
        has not improved, i.e. the archive has stalled and its per-program
        notebooks should be hidden from the worker prompt.

        Uses ``self._plateau_best_history`` (populated after every
        completed iteration by ``_update_plateau_history``). Independent
        of the F4 ``_detect_plateau``/``_plateau_boost_remaining`` pair —
        those drive exploration-ratio tilts at the sampler layer; this
        only gates notebook visibility in the worker prompt.

        Contract:
          * ``len(history) < K`` → False (not enough data to stall).
          * Else True iff ``max(history[-K:]) <= max(history[:-K]) + eps``
            with a small floating-point eps tolerance.
        """
        K = 3
        history = self._plateau_best_history
        if len(history) < K:
            return False
        # Split into pre-window reference vs. the window itself.
        pre = history[:-K]
        window = history[-K:]
        if not pre:
            return False
        eps = 1e-9
        return max(window) <= max(pre) + eps

    def _reps_build_dead_end_warnings(self) -> str:
        """Build dead-end warning text from annotated programs."""
        if not self._reps_enabled or not self.config.reps.annotations.dead_end_awareness:
            return ""

        dead_ends = []
        for pid, prog in self.database.programs.items():
            ann = prog.metadata.get("reps_annotations", {})
            if ann.get("dead_end"):
                hypothesis = ann.get("hypothesis", "unknown approach")
                outcome = ann.get("outcome", "failed")
                dead_ends.append(f"- {hypothesis}: {outcome}")

        if dead_ends:
            return (
                "KNOWN DEAD ENDS (do not repeat these approaches):\n"
                + "\n".join(dead_ends[:10])
            )
        return ""

    def _reps_build_iteration_config(
        self,
        island_id: int,
        prompt_extras: Dict[str, str],
        max_iterations: int,
    ) -> Optional["IterationConfig"]:
        """Build a REPS IterationConfig object for one iteration dispatch.

        Returns the dataclass directly now (no more asdict() across process
        boundaries). Returns None when REPS is disabled; caller falls back to
        a default IterationConfig.
        """
        if not self._reps_enabled:
            return None

        # F2: ε-Revisitation check
        if (
            self._reps_rng.random() < self._reps_epsilon
            and self.config.reps.revisitation.enabled
        ):
            target = self._reps_select_revisitation_target()
            if target is not None:
                alt_name = self._reps_worker_pool.get_alternative_worker_name(
                    target.metadata.get("reps_worker_name",
                        target.metadata.get("reps_worker_type", "exploiter"))
                )
                config = self._reps_worker_pool.build_iteration_config(
                    self.database,
                    prompt_extras,
                    override_name=alt_name,
                    target_island=island_id,
                )
                config.parent_id = target.id
                config.is_revisitation = True
                return config

        config = self._reps_worker_pool.build_iteration_config(
            self.database,
            prompt_extras,
            target_island=island_id,
        )

        # F5: Intelligence Contracts override — respect axis ownership
        if self._reps_contracts.enabled:
            contract = self._reps_contracts.select()
            if contract:
                worker_cfg = self._reps_worker_pool.get_worker_config(config.worker_name)
                if contract.model_id and not worker_cfg.owns_model:
                    config.model_id = contract.model_id
                if contract.temperature is not None and not worker_cfg.owns_temperature:
                    config.temperature = contract.temperature

        return config

    def _reps_select_revisitation_target(self) -> Optional[Program]:
        """Select a program with high score but low exploration for revisitation."""
        candidates = []
        for pid, prog in self.database.programs.items():
            if not prog.metrics:
                continue
            score = prog.metrics.get(
                "combined_score", safe_numeric_average(prog.metrics)
            )
            num_descendants = sum(
                1 for p in self.database.programs.values() if p.parent_id == pid
            )
            age = self._reps_batch_count - prog.metadata.get("reps_batch_found", 0)
            recency_window = self.config.reps.revisitation.recency_window
            recency_bonus = 1.5 if age < recency_window else 1.0

            priority = score * (1.0 / (1.0 + num_descendants)) * recency_bonus
            candidates.append((priority, prog))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _reps_update_epsilon(self, current_iteration: int, max_iterations: int):
        """Update ε with decay schedule."""
        if not self._reps_enabled or not self.config.reps.revisitation.enabled:
            return
        cfg = self.config.reps.revisitation
        progress = current_iteration / max(1, max_iterations)
        self._reps_epsilon = max(
            cfg.epsilon_end,
            cfg.epsilon_start * (1.0 - progress),
        )

    async def _reps_process_batch(self, batch_results: List):
        """Run all REPS controller-side modules at a batch boundary."""
        if not self._reps_enabled or not batch_results:
            return

        self._reps_batch_count += 1

        from reps.iteration_config import IterationResult

        reps_results = []
        for r in batch_results:
            meta = r.reps_meta or {}
            child_score = 0.0
            parent_score = meta.get("parent_score", 0.0)
            if r.child_program_dict and r.child_program_dict.get("metrics"):
                child_score = r.child_program_dict["metrics"].get(
                    "combined_score",
                    safe_numeric_average(r.child_program_dict["metrics"]),
                )
            improved = child_score > parent_score if r.error is None else False

            rr = IterationResult(
                child_program_dict=r.child_program_dict,
                parent_id=r.parent_id,
                iteration=r.iteration,
                error=r.error,
                target_island=r.target_island,
                prompt=r.prompt,
                llm_response=r.llm_response,
                artifacts=r.artifacts,
                diff=meta.get("diff", ""),
                worker_name=meta.get("worker_name", meta.get("worker_type", "exploiter")),
                is_revisitation=meta.get("is_revisitation", False),
                model_id=meta.get("model_id"),
                temperature=meta.get("temperature"),
                parent_score=parent_score,
                child_score=child_score,
                improved=improved,
                turns=meta.get("turns", []),
                iteration_time=r.iteration_time,
                tokens_in=meta.get("tokens_in", 0),
                tokens_out=meta.get("tokens_out", 0),
                wall_clock_seconds=meta.get("wall_clock_seconds", r.iteration_time),
            )
            reps_results.append(rr)

            self._reps_worker_pool.record_result(rr.worker_name, rr.improved)

        # F4: Convergence Monitor
        from reps.convergence_monitor import ConvergenceAction
        action = self._reps_convergence.update(reps_results)

        if action == ConvergenceAction.MILD_BOOST:
            self._reps_worker_pool.boost_explorer(0.2)
            self._reps_worker_pool.bump_temperatures(0.1)
        elif action == ConvergenceAction.MODERATE_DIVERSIFY:
            self._reps_worker_pool.force_explorer_majority(5)
        elif action == ConvergenceAction.SEVERE_RESTART:
            self._reps_epsilon = min(0.5, self._reps_epsilon * 2)

        # F6: SOTA-driven worker reallocation
        if self._reps_sota.enabled and self._reps_sota.target is not None:
            best = self.database.get_best_program()
            best_score = self._reps_get_sota_score(best)
            if best_score is not None:
                regime = self._reps_sota.get_regime(best_score)
                new_alloc = self._reps_sota.modulate_worker_allocation(regime)
                self._reps_worker_pool.set_allocation(new_alloc)

        # F5: Update contract posteriors
        if self._reps_contracts.enabled:
            for rr in reps_results:
                if rr.model_id and rr.temperature is not None:
                    self._reps_contracts.update(
                        rr.model_id, rr.temperature, rr.improved
                    )

        # F1: Reflection Engine
        if self._reps_reflection and self._reps_reflection_config.get("enabled", True):
            try:
                self._reps_current_reflection = await self._reps_reflection.reflect(
                    reps_results, self.database, self._reps_current_reflection,
                )
                if self._reps_metrics is not None:
                    self._reps_metrics.log_reflection(
                        self._reps_batch_count,
                        self._reps_current_reflection,
                        reflection_calls=self._reps_reflection.total_reflection_calls,
                        reflection_tokens=self._reps_reflection.total_reflection_tokens,
                    )
            except Exception as e:
                logger.warning(f"Reflection failed: {e}")

        # Metrics logging
        if self._reps_metrics is not None:
            self._reps_metrics.log_batch(
                batch_number=self._reps_batch_count,
                batch_results=reps_results,
                database=self.database,
                edit_entropy=self._reps_convergence.last_entropy,
                strategy_divergence=self._reps_convergence.last_divergence,
            )

        # F8: Annotate candidates
        if self.config.reps.annotations.enabled and self._reps_current_reflection:
            await self._reps_annotate_candidates(reps_results)

    async def _reps_annotate_candidates(self, reps_results: List):
        """F8: Annotate candidates with reflection-derived metadata."""
        reflection = self._reps_current_reflection or {}
        working = reflection.get("working_patterns", [])
        failing = reflection.get("failing_patterns", [])

        for rr in reps_results:
            if rr.child_program_dict is None:
                continue
            pid = rr.child_program_dict.get("id")
            if pid and pid in self.database.programs:
                prog = self.database.programs[pid]
                annotations = prog.metadata.get("reps_annotations", {})
                annotations["worker_name"] = rr.worker_name
                annotations["model_used"] = rr.model_id or "default"

                if not rr.improved and rr.child_score < rr.parent_score * 0.95:
                    annotations["dead_end"] = True
                    if failing:
                        annotations["outcome"] = failing[0][:200]

                if rr.improved and working:
                    annotations["hypothesis"] = working[0][:200]

                annotations["batch"] = self._reps_batch_count
                prog.metadata["reps_annotations"] = annotations
                if "reps_batch_found" not in prog.metadata:
                    prog.metadata["reps_batch_found"] = self._reps_batch_count

        # Per-program summaries moved to per-iteration path (see
        # _summarize_child_inline). Batch annotation only adds tags now.


