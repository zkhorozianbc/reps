# Tool-Calling Worker Primitive for REPS — Design

- Status: draft for implementation
- Date: 2026-04-21
- Author: zkhorozian (with Claude)

## 1. Goal

Replace the single-LLM-call compute node in REPS with a **swappable `Worker` primitive** so tool-calling agents (Anthropic tool-runner, DSPy ReAct, and future: Pydantic-AI, custom) can act as compute nodes interchangeably. Each worker declares its own config, tool set, and implementation. The rest of REPS (WorkerPool selection, Thompson bandit, convergence monitor, reflection) operates on named worker configs as the selection primitive.

## 2. Success criteria (verification only: live production testing)

The feature is considered done when:

1. Three concrete worker types implement the `Worker` interface:
   - `SingleCallWorker` — a straight port of today's one-shot LLM call path.
   - `AnthropicToolRunnerWorker` — native Anthropic tool-use loop with extended thinking.
   - `DSPyReActWorker` — DSPy `ReAct` program over the native Anthropic provider.
2. A YAML config declares all three as available worker types. A single live `reps-run` against a real benchmark (circle-packing) using `ANTHROPIC_API_KEY` executes iterations across all three worker types.
3. Per-program trace sidecars (`programs/<id>.trace.json`) contain the intermediate messages, including extended thinking content blocks preserved verbatim.
4. The final best program's `combined_score` exceeds the seed-program baseline by a non-trivial margin on the benchmark.

Verification is exclusively **live production testing**: running the harness end-to-end against the Anthropic API and observing improvement. New code is not gated on new unit tests; existing tests must still pass. Additional test coverage is nice-to-have and explicitly out of scope for the verification gate.

## 3. Scope / non-goals

In scope:

- Worker primitive, three implementations, registry-based config.
- Asyncio-only controller migration (drop `ProcessPoolExecutor`).
- Evaluator state isolation fixes required by concurrent asyncio iterations.
- Trace capture and persistence for tool-use and extended thinking.
- YAML config shape for worker declarations.
- WorkerPool/ContractSelector adaptation to worker-id as the selection axis.

Out of scope (v2):

- Multi-child-per-iteration workers (BestOfN). Contract closes this door for v1.
- Wall-clock / token-based batch boundaries. Iteration-based batching stays.
- Cost-aware bandit reward. Reward stays `improved: bool`.
- Sandboxing evolved code in a subprocess. Today it already runs in-thread via `run_in_executor(None, …)` — asyncio migration does not change this; call out as a known risk.
- A full migration off of `LLMInterface` for non-tool-calling paths — `SingleCallWorker` continues to use `LLMEnsemble`.

## 4. Current architecture (brief)

- `_run_iteration_worker` (`reps/controller.py:171-457`) runs in a `ProcessPoolExecutor` child; builds a prompt via `PromptSampler`, makes one `LLMEnsemble.generate_with_context` call inside `asyncio.run(...)`, parses a diff (SEARCH/REPLACE) or full rewrite, evaluates the child via `asyncio.run(...)` again, returns a `SerializableResult`.
- `IterationConfig` (`reps/iteration_config.py`) carries per-iteration bundle: `worker_type`, `model_id`, `temperature`, `generation_mode`, `prompt_extras`, `second_parent_id`, `target_island`.
- `WorkerPool` (`reps/worker_pool.py`) maps worker-type strings (`exploiter|explorer|crossover`) to `(temperature, generation_mode)` bundles plus second-parent sampling for crossover.
- `ContractSelector` (`reps/contract_selector.py`) runs a Thompson bandit over `(model_id, temperature)` arms, rewarded by `improved`.
- The evaluator (`reps/evaluator.py`) holds process-global state: `_pending_artifacts: Dict[program_id, …]`, `_current_program_id`, and it mutates `os.environ["REPS_PROGRAM_ID"]` at `:152-155`. Benchmark evaluators spawn subprocesses that inherit that env var — see `experiment/benchmarks/circle_packing/evaluator.py:145` which calls `subprocess.Popen` *without* `env=`.

## 5. Proposed architecture

### 5.1 Data shapes (new module `reps/workers/base.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Protocol, Union

SCHEMA_VERSION = 1

ErrorKind = Literal[
    "TOOL_ERROR", "MAX_TURNS_HIT", "PARSE_ERROR", "REFUSED", "TIMEOUT", "INTERNAL"
]

@dataclass
class WorkerError(Exception):
    kind: ErrorKind
    detail: Optional[str] = None

@dataclass
class WorkerConfig:
    name: str                              # unique id, e.g. "anthropic_tool_runner_exploiter"
    impl: str                              # "single_call" | "anthropic_tool_runner" | "dspy_react"
    role: str = "exploiter"                # "exploiter" | "explorer" | "crossover" — for analytics/reflection
    model_id: str = ""
    temperature: Optional[float] = None
    generation_mode: str = "diff"          # "diff" | "full"
    tools: List[str] = field(default_factory=list)
    max_turns: int = 1
    uses_evaluator: bool = False
    system_prompt_template: Optional[str] = None   # key into PromptSampler; defaults per impl
    impl_options: Dict[str, Any] = field(default_factory=dict)
    # Bandit axis ownership — ContractSelector overrides only axes the worker doesn't own
    owns_model: bool = True
    owns_temperature: bool = False
    # Resource hints
    expected_wall_clock_s: float = 15.0
    weight: float = 1.0                    # sampling weight in WorkerPool


@dataclass
class ContentBlock:
    type: Literal["text", "thinking", "redacted_thinking", "tool_use", "tool_result"]
    text: Optional[str] = None
    signature: Optional[str] = None                 # Anthropic signed thinking blob — verbatim
    data: Optional[str] = None                      # redacted_thinking base64 blob
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result_for_id: Optional[str] = None
    tool_result_content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_result_is_error: bool = False
    provider_extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    index: int
    role: Literal["system", "user", "assistant", "tool"]
    blocks: List[ContentBlock]
    model_id: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    stop_reason: Optional[str] = None
    started_at_ms: Optional[int] = None
    ended_at_ms: Optional[int] = None
    worker_type: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    impl_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerRequest:
    parent: "Program"
    inspirations: List["Program"]
    top_programs: List["Program"]
    second_parent: Optional["Program"]
    iteration: int
    language: str
    feature_dimensions: List[str]
    generation_mode: str                     # "diff" | "full"
    prompt_extras: Dict[str, str]
    temperature: Optional[float] = None
    model_id: Optional[str] = None


@dataclass
class WorkerContext:
    prompt_sampler: "PromptSampler"
    llm_factory: Callable[[str], "LLMInterface"]        # wraps LLMEnsemble model lookup
    dspy_lm_factory: Callable[["WorkerConfig"], Any]    # build configured dspy.LM
    evaluator: Optional["Evaluator"]                    # present iff WorkerConfig.uses_evaluator
    scratch_id_factory: Callable[[], str]               # UUIDs for intermediate tool-call evals
    final_child_id: str                                 # id assigned BEFORE worker.run
    config: "Config"
    iteration_config: "IterationConfig"


@dataclass
class WorkerResult:
    child_code: str
    changes_description: Optional[str] = None
    changes_summary: str = ""
    applied_edit: str = ""                    # feeds ConvergenceMonitor edit-entropy
    turns: List[TurnRecord] = field(default_factory=list)
    turn_count: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    error: Optional[WorkerError] = None


class Worker(Protocol):
    config: WorkerConfig
    @classmethod
    def from_config(cls, config: WorkerConfig) -> "Worker": ...
    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult: ...
```

### 5.2 Registry (new module `reps/workers/registry.py`)

```python
_IMPLS: Dict[str, Type[Worker]] = {}

def register(impl_name: str):
    def deco(cls: Type[Worker]) -> Type[Worker]:
        _IMPLS[impl_name] = cls
        return cls
    return deco

def build_worker(cfg: WorkerConfig) -> Worker:
    return _IMPLS[cfg.impl].from_config(cfg)
```

Each worker module registers itself via `@register("anthropic_tool_runner")`, etc. Defaults shipped in `reps/workers/defaults.py` with three canonical preset configs (exploiter/explorer/crossover) using `SingleCallWorker` for backward compatibility.

## 6. Asyncio-only controller migration

`reps/controller.py` is restructured:

- Dies: `_worker_init` (L53), `_lazy_init_worker_components` (L135), `ProcessParallelController._serialize_config` (L540), `_create_database_snapshot` (L621). Imports of `ProcessPoolExecutor`, `mp.Event`, `pickle`, `signal` are removed.
- `_run_iteration_worker` (L171-457) becomes `async def _run_iteration(self, iteration, parent_id, inspiration_ids, iteration_config) -> SerializableResult`, a method on a new `AsyncController` (kept as alias `ProcessParallelController = AsyncController` for one release).
- Main loop in `run_evolution` (L651) swaps `pending_futures: Dict[int, Future]` for `pending: Dict[int, asyncio.Task]`; the busy-poll at L727-741 becomes `done, _ = await asyncio.wait(pending.values(), return_when=FIRST_COMPLETED, timeout=per_iter_timeout)`. Tasks are created inside an `asyncio.TaskGroup` wrapped with an `asyncio.Semaphore(max_concurrent_iterations)`.
- `_submit_iteration` (L1014) becomes `async def _spawn_iteration(...)` returning an `asyncio.Task`. Error isolation is handled inside `_guarded_iteration` which catches and returns `SerializableResult(error=…)` so TaskGroup doesn't cancel siblings.
- One shared `LLMEnsemble`, `Evaluator`, `PromptSampler` instance is built in `AsyncController.start()` and reused by every iteration. `AsyncAnthropic` clients inside workers get real connection reuse.
- Shared `self.database`: asyncio is single-threaded, so serialized writes at the end of each `_run_iteration` don't race. No lock.
- Config: new `EvaluatorConfig.max_concurrent_iterations: int` in `reps/config.py:313`; legacy `parallel_evaluations` aliases onto it. `max_tasks_per_child` (controller.py:470) becomes a no-op with a deprecation log.
- Signal handling: `loop.add_signal_handler(SIGINT, self.request_shutdown)` sets an `asyncio.Event` that `run_evolution` checks; pair with a `TaskGroup` cancellation.

### Known risk called out in spec

All iterations share one Python thread (GIL). Fine while LLM-bound (default); if a worker accidentally blocks the loop (sync HTTP call, CPU-heavy parser), all concurrent iterations stall. Mitigation: enforce `asyncio.wait_for` on every I/O boundary inside `_run_iteration`; add a watchdog log when any task exceeds `config.evaluator.timeout + 30s`. Evolved code sandboxing via subprocess is a v2 item.

## 7. Evaluator isolation (required for #6 to be correct)

New public method `reps/evaluator.py`:

```python
@dataclass
class EvaluationOutcome:
    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]]
    program_id: str

async def evaluate_isolated(
    self,
    program_code: str,
    *,
    program_id: Optional[str] = None,       # None => generate scratch UUID
    scratch: bool = False,
    run_dir: Optional[str] = None,
) -> EvaluationOutcome:
    ...
```

Internal changes:

1. Replace `self._pending_artifacts: Dict` with a `contextvars.ContextVar[Dict[str, Any]]` named `_call_artifacts`. Each `evaluate_isolated` call does `token = _call_artifacts.set({})`, runs evaluation, reads the dict, `reset(token)` in `finally`. Asyncio propagates ContextVar per Task — concurrent calls see independent dicts automatically.
2. Delete `self._current_program_id` (dead field).
3. Delete mutation of `os.environ["REPS_PROGRAM_ID"]` at evaluator.py:152-155. Build a per-call env dict `call_env = {**os.environ, "REPS_PROGRAM_ID": pid, ...}`; thread it to benchmark evaluators via an optional `env` kwarg on `run_with_timeout(program_path, timeout_seconds, env=None)` (benchmark side) that passes `env=env` into `subprocess.Popen`.
4. New helper `reps/runtime.py` exposing `current_program_id() -> Optional[str]` backed by a ContextVar set by `evaluate_isolated`. Benchmarks update to prefer this helper; `os.environ["REPS_PROGRAM_ID"]` remains as the subprocess-facing channel only.
5. `asyncio.Semaphore(parallel_evaluations)` in `Evaluator.__init__` bounds concurrency at the evaluator layer; wrap `evaluate_isolated` body in `async with self._eval_semaphore:`.
6. Keep legacy `evaluate_program(code, program_id) -> Dict[str,float]` as a thin wrapper for the seed eval at `runner.py:67`.

`child_id` assignment moves: `child_id = str(uuid.uuid4())` happens in the controller **before** `worker.run(...)` and is passed in via `WorkerContext.final_child_id`. The controller's final call is `await self.evaluator.evaluate_isolated(result.child_code, program_id=final_child_id)`. Mid-run tool-call evaluations use `ctx.scratch_id_factory()` (auto-UUID) — their artifacts stay in the contextvar-scoped dict and are attached to the `TurnRecord.tool_result_content`, never persisted to the final Program's artifacts.

### Benchmark change

`experiment/benchmarks/circle_packing/evaluator.py:145` — add `env=env` kwarg to `subprocess.Popen`. `_dump_packing_markdown` (circle_packing/evaluator.py:193-194) reads program_id via `reps.runtime.current_program_id()` instead of `os.environ`. Similar updates for any other benchmark that reads `REPS_PROGRAM_ID`.

## 8. Worker implementations

### 8.1 `SingleCallWorker` (`reps/workers/single_call.py`)

A straight port of today's inner loop. Uses `ctx.llm_factory(cfg.model_id).generate_with_context(...)`. Emits two `TurnRecord`s: `role="user"` with the built prompt, `role="assistant"` with the raw response text. `applied_edit` = (diff-mode) `serialize_diff_blocks(extract_diffs(response))` or (full-mode) response text stripped via `parse_full_rewrite`.

### 8.2 `AnthropicToolRunnerWorker` (`reps/workers/anthropic_tool_runner.py`)

Native Anthropic tool-use loop; bypasses `LLMInterface` to preserve `thinking` / `tool_use` / `tool_result` content blocks. Owns an `anthropic.AsyncAnthropic` client per worker instance (reused across `run()` calls → real connection pooling).

**Tool set (hybrid, D1):**

- `submit_child(code: str, changes_description: str)` — terminal; required.
- `edit_file(search: str, replace: str)` — incremental edit; applies to the in-flight parent code; accumulates SEARCH/REPLACE blocks. Returns the updated code preview.
- `view_parent()` — returns current parent source.
- `view_program(program_id: str)` — returns source from inspirations/top programs by id.
- `run_tests(code: str)` — opt-in (requires `uses_evaluator=True`); calls `await ctx.evaluator.evaluate_isolated(code, program_id=ctx.scratch_id_factory(), scratch=True)`, returns metrics + truncated artifacts to the model as `tool_result`.

`applied_edit` rule: if any `edit_file` calls happened → `serialize_diff_blocks(accumulated_blocks)`; else → (diff-mode) synthesize a unified diff parent→submitted_code using `difflib.unified_diff`, or (full-mode) the submitted code verbatim. Consistent with the `SingleCallWorker` shape in the common case.

**Extended thinking handling:**

- For reasoning models (`any(p in model.lower() for p in REASONING_MODEL_PATTERNS)`), pass `thinking={"type":"enabled","budget_tokens":N}`; skip `temperature` (matches existing reps/llm/anthropic.py:26, 62-75).
- Signed thinking blocks returned in an assistant turn are **preserved verbatim** (including `signature`) and echoed back in the subsequent user turn alongside `tool_result` blocks. `ContentBlock.signature` carries this.
- Known LiteLLM bug applies only to DSPy's path (Agent 3 note); native Anthropic SDK is unaffected.

**Error loop:**

- `SyntaxError` on `compile(submitted_code, "<submitted>", "exec")` → feed back as `tool_result(is_error=True)` with the error; model retries within remaining `max_turns`.
- `stop_reason == "refusal"` → `WorkerError(REFUSED)`.
- `stop_reason == "end_turn"` without `submit_child` → `WorkerError(PARSE_ERROR)`.
- Network errors + 5xx → exponential backoff, `retries = impl_options.get("retries", 3)`.
- 4xx → fail fast, `WorkerError(INTERNAL)`.
- Exhausted `max_turns` → `WorkerError(MAX_TURNS_HIT)`.

Bump `anthropic>=0.60.0` in `pyproject.toml` to guarantee `AsyncAnthropic` + native `thinking`.

### 8.3 `DSPyReActWorker` (`reps/workers/dspy_react.py`)

Uses `dspy.ReAct(signature, tools, max_iters=cfg.max_turns)` wrapped in `with dspy.context(lm=lm):` to scope the LM thread-locally. Called from `run()` via `await asyncio.to_thread(_invoke)`.

Signatures:

```python
class EvolveProgramFull(dspy.Signature):
    """Given a parent program and evolutionary context, produce an improved child."""
    parent_code: str = dspy.InputField()
    language: str = dspy.InputField()
    iteration: int = dspy.InputField()
    inspirations: str = dspy.InputField()
    top_programs: str = dspy.InputField()
    second_parent_code: str = dspy.InputField()
    feature_dimensions: str = dspy.InputField()
    extras: str = dspy.InputField()
    child_code: str = dspy.OutputField(desc="complete rewritten child program")
    changes_description: str = dspy.OutputField()

class EvolveProgramDiff(EvolveProgramFull):
    child_code: str = dspy.OutputField(desc="unified diff against parent_code")
```

Tools are built from `cfg.tools`, delegating to the same implementations as the Anthropic worker where possible (shared `reps/workers/tools.py` module). `run_tests` calls `await ctx.evaluator.evaluate_isolated(...)` via an `asyncio.run_coroutine_threadsafe` shim (since the DSPy path lives inside `asyncio.to_thread`, not on the main loop).

`dspy_lm_factory`:

```python
def make_dspy_lm_factory(config: Config):
    def factory(wc: WorkerConfig):
        kwargs = {
            "model": f"anthropic/{wc.model_id}",
            "api_key": config.llm.api_key,       # already env-resolved
            "max_tokens": config.llm.max_tokens,
            "cache": False,                      # CRITICAL — default True kills diversity
        }
        if wc.temperature is not None:
            kwargs["temperature"] = wc.temperature
        if wc.impl_options.get("thinking"):
            kwargs["thinking"] = wc.impl_options["thinking"]
        return dspy.LM(**kwargs)
    return factory
```

Trace extraction: `pred.trajectory` dict contains `thought_i`, `tool_name_i`, `tool_args_i`, `observation_i` for each step. One `TurnRecord(role="assistant")` per step with a text block (thought) + optional tool_use block; one `TurnRecord(role="tool")` per observation. Raw `pred.trajectory` stored as opaque blob on the first assistant turn's `impl_specific["dspy_trace"]`.

Dependency: add `dspy>=3.1.3` to `pyproject.toml`.

Known caveats persisted in code comments:

- Extended thinking + tools has a known LiteLLM bug (BerriAI/litellm#14194). If `thinking` enabled and we see repeated 400s, fall back to plain tool use (strip `thinking` from kwargs).
- DSPy ignores `PromptSampler` templates entirely — it renders its own prompts from Signature docstrings. For DSPy workers, prompt tuning happens by editing Signature docs/field descriptions.

## 9. Integration with REPS

### 9.1 WorkerPool → named worker configs

`reps/worker_pool.py` refactored: `WorkerPool` now samples over **named worker configs** declared in YAML (list of `WorkerConfig` entries). The three default preset configs (`exploiter_single_call`, `explorer_single_call`, `crossover_single_call`) are still shipped so existing configs Just Work. `IterationConfig.worker_type` is renamed to `worker_name` (carrying a `WorkerConfig.name`); `worker_type` kept as a deprecated alias for one release. The crossover branch at `worker_pool.py:132` triggers when the selected config's `role == "crossover"` (not name-based).

### 9.2 ContractSelector → axis ownership

Arms become `(worker_id, temperature)` where `temperature` is only a bandit axis if the worker declares `owns_temperature=False`. Defaults: workers own their model, do not own temperature (bandit can still tune). The update call at `contract_selector.py:78-99` uses worker_id as key.

### 9.3 Controller shrinks

`_run_iteration` boils down to:

```python
async def _run_iteration(self, iteration, iteration_config, parent_id, inspiration_ids):
    final_child_id = str(uuid.uuid4())
    parent = self.database.programs[parent_id]
    request = self._build_request(iteration, iteration_config, parent, inspiration_ids)
    ctx = self._build_context(iteration_config, final_child_id)
    worker = self.worker_registry.build(iteration_config.worker_name)
    result = await worker.run(request, ctx)
    if result.error is not None:
        return SerializableResult(error=result.error.detail or result.error.kind, iteration=iteration)
    outcome = await self.evaluator.evaluate_isolated(result.child_code, program_id=final_child_id)
    child = Program(id=final_child_id, code=result.child_code, ..., metrics=outcome.metrics)
    return SerializableResult(
        child_program_dict=child.to_dict(),
        parent_id=parent.id,
        iteration=iteration,
        target_island=request_target_island,
        prompt=derive_prompt_from_turns(result.turns),
        llm_response=derive_llm_response_from_turns(result.turns),
        artifacts=outcome.artifacts,
        reps_meta={
            "worker_type": iteration_config.worker_name,
            "diff": result.applied_edit,
            "turns": [asdict(t) for t in result.turns],
            "usage": result.usage,
            "wall_clock_seconds": result.wall_clock_seconds,
            ...
        },
    )
```

### 9.4 Convergence monitor

Unchanged. Still reads `reps_meta["diff"]` → `classify_edit` sees `applied_edit`. Since `classify_edit` only checks length + keyword presence (convergence_monitor.py:30-70), swapping in the canonical REPS-format string is meaning-preserving and strictly cleaner (no prose/CoT noise).

### 9.5 Reflection / metrics

Untouched. `prompt` and `llm_response` survive as `@property` derivations on `SerializableResult` / `IterationResult`:

- `prompt = {"system": first_system_text_or_empty, "user": first_user_text_concatenated}`
- `llm_response = last_assistant_turn_text_blocks_concatenated`

Reflection engine and metrics logger consume these as today.

## 10. Trace persistence

- **Per-program sidecar**: `programs/<id>.trace.json` containing `{"schema_version": 1, "worker_type": ..., "turns": [...]}`. Gated by existing `DatabaseConfig.log_prompts`.
- **Evolution trace JSONL**: new `EvolutionTraceConfig.include_turns: bool = False`. When true, turns injected into the trace's `metadata["turns"]` via controller.py:782-794 (don't fork the vendored OpenEvolve tracer). Compression defaults to `true` when `include_turns=true` (existing `compress` flag honored).
- **Size caps**: new `DatabaseConfig.max_turns_persisted: Optional[int] = None`. If set, keep first N/2 and last N/2 turns, insert a single marker turn with `impl_specific={"truncated": true, "truncated_count": k}`.
- **Pretty printer**: `reps/workers/trace_render.py::render_trace(turns) -> str` (30-line helper) for CLI / post-hoc inspection.

## 11. YAML configuration surface

Additions to the Config schema. Each YAML lists one or more `WorkerConfig` entries under `reps.workers.types`. The verification runs each use **one entry per config file**; mixing multiple entries in one run is supported by the primitive (WorkerPool samples by `weight`) but is not part of the gate.

Example illustrating the schema with multiple types (non-gate demo):

```yaml
reps:
  workers:
    types:
      - name: exploiter_single_call
        impl: single_call
        role: exploiter
        model_id: claude-sonnet-4-5
        temperature: 0.3
        generation_mode: diff
        weight: 5

      - name: explorer_anthropic_tool_runner
        impl: anthropic_tool_runner
        role: explorer
        model_id: claude-opus-4-7
        generation_mode: full
        tools: [edit_file, view_parent, view_program, submit_child, run_tests]
        max_turns: 8
        uses_evaluator: true
        expected_wall_clock_s: 90
        impl_options:
          thinking_budget: 8000
          retries: 3
        weight: 2

      - name: refiner_dspy_react
        impl: dspy_react
        role: exploiter
        model_id: claude-sonnet-4-5-20250929
        temperature: 0.6
        generation_mode: full
        tools: [view_parent, run_tests]
        max_turns: 6
        uses_evaluator: true
        weight: 3
```

Verification YAMLs (three separate files, each declares one `types` entry):

- `experiment/configs/verify_single_call.yaml` — one `single_call` config (diff-mode, temp 0.3).
- `experiment/configs/verify_anthropic_tool_runner.yaml` — one `anthropic_tool_runner` config (full-mode, tool set includes `edit_file`, `submit_child`, `run_tests`, `view_parent`; `uses_evaluator: true`; thinking enabled if model is a reasoning model).
- `experiment/configs/verify_dspy_react.yaml` — one `dspy_react` config (full-mode, `cache=False` forced by impl, `max_turns: 6`, `tools: [view_parent, run_tests]`).

The legacy shape (no `types` list, just `initial_allocation`) remains supported via a shim that generates three default preset configs (`SingleCallWorker`-based).

## 12. Verification plan (live production only)

The core surface area being verified is the `Worker` contract and its three implementations. WorkerPool multi-worker-type sampling within a single run is a spec feature but **not** a gate — it can be demonstrated in a follow-up experiment.

The gate is **three independent `reps-run` executions**, one per worker impl, against the same benchmark (`circle_packing`). `ANTHROPIC_API_KEY` in env for all runs. Each run uses a YAML config that declares exactly one `WorkerConfig` entry with the target `impl`; role/temperature knobs per run are what we'd pick for that impl to stand on its own (e.g., the Anthropic tool-runner gets full-mode + `run_tests` enabled; the single-call gets diff-mode).

Run A — `SingleCallWorker`:
```
reps-run \
  experiment/benchmarks/circle_packing/initial_program.py \
  experiment/benchmarks/circle_packing/evaluator.py \
  --config experiment/configs/verify_single_call.yaml \
  --output /tmp/reps_verify_single_call \
  --iterations 40
```
Pass if: best `combined_score` in the output DB > seed `combined_score` by ≥5% relative; no tracebacks.

Run B — `AnthropicToolRunnerWorker`:
```
reps-run ... --config experiment/configs/verify_anthropic_tool_runner.yaml \
  --output /tmp/reps_verify_anthropic --iterations 40
```
Pass if: best score > seed by ≥5% relative; every `programs/<id>.trace.json` is present; at least one produced program's trace contains a `thinking` content block with a non-null `signature`; at least one contains a `tool_use` block for `edit_file` or `run_tests`; no tracebacks.

Run C — `DSPyReActWorker`:
```
reps-run ... --config experiment/configs/verify_dspy_react.yaml \
  --output /tmp/reps_verify_dspy --iterations 40
```
Pass if: best score > seed by ≥5% relative; every trace sidecar includes a non-empty `impl_specific.dspy_trace`; at least one trace shows a tool_use step; no tracebacks.

All three runs must pass. Existing `pytest tests/` must still pass as a code-health check (not a design gate).

## 13. File-level change list

New files:

- `reps/workers/__init__.py`
- `reps/workers/base.py` — data shapes
- `reps/workers/registry.py`
- `reps/workers/edit_serializer.py` — `serialize_diff_blocks`
- `reps/workers/single_call.py`
- `reps/workers/anthropic_tool_runner.py`
- `reps/workers/dspy_react.py`
- `reps/workers/tools.py` — shared tool impls used by both tool-runner and DSPy
- `reps/workers/defaults.py` — preset configs for legacy YAML shim
- `reps/workers/trace_render.py` — pretty printer
- `reps/runtime.py` — `current_program_id()` contextvar helper
- `reps/prompt_templates/system_message_tool_runner.txt`

Modified:

- `reps/controller.py` — asyncio rewrite: remove multiprocess pathway (L17-23 imports, L53-168 worker init, L540-614 init/start/stop, L621-649 snapshot, L171-457 worker → async method). Keep REPS feature methods `_reps_*` untouched aside from how they consume `reps_meta` (which now carries turns/usage).
- `reps/evaluator.py` — `evaluate_isolated` + `EvaluationOutcome`, contextvar for artifacts, delete `os.environ` mutation (:152-155), per-call env dict threaded into benchmark `evaluate()` via optional `env` kwarg, `asyncio.Semaphore` for concurrency cap.
- `reps/worker_pool.py` — refactor to named-config sampling; crossover keyed on `role`.
- `reps/contract_selector.py` — arm key becomes `(worker_id, temperature?)`, axis-ownership check.
- `reps/iteration_config.py` — rename `worker_type` → `worker_name` (keep alias field for one release); add `turns: List[Dict] = field(default_factory=list)` on `IterationResult`.
- `reps/config.py` — `EvaluatorConfig.max_concurrent_iterations`, `DatabaseConfig.max_turns_persisted`, `EvolutionTraceConfig.include_turns`, new `REPSWorkersConfig.types: List[WorkerConfig]`.
- `reps/runner.py` — seed evaluation keeps `evaluate_program` (legacy wrapper).
- `experiment/benchmarks/circle_packing/evaluator.py` — `env=env` on `subprocess.Popen` at :145; `current_program_id()` for markdown dump.
- `pyproject.toml` — bump `anthropic>=0.60.0`, add `dspy>=3.1.3`.

Tests:

- `tests/test_controller.py:92, 105, 124, 174` — rewrite for async controller. `:206` survives.
- New integration tests are explicitly out of scope for the verification gate. If added opportunistically: contextvar isolation test for evaluator, schema round-trip for TurnRecord.

## 14. Implementation ordering

1. **Foundation** — new modules `reps/workers/{base,registry,edit_serializer,trace_render}.py` + `reps/runtime.py`. No behavior change yet.
2. **Evaluator isolation** — `evaluate_isolated` + contextvar artifacts + env-dict threading + benchmark `env=` kwarg. Prerequisite for asyncio migration.
3. **Asyncio controller** — rewrite `controller.py`. Keep existing SingleCall behavior functional via the new Worker primitive.
4. **SingleCallWorker** — port today's behavior under new contract.
5. **AnthropicToolRunnerWorker** — the first "new" compute node.
6. **DSPyReActWorker**.
7. **WorkerPool + ContractSelector** adaptation.
8. **YAML shim** for legacy configs + defaults file.
9. **Live verification run** against circle_packing.

Each step leaves the harness in a runnable state. Git worktree recommended for the sequence (use `superpowers:using-git-worktrees`).

## 15. Open questions (v2)

- **Extended thinking + tools reliability on DSPy/LiteLLM path.** Monitor BerriAI/litellm#14194 and remove the experimental flag once fixed upstream.
- **Cost-aware bandit reward.** `improved: bool` ignores that a tool-runner iteration may cost 10× a single call. Add tokens/cost weighting in v2.
- **Wall-clock batch boundaries.** Bimodal latency between single-call (12s) and tool-loop (90s) workers can stall reflection. Revisit batch-as-iterations assumption.
- **Evolved-code sandboxing.** Asyncio migration removes the last process-isolation layer between evolved code and the harness. Add `asyncio.create_subprocess_exec` inside the evaluator for hostile-code resilience.
- **Multi-child workers (BestOfN).** Contract closes this for v1. Re-open if a concrete use case emerges.
