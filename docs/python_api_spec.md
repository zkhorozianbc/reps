# REPS Python API Spec

## Goals

Expose REPS as a Python package with an ergonomic, programmatic API.
For the LLM wrapper we follow `dspy.LM`'s shape; for the optimizer we
follow GEPA's `optimize_anything` (the artifact-text-in / score-out API
in the standalone `gepa-ai/gepa` package). A user should be able to:

```python
import reps

lm = reps.LM("anthropic/claude-sonnet-4.6")

def evaluate(code: str) -> reps.EvaluationResult:
    # run the code, score it, optionally produce per_instance_scores +
    # feedback for GEPA-style ASI.
    ...

optimizer = reps.REPS(
    lm=lm,
    max_iterations=50,
    selection_strategy="mixed",
    pareto_fraction=0.3,
    trace_reflection=True,
    merge=True,
)
result = optimizer.optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)
print(result.best_code, result.best_score)
```

…without writing a YAML file, an `evaluator.py` module, or running a CLI.

The evaluator takes the artifact text directly and returns a score (or a
richer `EvaluationResult` with per-instance scores and free-form
feedback for the Phase 1-5 ASI features). There is no separate "runner"
or "predict-then-score" decomposition. This matches the spirit of
`gepa.optimize_anything(seed_candidate, evaluator=...)` and aligns with
how REPS already works internally.

## Non-goals

- We are NOT modeling `dspy.GEPA.compile()` directly. That API requires
  a `dspy.Module` (a callable program) and a per-example metric, which
  pre-supposes a "predictor" abstraction REPS does not have. A
  `dspy.Module`-flavored `compile()` shim is a thin layer someone could
  build on top later — see "DSPy-compatibility shim" below.
- We are NOT replacing the existing `reps-run` CLI or the YAML config
  path. They remain the right entry point for batch experiments and
  power-user runs. The Python API is the right entry point for notebook
  prototyping, library integration, and CI checks.
- We are NOT trying to be GEPA. We borrow naming and shape where it's
  natural; we keep REPS's distinctive features (convergence monitor,
  SOTA steering, worker pool, all of Phases 1-5) intact.

## Status

This is a spec. Nothing is implemented yet. Phases 1-5 of the
[GEPA implementation plan](./gepa_implementation_plan.md) provide the
infrastructure that this API will surface; Phase 7 (the adapter pattern)
is the natural complement that lets `reps.REPS` optimize non-Python
artifacts.

## Working conventions (mirroring the GEPA plan)

- Branch: a new feature branch per phase of this work.
- Per-phase gate: unit tests pass, focused integration test exercises the
  new path, subagent review, commit.
- Default-off: existing CLI and YAML paths must keep working. The Python
  API is purely additive — a new entry point.
- Two-commit phases: pure module(s) first, wiring/integration second.

## Public surface (top-level `reps` namespace)

The `reps/__init__.py` module currently exports internal building blocks
(`ReflectionEngine`, `ConvergenceMonitor`, etc.). Those move to a
`reps.internal` submodule (still importable for power users) and the
top-level becomes user-facing:

```python
# Configuration & LLM
reps.LM                    # LLM wrapper, ~ dspy.LM
reps.configure             # process-wide defaults (lm, output_dir, log_level)

# Optimizer
reps.REPS                  # the headline optimizer class
reps.OptimizationResult    # what compile()/optimize() returns

# Adapter / artifact extension (Phase 7 work; sketched here for completeness)
reps.Adapter               # ABC for non-Python artifacts
reps.PythonSourceAdapter   # default — current REPS behavior

# Evaluation primitives (already exist; re-exported for convenience)
reps.EvaluationResult      # ASI-rich return shape from evaluators

# Internals (existing modules, kept for power users)
reps.internal.ReflectionEngine
reps.internal.WorkerPool
# ... etc.
```

`from reps import LM, REPS` is the canonical import.

## `reps.LM`

A thin wrapper around the existing `reps.llm` providers. Mirrors
`dspy.LM`'s constructor shape so users can copy patterns from the dspy
ecosystem.

### Signature

```python
reps.LM(
    model: str,                      # "<provider>/<model>" or just "<model>"
    *,
    api_key: Optional[str] = None,   # falls back to provider env var
    api_base: Optional[str] = None,  # for OpenRouter, Azure, custom gateways
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    retries: int = 2,
    retry_delay: int = 5,
    extended_thinking: Optional[str] = None,  # "off"|"low"|"medium"|"high"|"xhigh"|"max"
    **provider_kwargs: Any,          # passed through to provider client
)
```

### Behavior

- The `model` string is parsed as `"<provider>/<id>"` (e.g.
  `"anthropic/claude-sonnet-4.6"`) or just `"<id>"`. If the provider
  isn't given, fall back to `reps.llm.provider_of.provider_of_model(id)`
  which already does this inference.
- `api_key=None` ⇒ env-var fallback per provider:
  `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`. Fail
  loudly at construction if no key is resolvable (matches the existing
  `program_summarizer.build_summarizer_llm` behavior).
- `extended_thinking` is the existing `reasoning` knob renamed for
  consistency with how the field is described in user docs.
- `**provider_kwargs` is passed verbatim to the underlying client
  constructor so users can opt into provider-specific features
  (e.g. `cache_control`, `top_p`) without us enumerating them.
- The wrapper is async-first internally (existing `LLMInterface`) but
  exposes both sync and async call surfaces:

```python
lm = reps.LM("anthropic/claude-sonnet-4.6")

# Sync
text = lm("Hello")                            # __call__ is sync wrapper
text = lm.generate("Hello")                   # explicit sync
# Async
text = await lm.agenerate("Hello")
text = await lm.agenerate_with_context(system_message=..., messages=...)
```

`__call__` is the dspy-flavored shortcut. The `agenerate` /
`agenerate_with_context` methods preserve full async semantics for
callers that need them (e.g. tools or batch dispatch).

### Implementation notes

- Most of this exists in `reps/llm/{anthropic,openai_compatible,base}.py`.
  `reps.LM` is a thin façade over those (~50 lines).
- The sync wrappers use `asyncio.run` if no loop is running, else
  `asyncio.run_coroutine_threadsafe(...)` to avoid the "asyncio.run can't
  be called from a running event loop" trap.
- Existing `LLMEnsemble` stays as the multi-model abstraction; users who
  want ensembling can pass `reps.LM(model="ensemble", models=[...])`
  (see "Ensembles" below).

### Ensembles

```python
ensemble = reps.LM.ensemble([
    reps.LM("anthropic/claude-sonnet-4.6", weight=0.7),
    reps.LM("openai/gpt-5", weight=0.3),
])
optimizer = reps.REPS(lm=ensemble, ...)
```

`reps.LM.ensemble(...)` is a classmethod that returns an object
implementing the same interface as `reps.LM` but routing each call
through `LLMEnsemble`. Existing `LLMEnsemble` already does this — the
classmethod is a constructor sugar.

## `reps.REPS`

The headline optimizer. Mirrors `dspy.GEPA`'s pattern:

> ```python
> optimizer = dspy.GEPA(metric=..., max_metric_calls=150,
>                       reflection_lm="openai/gpt-5")
> optimized = optimizer.compile(student=MyProgram(),
>                               trainset=trainset, valset=valset)
> ```

…but with REPS's own concepts surfaced as constructor knobs.

### Signature

```python
reps.REPS(
    *,
    # LLMs
    lm: Union[reps.LM, "LLMEnsemble"],          # primary worker LLM
    reflection_lm: Optional[reps.LM] = None,    # F1 batch reflection (defaults to lm)
    summarizer_lm: Optional[reps.LM] = None,    # F8 per-iteration summary
    trace_reflection_lm: Optional[reps.LM] = None,  # Phase 3 directive (defaults to lm)

    # Search budget
    max_iterations: int = 100,
    max_metric_calls: Optional[int] = None,     # alternative budget (mirrors GEPA)
    target_score: Optional[float] = None,       # early-stop threshold
    early_stopping_patience: Optional[int] = None,

    # GEPA-style features (Phases 1-5)
    selection_strategy: str = "map_elites",     # "map_elites" | "pareto" | "mixed"
    pareto_fraction: float = 0.0,               # for "mixed"
    pareto_instance_keys: Optional[List[str]] = None,
    trace_reflection: bool = False,             # Phase 3
    lineage_depth: int = 3,                     # Phase 5
    merge: bool = False,                        # Phase 4
    minibatch_size: Optional[int] = None,       # Phase 6

    # Population / archive
    num_islands: int = 5,
    population_size: int = 1000,
    archive_size: int = 100,
    feature_dimensions: Optional[List[str]] = None,

    # REPS feature toggles (preserved)
    reflection: bool = True,                    # F1
    revisitation: bool = True,                  # F2
    convergence_monitor: bool = True,           # F4
    contracts: bool = True,                     # F5 Thompson bandit
    sota_steering: bool = False,                # F6

    # Workers
    workers: Optional[List[Dict[str, Any]]] = None,  # [{name, role, model_id, ...}]

    # Output / persistence
    output_dir: Optional[str] = None,           # None ⇒ tempdir, no persistence
    save_checkpoints: bool = False,
    checkpoint_interval: int = 50,
    log_level: str = "INFO",

    # Adapter (Phase 7)
    adapter: Optional[reps.Adapter] = None,     # None ⇒ PythonSourceAdapter
)
```

The constructor builds an internal `reps.config.Config` from these args.
Keeping the constructor explicit (rather than `**config: Any` or a
`Config` dataclass argument) gives users discoverability and IDE
autocomplete; it also forces us to evolve the public surface
deliberately.

### `optimize()` — primary entry point

This is the headline API and matches `gepa.optimize_anything`'s shape:
artifact text in, score out. The user supplies one callable that knows
how to run the artifact (whatever that means for their problem — execute
code, render a prompt, parse a config) and return a score.

```python
result = optimizer.optimize(
    initial: str,                                # seed artifact (code, prompt, config, ...)
    evaluate: Callable[..., reps.EvaluationResult | float | dict],
    *,
    seed: Optional[int] = None,                  # deterministic RNG seed
    callbacks: Optional[List["Callback"]] = None,
)
```

`evaluate` is called with the artifact text. Three accepted return
shapes, in order of richness:

1. `float` — just a scalar score. REPS treats it as `combined_score` and
   skips Pareto / trace-reflection / merge (no per-instance signal).
2. `dict` — must include `combined_score`; may include `validity`,
   `per_instance_scores`, `feedback`, `error`. Auto-wrapped in
   `EvaluationResult`.
3. `reps.EvaluationResult` — full ASI surface, recommended for users who
   want the Phase 1-5 features to actually fire.

The signature is permissive on inputs too — REPS introspects the
callable and passes whichever of these the function declares:

- `evaluate(code)` — minimum.
- `evaluate(code, *, env=None)` — for evaluators that spawn subprocesses
  with controlled env vars (matches today's benchmark `evaluator.py`
  pattern).
- `evaluate(code, *, env=None, instances=None)` — Phase 6 minibatch.
- `evaluate(program_path, ...)` — for legacy file-path-based evaluators.
  REPS detects the parameter name `program_path` and writes the artifact
  to a temp file before calling.

This is one explicit callable that turns the artifact into a score.
There is no "predict on each example, then score" decomposition — that
would force a "runner" abstraction we don't think REPS needs at the
public-API level.

### `aoptimize()` — async variant

Same signature, returning an awaitable. For notebook / library users who
already have an event loop running.

### `compile()` — DSPy-compatibility shim

A *thin* shim for users who already have a dspy-style metric and a
sample-by-sample evaluation pattern. NOT the primary API.

```python
result = optimizer.compile(
    initial: str,
    metric: Callable[[Example, Prediction], float],
    trainset: List[Example],
    valset: Optional[List[Example]] = None,
    runner: Callable[[str, Example], Any],         # required: how to run the artifact on one example
    *,
    seed: Optional[int] = None,
    callbacks: Optional[List["Callback"]] = None,
)
```

Three callables instead of one because dspy's pattern is
`predict-then-score`, and REPS — unlike `dspy.Module` — does not know how
to run an arbitrary string of Python source on an `Example`. The user
must say what "running the artifact" means via `runner`; the metric
scores the resulting prediction.

Internally this is a synthetic `evaluate` built on top of `optimize()`:
each example becomes one instance with key `f"ex_{i}"`,
`combined_score` is the mean of per-example scores,
`per_instance_scores` is the dict, and `feedback` is built from the
bottom-K example traces. So you get full ASI via the synthetic path —
just at the cost of three callables instead of one.

If you don't already have a dspy-shaped metric, prefer `optimize()`.

## Metric interface

Two metric interfaces are supported:

### dspy-flavored

```python
def metric(example, prediction, *, trace=None) -> float:
    """example: an item from trainset/valset.
       prediction: whatever the program produced for that example.
       trace: per-prediction trace (for ASI feedback)."""
```

REPS wraps this in a synthetic evaluator. Each example becomes one
instance with key `f"ex_{i}"`. The `combined_score` is the mean across
examples; `per_instance_scores` is the per-example score dict;
`feedback` is built from the bottom-K example traces (configurable).

This is the dspy-compatibility path — drop in a metric you already wrote
for `dspy.Evaluate`, get GEPA-style ASI for free.

### REPS-native

```python
def evaluate(program_path: str, env=None, instances=None) -> reps.EvaluationResult:
    """Returns an EvaluationResult with metrics, per_instance_scores,
       feedback, and (optionally) artifacts."""
```

This is what existing benchmarks already implement. The Python API just
exposes it without requiring a YAML config or CLI invocation.

## `reps.OptimizationResult`

```python
@dataclass
class OptimizationResult:
    best_code: str                                  # winning artifact text
    best_score: float                               # combined_score
    best_metrics: Dict[str, float]                  # full metrics dict
    best_per_instance_scores: Optional[Dict[str, float]]
    best_feedback: Optional[str]
    iterations_run: int
    total_metric_calls: int
    total_tokens: Dict[str, int]                    # {"in": N, "out": M}
    history: List["IterationRecord"]                # per-iteration record
    output_dir: Optional[str]                       # if persisted
    converged_early: bool
    early_stopping_reason: Optional[str]

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "OptimizationResult": ...
```

`history` is a list of compact iteration records (parent id, child id,
score delta, worker name, fidelity tag, optional directive). Useful for
plotting trajectories and post-hoc analysis without re-reading the
on-disk database.

## `reps.configure`

Mirrors `dspy.configure(lm=...)` for process-wide defaults:

```python
reps.configure(
    lm: Optional[reps.LM] = None,
    output_dir: Optional[str] = None,
    log_level: Optional[str] = None,
)
```

Sets module-level defaults. `reps.REPS()` constructor uses these when an
explicit value isn't passed. Useful in notebooks where users instantiate
many optimizers with the same LLM.

## Callbacks

```python
class Callback(Protocol):
    def on_iteration(self, record: "IterationRecord") -> None: ...
    def on_batch(self, batch: List["IterationRecord"]) -> None: ...
    def on_complete(self, result: "OptimizationResult") -> None: ...
```

Built-ins shipped with the package:
- `reps.callbacks.ProgressBar` — tqdm progress
- `reps.callbacks.JSONLogger` — append iteration records to a JSONL file
- `reps.callbacks.WandbLogger` — opt-in W&B integration
- `reps.callbacks.MLflowLogger` — opt-in MLflow integration

This is also the extension point for users to integrate with their own
experiment trackers without forking REPS.

## `reps.Adapter` (Phase 7 placeholder)

This API surface lands when Phase 7 ships. The spec is in the GEPA
implementation plan; this section pins the shape for the public API:

```python
class Adapter(ABC):
    @abstractmethod
    def evaluate(self, artifact: str, env=None, instances=None) -> EvaluationResult: ...

    @abstractmethod
    def parse_mutation(self, raw_response: str) -> str: ...

    @abstractmethod
    def initial_artifact(self) -> str: ...

    def render_in_prompt(self, artifact: str) -> str:
        return f"```\n{artifact}\n```"

    @property
    def language(self) -> str:
        return "text"
```

Users pick:
- `reps.PythonSourceAdapter()` (default) — current REPS behavior.
- `reps.PromptAdapter()` — optimize a single system prompt against a
  metric.
- `reps.ConfigAdapter(schema=...)` — optimize a JSON/YAML config.

…or implement their own. `reps.REPS(adapter=my_adapter, ...)` is the
hook point.

Until Phase 7 ships, `adapter=None` defaults to current code-only
behavior. The constructor accepts `adapter=...` from day one so the
field is forward-compatible.

## Worked examples

### Minimum viable — scalar evaluator

```python
import reps
import subprocess

def evaluate(code: str) -> float:
    # Run the candidate, return a scalar. No per-instance signal —
    # Pareto/trace-reflection/merge will degrade gracefully.
    result = subprocess.run(["python", "-c", code], capture_output=True)
    return 1.0 if result.returncode == 0 else 0.0

optimizer = reps.REPS(
    lm=reps.LM("anthropic/claude-sonnet-4.6"),
    max_iterations=20,
)
result = optimizer.optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)
print(result.best_score, result.best_code)
```

### GEPA-style (all features on)

```python
import reps

reps.configure(
    lm=reps.LM("anthropic/claude-sonnet-4.6"),
    output_dir="./runs/circle_packing",
)

def evaluate(code: str, env=None, instances=None) -> reps.EvaluationResult:
    # Custom evaluator emitting full ASI (per-objective scores + feedback).
    # `instances` is the Phase 6 minibatch subset (None ⇒ full set).
    ...

optimizer = reps.REPS(
    max_iterations=200,
    selection_strategy="mixed",
    pareto_fraction=0.3,
    trace_reflection=True,
    lineage_depth=3,
    merge=True,
    minibatch_size=2,                         # Phase 6
    num_islands=4,
    workers=[
        {"name": "exploiter", "role": "exploiter", "temperature": 0.3, "weight": 0.5},
        {"name": "explorer",  "role": "explorer",  "temperature": 1.2, "weight": 0.3},
        {"name": "merger",    "role": "crossover", "temperature": 0.6, "weight": 0.2},
    ],
    save_checkpoints=True,
)
result = optimizer.optimize(
    initial=open("initial_program.py").read(),
    evaluate=evaluate,
)
```

### Async / library

```python
async def run():
    optimizer = reps.REPS(
        lm=reps.LM("anthropic/claude-sonnet-4.6"),
        max_iterations=20,
    )
    return await optimizer.aoptimize(initial=seed, evaluate=evaluate)

asyncio.run(run())
```

### DSPy-compatibility shim

For users porting from dspy who already have a `metric(example, prediction)`
callable, `compile()` requires an explicit `runner` because REPS does not
know how to run a code string against a sample.

```python
def runner(code: str, example) -> str:
    # User-supplied: import the artifact and call whatever entry point
    # makes sense for their problem.
    ns: dict = {}
    exec(code, ns)
    return ns["answer"](example["question"])

def metric(example, prediction) -> float:
    return float(prediction.strip() == example["answer"])

optimizer = reps.REPS(lm=reps.LM("anthropic/claude-sonnet-4.6"), max_iterations=20)
result = optimizer.compile(
    initial=open("seed.py").read(),
    trainset=load_questions(),
    metric=metric,
    runner=runner,
)
```

If you don't already have a `metric(example, prediction)` callable, just
write `evaluate(code) -> EvaluationResult` and use `optimize()`.

### Migration from CLI

Before:
```bash
reps-run experiment/benchmarks/circle_packing/initial_program.py \
         experiment/benchmarks/circle_packing/evaluator.py \
         --config experiment/configs/circle_sonnet_reps.yaml \
         --output runs/cp \
         --iterations 100
```

After:
```python
import reps
import yaml

cfg = yaml.safe_load(open("experiment/configs/circle_sonnet_reps.yaml"))
optimizer = reps.REPS.from_config_dict(cfg)
optimizer.optimize_path(
    initial_program="experiment/benchmarks/circle_packing/initial_program.py",
    evaluator="experiment/benchmarks/circle_packing/evaluator.py",
    output_dir="runs/cp",
    iterations=100,
)
```

`reps.REPS.from_config_dict(...)` is the bridge for users with existing
YAML configs. `optimize_path(...)` is a convenience that loads
`initial_program` from disk and imports `evaluator` as a module.

## Implementation phases

### A — `reps.LM` + `reps.configure`

Pure shim over existing `reps.llm.*`. ~150 LOC + tests with mocked
HTTP. No internal changes. Lands first because it's a dependency for
phase B and is independently useful for users who only want REPS's
provider abstraction.

Files:
- New `reps/api/__init__.py` — public surface module.
- New `reps/api/lm.py` — `LM` class + ensemble factory.
- New `reps/api/configure.py` — process-wide defaults.
- Update `reps/__init__.py` — re-export `LM`, `configure`.
- Tests `tests/test_api_lm.py`.

### B — `reps.REPS` skeleton + `optimize()`

Builds the optimizer constructor + the primary `optimize()` entry point
(matching `gepa.optimize_anything`). Internally constructs a `Config`,
instantiates a controller + database + evaluator, and runs the existing
async event loop. The controller is unchanged — this is purely the new
entry path.

Includes the introspection logic that detects the `evaluate` callable's
signature and dispatches:
- `evaluate(code)` ⇒ pass artifact text directly.
- `evaluate(code, *, env=None, instances=None)` ⇒ pass through.
- `evaluate(program_path, ...)` ⇒ write artifact to a temp file first
  (legacy file-path-based evaluators).

Also includes the return-shape coercion: `float | dict |
EvaluationResult` — auto-wraps to `EvaluationResult` so the controller
sees a consistent type.

Files:
- New `reps/api/optimizer.py` — `REPS` class, `optimize()`,
  `aoptimize()`, `OptimizationResult`.
- New `reps/api/result.py` — `OptimizationResult` dataclass + helpers.
- New `reps/api/evaluate_dispatch.py` — signature introspection +
  return-shape coercion.
- Tests `tests/test_api_optimize.py` — end-to-end with a mock LLM and a
  trivial in-Python evaluator that doesn't need a benchmark file.
  Includes one test per signature variant and one per return shape.

### C — `compile()` DSPy-compat shim

The `compile()` shim — a `(runner, metric, trainset, valset)` quadruple
synthesized into an `optimize()`-style evaluator. The synthetic
evaluator iterates `trainset`, calls `runner(code, example)`,
calls `metric(example, prediction)`, builds `per_instance_scores` from
the per-example scores, and synthesizes `feedback` from bottom-K
example/prediction pairs.

Files:
- New `reps/api/synthetic_evaluator.py`.
- Update `reps/api/optimizer.py` — `compile()` method.
- Tests `tests/test_api_compile.py`.

This phase is independent of B; it could ship later or be deprioritized
entirely if the dspy-compatibility path doesn't earn its keep. Phase A
+ B is the minimum viable public API.

### D — Callbacks + result history + persistence

`Callback` Protocol, built-in callbacks, `OptimizationResult.save/load`,
`from_config_dict` for YAML migration.

Files:
- New `reps/api/callbacks.py`.
- Update `reps/api/optimizer.py` — callback dispatch hooks at iteration
  and batch boundaries.
- Update `reps/api/result.py` — JSON serialization.
- Tests `tests/test_api_callbacks.py`.

### E — Adapter integration (lands with Phase 7 of the GEPA plan)

`reps.REPS(adapter=...)` becomes meaningful once `reps.Adapter` and
`reps.PythonSourceAdapter` exist. Until then, the constructor accepts
the kwarg and ignores it — forward-compatible.

## Open design decisions

1. **Module layout.** Top-level `reps.LM` / `reps.REPS` (shortest user
   import) vs `reps.api.LM` / `reps.api.REPS` (clean separation from
   internals). Recommendation: top-level for users, with `reps.internal`
   re-exporting current internals for power users who already import
   `reps.ReflectionEngine` etc.

2. **Return shape.** `gepa.optimize_anything` returns a `GEPAResult`
   carrying the best candidate text + metadata. `dspy.GEPA.compile()`
   returns a *new `dspy.Module`* — a callable program. REPS optimizes
   raw text, so returning `OptimizationResult` (text + metadata)
   matches `optimize_anything`'s shape. No deviation. Keeping as the
   default; we can add `result.as_callable()` later if users ask for
   it (with a clear note that it `exec()`s arbitrary code).

3. **Sync wrappers.** `asyncio.run()` from inside a running loop raises
   `RuntimeError` in modern Python. We need `nest_asyncio`-style or
   thread-pool fallback. Recommendation: detect a running loop and
   raise a clear error pointing the user to `acompile()` /
   `aoptimize()`. Don't silently nest loops.

4. **`max_iterations` vs `max_metric_calls`.** GEPA uses metric-call
   budget; REPS has used iteration budget. Both are useful. Support
   both — the run terminates when *either* limit is hit. Document the
   difference clearly so users pick the right one.

5. **YAML compatibility.** `REPS.from_config_dict(cfg)` is the bridge.
   Should it be exact or lenient with old fields? Recommendation: exact
   — fail loudly on unrecognized keys with a helpful diff against the
   current schema, so users notice when their YAMLs go stale.

6. **Provider strings.** Should `reps.LM("claude-sonnet-4.6")` (no
   provider prefix) be allowed? Yes — fall back to
   `provider_of_model(model)`. But fail loudly if inference is
   ambiguous (e.g. a model name shared between providers).

7. **What about `dspy.Evaluate` / `dspy.assertion` interop?** Out of
   scope for this spec, but the `compile()` shim's
   `metric(example, prediction)` signature matches `dspy.Evaluate`'s so a
   user can lift a dspy metric into REPS by adding a `runner`. A formal
   `from_dspy_module(module)` helper that auto-derives the runner from
   a `dspy.Module`'s `__call__` is a follow-up.

8. **`compile()` worth shipping at all?** Phase C is independent of A+B
   and could be deprioritized. If `optimize()` covers the GEPA-style
   pattern most users want, `compile()` only matters for users with an
   existing dspy `metric(example, prediction)` callable they want to
   reuse. We could ship A+B first, gather feedback, and decide whether
   to invest in C based on demand.

## Append-only changelog

Updated as phases of this spec ship.

| Date | Phase | Commit | Notes |
|------|-------|--------|-------|
| —    | —     | —      | (spec only — no implementation yet) |
