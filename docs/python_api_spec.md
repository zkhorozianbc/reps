# REPS Python API Spec

## Goals

Expose REPS as a Python package with the smallest possible API:

```python
import reps

def evaluate(code: str) -> float:
    ...  # run the artifact, return a score

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4.6",          # or a built reps.Model
    api_key="sk-...",                              # optional; falls back to env var
    temperature=0.7,                               # any reps.ModelKwargs flow through
    max_iterations=50,
    selection_strategy="mixed",
    pareto_fraction=0.3,
    trace_reflection=True,
    merge=True,
).optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)

print(result.best_code, result.best_score)
```

**One outer call** (`optimize`), **one inner abstraction** (the `evaluate`
callable). Configure the model, set the knobs, hand over an evaluator,
let it run.

This shape is GEPA's `optimize_anything(seed_candidate, evaluator=...)`
with REPS's distinctive features (Pareto selection, trace reflection,
system-aware merge, ancestry-aware reflection — all of GEPA-plan
Phases 1-5) reachable as constructor kwargs.

## Non-goals

- Not modeling `dspy.Module` / `Predict` / `Signature`. REPS optimizes
  whole text artifacts.
- Not modeling `dspy.GEPA.compile()` directly — that requires a callable
  `dspy.Module` and a per-example metric. We borrow `dspy.LM`'s shape
  for the `Model` wrapper but follow `optimize_anything`'s shape for the
  optimizer.
- Not replacing the existing `reps-run` CLI or YAML configs. The Python
  API is the right entry point for notebooks, libraries, and CI; the
  CLI stays for batch experiments.

## v1 surface

Everything below is shipped in v1. Everything not listed is deferred to
v1.5 (see end of doc).

| Symbol | Purpose | Implemented in |
|---|---|---|
| `reps.Model` | LLM wrapper, similar shape to `dspy.LM` | [`reps/api/model.py:75`](../reps/api/model.py#L75) |
| `reps.ModelKwargs` | TypedDict of optional Model construction kwargs (used inline on `Optimizer`) | [`reps/api/model.py:33`](../reps/api/model.py#L33) |
| `reps.Example` | DSPy-style data record with explicit input keys | [`reps/api/example.py`](../reps/api/example.py) |
| `reps.Prediction` | Dict-like wrapper for an entrypoint's output | [`reps/api/example.py`](../reps/api/example.py) |
| `reps.Objective` | Compiles `(entrypoint, train_set, metric)` into the evaluator contract | [`reps/api/objective.py`](../reps/api/objective.py) |
| `reps.LLMJudge` | `Objective` that scores with an LLM judge | [`reps/api/objective.py`](../reps/api/objective.py) |
| `reps.PromptObjective` | `Objective` whose artifact is a *prompt template string*; REPS evolves the prompt | [`reps/api/objective.py`](../reps/api/objective.py) |
| `reps.Optimizer` | The optimizer | [`reps/api/optimizer.py:42`](../reps/api/optimizer.py#L42) |
| `reps.Optimizer.optimize(initial, evaluate)` | The single entry point | [`reps/api/optimizer.py:159`](../reps/api/optimizer.py#L159) |
| `reps.Optimizer.from_config(cfg)` | Power-user escape hatch | [`reps/api/optimizer.py:123`](../reps/api/optimizer.py#L123) |
| `reps.OptimizationResult` | What `optimize()` returns | [`reps/api/result.py:14`](../reps/api/result.py#L14) |
| `reps.EvaluationResult` | Optional rich return shape from the evaluator | [`reps/evaluation_result.py:11`](../reps/evaluation_result.py#L11) |

Top-level re-exports live in [`reps/__init__.py`](../reps/__init__.py)
and [`reps/api/__init__.py`](../reps/api/__init__.py); both kept in
sync via the same import lines.

That's it. Four classes (plus one TypedDict for kwarg typing), one
method on the optimizer. Internals are unchanged — Pareto, trace
reflection, merge, lineage, convergence monitor, SOTA controller, all
of it stays as it is in `reps/`. The public API is just a thin facade.

## `reps.Model`

**Implemented in:** [`reps/api/model.py:75`](../reps/api/model.py#L75) (class), constructor at line 83, sync `generate` at line 184, `__call__` at line 202.
**Tests:** [`tests/test_api_model.py`](../tests/test_api_model.py) (38 cases).

Thin facade over the existing `reps.llm.*` providers. Sync-only in v1
(async wrappers come in v1.5). Most users won't construct this directly
— passing a model-name string to `Optimizer(model=...)` builds it
inline. Use `reps.Model` directly when you want a standalone callable
(`model("hello")`) or to reuse one configured Model across multiple
optimizers.

### Signature

```python
reps.Model(
    model: str,                      # "<provider>/<id>" or "<id>"
    *,
    api_key: Optional[str] = None,   # falls back to provider env var
    api_base: Optional[str] = None,  # for OpenRouter, Azure, custom gateways
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
    retries: int = 2,
    retry_delay: int = 5,
    extended_thinking: Optional[str] = None,  # "off"|"low"|"medium"|"high"|"xhigh"|"max"
    **provider_kwargs: Any,
)
```

### Behavior

- `model` parsed as `"<provider>/<id>"` (e.g.
  `"anthropic/claude-sonnet-4.6"`) or just `"<id>"` (provider inferred
  via existing `reps.llm.provider_of.provider_of_model`). Splitting at
  [`reps/api/model.py:50`](../reps/api/model.py#L50)
  (`_split_provider`); resolution at
  [line 65](../reps/api/model.py#L65) (`_resolve_provider`).
- `api_key=None` ⇒ provider env-var fallback (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `OPENROUTER_API_KEY`). Fail loudly at construction
  if unresolved. See
  [`reps/api/model.py:101-108`](../reps/api/model.py#L101-L108).
- `extended_thinking` is the existing `reasoning` knob, mapped to
  `reasoning_effort` at
  [`reps/api/model.py:127`](../reps/api/model.py#L127).
- `**provider_kwargs` passed verbatim to the underlying SDK client.
  The wrapper rebuilds the SDK client when provider_kwargs are
  non-empty rather than threading kwargs through the internal
  `AnthropicLLM`/`OpenAICompatibleLLM` constructors. See
  [`reps/api/model.py:152-181`](../reps/api/model.py#L152-L181).

### Methods

```python
text = model("hello")            # __call__ — sync shortcut
text = model.generate("hello")   # explicit sync
```

That's it for v1. Async (`agenerate`, `agenerate_with_context`),
ensembles (`Model.ensemble`), and `reps.configure(model=...)` global
defaults all defer to v1.5.

## `reps.ModelKwargs`

**Implemented in:** [`reps/api/model.py:33`](../reps/api/model.py#L33).

A `TypedDict(total=False)` of the optional `Model` construction kwargs
(everything except `model` and `api_key`). Used as `**Unpack[ModelKwargs]`
on `reps.Optimizer` so users can configure the model inline without
building a `Model` instance first. Type checkers catch typos; runtime
users without type checking should know that unknown kwargs flow to
the underlying SDK constructor.

```python
class ModelKwargs(TypedDict, total=False):
    api_base: Optional[str]
    temperature: float
    max_tokens: Optional[int]
    timeout: int
    retries: int
    retry_delay: int
    extended_thinking: Optional[str]
```

## `reps.Optimizer`

**Implemented in:** [`reps/api/optimizer.py:42`](../reps/api/optimizer.py#L42) (class), constructor at line 52, `optimize()` at line 159, `from_config()` at line 123, kwarg → `Config` mapping at `_build_config` (line 259).
**Tests:** [`tests/test_api_optimize.py`](../tests/test_api_optimize.py) (78 cases including adversarial coverage and the inline-string-model path).

The optimizer. Constructor takes the model (or a model-name string) and
the optimization knobs; `optimize()` runs.

### Signature

```python
reps.Optimizer(
    *,
    # Model — either a built reps.Model, or a model-name string.
    model: reps.Model | str,
    api_key: Optional[str] = None,              # only used when model is a str

    # Search budget
    max_iterations: int = 100,

    # GEPA-style features (Phases 1-5)
    selection_strategy: str = "map_elites",     # "map_elites" | "pareto" | "mixed"
    pareto_fraction: float = 0.0,               # for "mixed"
    trace_reflection: bool | None = None,       # Phase 3; None auto-enables for objective=
    lineage_depth: int = 3,                     # Phase 5
    merge: bool = False,                        # Phase 4

    # Population
    num_islands: int = 5,

    # Output
    output_dir: Optional[str] = None,           # None ⇒ tempdir, no persistence

    # Inline Model construction (only used when `model` is a string).
    **model_kwargs: Unpack[ModelKwargs],
)
```

(Phase 6 `minibatch_size` was shipped then reverted — see
`docs/gepa_implementation_plan.md` "Phase 6 — reverted".)

### Behavior

- `model: str` — Optimizer constructs a `reps.Model(model, api_key=...,
  **model_kwargs)` internally. This is the most common path. Branch at
  [`reps/api/optimizer.py:80`](../reps/api/optimizer.py#L80).
- `model: reps.Model` — passing an already-built Model. In this case
  `api_key` and `**model_kwargs` MUST be empty; passing them raises
  `ValueError`. See
  [`reps/api/optimizer.py:82-89`](../reps/api/optimizer.py#L82-L89).
- Anything else — raises `TypeError`. See
  [`reps/api/optimizer.py:91-95`](../reps/api/optimizer.py#L91-L95).
- Field-level validation (`max_iterations >= 1`,
  `selection_strategy in {map_elites|pareto|mixed}`, `num_islands >= 1`)
  at [`reps/api/optimizer.py:96-104`](../reps/api/optimizer.py#L96-L104).
- Constructor → internal `Config` mapping happens lazily in
  [`_build_config`](../reps/api/optimizer.py#L259) at first
  `optimize()` call.

**Footgun to call out:** typos in the search knobs (e.g.
`max_itreations=...`) silently land in `**model_kwargs` and may pass
through to the SDK constructor where they fail with a confusing error.
Type checkers (with `Unpack[ModelKwargs]`) catch this; runtime users
without type checking won't until the SDK rejects the kwarg.

The constructor is **deliberately tight**. Everything that's not a
load-bearing GEPA knob is hard-defaulted (population_size,
archive_size, feature_dimensions, exploration_ratio, all the
F1-F8 toggles like reflection/convergence_monitor/sota_steering — these
ship enabled with REPS's existing defaults). Power users who want to
override them keep using YAML or a `Config` object directly (see
"escape hatches" below).

### `optimize()` — the single entry point

**Implemented in:** [`reps/api/optimizer.py:159`](../reps/api/optimizer.py#L159) (sync wrapper, running-loop guard at line 188), `_aoptimize_internal` at line 206 (the actual async pipeline).
**Evaluator dispatch:** [`reps/api/evaluate_dispatch.py`](../reps/api/evaluate_dispatch.py) — `register_user_evaluate` (line 67) stores the user's callable in a process-local registry; `write_shim` (line 85) writes a `_reps_user_evaluator.py` into the run dir; `dispatch_user_evaluate` (line 132) handles signature introspection (`_supported_kwargs`, line 94) and return-shape coercion (`coerce_return`, line 108).

```python
result: reps.OptimizationResult = optimizer.optimize(
    initial: str,
    evaluate: Callable[[str], float | dict | reps.EvaluationResult] | None = None,
    *,
    objective: reps.Objective | None = None,
    seed: Optional[int] = None,
)
```

Exactly one of `evaluate` or `objective` must be supplied. `objective=` is
the recommended entry point — it registers `objective.evaluate` through the
same dispatch shim. `evaluate=` is the power-user escape hatch described
next.

`evaluate` is the **single inner abstraction**. It's called with the
artifact text. Three accepted return shapes, in order of richness, all
coerced to `EvaluationResult` in
[`coerce_return`](../reps/api/evaluate_dispatch.py#L108):

1. **`float`** — just a scalar score. REPS treats it as `combined_score`
   and skips Pareto / trace-reflection / merge (no per-instance signal
   to act on).
2. **`dict`** — must include `combined_score`; may include `validity`,
   `per_instance_scores`, `feedback`, `error`. Auto-wrapped in
   `EvaluationResult` via
   [`EvaluationResult.from_dict`](../reps/evaluation_result.py#L41),
   which peels top-level `per_instance_scores` and `feedback` into
   the dedicated dataclass fields.
3. **`reps.EvaluationResult`** — full ASI surface, recommended if you
   want the Phase 1-5 features to actually fire.

That's the whole user contract: one callable in, one `OptimizationResult`
out.

### The objective layer

`reps.Objective` removes the need to hand-write `evaluate(code)`. You give it
a seed `entrypoint`, a `train_set` of `reps.Example` rows (each marked with
`.with_inputs(...)`), and a `metric`; it `exec`s each candidate, runs the
entrypoint per example, wraps the return as a `reps.Prediction`, scores it,
and returns an `EvaluationResult` in higher-is-better `combined_score` space.

- `reps.Objective.maximize(...)` / `reps.Objective.minimize(...)` — the
  classmethod picks the direction. Built-in metrics: `accuracy`,
  `exact_match` (maximize); `mae`, `mse`, `rmse` (minimize). A built-in used
  with the wrong classmethod raises `ValueError`.
- Custom metrics are callables `metric(example, pred, trace=None) -> bool |
  int | float`.
- `reps.LLMJudge(entrypoint=, train_set=, rubric=, model=, scale=)` — an
  `Objective` subclass that scores subjective outputs with an LLM judge,
  configurable independently of the mutation model. Judge calls are cached.

Passing `objective=` defaults `trace_reflection` on (an objective always
emits per-example feedback the reflection path consumes; an explicit
`trace_reflection=True/False` still wins). A raw `evaluate=` callable leaves
it off — it may emit no feedback at all.

The full contract is specified in
[`docs/objective_api_spec.md`](objective_api_spec.md).

### Escape hatches (power users)

For users who outgrow the simple constructor:

- `reps.Optimizer.from_config(cfg: reps.config.Config)` — accept the full
  internal `Config` dataclass and run it. The **only** escape
  hatch we expose in v1. Existing YAML-based configs can be loaded into
  `Config` via [`reps.runner.load_experiment_config`](../reps/runner.py),
  then passed in. Implemented at
  [`reps/api/optimizer.py:123`](../reps/api/optimizer.py#L123); uses a
  `_StubModel` ([line 408](../reps/api/optimizer.py#L408)) so the
  optimize() path has something to reference even when the LLM client
  isn't built directly through `reps.Model`.
- `reps.internal.*` re-exports the existing internals (`ReflectionEngine`,
  `WorkerPool`, etc.) — power users who already import these keep
  working. See [`reps/internal/__init__.py`](../reps/internal/__init__.py).

`from_config` is the safety valve: if a user needs a knob the simple
constructor doesn't expose, they drop down to `Config` rather than
asking us to grow the constructor. Keeps the v1 surface honest without
locking advanced users out.

## `reps.OptimizationResult`

**Implemented in:** [`reps/api/result.py:14`](../reps/api/result.py#L14).
**Built by:** `_collect_result` at [`reps/api/optimizer.py:325`](../reps/api/optimizer.py#L325).

```python
@dataclass
class OptimizationResult:
    best_code: str
    best_score: float
    best_metrics: Dict[str, float]
    best_per_instance_scores: Optional[Dict[str, float]]
    best_feedback: Optional[str]
    iterations_run: int
    total_metric_calls: int
    total_tokens: Dict[str, int]                    # {"in": N, "out": M}
    output_dir: Optional[str]                       # if persisted
```

That's all v1 needs. No `save`/`load`, no `as_callable()`, no `history`
list — those land in v1.5 if asked for.

## `reps.EvaluationResult`

**Implemented in:** [`reps/evaluation_result.py:11`](../reps/evaluation_result.py#L11).
**`from_dict` ASI peeling:** [`reps/evaluation_result.py:41`](../reps/evaluation_result.py#L41) — promotes top-level `per_instance_scores` and `feedback` keys into the dataclass fields without mutating the input dict.

Already exists from GEPA Phase 1.1. Re-exported at top level for
convenience.

```python
@dataclass
class EvaluationResult:
    metrics: Dict[str, float]                       # must include combined_score
    artifacts: Dict[str, Union[str, bytes]] = ...
    per_instance_scores: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None
```

## Worked examples

### Minimum viable

```python
import reps
import subprocess

def evaluate(code: str) -> float:
    proc = subprocess.run(["python", "-c", code], capture_output=True)
    return 1.0 if proc.returncode == 0 else 0.0

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4.6",       # api_key from $ANTHROPIC_API_KEY
    max_iterations=20,
).optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)
print(result.best_code, result.best_score)
```

15 lines. Three public symbols touched: `reps.Optimizer`, `optimize`,
`OptimizationResult` (its fields). No `reps.Model` instantiation needed
in the common case.

### Reusable Model

When you want to call the model directly (not just via the optimizer),
or to share one Model across multiple optimizer runs:

```python
import reps

model = reps.Model("anthropic/claude-sonnet-4.6", temperature=0.7)
print(model("hello"))                          # standalone callable

result = reps.Optimizer(model=model, max_iterations=20).optimize(...)
```

### GEPA-style (all features on)

```python
import reps

def evaluate(code: str) -> reps.EvaluationResult:
    # Custom evaluator emitting full ASI (per-objective scores + feedback).
    ...

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4.6",
    temperature=0.7,                            # any reps.ModelKwargs
    extended_thinking="high",
    max_iterations=200,
    selection_strategy="mixed",
    pareto_fraction=0.3,
    trace_reflection=True,
    lineage_depth=3,
    merge=True,
    minibatch_size=2,
    num_islands=4,
    output_dir="./runs/circle_packing",
).optimize(
    initial=open("initial_program.py").read(),
    evaluate=evaluate,
)
```

Same shape, more knobs.

### Escape hatch — full Config

```python
import reps
from reps.config import Config
from reps.runner import load_experiment_config

cfg: Config = load_experiment_config("path/to/run.yaml")
# tweak cfg here if you want
result = reps.Optimizer.from_config(cfg).optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)
```

For when the simple constructor doesn't have the knob you need.

## Implementation phases

### A — `reps.Model` ✅ Shipped

**Commits:** `3ef3bb4` (initial impl), `df77b02` (provider_kwargs forwarding fix + from_dict ASI peeling), `ab2517f` (LM → Model rename + ModelKwargs + Optimizer accepts string).

Pure shim over existing `reps.llm.*`. ~210 LOC of class + ~50 LOC of helpers + tests with mocked provider clients. No internal changes.

Files:
- [`reps/api/__init__.py`](../reps/api/__init__.py) — re-exports.
- [`reps/api/model.py`](../reps/api/model.py) — `Model` class (line 75) + `ModelKwargs` TypedDict (line 33).
- [`reps/__init__.py`](../reps/__init__.py) — top-level re-export of `Model`, `ModelKwargs`.
- [`tests/test_api_model.py`](../tests/test_api_model.py) — 38 cases covering provider routing, env-var fallback, kwarg propagation, sync wiring, running-loop guard, provider_kwargs forwarding to the SDK client, and `_to_model_config()` independent-copy semantics.

### B — `reps.Optimizer` + `optimize()` ✅ Shipped

**Commits:** `aaae09a` (initial impl), `9121c7b` (adversarial test coverage), `df77b02` (review-flagged fixes), `ab2517f` (Optimizer signature: `model: Model | str` + `**ModelKwargs`), `642ab15` (REPS → Optimizer rename).

Builds the constructor + the single entry point. Internally constructs
a `Config` from the kwargs, instantiates a controller + database +
evaluator, runs the existing async event loop, and packages the result.
Controller and internals are unchanged — this is purely the new entry
path.

Includes:
- Signature introspection on the user's `evaluate` callable at
  [`reps/api/evaluate_dispatch.py:94`](../reps/api/evaluate_dispatch.py#L94)
  (`_supported_kwargs`). The v1 contract is `Callable[[str], ...]`
  so we accept `evaluate(code)` and `evaluate(code, *, env=None)` —
  auto-detected so legacy benchmark evaluators work.
- Return-shape coercion at
  [`reps/api/evaluate_dispatch.py:108`](../reps/api/evaluate_dispatch.py#L108)
  (`coerce_return`): `float` → `dict` → `EvaluationResult`.
- Process-local registry + shim file pattern for the user's evaluator:
  [`register_user_evaluate`](../reps/api/evaluate_dispatch.py#L67),
  [`unregister_user_evaluate`](../reps/api/evaluate_dispatch.py#L79),
  [`write_shim`](../reps/api/evaluate_dispatch.py#L85),
  [`dispatch_user_evaluate`](../reps/api/evaluate_dispatch.py#L132).
- `from_config(cfg)` classmethod escape hatch at
  [`reps/api/optimizer.py:123`](../reps/api/optimizer.py#L123).

Files:
- [`reps/api/optimizer.py`](../reps/api/optimizer.py) — `Optimizer` class (line 42) + `optimize()` (line 159) + `_aoptimize_internal` (line 206) + `_build_config` (line 259) + `_collect_result` (line 325) + `_StubModel` (line 408).
- [`reps/api/result.py`](../reps/api/result.py) — `OptimizationResult` (line 14).
- [`reps/api/evaluate_dispatch.py`](../reps/api/evaluate_dispatch.py) — signature introspection + return-shape coercion + the registry/shim mechanism.
- [`reps/internal/__init__.py`](../reps/internal/__init__.py) — re-exports current internals (`ReflectionEngine`, `WorkerPool`, `ConvergenceMonitor`, etc.) so users who already import them keep working.
- [`tests/test_api_optimize.py`](../tests/test_api_optimize.py) — 78 cases including end-to-end with a mock LLM, the inline-string-model path, registry concurrency, env-var collision check, `from_config` round-trip, and post-construction validation.
- [`tests/test_evaluation_contract.py`](../tests/test_evaluation_contract.py) — `EvaluationResult.from_dict` ASI peeling tests.

That's the whole v1. Two phases.

## Open design decisions (v1 only)

1. **Sync-wrapper safety.** `optimize()` is sync but internals are async.
   `asyncio.run()` raises if a loop is already running. Recommendation:
   detect a running loop and raise a clear error pointing the user to
   wait for v1.5's `aoptimize()` (or to `asyncio.run` themselves on a
   new loop). Don't silently nest.

2. **Provider-string inference.** `reps.Model("claude-sonnet-4.6")` (no
   provider prefix) — allow it via `provider_of_model()`, but fail
   loudly with a helpful error if the model name is ambiguous between
   providers.

3. **`from_config` strictness.** Should `Optimizer.from_config(cfg)` accept
   a partial / under-specified `Config` (filling in defaults)? Or
   require fully-formed `Config`? Recommendation: accept partial — the
   `Config` dataclass has defaults for everything; users shouldn't need
   to know which fields are required.

## Deferred to v1.5+

This is the audit catch list — things in earlier drafts of this spec
that we're cutting from v1 to keep the surface lean. They land in v1.5
if real users ask. Keeping them visible here so we don't forget.

Optimizer surface:
- `aoptimize()` async variant.
- `compile(initial, metric, trainset, runner=, valset=)` — the dspy
  interop shim. Skipped entirely until users with existing dspy
  metrics show up.
- `optimize_path(initial_program=, evaluator=)` — file-loading
  convenience. Users can read files themselves in 2 lines.
- `OptimizationResult.save()` / `.load()`.
- `OptimizationResult.history` — per-iteration trajectory list.
- `OptimizationResult.as_callable()` — returns an executable wrapper
  around the best code.
- Callbacks (`Callback` Protocol, `ProgressBar`, `JSONLogger`, Wandb,
  MLflow). For v1, users poll the result after the fact.

LLM surface:
- `LM.ensemble([...])` — multi-model dispatch.
- `LM.agenerate()` / `LM.agenerate_with_context()` — async methods.
- `reps.configure(lm=...)` — process-wide defaults.

Constructor surface (Optimizer):
- `reflection_lm`, `summarizer_lm`, `trace_reflection_lm` — separate
  LLMs for individual roles. v1 uses `lm` for all of them.
- `workers=[...]` — explicit worker config. v1 uses REPS's existing
  default worker pool.
- `max_metric_calls`, `target_score`, `early_stopping_patience` —
  alternative budgets and early-stop knobs.
- `pareto_instance_keys` — restrict Pareto frontier to a subset of
  instance keys.
- `population_size`, `archive_size`, `feature_dimensions`,
  `exploration_ratio`, `exploitation_ratio` — search-tuning knobs.
  v1 uses defaults; power users go via `from_config`.
- `reflection`, `revisitation`, `convergence_monitor`, `contracts`,
  `sota_steering` — per-feature toggles. v1 ships with all on
  (matching today's defaults).
- `save_checkpoints`, `checkpoint_interval`, `log_level` — operational
  knobs. v1 uses sane defaults.
- `adapter=` (Phase 7 of the GEPA plan) — accepted but ignored in v1.

Evaluator contract:
- `evaluate(program_path: str, ...)` legacy file-path shim. v1 assumes
  `evaluate(code: str, ...)` (with optional `env` keyword). Users with
  file-path-based evaluators wrap them in 2 lines.
- `instances=[...]` kwarg for minibatch evaluation. Was shipped as GEPA
  Phase 6 then reverted; cascade evaluation covers the same use case
  without coupling the harness to an instance registry. See
  `docs/gepa_implementation_plan.md` "Phase 6 — reverted".

`reps.Optimizer.from_config_dict(dict)` — YAML-dict bridge. v1 has
`from_config(Config)` only; the dict version is just `Config(**dict)`.

## Append-only changelog

| Date       | Phase  | Commit     | Notes |
|------------|--------|------------|-------|
| 2026-05-04 | A      | `3ef3bb4`  | `reps.LM` initial impl (renamed to `Model` later) + 21 tests |
| 2026-05-04 | B      | `aaae09a`  | `reps.REPS + optimize() + OptimizationResult` (renamed to `Optimizer` later) + `evaluate_dispatch.py` registry/shim mechanism + 29 tests |
| 2026-05-04 | B+     | `9121c7b`  | Adversarial test coverage from test subagent: +62 tests across LM error paths, constructor wiring, dispatch edge cases, registry concurrency, env-var collision, `from_config` round-trip, internal back-compat |
| 2026-05-04 | A+B    | `df77b02`  | Review-flagged fixes: `provider_kwargs` actually forwarded to the underlying SDK client; `EvaluationResult.from_dict` peels top-level `per_instance_scores` and `feedback` into the dataclass fields |
| 2026-05-04 | rename | `642ab15`  | `reps.REPS` → `reps.Optimizer` (the package and class shared a name) |
| 2026-05-04 | rename | `ab2517f`  | `reps.LM` → `reps.Model` + new `reps.ModelKwargs` TypedDict + `Optimizer(model: Model \| str, api_key=, **ModelKwargs)` signature so the common case ("just give me a model name and an api key") avoids building a Model first |
| 2026-05-14 | C      | (this PR)  | Objective layer: `reps.Example`, `reps.Prediction`, `reps.Objective` (maximize/minimize + built-in metric registry), `reps.LLMJudge`; `optimize()` gains `objective=` (mutually exclusive with `evaluate=`). See `docs/objective_api_spec.md`. |
