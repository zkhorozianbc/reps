# REPS Python API Spec

## Goals

Expose REPS as a Python package with the smallest possible API:

```python
import reps

def evaluate(code: str) -> float:
    ...  # run the artifact, return a score

result = reps.Optimizer(
    lm=reps.LM("anthropic/claude-sonnet-4.6"),
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
callable). Configure the LM, set the knobs, hand over an evaluator,
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
  for the LLM wrapper but follow `optimize_anything`'s shape for the
  optimizer.
- Not replacing the existing `reps-run` CLI or YAML configs. The Python
  API is the right entry point for notebooks, libraries, and CI; the
  CLI stays for batch experiments.

## v1 surface

Everything below is shipped in v1. Everything not listed is deferred to
v1.5 (see end of doc).

| Symbol | Purpose |
|---|---|
| `reps.LM` | LLM wrapper, similar shape to `dspy.LM` |
| `reps.Optimizer` | The optimizer |
| `reps.Optimizer.optimize(initial, evaluate)` | The single entry point |
| `reps.OptimizationResult` | What `optimize()` returns |
| `reps.EvaluationResult` | Optional rich return shape from the evaluator (already exists) |

That's it. Four classes, one method on the optimizer. Internals are
unchanged — Pareto, trace reflection, merge, lineage, convergence
monitor, SOTA controller, all of it stays as it is in `reps/`. The
public API is just a thin facade.

## `reps.LM`

Thin facade over the existing `reps.llm.*` providers. Sync-only in v1
(async wrappers come in v1.5).

### Signature

```python
reps.LM(
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
  via existing `reps.llm.provider_of.provider_of_model`).
- `api_key=None` ⇒ provider env-var fallback (`ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `OPENROUTER_API_KEY`). Fail loudly at construction
  if unresolved.
- `extended_thinking` is the existing `reasoning` knob.
- `**provider_kwargs` passed verbatim to the underlying client.

### Methods

```python
text = lm("hello")            # __call__ — sync shortcut
text = lm.generate("hello")   # explicit sync
```

That's it for v1. Async (`agenerate`, `agenerate_with_context`),
ensembles (`LM.ensemble`), and `reps.configure(lm=...)` global defaults
all defer to v1.5.

## `reps.Optimizer`

The optimizer. Constructor takes the LM and the optimization knobs;
`optimize()` runs.

### Signature

```python
reps.Optimizer(
    *,
    # LLM
    lm: reps.LM,

    # Search budget
    max_iterations: int = 100,

    # GEPA-style features (Phases 1-5)
    selection_strategy: str = "map_elites",     # "map_elites" | "pareto" | "mixed"
    pareto_fraction: float = 0.0,               # for "mixed"
    trace_reflection: bool = False,             # Phase 3
    lineage_depth: int = 3,                     # Phase 5
    merge: bool = False,                        # Phase 4
    minibatch_size: Optional[int] = None,       # Phase 6 (placeholder until shipped)

    # Population
    num_islands: int = 5,

    # Output
    output_dir: Optional[str] = None,           # None ⇒ tempdir, no persistence
)
```

The constructor is **deliberately tight**. Everything that's not a
load-bearing GEPA knob is hard-defaulted (population_size,
archive_size, feature_dimensions, exploration_ratio, all the
F1-F8 toggles like reflection/convergence_monitor/sota_steering — these
ship enabled with REPS's existing defaults). Power users who want to
override them keep using YAML or a `Config` object directly (see
"escape hatches" below).

### `optimize()` — the single entry point

```python
result: reps.OptimizationResult = optimizer.optimize(
    initial: str,
    evaluate: Callable[[str], float | dict | reps.EvaluationResult],
    *,
    seed: Optional[int] = None,
)
```

`evaluate` is the **single inner abstraction**. It's called with the
artifact text. Three accepted return shapes, in order of richness:

1. **`float`** — just a scalar score. REPS treats it as `combined_score`
   and skips Pareto / trace-reflection / merge (no per-instance signal
   to act on).
2. **`dict`** — must include `combined_score`; may include `validity`,
   `per_instance_scores`, `feedback`, `error`. Auto-wrapped in
   `EvaluationResult`.
3. **`reps.EvaluationResult`** — full ASI surface, recommended if you
   want the Phase 1-5 features to actually fire.

That's the whole user contract: one callable in, one `OptimizationResult`
out.

### Escape hatches (power users)

For users who outgrow the simple constructor:

- `reps.Optimizer.from_config(cfg: reps.config.Config)` — accept the full
  internal `Config` dataclass and run it. This is the **only** escape
  hatch we expose in v1. Existing YAML-based configs can be loaded into
  `Config` via the existing `load_experiment_config` helper, then
  passed in.
- `reps.internal.*` re-exports the existing internals (`ReflectionEngine`,
  `WorkerPool`, etc.) — power users who already import these keep
  working.

`from_config` is the safety valve: if a user needs a knob the simple
constructor doesn't expose, they drop down to `Config` rather than
asking us to grow the constructor. Keeps the v1 surface honest without
locking advanced users out.

## `reps.OptimizationResult`

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

Already exists in `reps/evaluation_result.py`. Re-exported at top level
for convenience. Field shape unchanged from Phase 1.1:

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
    lm=reps.LM("anthropic/claude-sonnet-4.6"),
    max_iterations=20,
).optimize(
    initial=open("seed.py").read(),
    evaluate=evaluate,
)
print(result.best_code, result.best_score)
```

15 lines. Four public symbols touched: `reps.LM`, `reps.Optimizer`,
`optimize`, `OptimizationResult` (its fields).

### GEPA-style (all features on)

```python
import reps

def evaluate(code: str) -> reps.EvaluationResult:
    # Custom evaluator emitting full ASI (per-objective scores + feedback).
    ...

result = reps.Optimizer(
    lm=reps.LM("anthropic/claude-sonnet-4.6"),
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

### A — `reps.LM`

Pure shim over existing `reps.llm.*`. ~150 LOC + tests with mocked
provider clients. No internal changes.

Files:
- New `reps/api/__init__.py` — re-exports.
- New `reps/api/lm.py` — `LM` class.
- Update `reps/__init__.py` — re-export `LM`.
- Tests `tests/test_api_lm.py`.

### B — `reps.Optimizer` + `optimize()`

Builds the constructor + the single entry point. Internally constructs
a `Config` from the kwargs, instantiates a controller + database +
evaluator, runs the existing async event loop, and packages the result.
Controller and internals are unchanged — this is purely the new entry
path.

Includes:
- Signature introspection on the user's `evaluate` callable: the v1
  contract is `Callable[[str], ...]` so we accept `evaluate(code)`,
  `evaluate(code, *, env=None)`, and `evaluate(code, *, env=None,
  instances=None)` — auto-detected so legacy benchmark evaluators work.
- Return-shape coercion: `float` → wrapped in `dict` →
  `EvaluationResult`.
- `from_config(cfg)` classmethod escape hatch.

Files:
- New `reps/api/optimizer.py` — `Optimizer` class + `optimize()`.
- New `reps/api/result.py` — `OptimizationResult`.
- New `reps/api/evaluate_dispatch.py` — signature introspection +
  return-shape coercion.
- Update `reps/__init__.py` — re-export `Optimizer`, `OptimizationResult`,
  `EvaluationResult`.
- New `reps/internal/__init__.py` — re-exports current internals
  (`ReflectionEngine`, `WorkerPool`, `ConvergenceMonitor`, etc.) so
  users who already import them keep working.
- Tests `tests/test_api_optimize.py` — end-to-end with a mock LLM and
  a trivial in-memory evaluator.

That's the whole v1. Two phases, two commits.

## Open design decisions (v1 only)

1. **Sync-wrapper safety.** `optimize()` is sync but internals are async.
   `asyncio.run()` raises if a loop is already running. Recommendation:
   detect a running loop and raise a clear error pointing the user to
   wait for v1.5's `aoptimize()` (or to `asyncio.run` themselves on a
   new loop). Don't silently nest.

2. **Provider-string inference.** `reps.LM("claude-sonnet-4.6")` (no
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
  `evaluate(code: str, ...)` (with optional `env` and `instances`
  keywords). Users with file-path-based evaluators wrap them in 2 lines.

`reps.Optimizer.from_config_dict(dict)` — YAML-dict bridge. v1 has
`from_config(Config)` only; the dict version is just `Config(**dict)`.

## Append-only changelog

| Date | Phase | Commit | Notes |
|------|-------|--------|-------|
| —    | —     | —      | (spec only — no implementation yet) |
