# Optimizer / Engine Separation Spec

## Why this doc exists

REPS-the-public-package and REPS-the-internal-codebase have been
collapsed into one mental model, and that's been confusing every
conversation about it. This spec separates them, defines vocabulary,
answers the conceptual questions, and proposes a concrete boundary in
the code that we can implement incrementally.

This is a **design document**, not implementation. No code changes
land from this doc; once the design is agreed, we phase the migration.

## The conceptual questions, answered first

Before architecture: pin the words and the recursion concerns.

### "Should I be able to optimize a harness?"

Yes — and the recursion is fine because there are TWO different
harnesses involved, and they don't have to be the same code.

- **The outer harness** is the REPS package running on your machine.
  It implements the search loop, talks to LLMs, persists state.
- **The artifact under optimization** can be any text. If that text
  happens to be the source of *another* search loop (e.g., your own
  custom harness), REPS-the-outer optimizes it the same way it would
  optimize a math solver. There's no special "harness mode".

So: REPS optimizes programs. A program can be a harness. That's not a
contradiction — it's just that "program" is text and a harness is
text-shaped too. The outer REPS doesn't care what the inner program
does, only what the evaluator scores.

### "Does `optimize` taking a string mean it should just be runnable Python?"

The string can be **anything the evaluator knows how to score**:

- Python source — the common case.
- A system prompt — if the evaluator runs the prompt against a
  benchmark and grades the model's output.
- A JSON config — if the evaluator instantiates a system from it and
  measures.
- Multi-file projects (less elegant but workable — concatenate with
  delimiters, parse in the evaluator).

REPS doesn't impose any DSL. The artifact contract is just `str`. The
LLM-side prompts default to "rewrite the whole thing" mode for the
Python API; the EVOLVE-BLOCK markers exist only in the CLI/YAML mode
for benchmark authors who want to fix scaffolding and only let the
LLM edit a region.

### "Should there be more structure from agentic orchestration?"

Not in the public optimizer API. If your problem benefits from
multi-step agentic decomposition, that lives **inside the evaluator**
or **inside the prompt template**. The optimizer doesn't need to know.

If we ever do add structure, the right place is the (deferred)
`reps.Adapter` from the GEPA Phase 7 sketch — adapters can render
artifacts differently in prompts and parse mutations differently. But
that's still optional: the default adapter is "string in, string out".

### "Is there a simple non-DSL approach?"

We already have it. `evaluate(code: str) -> float`. No markup, no
schema, no interpreter. The text is whatever you want; the score is
whatever you decide.

### "Are we overloading by having both controller/harness AND optimize functionality?"

Yes, in the *code* (not the public API). Today:

- The `reps.Optimizer` public class is a thin facade.
- Underneath, `controller.py` (1800 lines) does both algorithm work
  (parent selection, acceptance, archive maintenance) AND I/O
  (calling evaluators, dispatching LLMs, coordinating workers).

The public surface is fine — `Optimizer.optimize(initial, evaluate)`
is one outer call, one inner abstraction. The internal tangle is the
problem this doc proposes to fix.

### "Is the harness the optimizer itself, in runnable Python?"

The harness is *a Python program* that *implements* an optimizer.
"Optimizer" is the abstract concept (an algorithm); "harness" is the
concrete machinery (the Python code that runs the algorithm). We've
been using both words for both things — that's where the confusion
came from.

After this spec, we use precise vocabulary:

| Word | Meaning |
|---|---|
| **Optimizer** | The search *algorithm* — selection, mutation, acceptance, archive. Pure decisions on `Program` data. |
| **Engine** | The *runtime* — calls LLMs, runs evaluators, persists state, dispatches workers. I/O effects. |
| **Harness** | The full Python package = Optimizer + Engine + glue. Avoid this word in the code; use "optimizer" or "engine" for precision. |
| **Artifact** | The text being optimized. Pure data. |
| **Evaluator** | User-supplied callable: artifact → score. Pure (from the optimizer's POV). |

Going forward, "harness" means "the whole REPS package" only. Inside
the code, we say "optimizer" (the algorithm) or "engine" (the I/O
layer).

## What's tangled today

Mapping today's modules to the cleaner layers:

| Today's file | Algorithm logic | Engine I/O |
|---|---|---|
| `reps/controller.py` (1831 lines) | parent selection routing, accept/reject, REPS feature gates (reflection, convergence, SOTA, merge, trace_reflection), revisitation | LLM dispatch, evaluator dispatch, worker pool management, batch coordination, output persistence |
| `reps/database.py` (2800 lines) | MAP-Elites archive, Pareto sampling, island migration, fitness ranking | program JSON read/write, artifact storage, lock-free state |
| `reps/evaluator.py` | (none) | calling user evaluator, cascade staging |
| `reps/workers/*.py` | per-worker prompt strategy hints | LLM call dispatch, diff parsing, tool runners |
| `reps/pareto.py` | (pure algorithm — frontier computation, complementary pair selection) | (none) |
| `reps/reflection_engine.py` | (none) | LLM call to summarize batch |
| `reps/convergence_monitor.py` | edit-entropy thresholds, escalation | (none — pure decisions) |
| `reps/sota_controller.py` | regime selection by gap-to-target | (none — pure decisions) |
| `reps/worker_pool.py` | weighted sampling over worker configs | (none) |
| `reps/contract_selector.py` | Thompson-sampling bandit | (none) |

`pareto.py`, `convergence_monitor.py`, `sota_controller.py`,
`worker_pool.py`, `contract_selector.py` are **already pure
algorithm modules**. The tangle is mostly in `controller.py`.

## The proposed boundary

Two protocols + one orchestrator + the existing typed data classes.

### `Algorithm` protocol — pure decisions

```python
class Algorithm(Protocol):
    """Owns the search algorithm. No I/O. Operates on Program data."""

    def select_parent(
        self,
        db: ProgramDatabase,
        iteration: int,
        island_id: int,
    ) -> Program: ...

    def select_inspirations(
        self,
        db: ProgramDatabase,
        parent: Program,
        n: int,
    ) -> List[Program]: ...

    def accept(self, child: Program, parent: Program) -> bool: ...

    def on_child_added(self, child: Program, db: ProgramDatabase) -> None:
        """Update internal state (convergence trackers, SOTA regime,
        bandit posteriors, etc.) after a child enters the database."""
        ...

    def on_batch_complete(
        self,
        recent: List[Program],
        db: ProgramDatabase,
    ) -> Dict[str, str]:
        """Optional batch-level work (reflection summaries,
        dead-end warnings, regime updates). Returns prompt_extras
        the engine should inject into the next batch's prompts."""
        ...
```

Today's REPS algorithm becomes one concrete implementation:
`EvolutionaryAlgorithm` — MAP-Elites + islands + Pareto + workers +
revisitation + convergence escalation + SOTA steering. Drop in others
later (e.g. random search, beam search, plain hill climbing) without
changing the engine.

### `Engine` protocol — pure effects

```python
class Engine(Protocol):
    """Owns I/O. No algorithm decisions. Talks to LLMs, evaluators,
    persistence. The algorithm asks for effects via this protocol."""

    async def propose(
        self,
        parent: Program,
        inspirations: List[Program],
        prompt_extras: Dict[str, str],
        worker_config: WorkerConfig,
    ) -> WorkerResult:
        """Generate one candidate child given a parent + context."""
        ...

    async def evaluate(self, code: str) -> EvaluationResult:
        """Run the user's evaluator on artifact text."""
        ...

    def persist(self, child: Program) -> None:
        """Add to database, save to disk if configured."""
        ...
```

Today's REPS engine becomes one concrete implementation:
`LLMWorkerEngine` — calls workers, runs `Evaluator`, writes to
`ProgramDatabase`. Drop in others later (e.g. `LocalEngine` for
testing without an LLM, `ModalEngine` for distributed evaluation).

### The orchestrator

```python
class Optimizer:
    """Wires an Algorithm and an Engine together; runs the loop."""

    def __init__(self, algorithm: Algorithm, engine: Engine, db: ProgramDatabase):
        self.algorithm = algorithm
        self.engine = engine
        self.db = db

    async def run(self, initial: str, max_iterations: int) -> OptimizationResult:
        # Seed the database
        seed_result = await self.engine.evaluate(initial)
        seed = Program(id="initial", code=initial, metrics=seed_result.metrics, ...)
        self.db.add(seed)

        # Main loop
        for i in range(max_iterations):
            # Algorithm decides who, engine produces what
            island = i % self.db.num_islands
            parent = self.algorithm.select_parent(self.db, i, island)
            inspirations = self.algorithm.select_inspirations(self.db, parent, n=5)
            extras = self.algorithm.on_batch_complete(...) if i % batch_size == 0 else {}

            # Engine produces a candidate
            worker_cfg = self.algorithm.pick_worker_config(self.db, i)
            result = await self.engine.propose(parent, inspirations, extras, worker_cfg)
            child_result = await self.engine.evaluate(result.child_code)
            child = Program(code=result.child_code, ..., metrics=child_result.metrics)

            # Algorithm decides accept/reject
            if self.algorithm.accept(child, parent):
                self.engine.persist(child)
                self.algorithm.on_child_added(child, self.db)

        return self._collect_result()
```

That's the whole loop. ~30 lines of orchestrator code. Today's
`controller.run_evolution` (~700 lines) collapses into this once the
algorithm and engine pieces live elsewhere.

## What this buys us

1. **Testability.** `Algorithm` is pure — test with a fake database
   and synthetic Programs. No LLM mocks, no temp dirs. Today many
   algorithm tests need a full controller setup.

2. **Swapability.** Want to try a different search algorithm with the
   same engine? Implement `Algorithm`, plug it in. Want to run on
   Modal instead of in-process? Implement `Engine` (the executor
   pattern from the previous discussion lives here naturally).

3. **Distribution.** With Engine factored out, the algorithm can run
   on one machine and the engine can shell out to many. Today's
   tangle makes this hard.

4. **Clearer public API story.** `reps.Optimizer(model=..., ...)` is
   sugar for "build a REPS-flavored EvolutionaryAlgorithm + an
   LLMWorkerEngine and run them". Power users can construct
   algorithm + engine separately for non-default combinations.

5. **Documentation.** "What does REPS do at iteration N?" becomes
   readable in 30 lines instead of grepping across 1800.

## What this does NOT change

- Public API surface. `reps.Optimizer.optimize(initial, evaluate)`
  works exactly the same; it's still the canonical entry point.
- The data types. `EvaluationResult`, `Program`, `OptimizationResult`
  stay where they are.
- The existing GEPA features (Phases 1-5). They get reorganized into
  algorithm or engine but the *behavior* is identical.
- The CLI. `reps-run` still works; it just constructs the same
  algorithm+engine pair under the hood.

## Migration phases

Each phase is independently committable, ships behind a feature flag
or via additive APIs (no public surface breakage), and ends with the
same `reps.Optimizer.optimize` call producing identical behavior.

### Phase S1 — Carve out `Algorithm`

Files:
- New `reps/algorithm/__init__.py` exposing `Algorithm` Protocol.
- New `reps/algorithm/evolutionary.py` — `EvolutionaryAlgorithm`
  class consolidating today's algorithm logic from `controller.py`.
- `controller.py` keeps everything; `EvolutionaryAlgorithm` is
  initially a *delegate* — its methods call back into the controller.

The seam is in place; behavior unchanged. ~200 LOC added, 0 deleted.

### Phase S2 — Carve out `Engine`

Files:
- New `reps/engine/__init__.py` exposing `Engine` Protocol.
- New `reps/engine/llm_worker.py` — `LLMWorkerEngine` consolidating
  today's I/O logic from `controller.py` + `evaluator.py`.
- Same delegate pattern: engine initially calls controller methods.

Seam in place; behavior unchanged. ~300 LOC added, 0 deleted.

### Phase S3 — Move logic across the seam

Now the actual untangling. Move the algorithm code from
`controller.py` into `EvolutionaryAlgorithm`; move I/O code into
`LLMWorkerEngine`. After this phase, `controller.py` is just the
loop (or is renamed `Optimizer`/`Orchestrator`).

Risk: this is the riskiest phase because it touches every existing
behavior. Run the full test suite (388 tests) at every step.

Estimated diff: -1500 / +1500 LOC (net wash; just relocation).

### Phase S4 — Reduce `controller.py` to the orchestrator

Once everything has moved, `controller.py` becomes the ~30-line loop
shown above. Delete the now-empty methods. Rename to
`reps/orchestrator.py` for clarity (the word "controller" was always
ambiguous).

### Phase S5 — Document the boundary publicly

Update `docs/python_api_spec.md` with an "internals architecture"
section pointing at `reps.algorithm.*` and `reps.engine.*` for power
users. The public `reps.Optimizer` surface stays unchanged.

Optionally: expose `algorithm` and `engine` as constructor kwargs on
`reps.Optimizer` so power users can swap them. Default to
`EvolutionaryAlgorithm` + `LLMWorkerEngine`.

## Open questions for sign-off

1. **Naming.** `Optimizer` vs `Orchestrator` vs `Loop` for the
   30-line glue class. Recommendation: keep the public class
   `reps.Optimizer` as today; rename the internal `controller.py` to
   `orchestrator.py` to signal it's just the loop.

2. **State ownership.** Today the database holds a lot of state
   (population, archive, islands). After the split, does the
   *algorithm* own the archive (because it's an algorithmic concept)
   or does the *engine* own it (because it persists)? Recommendation:
   the database stays as a shared dataclass passed to both — neither
   owns it, both read/write through it. Today's pattern.

3. **Algorithm batch operations.** `on_batch_complete` returns
   `prompt_extras` that the engine injects into the next batch's
   prompts. That's a tight coupling — the algorithm has opinions
   about the engine's prompts. Is the protocol shape right?
   Alternative: algorithm publishes events; engine subscribes and
   decides what to do with them.

4. **GEPA Phase 7 (Adapter) interaction.** The Adapter sits *inside
   the Engine* — it's how the engine renders artifacts in prompts and
   parses mutations. Confirm: Algorithm doesn't see Adapters.

5. **Async boundaries.** Algorithm methods sync (pure decisions);
   engine methods async (I/O). The orchestrator is async. Confirm
   this lines up with actual today's codepaths — if any algorithm
   decision today secretly does I/O, that's a bug to fix in S3.

6. **Tests.** Phase S3 is the risky one. Should we keep the existing
   integration tests as ground truth and write per-protocol unit
   tests in S1/S2 to lock down behavior before relocating in S3?
   Recommendation: yes — add property tests for the protocols first,
   then move code with confidence.

## Append-only changelog

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| —    | —     | spec only | (no implementation yet) |
