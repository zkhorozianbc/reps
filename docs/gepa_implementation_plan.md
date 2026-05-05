# GEPA-Inspired Improvements to REPS — Implementation Plan

## Context

REPS already had a strong evolutionary-search loop (MAP-Elites + islands,
worker diversity, convergence detection, SOTA steering). After comparing
its design to [GEPA](https://github.com/gepa-ai/gepa), we identified six
GEPA-style ideas that REPS could borrow:

1. **Per-instance scores** + free-form **ASI feedback** in the evaluator
   contract, instead of a scalar `combined_score`.
2. **Pareto-frontier selection** over per-instance scores, alongside
   MAP-Elites.
3. **Trace-grounded reflection**: a per-candidate LLM call that produces
   a concrete mutation directive from the parent's specific failures.
4. **System-aware merge**: pick crossover partners whose strengths
   complement the primary's weaknesses on disjoint instance dimensions.
5. **Ancestry-aware reflection**: extend (3) with the parent's recent
   lineage so the LLM sees what's been tried.
6. **Minibatch evaluation with promotion**: run evaluators on a subset of
   instances first; full-eval only on candidates that clear a threshold.

A seventh, deferred idea — extracting an adapter interface so REPS can
optimize non-Python artifacts — is sketched at the end.

The work is staged so each phase can be live-tested and reviewed before
the next begins. Existing REPS strengths (convergence monitor, SOTA
controller, worker pool, contract bandit) are preserved; nothing is
removed.

## Working conventions

- **Branch.** All work is on `claude/compare-gepa-design-Kmi8Y`.
- **Per-phase gate.** A phase isn't done until: unit tests pass, a focused
  live or integration test exercises the new path, a subagent reviews the
  diff (Explore for code audits, Plan for architectural questions, the
  `simplify` skill for duplication checks), and the work is committed.
- **Default-off.** Every new behavior ships behind a config flag with
  status-quo defaults. Existing benchmarks and configs keep working
  untouched.
- **Live tests deferred.** The plan called for short LLM runs at several
  gates; in practice we substituted end-to-end tests with mock LLMs to
  conserve API budget, leaving real-LLM validation for the user when
  enabling the features.
- **Two-commit phases.** When a phase has both a pure module and an
  integration step, we ship them as separate commits (`X.1` pure, `X.2`
  wiring) so a reviewer can read each layer independently.

## Status

| Phase | Title                                  | Status     | Commits |
|------:|----------------------------------------|------------|---------|
|     1 | Per-instance scores + ASI feedback     | ✅ Shipped | `102952e`, `eac9f83` |
|     2 | Pareto frontier sampler                | ✅ Shipped | `5dea23b`, `7e73e61` |
|     3 | Trace-grounded reflection              | ✅ Shipped | `c9bda73`, `1fdacfe` |
|     4 | System-aware merge                     | ✅ Shipped | `f89fb8f`, `b1db148` |
|     5 | Ancestry-aware reflection              | ✅ Shipped | `2420e69` |
|     6 | Minibatch evaluation with promotion    | ❌ Reverted | shipped + reverted — see below |
|     7 | Adapter pattern refactor (deferred)    | ⬜ Sketched | — |

### Phase 6 — reverted

Phase 6 shipped (commits `31e1482`, `877d57d`, `8441b72`, `2bed1a1`,
`13f6bf5`) and was then reverted in a single commit on review of the
public-API contract. The reasons:

- **Polluted the user-facing evaluator contract.** REPS's `Optimizer`
  is supposed to be a generic optimizer — `evaluate(code) -> score`,
  any callable, any logic. Phase 6 required benchmarks to expose an
  `INSTANCES = [...]` (or `list_instances()`) registry and accept an
  `instances=[...]` kwarg on `evaluate`. That couples the harness to a
  specific evaluator shape that most REPS benchmarks don't have.
- **Use case underdeveloped.** Minibatch shines on benchmarks with
  many independent test cases (math sets, RAG queries — dspy-eval-suite
  shape). REPS's existing benchmarks are single-program optimizations
  where evaluation cost is "run once + validate"; there's no natural
  subset to sample from.
- **Cascade evaluation already covers fast-fail.** `evaluate_stage1`
  (cheap) → `evaluate` (full) is a strategy the harness picks; the
  benchmark decides what "cheap" means. Same generic result protocol,
  no instance-registry coupling. If a future benchmark wants minibatch
  semantics, it can implement them inside `evaluate_stage1`.

**If we revisit:** the plan below stays in place as a reference. Phase 6
is the right shape *if* REPS picks up multi-instance benchmarks. Until
that's a concrete need, cascade is sufficient and the simpler public
contract wins.

## Completed phases (recap)

### Phase 1 — Per-instance scores + ASI feedback

- `EvaluationResult`, `EvaluationOutcome`, and `Program` all gained
  optional `per_instance_scores: dict | None` and `feedback: str | None`.
- `_evaluate_code_with_env` now returns `EvaluationResult` end-to-end so
  the new fields survive the cascade-merge step.
- `circle_packing` emits four sub-scores (`validity`, `boundary`,
  `overlap`, `sum_radii_progress`) with uncapped diagnostic totals so
  scores reflect ground truth even when the visible violation list is
  truncated for prompt size.
- Three island/migration copy sites in `database.py` propagate the new
  fields. Stage1+2 / stage3 cascade merges preserve them.

### Phase 2 — Pareto frontier sampler

- New `reps/pareto.py` — pure `compute_frontier`, `sample_pareto`,
  `dominates`, `program_score_vector`, `collect_instance_keys`. Falls
  back to broadcasting `combined_score` when programs lack per-instance
  data; NaN scores treated as `-inf` so failed evals don't pollute the
  frontier.
- `ProgramDatabase.sample_pareto_from_island` mirrors `sample_from_island`
  but routes parent selection through the frontier. Defensive fallbacks
  for empty island and empty frontier.
- `DatabaseConfig.selection_strategy: "map_elites" | "pareto" | "mixed"`
  (default `map_elites`), `pareto_fraction`, `pareto_instance_keys`. The
  controller's `_pick_iteration_inputs` gains a third branch gated by
  `_should_sample_pareto()`.

### Phase 3 — Trace-grounded reflection

- New `reps/trace_reflection.py` — `should_generate_directive`,
  `generate_directive`. Skips the LLM call entirely when feedback is
  absent/short, per-instance scores are empty, or a directive is already
  cached. LLM exceptions degrade silently.
- `Program.mutation_directive: str | None` cached lazily.
- `REPSTraceReflectionConfig` (default off). Wired into `REPSConfig`.
- Controller's `_maybe_generate_trace_directive` runs after parent fetch
  in `_run_iteration`, caches the directive, and threads a
  `## Suggested next change` block via `prompt_extras["trace_directive"]`.
  All four worker types propagate it (single_call / dspy_react via
  hardcoded extras lists; tool runners via generic loop).

### Phase 4 — System-aware merge

- `reps/pareto.py` gains `select_complementary_partner` and
  `select_complementary_pair`. Gain metric:
  `sum_k max(0, c.scores[k] − p.scores[k])`. Non-finite scores clamped to
  0 *for gain accounting only* so finite candidates can compensate for
  broken-eval primaries.
- `REPSMergeConfig` (default off): `enabled`, `instance_keys`,
  `strong_score_threshold`.
- Controller's `_maybe_select_complementary_partner` overrides the
  WorkerPool's random distant-island second_parent for crossover
  iterations when merge is enabled. `_build_crossover_context` renders a
  `## Crossover hint` block listing each parent's strong dimensions.
  Block fires ONLY when the override actually happens — legacy random
  crossovers keep their existing empty `crossover_context`.

### Phase 5 — Ancestry-aware reflection

- `ProgramDatabase.walk_lineage(program_id, max_depth=5) → List[Program]`
  walks parent_id pointers oldest-first with cycle protection and
  broken-link tolerance.
- `trace_reflection._build_lineage_block` renders one line per ancestor
  with score, changes_description, and prior cached directive. Truncated
  per field; depth 3 ⇒ ~600 chars worst case.
- `REPSTraceReflectionConfig.lineage_depth: int = 3` (0 disables, exact
  Phase 3 behavior). `generate_directive(... ancestors=...)` is the new
  optional kwarg. Cache key unchanged — directive caches on the parent
  regardless of which ancestors were visible at first generation.

## Phase 6 — Minibatch evaluation with promotion

### Goal

When evaluation cost dominates wall time, most candidates are duds and
shouldn't be paid for at full fidelity. Run the evaluator on a small
subset of test instances first; only re-run on the full set if the
minibatch score clears a promotion threshold.

GEPA reports ~35× fewer evaluations than RL-style optimization. Even a
2-3× saving on REPS would compound meaningfully across long runs.

### Why this is the riskiest remaining work

It touches the evaluation pipeline. Existing benchmarks must keep working
unchanged. The Pareto frontier and MAP-Elites archive must NOT mix
low-fidelity and high-fidelity scores or selection becomes meaningless.

### Compatibility with existing cascade evaluation

`EvaluatorConfig` already has `cascade_evaluation` + `cascade_thresholds`
with `evaluate_stage1` (fast filter) → `evaluate` / `evaluate_stage2`
(full). That's a coarser version of "minibatch with promotion": stages
can use entirely different evaluator functions, while minibatch runs the
same evaluator on a subset.

We won't replace cascade. Minibatch is the additional pattern: an
evaluator that exposes a single `evaluate(program_path, instances=...)`
API can opt into minibatch promotion without writing a separate stage1
function. Cascade remains for benchmarks that need stage-specific
evaluators.

### Step-by-step plan

#### Phase 6.1 — Evaluator contract extension (pure)

Files:
- `reps/evaluator.py`
- `reps/evaluation_result.py` (no change — `EvaluationResult` already
  carries the data we need; add a `fidelity` tag in metrics dict)
- `reps/config.py` — extend `EvaluatorConfig`

Config additions:
```python
@dataclass
class EvaluatorConfig:
    ...
    # Phase 6: minibatch evaluation with promotion. None disables.
    # When set, evaluate() is called twice: once with `instances` set to a
    # sampled subset of size minibatch_size, then (only if the minibatch
    # combined_score >= minibatch_promotion_threshold) on the full set
    # with instances=None.
    minibatch_size: Optional[int] = None
    minibatch_promotion_threshold: float = 0.5
    # Strategy for sampling instances. "fixed_subset" rotates a
    # deterministic subset each iteration so per-instance scores stay
    # comparable across candidates. "random" picks fresh each call —
    # cheaper to implement, harder to compare across the population.
    minibatch_strategy: str = "fixed_subset"
```

Pure helper module — `reps/minibatch.py`:
```python
def select_instances(all_keys: List[str], size: int, *,
                     iteration: int, strategy: str) -> List[str]:
    """Deterministic minibatch selection so per-instance scores are
    comparable across candidates evaluated in the same iteration window.
    Returns the sublist of keys to pass to evaluate(instances=...)."""
```

Tests:
- Empty `all_keys` → empty list.
- `size >= len(all_keys)` → all keys (skip the minibatch path).
- `fixed_subset` returns the same subset for the same iteration window
  (so two candidates in the same batch see the same instances).
- `random` produces different subsets across calls (with explicit RNG
  seeding so it's reproducible).

#### Phase 6.2 — Evaluator wiring

Files:
- `reps/evaluator.py` — new `evaluate_with_promotion` method that wraps
  `_evaluate_code_with_env`. Order of operations:
  1. If `cascade_evaluation` is enabled, defer to the existing cascade
     path. Minibatch and cascade are mutually exclusive.
  2. Otherwise, if `minibatch_size is None`, defer to the existing
     direct-evaluate path.
  3. Otherwise, sample `minibatch_size` instance keys from the union of
     `per_instance_scores` keys observed so far (or fall back to a
     benchmark-supplied registry — see "evaluator-side knowledge" below).
  4. Call `evaluate(program_path, instances=subset)`. Inspect
     `combined_score`. If below threshold: return the minibatch
     `EvaluationResult` tagged `metrics["fidelity"] = "minibatch"`.
  5. Otherwise: call `evaluate(program_path, instances=None)` for the
     full evaluation. Tag `metrics["fidelity"] = "full"`.

The benchmark evaluator's signature becomes:
```python
def evaluate(program_path, env=None, instances: Optional[List[str]] = None):
    ...
```

Benchmarks that don't accept `instances` continue to work — the harness
will detect via `inspect.signature` and skip the minibatch path with a
one-time warning.

#### Phase 6.3 — Archive / Pareto integrity

Critical: keep low-fidelity and high-fidelity scores from contaminating
selection.

Two policies, controllable per run:

```python
@dataclass
class EvaluatorConfig:
    ...
    minibatch_archive_policy: str = "promoted_only"
    # "promoted_only": only candidates with fidelity="full" enter the
    #   MAP-Elites archive and the Pareto frontier. Rejected
    #   minibatch-only candidates are still recorded for diagnostics
    #   but not eligible as parents.
    # "all_with_tag": all candidates enter; selection sees the fidelity
    #   tag and treats minibatch entries as a separate stratum.
```

`promoted_only` is the safe default. `all_with_tag` is more sample-
efficient on benchmarks where minibatch scores are reliable but
requires `sample_pareto_from_island` and `sample_from_island` to filter
or stratify by fidelity.

Implementation for `promoted_only`:
- `Program.metrics["fidelity"]` is set by the evaluator wiring step.
- `ProgramDatabase.add()` checks the fidelity tag. If `minibatch` and
  policy is `promoted_only`, store on a side dict
  (`self._minibatch_only` for diagnostics) and do NOT register in the
  archive or islands.
- All sampling paths (`sample`, `sample_from_island`,
  `sample_pareto_from_island`) only see archived programs.

#### Phase 6.4 — Pilot benchmark

Circle-packing is single-instance — bad pilot for minibatch. Two options:

1. Build a multi-instance variant: e.g.,
   `experiment/benchmarks/circle_packing_multistart/` that runs the
   solver from N different random initializations and treats each
   initialization as an "instance". `combined_score` is the mean across
   instances. Minibatch picks K of N. This is a real GEPA-style
   problem — variance across instances is the signal that minibatch
   exploits.
2. Pick or create a benchmark with naturally many instances (e.g., a set
   of math problems). Higher build cost, but more honest validation.

Recommendation: start with option 1 since the existing `circle_packing`
solver can be wrapped without a new problem domain.

#### Phase 6.5 — Live test gate

Mock-LLM unit tests cover:
- minibatch path called when configured, full eval only on promotion;
- disabled (`minibatch_size=None`) path is byte-identical to current
  behavior;
- `combined_score` recorded reflects the highest-fidelity eval that ran;
- `promoted_only` policy keeps minibatch-only programs out of the
  archive.

A real run on the pilot benchmark with `minibatch_size=2` and a
configurable threshold (e.g., 0.6) checks:
- wall-time savings vs Phase-5 baseline at equal iterations;
- best-score trajectory at least as good as baseline (no quality loss);
- archive contains only `fidelity=full` entries;
- per-iteration logs name which subset was evaluated.

#### Phase 6.6 — Subagent review

Plan agent — "audit whether minibatch scoring corrupts MAP-Elites bins
or the Pareto frontier (mixing low-fidelity and high-fidelity scores)."
Specifically check:

- Every code path that writes to the archive or frontier.
- The `Program` round-trip — does `fidelity` survive `from_dict`?
- Existing benchmarks (`circle_packing`, `circle_packing_n32`) still
  work with `minibatch_size=None` (the default).
- Cascade vs minibatch mutual exclusion is enforced — config validation
  raises if both are enabled.

### Open design decisions for Phase 6

1. **Instance registry.** Where does the harness learn the universe of
   instance keys? Options:
   a) Read from any `Program.per_instance_scores` already in the
      database (works after iteration 1, brittle when keys evolve).
   b) Benchmark exposes an `INSTANCES = [...]` module-level constant
      or `list_instances()` function.
   c) Evaluator returns the full instance list as part of its result.
   Recommendation: (b) — explicit and lets the benchmark version its
   instance set.

2. **Threshold semantics.** Compare minibatch `combined_score` to the
   threshold? Or compare to the parent's score (relative gate)? GEPA
   uses an absolute promotion threshold, so start there.

3. **Multi-stage minibatch.** GEPA can layer multiple promotion gates
   (small → medium → full). REPS's existing cascade already handles
   this for stage-based evaluators. Minibatch keeps a single gate to
   stay scoped; the cascade path remains for users who want more.

## Phase 7 — Adapter pattern refactor (deferred)

### Goal

Decouple "what's being optimized" from the harness. Today REPS assumes
Python source code with `EVOLVE-BLOCK` markers and a `combined_score`-
returning evaluator. GEPA's `GEPAAdapter` interface lets the same
optimizer optimize prompts, configs, vector graphics, RAG queries, tool
descriptions, etc.

### Why deferred

Largest blast radius of any phase. Lowest urgency — not worth doing
until Phase 6 lands and the new GEPA-style features prove their value
on existing benchmarks.

### Sketch (when we revisit)

A small interface in `reps/adapter.py`:

```python
class REPSAdapter(ABC):
    """Decouples the harness from the artifact under optimization."""

    @abstractmethod
    def evaluate(
        self,
        artifact: str,
        env: Optional[Dict[str, str]] = None,
        instances: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Score one artifact. May produce per_instance_scores +
        feedback (Phase 1 ASI). May accept an instance subset (Phase 6
        minibatch)."""

    @abstractmethod
    def parse_mutation(self, raw_response: str) -> str:
        """Convert LLM output into the next artifact (e.g. apply diffs,
        extract code from markdown, parse YAML)."""

    @abstractmethod
    def initial_artifact(self) -> str:
        """The seed for the search."""

    def render_in_prompt(self, artifact: str) -> str:
        """How the artifact appears in the worker prompt. Default:
        ```{language}\n{artifact}\n```. Adapters for prompts, configs,
        vector graphics override."""
        return f"```\n{artifact}\n```"

    @property
    def language(self) -> str:
        """Language hint for prompt rendering. Adapters override."""
        return "text"
```

A built-in `PythonSourceAdapter` recovers today's behavior. A new
`PromptAdapter` would optimize a single text prompt; a `JSONConfigAdapter`
would optimize a JSON document by editing keys; etc.

The harness would route via the adapter:
- Workers ask the adapter to `parse_mutation`.
- Evaluator calls `adapter.evaluate(...)`.
- Prompt sampler calls `adapter.render_in_prompt(...)` instead of
  hardcoding the Python codeblock.

This is GEPA's `GEPAAdapter` pattern, scoped down. Specific built-in
adapters (DSPy program, Generic RAG, MCP, etc.) are not in scope — REPS
isn't trying to be GEPA. The point is to make non-Python optimizations
*possible* without touching `controller.py`.

### Estimated effort

Two to four phased commits, similar to Phases 2-3:

- 7.1: Define `REPSAdapter` ABC + `PythonSourceAdapter` matching today's
  behavior. Pure module + tests. No integration.
- 7.2: Wire `Evaluator` and `PromptSampler` to call through the adapter.
  Existing benchmarks register the default Python adapter via config or
  convention.
- 7.3: Add a second adapter (probably `PromptAdapter` for system-prompt
  optimization) and a small example benchmark. Validates the abstraction.

### Risks

- Existing benchmarks must continue to work without code changes — the
  default adapter is critical.
- Worker implementations (single_call, dspy_react, two tool runners) all
  contain Python-source-aware code (diff parsing, EVOLVE-BLOCK
  extraction). The adapter has to abstract those without forcing every
  worker to know every adapter.
- The two-stage parse (full rewrite vs diff) may not generalize cleanly.
  Adapters might need to declare which mutation modes they support.

## Append-only changelog

This section is updated as phases ship. Each row references the commit
SHA and a one-line description of what landed.

| Date       | Phase | Commit     | Notes |
|------------|-------|------------|-------|
| 2026-05-01 | 1.1   | `102952e`  | EvaluationResult/Program gain optional ASI fields |
| 2026-05-01 | 1.2   | `eac9f83`  | circle_packing emits per_instance_scores + feedback |
| 2026-05-01 | 2.1   | `5dea23b`  | Pure pareto module |
| 2026-05-01 | 2.2   | `7e73e61`  | Pareto sampler wired into database + controller |
| 2026-05-01 | 3.1   | `c9bda73`  | trace_reflection module + Program.mutation_directive |
| 2026-05-01 | 3.2   | `1fdacfe`  | Controller + workers thread mutation_directive into prompts |
| 2026-05-01 | 4.1   | `f89fb8f`  | select_complementary_partner / _pair |
| 2026-05-01 | 4.2   | `b1db148`  | System-aware merge wired into crossover dispatch |
| 2026-05-01 | 5     | `2420e69`  | walk_lineage + ancestry block in directive prompt |
| 2026-05-05 | 6.1-3 | `31e1482`, `877d57d`, `8441b72`, `2bed1a1`, `13f6bf5` | Phase 6 shipped (minibatch with promotion + archive integrity + adversarial tests + review fixes) |
| 2026-05-05 | 6     | revert | Phase 6 reverted: polluted the user-facing evaluator contract; cascade evaluation covers fast-fail without coupling the harness to an instance registry. See "Phase 6 — reverted" section above. |
