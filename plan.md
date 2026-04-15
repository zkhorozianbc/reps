# REPS: Implementation Plan over OpenEvolve

**Recursive Evolutionary Program Search — Concrete Build Plan**

*Starting point: OpenEvolve v0.2.26 (`algorithmicsuperintelligence/openevolve`)*
*Tech stack: macOS (Apple Silicon), `uv` Python environment, Gemini API key + Anthropic API key*

---

## Feature Reference

Every mechanism described below is implemented in one of the five phases. This section is the canonical definition of each feature so that phase descriptions can reference them by name without re-explaining.

### F1 — Reflection Engine

A dedicated LLM pass that runs in the **controller process** after each batch of iteration results returns from the process pool. It produces structured analysis of what worked, what failed, and what to try next. The reflection is serialized into the `prompt_extras` dict of `IterationConfig` objects for the next batch, injected into prompts via a `{reflection}` template field. Unlike OpenEvolve's raw artifact feedback (stderr, scores), reflection is *synthesized reasoning* about the search trajectory — it identifies patterns across multiple candidates rather than reporting on one.

**Mechanism:** After a batch of iterations completes and results are merged, select the top-k and bottom-k candidates from that batch. Call an LLM (in the controller process, using the `openevolve/llm/` module directly) with: (a) the current best program, (b) the top-k diffs + scores, (c) the bottom-k diffs + scores, (d) the previous reflection. The output is a structured JSON with fields: `working_patterns` (what kinds of edits improved scores), `failing_patterns` (what kinds of edits hurt or stalled), `hypotheses` (causal claims about why), `suggested_directions` (concrete next steps). This JSON is stored in the controller and included in `prompt_extras` for all `IterationConfig` objects in the next batch.

**Source paper:** MR-Search — 9–19% relative improvement from cross-episode self-reflection.

### F2 — ε-Revisitation

With probability ε (default 0.15, decaying over generations), instead of generating a fresh candidate, the system revisits a promising-but-underexplored parent program using a different worker configuration. This is amortized critique — empirically testing whether an abandoned path was truly exhausted or just poorly sampled by a single LLM roll.

**Mechanism:** Maintain a `revisitation_priority` score for each program in the database: `priority = score × (1 / (1 + num_descendants)) × recency_bonus`. Programs with high scores but few children are prioritized. When a revisitation fires, select the target program and assign it to a *different* worker type or model than the one that originally generated its children. If the revisitation produces a score improvement, boost that lineage's allocation; if not, decay the target's revisitation priority permanently.

**Design rationale:** Cheaper than actor-critic on every call; targets the specific failure mode where a good semantic direction gets one bad LLM sample and is never retried.

### F3 — Worker Type Diversity

OpenEvolve has one generation strategy (diff or full rewrite, set by config). REPS introduces structurally different worker types that operate on programs in fundamentally different ways:

- **Exploiter:** Small, targeted diffs. Low temperature (0.2–0.5). Operates within EVOLVE-BLOCK regions via surgical diff mode. The default mode, similar to OpenEvolve's current behavior.
- **Explorer:** Full rewrites within EVOLVE-BLOCK regions. High temperature (0.8–1.2). Replaces entire block contents rather than applying surgical diffs, but does **not** expand the mutation surface beyond marked blocks — the evaluator contract depends on immutable context outside the blocks. The difference from Exploiter is scope-of-change within the allowed region, not scope-of-region.
- **Crossover:** Selects two parent programs from different MAP-Elites niches and merges them. The LLM receives both parents with instructions to combine the best aspects of each within the EVOLVE-BLOCK regions. Introduces recombination that pure mutation cannot achieve.
- **RLM Worker (Phase 6 only):** For programs >2000 lines. Offloads the program to a Python REPL, uses recursive sub-LLM calls to analyze and modify sections. Not needed for the AlphaEvolve math benchmarks but critical for real-world codebases.

Workers are managed by a `WorkerPool` that **wraps** OpenEvolve's `openevolve/llm/` module — the underlying LLM package (with its retry logic, async generation, model fallback, and provider abstraction) is preserved. The WorkerPool's job is to decide *which* call to make (worker type, model, temperature) and encode that decision into an `IterationConfig` that gets dispatched to the process pool. It does not make LLM calls directly. Each worker type maintains independent yield statistics tracked by the controller.

### F4 — Convergence Monitor

An active monitoring system that runs in the **controller process** at REPS batch boundaries (not per-iteration). It detects when the evolutionary search is collapsing — i.e., when different workers and different lineages are converging to produce the same kinds of edits, indicating premature loss of diversity.

**Two metrics tracked over a sliding window (default: last 20 batches of returned results):**

- **Edit entropy:** Compute a bag-of-edits representation (AST diff categories, or simpler: character-level diff clustering) for each candidate in the window. Measure Shannon entropy over the distribution of edit types. When entropy drops below a threshold, edits are becoming homogeneous.
- **Strategy divergence:** Measure the KL divergence between the output distributions of different worker types. When different workers start producing the same edits, the benefit of worker diversity has collapsed.

The monitor operates on **batch results after they return from the process pool** — it reads the diffs and worker types from completed iterations, not from in-flight workers. Its output (a `ConvergenceAction`) modifies the `IterationConfig` parameters for the *next* batch dispatched to the pool.

**Three escalation levels:**

1. **Mild collapse** (entropy < threshold_1): Increase Explorer allocation by 20%, bump temperatures by 0.1.
2. **Moderate collapse** (entropy < threshold_2): Inject random restart candidates from the most distant occupied MAP-Elites niches. Force the next 5 generations to use at least 50% Explorer workers.
3. **Severe collapse** (entropy < threshold_3): Switch primary model, double ε-revisitation rate, force crossover from archived programs >50 generations old.

### F5 — Intelligence Contracts

Instead of fixed model weights (OpenEvolve: "model A with probability 0.6, model B with 0.4"), each LLM call has a learnable contract tuple: `(model_id, temperature, max_tokens)`. A Thompson-sampling bandit with one arm per contract configuration learns which configs produce the best yield for different program states.

**State features for the bandit:** Current best score, score gap to known SOTA (if available), generation number, parent program complexity (LOC, cyclomatic complexity), worker type, number of prior failed attempts on this parent.

**Contract arms (initial):** `{gemini-2.5-pro, gemini-2.5-flash, claude-sonnet-4} × {0.3, 0.7, 1.0} = 9 arms`. Each arm maintains a Beta distribution posterior; the bandit samples from posteriors and selects the arm with the highest sample (Thompson sampling). Posteriors update on success (score improvement) / failure (no improvement).

### F6 — SOTA-Distance Steering

For benchmark problems with a known state-of-the-art score or theoretical bound, the system uses the gap between the current best and the target as a continuous control signal that modulates search behavior.

**Control mapping:**

| Gap to SOTA | Behavior |
|---|---|
| >30% | "Far regime" — maximize Explorer allocation, high temperatures, prompt says "try fundamentally different algorithms" |
| 10–30% | "Mid regime" — balanced explore/exploit, prompt says "the current approach has merit but needs structural improvements" |
| <10% | "Near regime" — maximize Exploiter allocation, low temperatures, prompt says "focus on surgical parameter tuning and micro-optimizations" |
| <2% | "Polishing regime" — run multi-seed evaluation to confirm improvements are real, increase eval rigor |

This is injected into prompts as `{sota_injection}` and also modulates the convergence monitor thresholds and worker allocation ratios.

### F7 — Compute Signature Tracking

Every successful improvement records a `ComputeSignature`: total tokens consumed, model used, worker type, number of attempts before success, wall-clock time. These signatures are stored in the program database alongside the program and its scores.

**Analysis dimensions:**

| | High novelty (large score jump) | Low novelty (small score delta) |
|---|---|---|
| **Low tokens** | Cheap breakthrough — fan out from this region | Normal polish — business as usual |
| **High tokens** | Expensive discovery — region is rich, keep investing | Expensive grind — stuck, redirect to crossover/restart |

The meta-prompter reads these signatures to modulate budget allocation. Cheap breakthroughs get amplified (DG-style delight weighting). Expensive grinds trigger strategy switches.

### F8 — Enriched Program Annotations (OrgEvolve)

Extend the program database record from `{code, scores}` to include structured metadata. Annotations are stored as a **separate lightweight dict** alongside but independent of OpenEvolve's artifact system (which has a 10KB inline threshold). Annotations are typically ~500 bytes and never hit that threshold — they are serialized with the database snapshot and available to iteration workers for prompt building.

```python
{
    "program": "<code>",
    "scores": {"primary": 0.85, "auxiliary": [...]},
    # --- REPS annotations (separate from artifacts) ---
    "annotations": {
        "worker_type": "explorer",
        "model_used": "gemini-2.5-pro",
        "hypothesis": "blocking memory layout reduces cache misses",
        "outcome": "12% improvement but numerical instability above dim=1024",
        "dead_end_flag": False,
        "salvageable_components": ["blocking logic lines 40-67"],
        "related_failures": ["gen-12/candidate-7"],
        "compute_signature": {...}
    }
}
```

The hypothesis and outcome fields are populated by the Reflection Engine. The `related_failures` field enables cross-referencing so new candidates don't repeat known dead ends. The prompt sampler queries these annotations when building context for the next batch — transforming the database from a passive gene pool into an active knowledge base.

### F9 — Meta-Level World Model (Future)

A learned dynamics model over the archive state that enables planning rather than reactive worker allocation. Given `(archive_state, worker_allocation_strategy)`, predicts `(fitness_improvement, niche_coverage_delta)` N steps into the future. The meta-RL layer (GRPO) plans through this model — simulating allocation strategies and picking the predicted-best one before executing real evaluations. This is Dyna-Q at the meta-level. Deferred to Phase 5.

---

## Benchmark Suite

All benchmarks come from OpenEvolve's `examples/` directory and run locally on a MacBook. No GPUs, no external datasets. Each evaluation is milliseconds to low seconds.

### Primary Benchmarks

| Problem | Eval time | Known SOTA | What we evolve | OpenEvolve ships it? |
|---|---|---|---|---|
| **Circle packing (n=26)** | ~seconds | 2.635 (AlphaEvolve) | Placement algorithm | Yes |
| **Function minimization** | ~ms | Known global minima | Optimization algorithm | Yes |
| **Symbolic regression** | ~ms | Exact equations | Equation discovery | Yes |

### Ablation Protocol

Every phase produces a comparison table with these columns:

| Variant | Best score | Generations to 90% of best | Total LLM cost ($) | Final diversity (niche occupancy) |
|---|---|---|---|---|

Variants per phase:
- `baseline` — unmodified OpenEvolve
- `+feature_X` — OpenEvolve + the single feature added in that phase
- `+all_prior` — cumulative: all features from prior phases
- `+all_prior+feature_X` — cumulative + the new feature

Minimum 3 seeds per variant for statistical significance. Report mean ± std.

### Measurement Infrastructure

Add to OpenEvolve's existing checkpoint system:

- **Score trajectory CSV:** `batch, best_score, mean_score, worst_score, num_improvements, timestamp`
- **Worker yield CSV:** `batch, worker_type, num_candidates, num_improvements, yield_rate`
- **Diversity CSV:** `batch, edit_entropy, strategy_divergence, niche_occupancy, unique_edit_types`
- **Cost CSV:** `batch, model, tokens_in, tokens_out, cost_usd, wall_clock_seconds`
- **Reflection log:** `batch, reflection_json` (for qualitative analysis)

All CSVs written to `openevolve_output/metrics/` alongside existing checkpoints.

---

## Environment Setup

```bash
# Clone and set up
git clone https://github.com/algorithmicsuperintelligence/openevolve.git
cd openevolve

# Create uv environment
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# API keys
export OPENAI_API_KEY="your-gemini-api-key"           # Gemini via OpenAI-compat
export ANTHROPIC_API_KEY="your-anthropic-api-key"     # For Claude models in ensemble

# Verify installation
python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 10
```

### Directory Structure (REPS additions)

```
openevolve/
├── openevolve/
│   ├── controller.py          # ← modify: dispatch IterationConfig, batch hooks for reflection/convergence/SOTA
│   ├── iteration.py           # ← modify: accept IterationConfig, apply worker-type-specific prompt/temp/mode
│   ├── prompt_sampler.py      # ← modify: add reflection, SOTA injection, dead-end warning template fields
│   ├── llm/                   # ← KEEP AS-IS: retry logic, async gen, model fallback, provider abstraction
│   ├── database.py            # ← modify: add annotations, revisitation priority, compute signatures
│   ├── evaluator.py           # ← minor: add multi-seed eval option
│   ├── reflection_engine.py   # NEW: post-batch reflection (runs in controller process)
│   ├── convergence_monitor.py # NEW: edit entropy + strategy divergence (runs in controller process)
│   ├── worker_pool.py         # NEW: worker type selection + IterationConfig builder (wraps llm/)
│   ├── contract_selector.py   # NEW: Thompson-sampling intelligence contracts (runs in controller process)
│   ├── sota_controller.py     # NEW: gap-aware search modulation (runs in controller process)
│   ├── iteration_config.py    # NEW: dataclass passed to iteration workers
│   ├── metrics_logger.py      # NEW: CSV logging for ablations
│   └── config.py              # ← modify: add REPS-specific config fields
├── configs/
│   ├── reps_circle_packing.yaml
│   ├── reps_function_min.yaml
│   └── reps_symbolic_regression.yaml
├── scripts/
│   ├── run_ablation.py        # NEW: automated ablation runner
│   └── plot_results.py        # NEW: generate comparison figures
└── results/                   # NEW: ablation outputs
```

---

## OpenEvolve Execution Model (Architectural Constraints)

Every REPS modification must respect these execution realities. Ignoring them leads to code that looks correct but silently breaks parallelism or data flow.

### Process isolation via `iteration.py`

OpenEvolve does **not** run LLM calls inline in the controller. The controller dispatches work to a `ProcessPoolExecutor`. Each iteration runs in a **separate OS process** that receives a serialized **snapshot** of the database, not a live reference. The iteration process independently: samples a parent from its DB snapshot, builds a prompt, calls the LLM, applies the diff, runs the evaluator, and returns results. The controller then merges results back into the live database.

**Implication for REPS:** Worker type selection, contract selection, ε-revisitation decisions, and prompt injection (reflection, SOTA) must all happen **before** dispatching to the process pool. These decisions are encoded into an `IterationConfig` object that the worker process receives alongside the DB snapshot. The worker process itself is stateless — it executes whatever config it's given. REPS modules like `ReflectionEngine`, `ConvergenceMonitor`, and `SOTAController` live in the controller process, **not** in the iteration workers.

```
Controller (single process, stateful)
├── ReflectionEngine      ← runs between batches
├── ConvergenceMonitor    ← runs between batches
├── SOTAController        ← runs between batches
├── ContractSelector      ← queried before dispatching each iteration
└── dispatches IterationConfig → ProcessPoolExecutor
    └── iteration.py (stateless worker process)
        ├── receives: DB snapshot + IterationConfig
        ├── IterationConfig includes: worker_type, model_id, temperature,
        │   prompt_extras (reflection text, SOTA injection, dead-end warnings)
        └── returns: candidate code, eval results, compute signature
```

### Batch-oriented generation, not synchronous generations

OpenEvolve does not have clean "generation" boundaries. The controller dispatches N iterations in parallel; they return asynchronously. There is no moment where "generation g is complete and generation g+1 starts." Instead, results trickle back and get merged into the database continuously.

**Implication for REPS:** When this document says "after each generation," it means "after a batch of K iterations has completed and results are merged." We define a **REPS batch** as the `num_workers` iterations dispatched in one wave of the controller's main loop. The reflection engine, convergence monitor, and SOTA controller all run at batch boundaries, not per-iteration.

### `openevolve/llm/` is a module, not a single file

The LLM integration lives in `openevolve/llm/` — a package with retry logic, async generation, model fallback chains, and provider abstraction (OpenAI-compatible). REPS's `WorkerPool` **wraps** this module; it does not replace it. Workers still call the underlying `llm/` package for actual LLM API communication. The WorkerPool's job is to select *which* call to make (worker type, model, temperature), not to make the call itself.

### EVOLVE-BLOCK markers define the mutation surface

Users mark mutable regions with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`. The LLM is constrained to proposing changes within these regions. The rest of the codebase is immutable context.

**Implication for REPS worker types:** The Exploiter operates within EVOLVE-BLOCKs (targeted diffs). The Explorer also operates within EVOLVE-BLOCKs but with higher temperature and full-rewrite mode (replacing the entire block contents rather than applying surgical diffs). The Explorer does **not** expand the mutation surface beyond the marked blocks — that would break the evaluator contract. The difference is scope-of-change within the allowed region, not scope-of-region.

### Artifact size threshold

Artifacts <10KB are stored inline in the database; larger artifacts go to disk. REPS's enriched annotations (F8) are stored as a separate `annotations` dict alongside but independent of the artifact system. Annotations are lightweight metadata (~500 bytes typical) and never hit the 10KB threshold. They are serialized with the database snapshot and available to iteration workers for prompt building.

### Lazy island migration

Islands migrate programs between each other based on per-island generation counts, not global iteration counts. Migration happens when `island.generation_count % migration_interval == 0`. This is asynchronous across islands — island A might migrate while island B hasn't yet. REPS batch boundaries are independent of migration timing. The convergence monitor and reflection engine operate on all islands collectively; SOTA steering applies globally.

---

## Phase 1: Baselines + Measurement Infrastructure (Days 1–2)

**Goal:** Establish reproducible baselines on all three benchmarks; build the metrics logging that every subsequent phase depends on.

### Concrete Changes

**1.0 — Audit OpenEvolve's execution model**

Before modifying anything, read and annotate the actual control flow. Trace these specific paths through the code:

- `controller.py`: How does the main loop dispatch iterations? What is passed to `ProcessPoolExecutor`? What comes back?
- `iteration.py`: What state does a worker process receive (DB snapshot format)? What does it return? Where does prompt building happen — in the controller or the worker?
- `llm/`: What's the module structure? Which class handles retries, which handles model fallback?
- `database.py`: How is the DB snapshot serialized for worker processes? How are results merged back?
- `prompt_sampler.py`: Where are template fields defined? What context is available at prompt-build time?

Document the answers in `docs/execution_model.md`. This document becomes the source of truth for where each REPS module can and cannot run. Every subsequent phase references it.

**1.1 — Run OpenEvolve baselines**

Run each benchmark 3× with different seeds (42, 123, 456). Record:
- Final best score per seed
- Score-vs-generation curve
- Total tokens consumed
- Wall-clock time
- Final MAP-Elites niche occupancy

```bash
# Circle packing baseline
for seed in 42 123 456; do
  python openevolve-run.py \
    examples/circle_packing/initial_program.py \
    examples/circle_packing/evaluator.py \
    --config examples/circle_packing/config.yaml \
    --iterations 200 \
    --seed $seed \
    --output results/baseline/circle_packing/seed_$seed
done
```

**1.2 — Add `metrics_logger.py`**

Create a new module that hooks into OpenEvolve's controller loop. After each batch of iterations returns, write rows to the CSV files described in Measurement Infrastructure above. This module is purely additive — no changes to existing logic.

```python
# openevolve/metrics_logger.py
class MetricsLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_csvs()

    def log_batch(self, batch_number: int, batch_results: list, 
                  database, timing: dict):
        """Called in controller after each batch of iterations returns."""
        self._write_score_row(batch_number, batch_results)
        self._write_diversity_row(batch_number, batch_results, database)
        self._write_cost_row(batch_number, timing)
        self._write_worker_yield_row(batch_number, batch_results)
```

**1.3 — Modify `controller.py` to call the logger**

Single insertion point: after batch results are merged back into the database, call `self.metrics_logger.log_batch(...)`. No behavioral change to the search.

**1.4 — Add `run_ablation.py` script**

Automates running N seeds × M variants and collects results into a single comparison CSV.

```python
# scripts/run_ablation.py
"""
Usage: python scripts/run_ablation.py \
    --benchmark circle_packing \
    --variants baseline,+reflection \
    --seeds 42,123,456 \
    --iterations 200
"""
```

### Benchmark Targets

| Benchmark | Expected OpenEvolve baseline (200 iter) | Notes |
|---|---|---|
| Circle packing (n=26) | ~2.62–2.635 | OpenEvolve claims SOTA match at ~460 iter |
| Function minimization | Converges to near-global minimum | Baseline transforms random → simulated annealing |
| Symbolic regression | Discovers exact or near-exact equations | Problem-dependent |

### Exit Criteria

- All three benchmarks produce reproducible results across seeds
- Metrics CSVs are generating correctly
- Ablation script can run two variants and produce a comparison table

---

## Phase 2: Reflection Engine + ε-Revisitation (Days 3–5)

**Goal:** Add the two highest-impact, lowest-effort mechanisms (F1, F2). Measure their individual and combined effect against the Phase 1 baseline.

### Concrete Changes

**2.1 — Implement `reflection_engine.py` (F1)**

Runs in the controller process after each batch returns. Uses the `openevolve/llm/` module directly (not through the process pool) for the reflection LLM call.

```python
class ReflectionEngine:
    def __init__(self, llm_client, config):
        self.llm = llm_client  # openevolve/llm/ module, called in controller process
        self.max_top_k = config.get("reflection_top_k", 3)
        self.max_bottom_k = config.get("reflection_bottom_k", 2)

    async def reflect(self, batch_results: list,
                      previous_reflection: dict) -> dict:
        """
        Analyze the batch's results and produce structured reflection.
        Called in the controller process after batch results are merged.
        Returns: {
            "working_patterns": [...],
            "failing_patterns": [...],
            "hypotheses": [...],
            "suggested_directions": [...],
            "batch_number": int
        }
        """
        top_k = select_top_k(batch_results, self.max_top_k)
        bottom_k = select_bottom_k(batch_results, self.max_bottom_k)
        
        prompt = self._build_reflection_prompt(
            top_k, bottom_k, previous_reflection
        )
        response = await self.llm.generate(prompt)
        return self._parse_reflection(response)
```

**2.2 — Modify `prompt_sampler.py` to support `{reflection}` template field**

The prompt sampler already builds prompts inside `iteration.py` (worker process). The reflection text arrives via `IterationConfig.prompt_extras["reflection"]`. The sampler checks for this key and injects formatted reflection text into the prompt template.

```python
# In prompt_sampler.py, extend the prompt building:
reflection_text = prompt_extras.get("reflection", "")
if reflection_text:
    prompt = prompt.replace("{reflection}", reflection_text)
else:
    prompt = prompt.replace("{reflection}", "")
```

**2.2b — Define `iteration_config.py`**

This is the bridge between the controller (where REPS decisions live) and the iteration worker (stateless process). Every REPS mechanism that affects how an iteration runs must encode its decision into this config.

```python
# openevolve/iteration_config.py
@dataclass
class IterationConfig:
    """Serializable config passed from controller to iteration worker process."""
    # Parent selection
    parent_id: str | None          # If set, use this parent; if None, sample from DB snapshot
    
    # Worker type (F3)
    worker_type: str               # "exploiter", "explorer", "crossover"
    
    # Model/generation params (F5 intelligence contracts)
    model_id: str | None           # Override model selection; None = use default
    temperature: float             # LLM temperature for this iteration
    
    # Prompt extras (F1 reflection, F6 SOTA, F8 dead-end warnings)
    prompt_extras: dict            # Keys: "reflection", "sota_injection", "dead_end_warnings"
    
    # Crossover-specific (F3)
    second_parent_id: str | None   # For crossover worker: ID of second parent
    
    # Metadata
    is_revisitation: bool          # F2: whether this is an ε-revisitation
    generation_mode: str           # "diff" or "full" — Explorer uses "full", Exploiter uses "diff"

@dataclass
class IterationResult:
    """Returned from iteration worker process to controller."""
    child: str                     # Generated program code
    results: dict                  # Evaluation results (scores, artifacts)
    diff: str                      # Raw LLM output (for convergence monitor edit entropy)
    worker_type: str               # Which worker type produced this
    is_revisitation: bool          # Whether this was an ε-revisitation
    model_id: str | None           # Which model was used
    temperature: float             # What temperature was used
    parent_score: float            # Parent's score (for computing improvement)
    parent_id: str                 # Parent program ID (for genealogy tracking)
    tokens_in: int                 # Input tokens consumed
    tokens_out: int                # Output tokens consumed
    wall_clock_seconds: float      # Wall-clock time for the iteration
    improved: bool = False         # Set by controller after comparing to parent_score
```

**2.3 — Implement ε-revisitation in `controller.py` (F2)**

Modify the controller's iteration dispatch logic: before building the `IterationConfig` for a worker process, roll a random number. If < ε, override the config to target a revisitation candidate instead of a freshly sampled parent. The decision happens in the controller; the worker process just executes whatever config it receives.

```python
# In controller.py, when building configs for the next batch:
iteration_configs = []
for i in range(batch_size):
    if random.random() < self.epsilon:
        # Revisitation: pick underexplored parent, use alternative worker config
        target = self.database.select_revisitation_target()
        config = IterationConfig(
            parent_id=target.id,
            worker_type=self._get_alternative_worker_type(target),
            temperature=self._get_alternative_temperature(target),
            prompt_extras=self._build_prompt_extras(),  # reflection, SOTA, etc.
            is_revisitation=True,
        )
    else:
        # Normal iteration
        config = IterationConfig(
            parent_id=None,  # worker will sample from DB snapshot
            worker_type=self._sample_worker_type(),
            temperature=self._sample_temperature(),
            prompt_extras=self._build_prompt_extras(),
            is_revisitation=False,
        )
    iteration_configs.append(config)

# Dispatch batch to process pool
results = await self.pool.map(run_iteration, iteration_configs)
```

**2.4 — Add revisitation priority scoring to `database.py`**

```python
def select_revisitation_target(self) -> Program:
    """Select program with high score but low exploration."""
    priorities = []
    for program in self.programs:
        priority = program.score * (1.0 / (1.0 + program.num_descendants))
        if program.age < self.revisitation_recency_window:
            priority *= 1.5  # recency bonus
        priorities.append((priority, program))
    priorities.sort(reverse=True)
    return priorities[0][1]
```

**2.5 — Add ε decay schedule**

```python
# ε starts at 0.15 and decays to 0.05 over the run
self.epsilon = max(0.05, 0.15 * (1 - generation / max_iterations))
```

### Ablation Matrix

Run all of these on circle packing + function minimization, 3 seeds each, 200 iterations:

| Variant | Description |
|---|---|
| `baseline` | Unmodified OpenEvolve |
| `+reflection` | Baseline + reflection engine only |
| `+epsilon` | Baseline + ε-revisitation only |
| `+reflection+epsilon` | Both features |

### Expected Outcomes

- Reflection alone should show faster early convergence (the search doesn't repeat known failures)
- ε-revisitation alone should show higher final scores (abandoned good paths get a second chance)
- Combined should compound: reflection identifies promising directions, ε-revisitation ensures they're adequately explored

### Exit Criteria

- Ablation table shows at least one variant statistically beats baseline (p < 0.05 via paired t-test across seeds)
- Reflection logs show coherent, non-degenerate analysis (qualitative check)
- ε-revisitation is firing at the expected rate (~15% early, ~5% late)

---

## Phase 3: Worker Diversity + Convergence Monitor (Days 6–10)

**Goal:** Add the structural worker diversity mechanism (F3) and the convergence monitor (F4) that prevents diversity collapse. This is the biggest structural change to the codebase.

### Concrete Changes

**3.1 — Implement `worker_pool.py` (F3)**

The WorkerPool does **not** call the LLM directly. It lives in the controller process and its job is to build `IterationConfig` objects that encode worker-type-specific decisions. The actual LLM calls happen inside `iteration.py` (worker process) using the existing `openevolve/llm/` module.

```python
class WorkerPool:
    def __init__(self, config):
        # Initial allocation: 60% exploiter, 25% explorer, 15% crossover
        self.allocation = {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15}
        self.yield_tracker = {k: deque(maxlen=100) for k in self.allocation}
        self.worker_configs = {
            "exploiter": {"temperature": 0.3, "generation_mode": "diff"},
            "explorer":  {"temperature": 1.0, "generation_mode": "full"},
            "crossover": {"temperature": 0.7, "generation_mode": "full"},
        }

    def build_iteration_config(self, database, prompt_extras: dict,
                                override_type: str = None) -> IterationConfig:
        """Build an IterationConfig for one iteration dispatch."""
        worker_type = override_type or self._sample_worker_type()
        wconf = self.worker_configs[worker_type]
        
        config = IterationConfig(
            parent_id=None,  # worker samples from DB snapshot
            worker_type=worker_type,
            model_id=None,   # default model selection (overridden by F5 contracts)
            temperature=wconf["temperature"],
            prompt_extras=prompt_extras,
            second_parent_id=None,
            is_revisitation=False,
            generation_mode=wconf["generation_mode"],
        )
        
        # Crossover needs a second parent selected from a distant niche
        if worker_type == "crossover":
            second = database.sample_distant_niche()
            config.second_parent_id = second.id
        
        return config

    def record_result(self, worker_type: str, improved: bool):
        """Track yield per worker type for allocation rebalancing."""
        self.yield_tracker[worker_type].append(1.0 if improved else 0.0)
    
    def _sample_worker_type(self) -> str:
        types, weights = zip(*self.allocation.items())
        return random.choices(types, weights=weights, k=1)[0]
```

**3.2 — Modify `iteration.py` to interpret `IterationConfig`**

This is the critical integration point. The iteration worker process currently hardcodes its generation strategy. We modify it to read `worker_type`, `generation_mode`, `temperature`, `prompt_extras`, and `second_parent_id` from the `IterationConfig` and adjust its behavior accordingly.

```python
# In iteration.py (worker process):
def run_iteration(config: IterationConfig, db_snapshot, llm_module):
    # Select parent
    if config.parent_id:
        parent = db_snapshot.get(config.parent_id)
        inspirations = db_snapshot.sample_inspirations(exclude=config.parent_id)
    else:
        parent, inspirations = db_snapshot.sample()
    
    # For crossover: retrieve second parent's code from DB snapshot
    second_parent = None
    if config.second_parent_id:
        second_parent = db_snapshot.get(config.second_parent_id)
    
    # Build prompt with REPS extras
    prompt = build_prompt(
        parent, inspirations,
        mode=config.generation_mode,       # "diff" or "full"
        extras=config.prompt_extras,       # reflection, SOTA injection, dead-end warnings
        second_parent=second_parent,       # None unless crossover
    )
    
    # Call LLM via existing llm/ module (retries, fallback preserved)
    response = llm_module.generate(
        prompt, 
        model=config.model_id,      # None = use default from config
        temperature=config.temperature,
    )
    
    # Apply diff and evaluate (unchanged from OpenEvolve)
    child = apply_diff(parent, response)
    results = evaluate(child)
    
    return IterationResult(
        child=child, results=results,
        diff=response,                     # raw LLM output for convergence monitor
        worker_type=config.worker_type,
        is_revisitation=config.is_revisitation,
        model_id=config.model_id,
        temperature=config.temperature,
        parent_score=parent.best_score,
        # compute signature fields filled from timing
    )
```

**3.3 — Modify `controller.py` batch dispatch to use WorkerPool**

Replace the controller's iteration dispatch with WorkerPool-driven config building.

```python
# In controller.py main loop:
prompt_extras = self._build_prompt_extras()  # reflection, SOTA, dead-ends

configs = []
for i in range(batch_size):
    if random.random() < self.epsilon:
        # ε-revisitation (from Phase 2)
        target = self.database.select_revisitation_target()
        config = self.worker_pool.build_iteration_config(
            self.database, prompt_extras,
            override_type=self._get_alternative_worker_type(target)
        )
        config.parent_id = target.id
        config.is_revisitation = True
    else:
        config = self.worker_pool.build_iteration_config(
            self.database, prompt_extras
        )
    configs.append(config)

# Dispatch to process pool
results = await self.dispatch_batch(configs)

# Post-batch: set improved flags, update yield stats, run convergence monitor
for result in results:
    result.improved = result.results.get("primary", 0) > result.parent_score
    self.worker_pool.record_result(result.worker_type, result.improved)

# Run convergence monitor on completed batch
action = self.convergence_monitor.update(results)
# ... handle action (see 3.5)

# Run reflection engine on completed batch
self.current_reflection = await self.reflection_engine.reflect(
    results, self.current_reflection
)
```

**3.4 — Implement `convergence_monitor.py` (F4)**

Runs in the controller process **after** a batch of iteration results returns from the process pool. Reads diffs and worker types from completed `IterationResult` objects.

```python
class ConvergenceMonitor:
    def __init__(self, config):
        self.window_size = config.get("convergence_window", 20)  # in batches
        self.edit_history = deque(maxlen=self.window_size * 50)
        self.thresholds = {
            "mild": config.get("entropy_threshold_mild", 0.5),
            "moderate": config.get("entropy_threshold_moderate", 0.3),
            "severe": config.get("entropy_threshold_severe", 0.15),
        }

    def update(self, batch_results: list) -> ConvergenceAction:
        """Called after each batch returns from process pool."""
        for result in batch_results:
            self.edit_history.append({
                "diff": result.diff,
                "worker_type": result.worker_type,
                "improved": result.improved,
            })
        
        edit_entropy = self._compute_edit_entropy()
        strategy_div = self._compute_strategy_divergence()
        
        if edit_entropy < self.thresholds["severe"]:
            return ConvergenceAction.SEVERE_RESTART
        elif edit_entropy < self.thresholds["moderate"]:
            return ConvergenceAction.MODERATE_DIVERSIFY
        elif edit_entropy < self.thresholds["mild"]:
            return ConvergenceAction.MILD_BOOST
        return ConvergenceAction.NONE

    def _compute_edit_entropy(self) -> float:
        """Shannon entropy over edit type distribution across the window."""
        edit_types = [classify_edit(e["diff"]) for e in self.edit_history]
        counts = Counter(edit_types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * log2(p) for p in probs if p > 0)
```

**3.5 — Wire convergence actions into controller**

After each batch returns and results are merged, call `self.convergence_monitor.update(batch_results)`. The returned action modifies worker pool allocation for the *next* batch to be dispatched.

```python
# In controller.py, after batch results are merged:
action = self.convergence_monitor.update(batch_results)
if action == ConvergenceAction.MILD_BOOST:
    self.worker_pool.boost_explorer(0.2)
    self.worker_pool.bump_temperatures(0.1)
elif action == ConvergenceAction.MODERATE_DIVERSIFY:
    self.worker_pool.force_explorer_majority(5)  # next 5 batches
    self.database.inject_distant_restarts(3)
elif action == ConvergenceAction.SEVERE_RESTART:
    self.epsilon *= 2
    self.worker_pool.force_model_switch()
```

**3.6 — Add worker yield tracking to metrics logger**

Extend the CSV logging to record per-worker-type yield rates per batch.

### Ablation Matrix

| Variant | Description |
|---|---|
| `phase2_best` | Best variant from Phase 2 (reflection + ε) |
| `+workers_no_monitor` | Add worker diversity but no convergence monitor |
| `+monitor_no_workers` | Single worker type but with convergence monitor (adjusts temperature only) |
| `+workers+monitor` | Both: full worker diversity with active convergence monitoring |

### Expected Outcomes

- Worker diversity alone should show higher final scores through better search coverage, but may show instability (explorer wastes compute on bad rewrites when close to SOTA)
- Convergence monitor alone should prevent the plateau behavior visible in OpenEvolve's later generations
- Combined should show the best of both: diverse search that self-corrects when it starts to collapse

### Exit Criteria

- Worker yield CSV shows all three worker types producing improvements (none is dead weight)
- Convergence monitor fires at least once during a 200-iteration run (proving it detects real collapse)
- Combined variant beats Phase 2 best on at least 2 of 3 benchmarks

---

## Phase 4: Intelligence Contracts + SOTA Steering (Days 11–15)

**Goal:** Add the adaptive compute allocation (F5) and SOTA-distance control signal (F6). These are the mechanisms that make REPS cost-efficient, not just score-efficient.

### Concrete Changes

**4.1 — Implement `contract_selector.py` (F5)**

```python
class ContractSelector:
    def __init__(self, config):
        self.arms = self._build_arms(config)
        # Each arm is a (model_id, temperature) pair
        # Beta(alpha, beta) posterior per arm
        self.posteriors = {
            arm: {"alpha": 1.0, "beta": 1.0} 
            for arm in self.arms
        }

    def select(self, context: dict) -> Contract:
        """Thompson sampling: sample from each posterior, pick highest."""
        samples = {}
        for arm, params in self.posteriors.items():
            samples[arm] = np.random.beta(params["alpha"], params["beta"])
        best_arm = max(samples, key=samples.get)
        return Contract(model_id=best_arm[0], temperature=best_arm[1])

    def update(self, arm: tuple, success: bool):
        """Update posterior with observation."""
        if success:
            self.posteriors[arm]["alpha"] += 1.0
        else:
            self.posteriors[arm]["beta"] += 1.0

    def _build_arms(self, config) -> list:
        models = config.get("models", ["gemini-2.5-flash", "gemini-2.5-pro"])
        temperatures = config.get("temperatures", [0.3, 0.7, 1.0])
        return list(product(models, temperatures))
```

**4.2 — Wire contracts into IterationConfig dispatch**

The contract selector runs in the controller, **before** dispatching each iteration to the process pool. It overrides the `model_id` and `temperature` fields in the `IterationConfig` that `WorkerPool.build_iteration_config()` produces.

```python
# In controller.py, after WorkerPool builds the base config:
for config in configs:
    context = {
        "current_best": self.database.best_score,
        "sota_gap": self.sota_controller.current_gap if self.sota_controller else None,
        "batch_number": self.batch_count,
        "parent_complexity": self.database.get_complexity(config.parent_id),
        "worker_type": config.worker_type,
    }
    contract = self.contract_selector.select(context)
    config.model_id = contract.model_id
    config.temperature = contract.temperature  # overrides worker default

# After batch returns:
for result in batch_results:
    arm = (result.model_id, result.temperature)
    improved = result.scores["primary"] > result.parent_score
    self.contract_selector.update(arm, improved)
```

**4.3 — Implement `sota_controller.py` (F6)**

```python
class SOTAController:
    def __init__(self, target_score: float, config):
        self.target = target_score
        self.gap_history = []

    def get_regime(self, current_best: float) -> SearchRegime:
        gap = (self.target - current_best) / self.target
        self.gap_history.append(gap)
        
        if gap > 0.30:
            return SearchRegime.FAR
        elif gap > 0.10:
            return SearchRegime.MID
        elif gap > 0.02:
            return SearchRegime.NEAR
        else:
            return SearchRegime.POLISHING

    def get_prompt_injection(self, regime: SearchRegime) -> str:
        injections = {
            SearchRegime.FAR: (
                "You are far from the best known result. "
                "Try fundamentally different algorithmic approaches. "
                "Radical restructuring is encouraged."
            ),
            SearchRegime.MID: (
                "The current approach has merit but needs structural improvements. "
                "Look for algorithmic inefficiencies and consider hybrid strategies."
            ),
            SearchRegime.NEAR: (
                "You are close to the best known result. "
                "Focus on surgical parameter tuning and micro-optimizations. "
                "Small, precise changes are more likely to help than large rewrites."
            ),
            SearchRegime.POLISHING: (
                "You are within 2% of the best known result. "
                "Only make changes you are highly confident will improve the score. "
                "Verify numerical precision and edge cases."
            ),
        }
        return injections[regime]

    def modulate_worker_allocation(self, regime, worker_pool):
        """Adjust worker allocation based on SOTA gap regime."""
        allocations = {
            SearchRegime.FAR:       {"exploiter": 0.3, "explorer": 0.5, "crossover": 0.2},
            SearchRegime.MID:       {"exploiter": 0.5, "explorer": 0.3, "crossover": 0.2},
            SearchRegime.NEAR:      {"exploiter": 0.7, "explorer": 0.15, "crossover": 0.15},
            SearchRegime.POLISHING: {"exploiter": 0.85, "explorer": 0.05, "crossover": 0.10},
        }
        worker_pool.set_allocation(allocations[regime])
```

**4.4 — Add compute signature tracking (F7)**

Extend the database record to include `ComputeSignature` and log it per candidate.

```python
@dataclass
class ComputeSignature:
    tokens_in: int
    tokens_out: int
    model_id: str
    worker_type: str
    temperature: float
    wall_clock_seconds: float
    attempts_before_success: int
```

**4.5 — Modify prompt sampler for `{sota_injection}` field**

Add the SOTA controller's prompt injection to the template alongside the existing `{reflection}` field.

### Ablation Matrix

| Variant | Description |
|---|---|
| `phase3_best` | Best cumulative variant from Phase 3 |
| `+contracts` | Add intelligence contracts (Thompson sampling over model × temp) |
| `+sota_steering` | Add SOTA-distance prompt injection + worker reallocation |
| `+contracts+sota` | Both |

### Expected Outcomes

- Intelligence contracts should reduce cost per improvement (the system learns to use Flash for easy improvements and Pro for hard ones)
- SOTA steering should improve final scores on problems with known targets (circle packing especially, since AlphaEvolve's SOTA is published)
- Cost tracking should show the combined system reaching the same scores as Phase 3 with fewer total tokens

### Exit Criteria

- Thompson sampling posteriors visibly differentiate between arms (not all equal after 200 iterations)
- SOTA controller transitions through at least 2 regimes during a circle packing run
- Cost per improvement metric ($/score_delta) is lower than Phase 3 best

---

## Phase 5: Enriched Annotations + Integration Hardening (Days 16–20)

**Goal:** Add the OrgEvolve-style program annotations (F8), integrate all mechanisms into a polished system, run final paper-ready benchmarks with extended iterations.

### Concrete Changes

**5.1 — Extend database records with annotations (F8)**

Modify `database.py` to store the enriched record format described in F8. The Reflection Engine now writes `hypothesis` and `outcome` fields per candidate, not just per batch.

```python
# After reflection runs, annotate individual candidates:
for candidate, reflection in zip(top_candidates, candidate_reflections):
    candidate.annotations = {
        "hypothesis": reflection.get("hypothesis", ""),
        "outcome": reflection.get("outcome", ""),
        "salvageable": reflection.get("salvageable_components", []),
        "dead_end": reflection.get("dead_end", False),
    }
    self.database.update_annotations(candidate.id, candidate.annotations)
```

**5.2 — Add dead-end awareness to prompt sampler**

When building prompts inside the iteration worker process, the sampler queries the **DB snapshot** (which includes annotations serialized from the controller's live database) for dead-end flags and related failures. This context is included in the prompt as negative examples.

```python
# In prompt_sampler.py (runs in iteration worker with DB snapshot):
def _build_negative_context(self, parent, db_snapshot):
    """Gather known dead ends related to this parent's lineage."""
    dead_ends = db_snapshot.get_related_dead_ends(parent.id)
    if dead_ends:
        return "KNOWN DEAD ENDS (do not repeat these approaches):\n" + \
               "\n".join(f"- {de.hypothesis}: {de.outcome}" for de in dead_ends)
    return ""
```

**Note:** This requires that `database.py`'s snapshot serialization includes the `annotations` dict for each program. Verify in Phase 1 step 1.0 (execution model audit) that the snapshot mechanism can carry this additional data without exceeding serialization limits.

**5.3 — Full integration test**

Run the complete REPS stack (all features from Phases 2–5) on all three benchmarks with extended iterations:

```bash
python scripts/run_ablation.py \
    --benchmark circle_packing,function_minimization,symbolic_regression \
    --variants baseline,phase2_best,phase3_best,phase4_best,full_reps \
    --seeds 42,123,456,789,1337 \
    --iterations 500
```

**5.4 — Generate final comparison figures**

```bash
python scripts/plot_results.py \
    --results_dir results/ \
    --output figures/ \
    --plots score_curves,ablation_table,worker_yield,convergence_events,contract_posteriors,cost_efficiency
```

### Final Ablation Table (Paper-Ready)

| Variant | Circle Packing (n=26) | Func Min | Symb Reg | Cost ($) | Tokens (M) |
|---|---|---|---|---|---|
| OpenEvolve baseline | | | | | |
| + Reflection (F1) | | | | | |
| + ε-Revisitation (F2) | | | | | |
| + Workers (F3) | | | | | |
| + Convergence Monitor (F4) | | | | | |
| + Intelligence Contracts (F5) | | | | | |
| + SOTA Steering (F6) | | | | | |
| + Annotations (F8) | | | | | |
| **Full REPS** | | | | | |

Each cell: mean ± std across 5 seeds.

### Exit Criteria

- Full REPS beats OpenEvolve baseline on all three benchmarks
- At least 3 individual features show statistically significant improvement over their respective baselines
- No individual feature hurts performance (justifying inclusion in the full stack)
- Total cost for a 500-iteration circle packing run is documented

---

## Phase 6 (Future): Meta-Level World Model + RL Fine-Tuning

**Deferred until Phases 1–5 produce sufficient trajectory data.**

### F9 — World Model

Train a small MLP on `(archive_state_features, worker_allocation) → (predicted_fitness_delta, predicted_niche_delta)` using trajectory data from Phase 5 runs. Use it for Dyna-Q style planning over worker allocation.

### DG + GRPO Fine-Tuning

Collect search trajectories from all Phase 5 runs. Fine-tune a small model using Delightful Policy Gradient with GRPO for group-relative advantage estimation. The fine-tuned model replaces one of the ensemble members, specializing in the specific edit patterns that produce score improvements on these benchmarks.

---

## Config Template

```yaml
# configs/reps_circle_packing.yaml
max_iterations: 200
random_seed: 42

# LLM ensemble (used by all workers)
llm:
  models:
    - name: "gemini-2.5-pro"
      api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
      weight: 0.5
    - name: "gemini-2.5-flash"
      api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
      weight: 0.5

# REPS-specific settings
reps:
  # Batch size: number of iterations dispatched per wave to the process pool.
  # All REPS controller-side modules (reflection, convergence, SOTA) run at batch boundaries.
  batch_size: 10

  # F1: Reflection
  reflection:
    enabled: true
    top_k: 3
    bottom_k: 2
    model: "gemini-2.5-flash"  # use cheap model for reflection

  # F2: ε-revisitation
  revisitation:
    enabled: true
    epsilon_start: 0.15
    epsilon_end: 0.05
    decay: "linear"
    recency_window: 50

  # F3: Worker diversity
  workers:
    types: ["exploiter", "explorer", "crossover"]
    initial_allocation:
      exploiter: 0.6
      explorer: 0.25
      crossover: 0.15
    exploiter_temperature: 0.3
    explorer_temperature: 1.0

  # F4: Convergence monitor
  convergence:
    enabled: true
    window_size: 20
    entropy_threshold_mild: 0.5
    entropy_threshold_moderate: 0.3
    entropy_threshold_severe: 0.15

  # F5: Intelligence contracts
  contracts:
    enabled: true
    models: ["gemini-2.5-flash", "gemini-2.5-pro"]
    temperatures: [0.3, 0.7, 1.0]

  # F6: SOTA steering
  sota:
    enabled: true
    target_score: 2.635  # AlphaEvolve's published result for n=26

  # F8: Annotations
  annotations:
    enabled: true
    dead_end_awareness: true

# Standard OpenEvolve settings
database:
  population_size: 500
  num_islands: 5
  migration_interval: 20

evaluator:
  enable_artifacts: true
  cascade_evaluation: true
```

---

## Timeline Summary

| Phase | Days | Features Added | Key Deliverable |
|---|---|---|---|
| **1** | 1–2 | Baselines + metrics logging | Reproducible baseline scores; ablation infrastructure |
| **2** | 3–5 | Reflection (F1) + ε-revisitation (F2) | First ablation table; proof that REPS mechanisms help |
| **3** | 6–10 | Worker diversity (F3) + Convergence monitor (F4) | Structural improvement; diversity collapse prevention |
| **4** | 11–15 | Intelligence contracts (F5) + SOTA steering (F6) + Compute signatures (F7) | Cost-efficient search; adaptive compute allocation |
| **5** | 16–20 | Annotations (F8) + integration + final benchmarks | Paper-ready ablation tables and figures |
| **6** | Future | World model (F9) + DG/GRPO fine-tuning | Meta-level planning; learned search policy |