# REPS: Recursive Evolutionary Program Search

REPS extends [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) with self-improving meta-cognition for evolutionary code search. Instead of blindly sampling LLM mutations, the controller reflects on what's working, diversifies search strategies, and steers compute toward productive directions.

## Result: Circle Packing n=26

| System | sum_radii | Iterations | Phases | Model |
|---|---|---|---|---|
| OpenEvolve (shipped best) | 2.634292402141039 | 470 | 2 | gemini-2.0-flash + claude-3.7-sonnet |
| AlphaEvolve (paper) | 2.635896 | — | — | Gemini 2.0 Pro |
| **REPS** | **2.63598308491761** | **100** | **1** | **claude-sonnet-4.6** |

REPS exceeds the AlphaEvolve paper result by +0.00008708491761 sum_radii, in 100 iterations with a single phase. OpenEvolve required 470 iterations across 2 manually-configured phases to reach 2.634292402141039.

![REPS Circle Packing](experiment/results/circle_sonnet_reps/packing.png)

## What REPS Does

Every iteration is still a single LLM call — one prompt in, one code edit out. REPS makes the controller smarter about orchestrating those calls:

- **Reflection (F1):** Every 5 iterations, an LLM analyzes what edits worked/failed and injects structured insight into future prompts
- **Worker Diversity (F3):** Three mutation strategies — exploiter (small diffs, low temp), explorer (full rewrites, high temp), crossover (merges two parents from different islands)
- **Convergence Detection (F4):** Measures Shannon entropy over edit types; forces diversification when the search collapses
- **Contract Selection (F5):** Thompson-sampling bandit learns which model/temperature combos produce improvements
- **SOTA Steering (F6):** When the target score is known, modulates exploration vs exploitation based on the gap
- **Epsilon-Revisitation (F2):** Revisits high-scoring but underexplored programs with alternative worker types
- **Annotations (F8):** Tags programs with dead-end warnings so future prompts avoid known failures

## Benchmarks

### Circle Packing (n=26)

Pack 26 non-overlapping circles into a unit square, maximizing sum of radii.

| Variant | sum_radii | Iterations | Model |
|---|---|---|---|
| Baseline (unmodified OpenEvolve) | 2.497400 | 100 | claude-sonnet-4.6 |
| **REPS** | **2.63598308491761** | **100** | **claude-sonnet-4.6** |

Same model, same config, same seed. REPS achieves +5.6% higher sum_radii.

With Gemini flash models (cheaper, weaker):

| Variant | sum_radii | Iterations | Model |
|---|---|---|---|
| Baseline (unmodified OpenEvolve) | 2.128300 | 100 | gemini-2.0-flash + gemini-2.5-flash-lite |
| REPS | 2.219700 | 100 | gemini-2.0-flash + gemini-2.5-flash-lite |

### Function Minimization

Find the global minimum of f(x,y) = sin(x)cos(y) + sin(xy) + (x^2+y^2)/20.

Both baseline and REPS saturate at combined_score ~1.4995 within 100 iterations. This benchmark is too easy to differentiate the approaches — both find simulated annealing and converge to the same ceiling.

## Architecture

```
Controller (single process, stateful)
├── WorkerPool          ← builds IterationConfig per dispatch
├── ReflectionEngine    ← LLM call between batches
├── ConvergenceMonitor  ← edit entropy tracking
├── ContractSelector    ← Thompson-sampling model selection
├── SOTAController      ← gap-aware regime switching
├── MetricsLogger       ← CSV output
└── dispatches IterationConfig → ProcessPoolExecutor
    └── _run_iteration_worker (stateless)
        ├── receives: DB snapshot + IterationConfig
        ├── interprets: worker_type, generation_mode, temperature, prompt_extras
        └── returns: SerializableResult + REPS metadata
```

REPS modules run in the controller process at batch boundaries. Workers are stateless — they execute whatever config they receive. The underlying OpenEvolve LLM module, evaluator, and database are unmodified.

## Setup

```bash
cd openevolve_src
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

## Running

```bash
# Baseline (unmodified OpenEvolve)
cd openevolve_clean
uv run python openevolve-run.py \
  examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config <config.yaml> \
  --iterations 100

# REPS
cd openevolve_src
uv run python openevolve-run.py \
  examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config <reps_config.yaml> \
  --iterations 100
```

## Experiment Configs

All configs are in `experiment/configs/`. The baseline and REPS configs are identical except for the `reps:` section — verified by YAML parse comparison.

- `circle_base.yaml` — shipped OpenEvolve config, only api_base changed for OpenRouter
- `circle_reps.yaml` — same + REPS features enabled
- `circle_sonnet_base.yaml` — Sonnet 4.6 baseline
- `circle_sonnet_reps.yaml` — Sonnet 4.6 + REPS

## Tests

```bash
cd openevolve_src
uv run python -m pytest tests/ --ignore=tests/integration
# 350 passed
```
