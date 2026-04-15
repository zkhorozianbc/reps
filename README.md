<h1 align="center">REPS</h1>

<p align="center">A self-improving evolutionary code search agent that reflects, diversifies, and steers.</p>

<p align="center">
  <a href="https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb"><img src="https://img.shields.io/badge/circle%20packing%20n%3D26-2.6359831%20%E2%9C%93-brightgreen" alt="Circle Packing Score"></a>
  <a href="https://github.com/algorithmicsuperintelligence/openevolve"><img src="https://img.shields.io/badge/built%20on-OpenEvolve-blue" alt="Built on OpenEvolve"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
</p>

REPS extends [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) with adaptive meta-cognition for evolutionary code search. Between iterations, the controller reflects on which mutations worked, rebalances search strategies, and steers compute based on distance to known targets.

## Result: Circle Packing n=26

| System | sum_radii | Iterations | Model |
|---|---|---|---|
| Prior SOTA | 2.634 | — | — |
| OpenEvolve (shipped best) | 2.6342924 | 470 | gemini-2.0-flash + claude-3.7-sonnet |
| AlphaEvolve (paper) | 2.6358628 | — | Gemini 2.0 Pro |
| FICO Xpress Solver | 2.6359155 | — | — |
| **REPS** | **2.6359831** | **100** | **claude-sonnet-4.6** |

Verified against [DeepMind's official validator](https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb). Full precision: **2.6359830849173465**.

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
| **REPS** | **2.6359831** | **100** | **claude-sonnet-4.6** |

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

## Quickstart

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), an [OpenRouter](https://openrouter.ai/) API key.

```bash
git clone https://github.com/zkhorozianbc/reps.git
cd reps/openevolve
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
```

Set your API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Run REPS on circle packing (n=26):

```bash
uv run python openevolve-run.py \
  examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config ../experiment/configs/circle_sonnet_reps.yaml \
  --iterations 100
```

Results go to `openevolve_output/`. Best program is in `openevolve_output/best/`.

## Configs

All configs are in `experiment/configs/`. Each pair is identical except for the `reps:` section.

| Config | Model | REPS |
|---|---|---|
| `circle_sonnet_base.yaml` | claude-sonnet-4.6 | off |
| `circle_sonnet_reps.yaml` | claude-sonnet-4.6 | on |
| `circle_base.yaml` | gemini-2.0-flash | off |
| `circle_reps.yaml` | gemini-2.0-flash | on |

All configs use OpenRouter as the API provider. To use a different provider, change `api_base` and `api_key` in the config.

## Tests

```bash
cd openevolve
uv run python -m pytest tests/ --ignore=tests/integration
```
