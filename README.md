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
reps/
├── config.py          # All config (evolution + REPS features)
├── controller.py      # Evolution loop with REPS orchestration
├── database.py        # Program database
├── evaluator.py       # Program evaluator
├── runner.py          # CLI entry point
├── llm/               # LLM providers
│   ├── openrouter.py  # OpenAI-compatible (OpenRouter, etc.)
│   ├── anthropic.py   # Native Anthropic API
│   └── ensemble.py    # Model ensemble
├── prompt_sampler.py  # Prompt template building
├── reflection_engine.py   # F1
├── worker_pool.py         # F3
├── convergence_monitor.py # F4
├── contract_selector.py   # F5
├── sota_controller.py     # F6
└── metrics_logger.py      # Metrics CSV logging
```

REPS modules run in the controller process at batch boundaries. Workers are stateless — they execute whatever config they receive.

## Quickstart

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zkhorozianbc/reps.git
cd reps
uv venv .venv --python 3.12
uv pip install -e .
# Optional: install OpenEvolve for baseline comparison
uv pip install openevolve
```

Set your API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
# Or, to use the native Anthropic API instead:
export ANTHROPIC_API_KEY=sk-ant-...
```

Run REPS on circle packing (n=26):

```bash
reps-run \
  experiment/benchmarks/circle_packing/initial_program.py \
  experiment/benchmarks/circle_packing/evaluator.py \
  --config experiment/configs/circle_sonnet_reps.yaml \
  --output output/ \
  --iterations 100
```

Results go to the directory specified by `--output`. Best program is in `output/best/`.

To run a baseline (unmodified OpenEvolve), use a config with `harness: openevolve`.

## Adding a Benchmark

Benchmarks live in `experiment/benchmarks/<name>/`. Each benchmark needs two files:

```
experiment/benchmarks/<name>/
├── initial_program.py    # Seed program the LLM will evolve
└── evaluator.py          # Scores each candidate program
```

### `evaluator.py`

Must define an `evaluate(program_path)` function that returns a dict with at least `combined_score`:

```python
import importlib.util
import traceback

def evaluate(program_path):
    """Load a candidate program, run it, return metrics.

    Args:
        program_path: path to the .py file to evaluate

    Returns:
        dict with at least 'combined_score' (float, higher is better)
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.solve()  # call whatever your program defines

        score = ...  # compute how good the result is

        return {"combined_score": score}
    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
```

The evaluator is loaded once per worker process. It receives the path to a temporary `.py` file containing the candidate code, loads it dynamically, runs it, and returns a score. No relative imports — the evaluator must be a standalone file.

For cascade evaluation (early rejection of bad candidates), optionally define `evaluate_stage1(program_path)` and `evaluate_stage2(program_path)` alongside the main `evaluate()`.

### `initial_program.py`

The seed code the LLM starts evolving from. Wrap the evolvable portion in `EVOLVE-BLOCK` markers:

```python
# EVOLVE-BLOCK-START
import numpy as np

def solve():
    """The function the LLM will improve."""
    # naive starting solution
    return naive_result
# EVOLVE-BLOCK-END

# Code outside the block is fixed and won't be modified
if __name__ == "__main__":
    print(solve())
```

### Config

Create a YAML config in `experiment/configs/` that references your benchmark. The initial program and evaluator are passed as CLI args, not in the config. The config controls the LLM, REPS features, and evolution parameters:

```yaml
harness: reps              # "reps" or "openevolve"
provider: openrouter       # "openrouter" or "anthropic"
max_iterations: 100

llm:
  primary_model: "anthropic/claude-sonnet-4.6"
  api_key: "${OPENROUTER_API_KEY}"
  temperature: 0.7
  max_tokens: 8192
  timeout: 120

prompt:
  system_message: |
    You are an expert at <domain>. Improve the solve() function to ...
  num_top_programs: 3

evaluator:
  timeout: 60
  parallel_evaluations: 4

reps:
  enabled: true
  batch_size: 5
  # ... (see experiment/configs/ for full examples)
```

### Run

```bash
reps-run \
  experiment/benchmarks/<name>/initial_program.py \
  experiment/benchmarks/<name>/evaluator.py \
  --config experiment/configs/<your_config>.yaml \
  --output experiment/results/<name>/ \
  --iterations 100
```

See `experiment/benchmarks/circle_packing/` for a complete working example.

## Configs

All configs are in `experiment/configs/`. Each pair is identical except for the `reps:` section.

| Config | Model | REPS |
|---|---|---|
| `circle_sonnet_base.yaml` | claude-sonnet-4.6 | off |
| `circle_sonnet_reps.yaml` | claude-sonnet-4.6 | on |
| `circle_base.yaml` | gemini-2.0-flash | off |
| `circle_reps.yaml` | gemini-2.0-flash | on |

All configs use OpenRouter as the default provider. To use the native Anthropic API, set `provider: anthropic` in the config and export `ANTHROPIC_API_KEY`.

## Tests

```bash
uv run python -m pytest tests/
```
