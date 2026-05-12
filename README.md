<h1 align="center">REPS</h1>

<p align="center">Recursive Evolutionary Program Search for LLM-driven code optimization.</p>

<p align="center">
  <a href="https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb"><img src="https://img.shields.io/badge/circle%20packing%20n%3D26-2.6359831%20%E2%9C%93-brightgreen" alt="Circle Packing Score"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
</p>

REPS evolves whole Python programs. You give it a seed program and an
evaluator; it generates candidate edits, scores them, reflects on what worked,
keeps a diverse population, and steers compute toward better programs.

## At a Glance

- **Use it as a library** with `reps.Optimizer(...).optimize(initial, evaluate)`.
- **Use it as a harness** with `reps-run --config <config.yaml>` for reproducible experiments.
- **Score simply or richly**: evaluators can return a float, a metrics dict, or `reps.EvaluationResult` with per-instance scores and feedback.
- **Search with memory**: REPS includes island diversity, reflection, revisitation, convergence monitoring, Pareto selection, trace reflection, and merge-aware crossover.

## Install

Requires Python 3.12+.

```bash
pip install reps-py
```

For local development:

```bash
git clone https://github.com/zkhorozianbc/reps.git
cd reps
uv venv .venv --python 3.12
uv pip install -e ".[dev,benchmarks]"
```

Set the API key for the provider you use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...
```

A sibling `.env` file is auto-loaded by the CLI.

## Quick Start

```python
import reps

seed = """
def solve():
    return 1
"""

TARGET = 42

def evaluate(code: str) -> float:
    namespace = {}
    try:
        exec(code, namespace)  # For demos only; sandbox untrusted code in real runs.
        return -abs(TARGET - float(namespace["solve"]()))
    except Exception:
        return -1_000_000.0

result = reps.Optimizer(
    model="anthropic/claude-sonnet-4-6",
    max_iterations=10,
).optimize(initial=seed, evaluate=evaluate)

print(result.best_score)
print(result.best_code)
```

See [`examples/basic_optimizer.py`](examples/basic_optimizer.py) for the same
shape as a runnable script.

## Benchmark Achievements

REPS' headline run improves the Circle Packing n=26 benchmark in 100
iterations, verified against DeepMind's official validator.

| System | sum_radii | Iterations | Model |
|---|---:|---:|---|
| Prior SOTA | 2.634 | n/a | n/a |
| OpenEvolve shipped best | 2.6342924 | 470 | gemini-2.0-flash + claude-3.7-sonnet |
| AlphaEvolve paper | 2.6358628 | n/a | Gemini 2.0 Pro |
| FICO Xpress Solver | 2.6359155 | n/a | n/a |
| **REPS** | **2.6359831** | **100** | **claude-sonnet-4.6** |

Verified with the [DeepMind validator](https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb).

![REPS Circle Packing](experiment/results/circle_sonnet_reps/packing.png)

## Core Concepts

- **Seed program**: the first candidate. For CLI benchmarks, wrap the evolvable region in `EVOLVE-BLOCK` markers.
- **Evaluator**: a callable or `evaluator.py` that runs a candidate and returns a higher-is-better score.
- **Population**: candidates are stored across islands so the search can keep multiple promising directions alive.
- **Workers**: exploiters refine good parents, explorers try larger pivots, and crossover workers merge ideas.
- **Reflection**: REPS summarizes wins, failures, and lineage context so later candidates are not blind retries.

## Evaluators

For the Python API, an evaluator is any callable that accepts candidate code:

```python
def eval_float(code: str) -> float:
    return 1.0

def eval_dict(code: str) -> dict:
    return {
        "combined_score": 0.9,
        "per_instance_scores": {"case_a": 1.0, "case_b": 0.8},
        "feedback": "case_b is the weak spot",
    }

def eval_full(code: str) -> reps.EvaluationResult:
    return reps.EvaluationResult(
        metrics={"combined_score": 0.9},
        per_instance_scores={"case_a": 1.0, "case_b": 0.8},
        feedback="case_b is the weak spot",
    )
```

Use `per_instance_scores` with `selection_strategy="pareto"` or `"mixed"` when
you want REPS to preserve candidates that solve different parts of the task.
Use `feedback` with `trace_reflection=True` when evaluator diagnostics should
guide the next mutation.

## Main Knobs

| Kwarg | Effect | Default |
|---|---|---|
| `max_iterations` | Search budget | `100` |
| `num_islands` | Independent population islands | `5` |
| `selection_strategy` | `"map_elites"`, `"pareto"`, or `"mixed"` | `"map_elites"` |
| `pareto_fraction` | Blend ratio for `"mixed"` selection | `0.0` |
| `trace_reflection` | Reflect on per-instance scores and feedback | `False` |
| `lineage_depth` | Ancestors included in trace reflection | `3` |
| `merge` | Complementarity-aware crossover parent selection | `False` |
| `output_dir` | Persist run artifacts; `None` uses a temp directory | `None` |

The full public surface is documented in
[`docs/python_api_spec.md`](docs/python_api_spec.md).

## Examples

| Path | Shows |
|---|---|
| [`examples/basic_optimizer.py`](examples/basic_optimizer.py) | Minimal Python API run with a scalar score |
| [`examples/rich_evaluator.py`](examples/rich_evaluator.py) | `EvaluationResult`, per-instance scores, feedback, Pareto/mixed selection |
| [`examples/reuse_model.py`](examples/reuse_model.py) | Sharing one `reps.Model` across optimizer runs |
| [`examples/custom_benchmark/`](examples/custom_benchmark/) | A CLI benchmark directory with `initial_program.py`, `evaluator.py`, `system_prompt.md`, and `config.yaml` |

## CLI Experiments

Use the CLI when you want YAML configs, repeatable runs, worker pools, and
artifact directories:

```bash
reps-run --config experiment/configs/circle_sonnet_reps.yaml
reps-run --config experiment/configs/circle_sonnet_reps.yaml --iterations 50
reps-run --config experiment/configs/circle_sonnet_reps.yaml -o llm.temperature=0.9
```

Results land in `experiment/results/<config-stem>/run_NNN/` by default. Each
run saves `best_program.py`, serialized candidates, metrics, logs, and optional
visualizations.

To add a benchmark, create:

```text
my_benchmark/
├── initial_program.py
├── evaluator.py
└── system_prompt.md
```

Then point a config at it:

```yaml
task: ./my_benchmark
```

See [`examples/custom_benchmark/`](examples/custom_benchmark/) for a complete
small version, and [`experiment/benchmarks/`](experiment/benchmarks/) for the
larger bundled benchmarks.

## Bundled Benchmarks

- [`experiment/benchmarks/circle_packing/`](experiment/benchmarks/circle_packing/) - Circle Packing n=26.
- [`experiment/benchmarks/circle_packing_n32/`](experiment/benchmarks/circle_packing_n32/) - Circle Packing n=32.
- [`experiment/benchmarks/online_bin_packing/`](experiment/benchmarks/online_bin_packing/) - Online 1-D bin packing with FunSearch-style datasets.

Reference configs live in [`experiment/configs/`](experiment/configs/).
`reps/config.py` is the source of truth for every YAML field.

## Status

REPS is pre-1.0. Minor version bumps may include breaking changes until
`1.0.0`; pin to a minor series such as `reps-py==0.2.*` when you need stable
upgrades. See [`docs/release_spec.md`](docs/release_spec.md).

## Tests

```bash
uv run python -m pytest tests/
```

## Design Docs

- [`docs/python_api_spec.md`](docs/python_api_spec.md) - public API contract.
- [`docs/gepa_implementation_plan.md`](docs/gepa_implementation_plan.md) - Pareto selection, trace reflection, merge, and lineage rollout.
- [`docs/optimizer_engine_separation_spec.md`](docs/optimizer_engine_separation_spec.md) - public facade and runtime engine split.
- [`docs/release_runbook.md`](docs/release_runbook.md) - release checklist.

## Acknowledgements

Forked from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve);
now self-contained.
