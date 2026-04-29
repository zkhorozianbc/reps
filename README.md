<h1 align="center">REPS</h1>

<p align="center">A self-improving evolutionary code search agent that reflects, diversifies, and steers.</p>

<p align="center">
  <a href="https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb"><img src="https://img.shields.io/badge/circle%20packing%20n%3D26-2.6359831%20%E2%9C%93-brightgreen" alt="Circle Packing Score"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
</p>

REPS evolves programs with an LLM-driven loop that reflects between batches, balances explorer/exploiter workers, detects convergence, and steers compute by distance to a known target. Forked from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve); now self-contained.

## Result: Circle Packing n=26

| System | sum_radii | Iterations | Model |
|---|---|---|---|
| Prior SOTA | 2.634 | — | — |
| OpenEvolve (shipped best) | 2.6342924 | 470 | gemini-2.0-flash + claude-3.7-sonnet |
| AlphaEvolve (paper) | 2.6358628 | — | Gemini 2.0 Pro |
| FICO Xpress Solver | 2.6359155 | — | — |
| **REPS** | **2.6359831** | **100** | **claude-sonnet-4.6** |

Verified against [DeepMind's official validator](https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb).

![REPS Circle Packing](experiment/results/circle_sonnet_reps/packing.png)

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zkhorozianbc/reps.git
cd reps
uv venv .venv --python 3.12
uv pip install -e .
```

Set the API key matching your config's `provider:`

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # provider: anthropic
export OPENROUTER_API_KEY=sk-or-...      # provider: openrouter
export OPENAI_API_KEY=sk-...             # provider: openai
```

A sibling `.env` file is auto-loaded.

## Run

Everything lives in the YAML — point `reps-run` at a config and go:

```bash
reps-run --config experiment/configs/circle_sonnet_reps.yaml
```

Results land in `experiment/results/<config-stem>/run_NNN/` (auto-versioned). The best program is saved as `best_program.py`; per-iteration metrics under `metrics/`.

Common overrides:

```bash
reps-run --config <yaml> --iterations 50 --output my_runs/
reps-run --config <yaml> -o llm.temperature=0.9 -o reps.batch_size=10
```

The config decides everything else — model, workers, harness (`reps` or `openevolve`), and which benchmark to evolve (via `task:`).

## Add a Benchmark

Drop two files into `experiment/benchmarks/<name>/`:

```
experiment/benchmarks/<name>/
├── initial_program.py    # seed code (wrap evolvable region in EVOLVE-BLOCK markers)
└── evaluator.py          # defines evaluate(program_path) -> {"combined_score": float, ...}
```

`initial_program.py`:

```python
# EVOLVE-BLOCK-START
def solve():
    return naive_result
# EVOLVE-BLOCK-END
```

`evaluator.py`:

```python
def evaluate(program_path):
    # import program_path, run it, score it
    return {"combined_score": score}
```

Optional files in the same directory:
- `system_prompt.md` — task-specific system prompt (auto-loaded)
- `visualize.py` — `visualize_from_program(path, save_path)` for best-program plots

Then point a config at it:

```yaml
task: ../benchmarks/<name>     # resolved relative to this YAML
max_iterations: 100
provider: anthropic
# ... see experiment/configs/circle_sonnet_reps.yaml for a full example
```

Run it: `reps-run --config experiment/configs/<your_config>.yaml`.

For cascade evaluation, also define `evaluate_stage1` / `evaluate_stage2`. If the primary objective metric isn't `combined_score`, set `reps.sota.target_metric:` so SOTA steering compares the right value.

## Configs

Reference configs in `experiment/configs/`:

- `circle_sonnet_reps.yaml`, `circle_opus47_anthropic.yaml`, `reps_full.yaml` — full REPS runs
- `verify_*.yaml` — minimal smoke tests, one per worker impl
- `circle_base.yaml`, `circle_sonnet_base.yaml` — `harness: openevolve` baselines (`uv pip install openevolve`)

`reps/config.py` is the source of truth for every field and default.

## Tests

```bash
uv run python -m pytest tests/
```
