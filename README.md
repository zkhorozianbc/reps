<h1 align="center">REPS</h1>

<p align="center">A self-improving evolutionary code search agent that reflects, diversifies, and steers.</p>

<p align="center">
  <a href="https://colab.research.google.com/github/google-deepmind/alphaevolve_repository_of_problems/blob/main/experiments/packing_circles_max_sum_of_radii/packing_circle_max_sum_of_radii.ipynb"><img src="https://img.shields.io/badge/circle%20packing%20n%3D26-2.6359831%20%E2%9C%93-brightgreen" alt="Circle Packing Score"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
</p>

REPS is an evolutionary code search harness with adaptive meta-cognition. Between iterations, the controller reflects on which mutations worked, rebalances search strategies, and steers compute based on distance to known targets. It started as a fork of [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve); the harness is now self-contained.

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

The controller orchestrates one Worker per iteration (see [Workers](#workers) for the four implementations). Between batches it runs adaptive features:

- **Reflection (F1):** Every batch, an LLM analyzes which edits worked or failed and injects structured insight into future prompts.
- **Worker Diversity (F3):** Multiple worker configs sampled by weight (e.g. exploiter / explorer / crossover) so the search keeps multiple strategies in play.
- **Convergence Detection (F4):** Measures Shannon entropy over edit types; forces diversification when the search collapses.
- **Contract Selection (F5):** Thompson-sampling bandit learns which model/temperature combos produce improvements.
- **SOTA Steering (F6):** When the target score is known, modulates exploration vs exploitation based on the gap.
- **Epsilon-Revisitation (F2):** Revisits high-scoring but underexplored programs with alternative worker types.
- **Annotations (F8):** Tags programs with dead-end warnings so future prompts avoid known failures.

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

## Architecture

```
reps/
├── runner.py              # CLI entry (`reps-run`); dotenv, output versioning
├── controller.py          # Async evolution loop with REPS orchestration
├── config.py              # Dataclass-based YAML config (single source of truth)
├── database.py            # In-memory MAP-Elites + island archive (with checkpoints)
├── evaluator.py           # Cascade evaluator with per-iteration semaphore
├── prompt_sampler.py      # Builds diff/full-rewrite prompts from templates
├── prompt_templates.py    # Loads prompt_templates/ *.txt + fragments.json
├── prompt_templates/      # The actual template text files
├── runtime.py             # ContextVar program-id propagation for asyncio
├── async_utils.py         # TaskPool helpers
├── reflection_engine.py   # F1
├── worker_pool.py         # F3 sampler / allocation / mutators
├── convergence_monitor.py # F4
├── contract_selector.py   # F5 Thompson sampler over (model, temp) arms
├── sota_controller.py     # F6
├── program_summarizer.py  # Per-iteration LLM summary used by F8 annotations
├── metrics_logger.py      # CSV + JSONL emitters under output_dir/metrics/
├── embedding.py           # OpenAI/Azure embeddings (optional novelty path)
├── novelty_judge.py       # LLM-as-judge novelty check (optional)
├── llm/
│   ├── base.py            # LLMInterface ABC + retry helper
│   ├── anthropic.py       # Native Anthropic client
│   ├── openai_compatible.py  # OpenAI / OpenRouter / any OpenAI-shape gateway
│   ├── ensemble.py        # Weighted-sample model ensemble
│   ├── provider_of.py     # Heuristic: model_id → provider
│   └── stream_print.py    # Pretty-print streaming chunks to stderr
└── workers/
    ├── base.py            # Worker Protocol + dataclasses (Config/Request/Result/TurnRecord)
    ├── registry.py        # @register decorator + build_worker
    ├── single_call.py     # One-shot LLM call (default minimal)
    ├── anthropic_tool_runner.py  # Native Anthropic tool-use loop
    ├── openai_tool_runner.py     # OpenAI Responses API tool-use loop
    ├── dspy_react.py      # DSPy ReAct (Anthropic via LiteLLM)
    ├── tools.py           # view_parent / edit_file / run_tests / submit_child / mark_converged
    ├── edit_serializer.py # SEARCH/REPLACE block formatter
    ├── _runner_common.py  # Shared helpers between the two tool runners
    └── trace_render.py    # Pretty-print TurnRecord traces for logs
```

REPS modules run in the controller process at batch boundaries. Workers handle one iteration each — single_call is one prompt → one edit; the tool-runner workers loop with tools (view_parent, edit_file, run_tests, submit_child) until they `submit_child` or hit `max_turns`.

## Quickstart

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zkhorozianbc/reps.git
cd reps
uv venv .venv --python 3.12
uv pip install -e .
# Optional: install OpenEvolve for baseline comparison via `harness: openevolve`
uv pip install openevolve
```

Set your API key — depending on provider:

```bash
export OPENROUTER_API_KEY=sk-or-...      # provider: openrouter
export ANTHROPIC_API_KEY=sk-ant-...      # provider: anthropic
export OPENAI_API_KEY=sk-...             # provider: openai (also used by openai_tool_runner)
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

Results go to `--output/run_NNN/` (auto-versioned). The best program code lives at `<run>/best_program.py`.

To run a baseline (unmodified OpenEvolve), use a config with `harness: openevolve` (requires `pip install openevolve`).

## Workers

A *worker* runs one iteration: receives a parent program + prompt context, returns a child program. The worker is selected per-iteration by F3 (worker pool sampling). Configure under `reps.workers.types` in YAML.

| Worker | When to use | Key options (`impl_options`) |
|---|---|---|
| `single_call` | One-shot LLM call. Cheapest, fastest. | `provider`, `api_key`, `api_base`, `model`, `temperature`, `max_tokens` |
| `anthropic_tool_runner` | Native Anthropic tool-use loop. Best correctness via `edit_file` + `run_tests` + `submit_child`. Supports thinking blocks, server-side `code_execution`, and `web_search`. | `thinking_effort` (`low`/`medium`/`high`/`xhigh`), `code_execution` (bool), `web_search` (bool), `task_budget_total`, `max_tokens`, `timeout` |
| `openai_tool_runner` | OpenAI Responses-API tool-use loop. Same protocol shape as anthropic_tool_runner but provider-native. | `reasoning_effort` (`low`/`medium`/`high`), `max_tokens`, `timeout` |
| `dspy_react` | DSPy ReAct over Anthropic via LiteLLM. Research / experimental. | `temperature`, `max_tokens` |

Each entry in `reps.workers.types` is a `WorkerConfig`:

```yaml
reps:
  workers:
    types:
      - name: exploiter_atr
        impl: anthropic_tool_runner
        role: exploiter            # exploiter | explorer | crossover
        weight: 0.7
        model_id: claude-sonnet-4-6
        temperature: 0.4
        max_turns: 12
        generation_mode: diff      # "diff" or "full"
        tools: [view_parent, edit_file, run_tests, submit_child]
        impl_options:
          thinking_effort: high
          code_execution: false
      - name: explorer_single
        impl: single_call
        role: explorer
        weight: 0.3
        model_id: claude-sonnet-4-6
        temperature: 0.9
        generation_mode: full
```

See `experiment/configs/verify_*.yaml` for a working example of each worker impl.

## Configuration

All configuration is a single dacite-deserialized dataclass tree rooted at `Config` in `reps/config.py`. The major sections:

| YAML key | Purpose |
|---|---|
| `provider`, `harness`, `reasoning`, `task` | Top-level run shape |
| `max_iterations`, `checkpoint_interval`, `random_seed` | Run lifecycle |
| `llm:` | Default model(s), API base/key, retries, ensemble |
| `prompt:` | Template directory, num_top/diverse programs, artifact rendering |
| `database:` | Population/archive sizes, MAP-Elites feature dimensions, island migration, embedding-based novelty |
| `evaluator:` | Timeout, cascade thresholds, `parallel_evaluations`, LLM feedback |
| `reps.reflection` (F1) | `enabled`, `top_k`, `bottom_k`, `model` |
| `reps.revisitation` (F2) | `enabled`, `epsilon_start/end`, `decay`, `recency_window` |
| `reps.workers` (F3) | `types: [WorkerConfig, ...]` (see above) |
| `reps.convergence` (F4) | `enabled`, `window_size`, entropy thresholds |
| `reps.contracts` (F5) | `enabled`, `models`, `temperatures` |
| `reps.sota` (F6) | `enabled`, `target_score`, `target_metric` |
| `reps.annotations` (F8) | `enabled`, `dead_end_awareness` |
| `reps.summarizer` | Per-iteration program summarizer model (`model_id`, `task_instructions`, optional `provider`/`api_key`) |

`reps/config.py` is the source of truth for every field, default, and forward compat note.

### Providers

- `provider: openrouter` → `OPENROUTER_API_KEY`, served via `reps/llm/openai_compatible.py`.
- `provider: anthropic` → `ANTHROPIC_API_KEY`, served via `reps/llm/anthropic.py`.
- `provider: openai` → `OPENAI_API_KEY`, served via `reps/llm/openai_compatible.py` (api_base defaults to OpenAI Direct).

The native tool-runner workers (`anthropic_tool_runner`, `openai_tool_runner`) bypass `LLMInterface` and use their provider's SDK directly to access raw content blocks (thinking, tool_use, tool_result) and Responses-API features. They take their own API key via `impl_options.api_key` (falls back to env).

## Metrics output

Each run writes to `<output>/run_NNN/metrics/`:

| File | Contents |
|---|---|
| `score_trajectory.csv` | Per-iteration best/last scores |
| `cost.csv` | Per-iteration cumulative tokens in/out (per provider) |
| `worker_yield.csv` | Per-iteration yield rate per worker |
| `diversity.csv` | Per-batch edit-entropy and worker-allocation snapshot |
| `convergence_events.jsonl` | Each time the convergence monitor escalates (MILD_BOOST, MODERATE_DIVERSIFY, SEVERE_RESTART) |
| `reflection_log.jsonl` | F1 reflection JSON per batch |

The full program archive lives at `<run>/programs/<id>.json` (one file per program) plus the snapshot `database.json`. Best program code: `<run>/best_program.py`.

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

If you have a primary objective metric distinct from `combined_score` (e.g. `sum_radii`), set `reps.sota.target_metric: <name>` in the config so F6 SOTA steering compares against the right value.

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

All configs are in `experiment/configs/`. Examples worth knowing:

- **Reference / starting points**: `reps_full.yaml`, `circle_sonnet_reps.yaml`, `circle_opus47_anthropic.yaml`.
- **Worker-impl smoke configs**: `verify_single_call.yaml`, `verify_anthropic_tool_runner.yaml`, `verify_openai_tool_runner.yaml`, `verify_dspy_react.yaml` — minimal configs exercising one worker each.
- **OpenEvolve baselines**: `circle_base.yaml`, `circle_sonnet_base.yaml` — `harness: openevolve`.
- **n=32 variants**: `circle_*_n32.yaml`.
- **SOTA-push runs**: `push_sota_opus.yaml`, `push_sota_sonnet.yaml`.

Provider can be `openrouter`, `anthropic`, or `openai`. Configs that target the native Anthropic API set `provider: anthropic` and rely on `ANTHROPIC_API_KEY`.

## Tests

```bash
uv run python -m pytest tests/
```
