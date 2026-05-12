# REPS Examples

These examples are intentionally small. They show the shape of REPS programs,
evaluators, and configs without requiring the bundled research benchmarks.

## Python API

- [`basic_optimizer.py`](basic_optimizer.py) starts from a tiny `solve()` seed and scores by distance from a target number.
- [`rich_evaluator.py`](rich_evaluator.py) returns `reps.EvaluationResult` with per-instance scores and feedback for trace reflection.
- [`reuse_model.py`](reuse_model.py) builds one `reps.Model` and shares it across optimizer instances.

Run one with:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/basic_optimizer.py
```

Set `REPS_MODEL` to use a different model string:

```bash
REPS_MODEL=openrouter/anthropic/claude-sonnet-4.6 uv run python examples/basic_optimizer.py
```

## CLI Benchmark

[`custom_benchmark/`](custom_benchmark/) is a complete benchmark scaffold for
`reps-run`:

```bash
reps-run --config examples/custom_benchmark/config.yaml
```

It contains the same pieces a larger benchmark needs: `initial_program.py`,
`evaluator.py`, `system_prompt.md`, and a YAML config.
