# REPS Examples

These examples are intentionally small. They show the shape of REPS programs,
evaluators, and configs without requiring the bundled research benchmarks.

## Python API

- [`basic_optimizer.py`](basic_optimizer.py) is the first-run path: a `reps.Objective` scores a `predict(x)` entrypoint against a train set with the `mae` metric — no hand-written evaluator.
- [`llm_judge.py`](llm_judge.py) uses `reps.LLMJudge` to score subjective `answer(question)` outputs with a separate judge model.
- [`rich_evaluator.py`](rich_evaluator.py) shows the `evaluate=` escape hatch: a raw callable returning `reps.EvaluationResult` with per-instance scores and feedback.
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
