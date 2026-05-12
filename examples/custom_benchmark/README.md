# Custom Benchmark Example

This directory is a minimal `reps-run` benchmark. It asks REPS to evolve
`predict(x)` so it approximates:

```python
x * x - 3 * x + 2
```

Run it from the repository root:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
reps-run --config examples/custom_benchmark/config.yaml
```

The benchmark pieces are:

- `initial_program.py` - seed code with an `EVOLVE-BLOCK`.
- `evaluator.py` - imports a candidate file and returns metrics, per-instance scores, and feedback.
- `system_prompt.md` - task-specific instructions for the mutation model.
- `config.yaml` - small REPS run using the `single_call` worker.
