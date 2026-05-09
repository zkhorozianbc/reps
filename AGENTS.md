# AGENTS.md

## Project

REPS (Recursive Evolutionary Program Search) — a self-contained harness for evolutionary code search with adaptive meta-cognition. Uses LLMs to evolve programs through reflection, worker diversity, convergence detection, and SOTA steering.

## Running

```bash
uv pip install -e .
reps-run <initial_program> <evaluator> --config <config.yaml> --output <output_dir> --iterations N
```

Tests: `uv run python -m pytest tests/`

## Rules

### Never ignore errors or warnings

When you see an error or warning in logs, test output, or runtime output:

1. **Stop and investigate immediately.** Read the full traceback. Understand the root cause.
2. **Ask: is this a harness bug or genuinely expected behavior?** If the seed program itself would error (e.g., missing dependency), that's infrastructure — not "LLM noise."
3. **Fix before continuing.** Don't waste API credits running a broken pipeline.
4. **Verify the fix.** Run the failing case again and confirm the error is gone.
5. **Document the investigation** in the commit message so the reasoning is preserved.

This applies to every error, every warning, every unexpected output. There is no such thing as "expected noise" that you can skip investigating. If it looks like noise, prove it by understanding the cause — don't assume.

### Pre-run validation

Before launching an experiment run:

1. Verify the seed program evaluates successfully: `uv run python -c "import sys; sys.path.insert(0, '<evaluator_dir>'); from evaluator import evaluate; print(evaluate('<initial_program>'))"` 
2. Verify all imports the seed program uses are installed (`scipy`, `numpy`, etc.)
3. Check that the config's `provider` and `api_key` match (anthropic provider needs `ANTHROPIC_API_KEY`, openrouter needs `OPENROUTER_API_KEY`)

### Code structure

- `reps/` — self-contained harness (config, controller, LLM providers, evaluator, database, REPS features)
- `experiment/benchmarks/` — benchmark problems (evaluator.py + initial_program.py per benchmark)
- `experiment/configs/` — YAML experiment configs
- `tests/` — pytest suite (111 tests)
