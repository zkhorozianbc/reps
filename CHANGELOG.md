# Changelog

All notable changes to REPS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0.0. Until then, the project follows the pre-1.0 policy
documented in [`docs/release_spec.md`](docs/release_spec.md): minor bumps
may include breaking changes; only patch bumps are safe to consume blindly.

## [Unreleased]

### Added
- `reps.Example` and `reps.Prediction` — DSPy-inspired data primitives with
  explicit input keys, dot/item access, and `.with_inputs()` / `.inputs()` /
  `.labels()`. `reps.Example` accepts any dict-like object (`keys()` +
  `__getitem__`), so a `dspy.Example` from a DSPy built-in dataset
  (`Colors`, `HotPotQA`, …) drops straight in.
- `reps.Objective` — compiles a seed `entrypoint` + `train_set` + `metric`
  into the evaluator contract; `Objective.maximize` / `Objective.minimize`
  classmethods with built-in metrics (`accuracy`, `exact_match`, `mae`,
  `mse`, `rmse`) and custom metric callables. `EvaluationResult.feedback`
  now carries a per-example `input → predicted vs expected → metric`
  breakdown (for both maximize and minimize), not just aggregate losses.
- `reps.LLMJudge` — an `Objective` that scores subjective outputs with an
  LLM judge, with a configurable rubric, scale, and judge model.
- `reps.PromptObjective` — an `Objective` whose artifact is a *prompt
  template string* (not Python code). At evaluation the harness fills
  `{field}` placeholders with each example's inputs, calls a configured
  inference-time LLM, optionally runs a `parse=` callable, and scores with
  the metric. REPS' mutation worker then evolves the prompt itself —
  putting REPS in the same optimization space as DSPy's
  prompt-tuning optimizers (BootstrapFewShot, MIPROv2, …).

### Changed
- `Optimizer.optimize` accepts `objective=` (a `reps.Objective` / `LLMJudge`)
  as the recommended alternative to a raw `evaluate=` callable; exactly one
  of the two must be supplied. Existing `evaluate=` callers are unaffected.
- `Optimizer`'s `trace_reflection` now defaults to `None` ("auto"): a
  `objective=` run enables it (the objective always emits per-example
  feedback the reflection path consumes), an `evaluate=` run leaves it off.
  An explicit `trace_reflection=True/False` still wins.

### Deprecated

### Removed

### Fixed
- `Optimizer.optimize` returns `combined_score=0.0` (not `-0.0`) for a
  perfect minimize objective — IEEE 754 negative zero from unary negation
  was a cosmetic wart in the public headline number.

### Security

## [0.2.0] - 2026-05-12

### Added
- OpenRouter end-to-end compatibility across the harness, including a
  Sonnet-over-OpenRouter smoke config and battletest results.
- Per-run health tracking and structured observability warnings;
  score-plateau convergence detector alongside niche-occupancy growth.
- New tests: `test_database_persistence`, `test_tool_runners`,
  `test_worker_pool`; broader coverage in controller/config/runner/API.
- `db.metric_call_count` surfaced through `OptimizationResult`.
- Default worker preset auto-applied when `reps.enabled=True` but no
  workers are configured, so REPS knobs work without internal YAML.

### Changed
- Stricter config validation: providers, harnesses, worker impls/roles,
  reasoning levels, and selection/diversity strategies are now enum-
  checked with clearer error messages.
- `Optimizer.optimize` writes the per-run dispatch shim with the
  evaluator registry id baked in, removing the previous
  `REPS_USER_EVALUATOR_ID` env-var hand-off.
- Anthropic/OpenAI tool runners and ensemble LLM hardened against
  partial provider responses.

### Fixed
- Suppress the misleading no-`combined_score` warning when evaluators
  legitimately omit it.
- Eliminate spurious warnings during normal runs (carried in from
  `f934b2c`).
- `dspy-react` test is skipped when `dspy` is not installed.

## [0.1.0] - 2026-05-05

First public release. Ships the v1 Python API
([`docs/python_api_spec.md`](docs/python_api_spec.md)) and GEPA Phases 1-5
([`docs/gepa_implementation_plan.md`](docs/gepa_implementation_plan.md)).

### Added

- `reps.Optimizer` — single-class entry point for evolutionary code
  search; constructed with `model=...` plus optional knobs, runs via
  `optimizer.optimize(initial, evaluate)` (`aaae09a`, renamed from
  `reps.REPS` in `642ab15`).
- `reps.Model` — sync LLM facade over `reps.llm.*` providers
  (Anthropic, OpenAI, OpenRouter); model strings parsed as
  `"<provider>/<id>"` with env-var fallback for `api_key` (`3ef3bb4`,
  renamed from `reps.LM` in `ab2517f`).
- `reps.ModelKwargs` — `TypedDict(total=False)` of optional `Model`
  construction kwargs; spread as `**Unpack[ModelKwargs]` on
  `Optimizer` so common-case users skip the `Model` constructor
  entirely (`ab2517f`).
- `reps.OptimizationResult` — return type from `optimize()`; carries
  `best_code`, `best_score`, `best_metrics`, `best_per_instance_scores`,
  `best_feedback`, `iterations_run`, `total_metric_calls`,
  `total_tokens`, `output_dir` (`aaae09a`).
- `reps.EvaluationResult` — re-exported at top level; evaluators may
  return `float`, `dict`, or `EvaluationResult` and the harness coerces
  consistently (`102952e`).
- `EvaluationResult.per_instance_scores` and `EvaluationResult.feedback`
  fields — power Pareto selection, trace reflection, and merge (GEPA
  Phase 1.1, `102952e`).
- `Optimizer.from_config(cfg: reps.config.Config)` — escape hatch for
  power users who need knobs the simple constructor doesn't expose
  (`aaae09a`).
- `reps.internal.*` — documented re-export surface for advanced
  internals (`ReflectionEngine`, `WorkerPool`, `ConvergenceMonitor`,
  etc.) so existing direct importers keep working (`aaae09a`).
- Pareto-frontier selection — `selection_strategy="pareto"` or
  `"mixed"` with `pareto_fraction=...` on `Optimizer`; chooses parents
  by per-instance domination instead of MAP-Elites bins (GEPA Phase 2,
  `5dea23b`, `7e73e61`).
- Trace-grounded reflection — `trace_reflection=True` on `Optimizer`
  emits a per-candidate LLM-generated mutation directive from the
  parent's specific failures (GEPA Phase 3, `c9bda73`, `1fdacfe`).
- System-aware merge — `merge=True` on `Optimizer` selects crossover
  partners whose strengths complement the primary's weaknesses on
  disjoint instance dimensions (GEPA Phase 4, `f89fb8f`, `b1db148`).
- Ancestry-aware reflection — `lineage_depth=N` on `Optimizer` extends
  trace reflection with N generations of parent context (GEPA Phase 5,
  `2420e69`).
- `circle_packing` benchmark emits four sub-scores (`validity`,
  `boundary`, `overlap`, `sum_radii_progress`) as
  `per_instance_scores` plus `feedback` (`eac9f83`).

### Fixed

- `provider_kwargs` are now actually forwarded to the underlying SDK
  client when constructing `reps.Model` (previously declared but
  swallowed) (`df77b02`).
- `EvaluationResult.from_dict` peels top-level `per_instance_scores`
  and `feedback` keys into the dedicated dataclass fields rather than
  leaving them buried in `metrics` (`df77b02`).

### Removed

- GEPA Phase 6 (minibatch evaluation with promotion) — shipped in
  `31e1482`, `877d57d`, `8441b72`, `2bed1a1`, `13f6bf5`, then reverted
  in `d0ad5c0` because the `evaluate(code, instances=...)` contract
  coupled the harness to a benchmark-side instance registry that most
  REPS benchmarks don't have. Cascade evaluation
  (`evaluate_stage1` → `evaluate`) covers the same fast-fail use case
  without polluting the public contract. See
  [`docs/gepa_implementation_plan.md` "Phase 6 — reverted"](docs/gepa_implementation_plan.md).

[Unreleased]: https://github.com/zkhorozianbc/reps/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zkhorozianbc/reps/releases/tag/v0.1.0
