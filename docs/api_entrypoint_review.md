# REPS Package Entrypoint/API Review

Date: 2026-05-12

Scope: every package entrypoint/API a user is likely to hit in the installed package or source checkout:

- Public Python facade: `reps`, `reps.api`, `reps.Model`, `reps.Optimizer`, `reps.OptimizationResult`, `reps.EvaluationResult`.
- Public evaluator contract: `Optimizer.optimize(initial, evaluate)`, generated evaluator shim, `Evaluator`.
- CLI/YAML: `reps-run`, `Config.from_yaml`, `Config.from_dict`, `-o/--override`, config files.
- LLM and worker extension surfaces: provider wrappers, `WorkerConfig`, worker registry, tool-runner workers.
- Persistence/result surfaces: `ProgramDatabase`, artifacts, saved run directories, result aggregation.
- Back-compat/internal exports reachable from `reps.*` and `reps.internal.*`.

Method: local source inspection plus five parallel subagent reviews split by API surface. No production files were edited as part of this review.

## Executive Summary

The public facade is promising: `Model`, `Optimizer`, `OptimizationResult`, and `EvaluationResult` are compact, readable, and covered by useful unit tests. The main issue is integration drift. The polished facade advertises behavior that the older runner/controller/evaluator layers do not always deliver.

The highest-risk problems are:

1. Public Python API knobs (`trace_reflection=True`, `merge=True`) can enable REPS with no workers configured and crash before a run starts.
2. Evaluator failures, seed failures, and some harness/persistence errors are logged and converted into sentinel metrics instead of failing fast.
3. The public evaluator shim uses a process-global env var, so overlapping `Optimizer.optimize()` calls can call the wrong user evaluator.
4. Timeout semantics do not actually stop user evaluator code because the evaluator runs in a thread.
5. CLI/provider/config validation is too permissive for expensive runs; OpenRouter configs can silently route through OpenAI Direct unless `api_base` is set.

## What Looks Well Engineered

- The facade shape is small and learnable: `Model`, `Optimizer`, `OptimizationResult`, `EvaluationResult`.
- Constructor validation catches core mistakes for `max_iterations`, `selection_strategy`, and `num_islands`.
- `EvaluationResult.from_dict()` peels `per_instance_scores` and `feedback` without mutating the input dict.
- The package metadata is in decent shape: `readme`, license metadata, URLs, optional extras, `py.typed`, and prompt template package data exist.
- `${ENV_VAR}` references in YAML fail loudly during config load.
- `task` and `prompt.template_dir` relative resolution is good UX and should be extended to other paths.
- Runner log-handler cleanup is tested.
- Optional `dspy` registration is gated so the base package can import without the extra.

## P0 / Correctness And Credit-Burn Risks

### 1. Advertised Python API REPS Knobs Can Crash Before The Run

`Optimizer._build_config()` flips `cfg.reps.enabled = True` when `trace_reflection` or `merge` is enabled, but it does not install any default `reps.workers.types`.

References:

- `reps/api/optimizer.py:278-285`
- `reps/config.py:425-432`
- `reps/worker_pool.py:38-43`
- `reps/controller.py:220-233`
- `README.md:29`, `README.md:102`

Why it matters:

- README and API docs advertise `trace_reflection=True` and `merge=True` as simple constructor knobs.
- In practice, enabling either path can make `ProcessParallelController.__init__` build `WorkerPool(reps.workers)` and raise because the worker list is empty.

Suggested improvement:

- Either install a default worker preset when the simple API enables REPS, or do not expose these knobs until explicit workers are configured.
- Add an end-to-end public API test for `Optimizer(..., trace_reflection=True).optimize(...)` and `merge=True`, not just config mapping tests.

### 2. Seed And Evaluator Failures Continue Into Expensive Runs

Evaluator exceptions and timeouts are converted to metrics such as `{"error": 0.0}` or `{"timeout": True}`. The runner warns if the seed scores zero but still starts the controller.

References:

- `reps/evaluator.py:355-409`
- `reps/runner.py:145-154`
- `reps/controller.py:643-655`, `reps/controller.py:723-735`

Why it matters:

- This violates the project rule in `AGENTS.md`: seed/evaluator failures are infrastructure problems, not LLM noise.
- A broken evaluator or missing dependency can burn API credits.

Suggested improvement:

- Fail fast before controller startup when the seed has empty metrics, timeout metrics, `error`, missing primary score, or evaluator infrastructure exceptions.
- Reserve sentinel metrics for candidate runtime failures only after the seed and harness have been proven healthy.
- Preserve full traceback in artifacts/logs and include a clear fatal error message.

### 3. Harness/Persistence Exceptions Are Swallowed

`run_evolution()` wraps result handling, including `database.add(...)`, in a broad `except Exception` and logs before continuing.

References:

- `reps/controller.py:1195-1219`
- `reps/controller.py:1383-1387`
- Example underlying infrastructure error source: `reps/database.py:1025-1031`

Why it matters:

- Invalid feature dimensions, persistence failures, and other harness bugs can be treated like an iteration-level error.
- The run continues in a corrupted or misleading state.

Suggested improvement:

- Separate soft worker/evaluator result errors from hard harness/persistence errors.
- Let database/config/persistence exceptions fail the run, or set shutdown and return a fatal status.
- Add tests that invalid `feature_dimensions` and failed per-iteration persistence do not silently continue.

### 4. Concurrent `Optimizer.optimize()` Calls Can Cross-Wire Evaluators

The generated shim reads `REPS_USER_EVALUATOR_ID` from `os.environ`, and each optimize call mutates that same process-global variable.

References:

- `reps/api/optimizer.py:229-251`
- `reps/api/evaluate_dispatch.py:61-63`
- `tests/test_api_optimize.py:751-768`

Why it matters:

- Two overlapping optimizers in the same process can dispatch a candidate from run A into run B's user evaluator.
- Current tests verify distinct registry IDs and cleanup, but not overlapping real runs.

Suggested improvement:

- Embed the registry ID into the per-run shim source instead of reading a process-global env var.
- Alternatively pass the ID through a per-run object or serialize public `optimize()` calls with a documented lock.
- Add an overlapping threaded optimize integration test.

### 5. Evaluator Timeouts Do Not Stop User Code

The evaluator runs the loaded `evaluate()` in a thread executor and wraps the future with `asyncio.wait_for`.

References:

- `reps/evaluator.py:471-480`
- Cascade repeats: `reps/evaluator.py:528-535`, `reps/evaluator.py:573-580`, `reps/evaluator.py:647-654`

Why it matters:

- Python cannot kill a running thread. After timeout, user code may still be executing, spawning child processes, mutating files, or consuming CPU.
- The system records timeout metrics even though the underlying execution may still be alive.

Suggested improvement:

- Run evaluator/user code in a killable subprocess or process pool with process-group cleanup.
- If keeping thread execution, document clearly that `EvaluatorConfig.timeout` is not a sandbox or hard resource boundary.
- Provide safe evaluator templates with subprocess timeout, cwd isolation, controlled env, and cleanup.

### 6. `provider: openrouter` Does Not Guarantee OpenRouter Routing

`Config.provider` defaults to `openrouter`, but `llm.api_base` defaults to `None`; `LLMEnsemble` routes `provider=None` through `OpenAICompatibleLLM` with `base_url=None`, which is OpenAI Direct.

References:

- `reps/config.py:99-103`
- `reps/config.py:548-550`
- `reps/runner.py:95-99`
- `reps/llm/ensemble.py:69-73`
- `reps/llm/openai_compatible.py:41-46`

Why it matters:

- A shorthand OpenRouter config can accidentally call OpenAI Direct with the wrong key/base.
- There is no upfront provider/api-key compatibility check.

Suggested improvement:

- Add a config finalization phase that validates `provider in {"openrouter", "anthropic", "openai"}`.
- Stamp per-model provider from top-level provider.
- Set OpenRouter default base URL when provider is `openrouter`.
- Fail fast when provider/key/base combinations are inconsistent.

## P1 / Public Contract Mismatches

### 7. `env` / `instances` Forwarding Is Documented But Broken In The Real API Path

The public docs and optimizer docstring say optional `env` and `instances` are forwarded to user evaluators. The real generated shim has `def evaluate(program_path, **kwargs)`. The internal evaluator only passes `env` when the loaded function signature explicitly contains `env`, so the shim never receives it.

References:

- `reps/api/optimizer.py:170-172`
- `reps/api/evaluate_dispatch.py:61-63`, `reps/api/evaluate_dispatch.py:151-155`
- `reps/evaluator.py:465-470`
- `docs/python_api_spec.md:238-260`, contradictory note around `docs/python_api_spec.md:529-532`
- Tests call `dispatch_user_evaluate(..., env=...)` directly around `tests/test_api_optimize.py:112`, missing the real shim/evaluator integration.

Suggested improvement:

- Generate the shim as `def evaluate(program_path, *, env=None, instances=None, **kwargs)`.
- Add a real `Optimizer`/`Evaluator` integration test that a user callable with `env` receives it.
- Either implement `instances` end-to-end or remove it from public docs.

### 8. `current_program_id()` Does Not Reach Evaluator Threads

`evaluate_isolated()` sets a ContextVar, but the evaluator function runs inside `run_in_executor`; ContextVars do not automatically propagate to executor threads.

References:

- `reps/evaluator.py:190-213`
- `reps/evaluator.py:471-476`
- `reps/runtime.py`

Suggested improvement:

- Use `contextvars.copy_context().run(...)` inside the executor lambda.
- Or document that `current_program_id()` is only reliable in async harness context, not inside threaded evaluator code.

### 9. Dict Evaluator Returns Do Not Enforce The Documented `combined_score`

Docs say dict returns must include `combined_score`. Implementation passes dicts through unchanged, and result collection defaults `best_score` to `0.0` if missing.

References:

- `docs/python_api_spec.md:251-260`
- `reps/api/evaluate_dispatch.py:116-125`
- `reps/evaluation_result.py:50-63`
- `reps/api/optimizer.py:347`
- Fallback ranking path: `reps/database.py:627-631`, `reps/controller.py:1274-1290`

Suggested improvement:

- Decide: hard-require `combined_score` for public API dict returns, or formally document fallback scoring.
- If fallback is retained, compute `OptimizationResult.best_score` with the same fallback as the database.

### 10. `OptimizationResult.total_metric_calls` Is Not Actual Metric Calls

Docs say it counts evaluator invocations. Implementation uses `len(db.programs)`.

References:

- `reps/api/result.py:17-21`
- `reps/api/optimizer.py:373-377`
- Tests codify program count around `tests/test_api_optimize.py:459`

Why it matters:

- It undercounts retries and cascade stages.
- It can overcount stale programs if an output directory is reused.

Suggested improvement:

- Rename to `programs_evaluated`, or instrument actual evaluator calls and store the true count.

### 11. `Model` Provider Kwargs And Client Reuse Are Lost In `Optimizer`

`Model` can forward `provider_kwargs` to its SDK client, but `Optimizer` only copies `_to_model_config()` into `Config`; `_to_model_config()` omits provider kwargs, and the run builds fresh clients through `LLMEnsemble`.

References:

- `reps/api/model.py:124`, `reps/api/model.py:152-180`, `reps/api/model.py:207-230`
- `reps/api/optimizer.py:287-304`
- `reps/controller.py:342-350`
- README reuse claim: `README.md:119-124`

Suggested improvement:

- Either make `Optimizer` actually use the provided `Model` client, or carry provider kwargs/custom client config into `LLMModelConfig`.
- Otherwise document that only basic model configuration is reused, not the configured SDK client.

### 12. `minibatch_size` Is Public But A No-Op

`minibatch_size` is a constructor kwarg and maps into config, but `EvaluatorConfig` says it is currently unused. Docs say Phase 6 was reverted but still show `minibatch_size=2` in examples.

References:

- `reps/api/optimizer.py:69`, `reps/api/optimizer.py:281`
- `reps/config.py:347-351`
- `docs/python_api_spec.md:197-198`, `docs/python_api_spec.md:386`

Suggested improvement:

- Remove from v1 until implemented, or make no-op status explicit and remove from examples.

### 13. Public/Internal API Story Is Contradictory

Top-level `reps` exports internals such as `WorkerPool`, `ContractSelector`, and `MetricsLogger`, while docs say the v1 surface is only the small facade. `reps.internal` says symbols may change without deprecation, while release docs discuss public API versioning.

References:

- `reps/__init__.py:15-31`, `reps/__init__.py:33-50`
- `reps/internal/__init__.py:1-13`
- `docs/python_api_spec.md:51-73`
- `docs/release_spec.md`

Suggested improvement:

- Before 1.0, decide whether top-level internals are stable public API.
- Prefer moving legacy names behind `reps.internal` plus lazy/deprecated `__getattr__`, or document that top-level also includes legacy unstable exports.

## P1 / CLI And YAML UX

### 14. Documented CLI Overrides Can Be Ineffective

`README` advertises `-o llm.temperature=0.9`, but overrides apply after `LLMConfig.__post_init__` has already copied shared LLM fields into model configs. The actual `llm.models[*]` can remain stale. Typos on the final path segment also create ad-hoc attributes silently.

References:

- `README.md:149`
- `reps/runner.py:317-325`
- `reps/runner.py:269-285`
- `reps/config.py:173-186`

Suggested improvement:

- Apply overrides to the raw YAML dict before `Config.from_dict`, or add a validated override/finalize phase.
- Forbid unknown override paths.
- Re-run model shorthand propagation after overrides when shared LLM fields change.

### 15. YAML Validation Is Too Permissive For Expensive Runs

Unknown YAML fields are silently ignored by the current `dacite.from_dict` config. Existing configs contain `allow_full_rewrites`, which has no implementation hit. A misspelled `harness` falls through to REPS because runner only special-cases `"openevolve"`.

References:

- `reps/config.py:622-628`
- `reps/runner.py:348-351`
- `experiment/configs/circle_sonnet_reps.yaml:42`

Suggested improvement:

- Use strict config loading or explicitly collect unknown keys into a fatal validation error.
- Validate enum-like fields (`provider`, `harness`, `selection_strategy`, worker `impl`, generation modes) before creating run directories or API clients.

### 16. Config Paths Are Inconsistent Or Dead

`Config.output` exists but runner ignores it. `task` is resolved relative to the config file, but `initial_program` and `evaluator_path` are not. `.env` loading walks from CWD before parsing `--config`, not from the config directory.

References:

- `reps/config.py:535-537`
- `reps/runner.py:243-266`
- `reps/runner.py:321-334`
- `reps/config.py:598-603`

Suggested improvement:

- Use `args.output or config.output or default`.
- Resolve all YAML paths consistently relative to the config file.
- Load `.env` from the config directory plus CWD ancestors, with shell env still taking precedence.

### 17. PyPI README Benchmark Story Is Mismatched

The wheel ships only `reps/`, but README says `[benchmarks]` supports bundled circle-packing benchmark/config paths under `experiment/`.

References:

- `pyproject.toml:55`
- `README.md:55`, `README.md:141-181`
- `docs/packaging_spec.md`

Suggested improvement:

- Either ship examples/benchmarks explicitly, or label those CLI examples as source-checkout-only.
- Provide an installed-package example benchmark users can run after `pip install reps-py`.

### 18. Minor CLI Brittleness

`_next_run_dir()` can crash if `run_*` entries exist but none have numeric suffixes. `--iterations 0` is ignored because the code checks truthiness, while negative values are accepted.

References:

- `reps/runner.py:40-54`
- `reps/runner.py:323-324`

Suggested improvement:

- Match `^run_(\d+)$`, skip nonmatching directories, and retry `mkdir` on collision.
- Add a positive-int argparse type, or document and implement explicit zero-run semantics.

## P1 / Workers And LLM Providers

### 19. Native Tool-Runner Workers Bypass Shared LLM Config

`anthropic_tool_runner` and `openai_tool_runner` read only `impl_options.api_key` or standard env vars. They ignore shared `config.llm.api_key`, `api_base`, timeout/retries, and top-level `reasoning`.

References:

- `reps/workers/anthropic_tool_runner.py:70-90`
- `reps/workers/openai_tool_runner.py:77-90`
- `reps/workers/registry.py:21-29`
- `reps/runner.py:124-133`

Suggested improvement:

- Build provider clients lazily in `run()` using `ctx.config`, with `impl_options` as overrides.
- Align tool-runner defaults with `LLMModelConfig` and provider finalization.

### 20. Anthropic Temperature Handling Contradicts Its Own Comment

The class-level override says all Claude models should keep temperature, but `generate_with_context()` checks the module-level `REASONING_MODEL_PATTERNS` and skips temperature for Claude 4.6/4.7.

References:

- `reps/llm/anthropic.py:43-50`
- `reps/llm/anthropic.py:95`
- `reps/llm/anthropic.py:116-119`

Suggested improvement:

- Use `self.REASONING_MODEL_PATTERNS` if the override is intentional, or remove the override/comment.

### 21. Model Override Matching Is Brittle Across Provider Prefixes

Anthropic strips `anthropic/` from model names, but worker `model_id` is compared exactly against provider-normalized names. A prefixed Anthropic worker override can fail even when the model exists.

References:

- `reps/llm/anthropic.py:60-62`
- `reps/worker_pool.py:112`
- `reps/workers/single_call.py:104-109`
- `reps/llm/ensemble.py:124-138`

Suggested improvement:

- Canonicalize aliases in `LLMEnsemble`, or keep both raw and provider-native model names for override matching.

### 22. Tool-Runner Auto-Reevaluation Errors Are Returned As Accepted

When submit-child auto-reevaluation raises, the tool result still says `accepted` and is not marked as an error.

References:

- `reps/workers/anthropic_tool_runner.py:381-396`
- `reps/workers/openai_tool_runner.py:298-313`

Suggested improvement:

- Mark these tool results as errors and allow the worker to retry, or propagate a `WorkerError` for infrastructure exceptions.

### 23. Explicit Tool-Runner Template Misses Are Silently Swallowed

Both tool runners catch any template read failure and fall back to sampler system text, even when `system_prompt_template` was explicitly configured.

References:

- `reps/workers/anthropic_tool_runner.py:530-555`
- `reps/workers/openai_tool_runner.py:451-474`

Suggested improvement:

- Fail loudly for explicitly configured template names.
- At minimum log the missing path and whether fallback was used.

### 24. WorkerPool Validation And Reproducibility Hazards

Duplicate worker names collapse silently into a dict. All-zero weights normalize to all zero and can fail later during sampling. It reads `workers_config.random_seed`, but `REPSWorkersConfig` has no such field. The module-level RNG is shared across pools.

References:

- `reps/worker_pool.py:18`
- `reps/worker_pool.py:35-56`
- `reps/worker_pool.py:137-146`
- `reps/config.py:425-432`

Suggested improvement:

- Validate unique names and positive finite weights.
- Use an instance RNG seeded from `Config.random_seed`.
- Add `random_seed` to `REPSWorkersConfig` only if it is intended to be YAML-facing.

## P1 / Persistence And Result Integrity

### 25. Reusing `output_dir` Can Contaminate New API Results

The API uses `mkdir(exist_ok=True)` and reuses the caller directory. `ProgramDatabase.save()` writes current programs but does not clear old `programs/*.json`. Result collection loads every JSON file present.

References:

- `reps/api/optimizer.py:219-227`
- `reps/database.py:727-739`
- `reps/database.py:799-813`
- `reps/api/optimizer.py:335-379`

Suggested improvement:

- Create a fresh run subdirectory, fail on non-empty `output_dir`, or clear owned files before saving.
- Document the exact API persistence semantics.

### 26. Novelty Rejection Leaks Rejected Programs

`ProgramDatabase.add()` inserts into `self.programs` before novelty checking. If novelty rejects, it returns without rolling back. Controller still treats the child as accepted and may store artifacts/persist it.

References:

- `reps/database.py:253-260`
- `reps/database.py:295-300`
- `reps/controller.py:1195-1219`

Suggested improvement:

- Move novelty before insertion, or return an accepted/rejected status from `add()`.
- Have the controller skip follow-up persistence/artifact storage for rejected programs.

### 27. Artifact Round-Trips Are Not Lossless

Small bytes artifacts are JSON-encoded with a sentinel, but `get_artifacts()` uses plain `json.loads` and never calls the deserializer. Large artifact keys are sanitized into filenames without a manifest, so names can change or collide.

References:

- `reps/database.py:2521-2524`
- `reps/database.py:2534-2563`
- `reps/database.py:2574-2584`
- `reps/database.py:2640-2686`

Suggested improvement:

- Use `json.loads(..., object_hook=self._artifact_deserializer)`.
- Add a manifest for large artifacts preserving original keys, content type, and filename.
- Test `str`, `bytes`, unsafe names, and collision cases.

### 28. `OptimizationResult.best_score` Can Contradict Search Ranking

The database ranks programs with `get_fitness_score(...)` when `combined_score` is missing; result collection reports only `combined_score` and defaults to `0.0`.

References:

- `reps/api/optimizer.py:347`
- `reps/database.py:627-631`
- `reps/controller.py:1274-1290`

Suggested improvement:

- Enforce `combined_score`, or compute the result score with the same fallback used by the database.

## P2 / Bloat, Confusion, And Cleanup

### 29. README Overstates Python API Defaults

README says REPS reflects, balances workers, detects convergence, and has convergence/SOTA steering on by default. The simple Python API leaves `cfg.reps.enabled=False` unless `trace_reflection` or `merge` are set; SOTA defaults disabled.

References:

- `README.md:10`, `README.md:32`
- `reps/api/optimizer.py:283-285`
- `reps/config.py:514`
- `reps/config.py:454-462`

Suggested improvement:

- Distinguish CLI/YAML full-REPS presets from Python API defaults.
- Clarify which features require `from_config` or a worker config.

### 30. Quick-Start Evaluator Encourages Unsafe In-Process Execution

README quick start uses `exec(code, namespace)` directly.

References:

- `README.md:74-80`
- Safer but still incomplete subprocess example: `docs/python_api_spec.md:335-342`

Suggested improvement:

- Use a subprocess example with timeout, isolated temp dir, and controlled env.
- Add an explicit note that REPS executes generated code and is not a sandbox.

### 31. Convergence Knobs Exist Internally But Are Not Config-Reachable

`ConvergenceMonitor` accepts additional thresholds, but `REPSConvergenceConfig` exposes only entropy thresholds and controller passes `asdict(reps.convergence)`.

References:

- `reps/convergence_monitor.py:153`, `reps/convergence_monitor.py:189`
- `reps/config.py:436-443`
- `reps/controller.py:234`

Suggested improvement:

- Add the missing dataclass/YAML fields, or remove the dead internal knobs.
- Add an integration test through `Config.from_yaml`.

### 32. Contract Updates Can Credit The Wrong Model

Unknown `(model_id, temperature)` updates fall back to the closest arm by temperature, even across different models.

References:

- `reps/contract_selector.py:90-95`
- `reps/contract_selector.py:102-112`

Suggested improvement:

- Prefer exact model match; otherwise log and no-op.
- Include `worker_name` in the arm key if contract learning is meant to be worker-aware.

## Recommended Fix Order

1. Make seed/evaluator validation fatal before any LLM calls.
2. Fix `Optimizer` REPS knob behavior by installing default workers or removing those simple knobs.
3. Replace the process-global evaluator shim ID with per-run shim state.
4. Add provider/config finalization and strict YAML/override validation.
5. Decide and enforce the `combined_score` contract.
6. Fix timeout semantics or document the current limitation prominently.
7. Make output directories fresh or fail on reuse.
8. Align workers/tool-runners with shared LLM config.
9. Clean up docs/README around PyPI benchmarks, feature defaults, no-op `minibatch_size`, and unsafe evaluator examples.
10. Tackle persistence correctness: novelty rejection, artifacts, and `total_metric_calls`.

## Suggested Test Additions

- Public API run with `trace_reflection=True` and no explicit workers.
- Public API run with `merge=True` and no explicit workers.
- Full shim/evaluator integration proving `env` reaches `evaluate(code, *, env=None)`.
- Overlapping threaded `Optimizer.optimize()` calls with different evaluator functions.
- Seed evaluator missing dependency raises before controller startup.
- CLI `-o llm.temperature=...` changes actual `llm.models[*].temperature`.
- Unknown YAML key and misspelled `harness` fail before run directory creation.
- OpenRouter shorthand config sets provider/base/key consistently.
- Reused `output_dir` does not load stale programs.
- Artifact round-trip for bytes, text, unsafe names, and colliding names.
- `WorkerPool` duplicate names and all-zero weights fail at construction.
