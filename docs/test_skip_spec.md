# Test skip hygiene — design spec

## 1. Goals

Eliminate the two pre-existing failures in `tests/test_controller.py` so the
full pytest suite runs green without `ANTHROPIC_API_KEY` set in the
environment. CI and downstream packagers (PyPI release pipeline, Docker
images, contributors on a fresh checkout) must be able to run
`uv run python -m pytest tests/` and see 0 failures. The fix must be
minimal — no sweeping refactor of the summarizer subsystem — and must not
silently change runtime behavior of an actual REPS run that *does* have a
key configured.

## 2. Diagnosis

### Test A: `test_controller_with_reps_does_not_create_default_output_dir`

**Purpose (per docstring):** "Direct controller construction should not
create a fallback output dir." The test:

1. Sets `self.config.reps.enabled = True`
2. Installs the default 3-worker preset
3. `chdir`s into a temp dir
4. Constructs `ProcessParallelController(...)`
5. Asserts `controller.output_dir is None`, `controller._reps_metrics is None`,
   and that no `openevolve_output` directory was created on disk

**Observed traceback (tail):**

```
tests/test_controller.py:121: in test_controller_with_reps_does_not_create_default_output_dir
    controller = ProcessParallelController(self.config, self.eval_file, self.database)
reps/controller.py:256: in __init__
    self._reps_summarizer_llm = build_summarizer_llm(summarizer_cfg)
reps/program_summarizer.py:56: in build_summarizer_llm
    raise ValueError(
E   ValueError: summarizer: provider='anthropic' but no api_key set
    (neither reps.summarizer.api_key nor ANTHROPIC_API_KEY in env)
```

**Why it requires the key:** the `Config()` constructor defaults
`reps.summarizer.enabled = True` and `reps.summarizer.model_id = "claude-opus-4-7"`
(see `reps/config.py:493-494`). The controller's `__init__`
unconditionally builds the summarizer LLM if both `reps.enabled` and
`reps.summarizer.enabled` are true (`reps/controller.py:252-260`):

```python
self._reps_summarizer_llm = None
summarizer_cfg = getattr(reps, "summarizer", None)
if summarizer_cfg is not None and summarizer_cfg.enabled:
    from reps.program_summarizer import build_summarizer_llm
    self._reps_summarizer_llm = build_summarizer_llm(summarizer_cfg)
```

`build_summarizer_llm` then resolves the api key from the env var and
raises `ValueError` if it cannot find one (`reps/program_summarizer.py:54-59`).

The test does *not* care about the summarizer at all. It is testing one
specific thing: that no implicit `openevolve_output` directory is created
when `output_dir` is unspecified.

### Test B: `test_sota_steering_uses_same_raw_metric_for_prompt_and_reallocation`

**Purpose (per docstring):** "F6 should use the same score basis across
both steering callsites." The test:

1. Sets `reps.enabled = True`, `reps.sota.enabled = True`,
   target_metric `sum_radii`, target_score `2.635`
2. Disables convergence, contracts, reflection — but leaves summarizer
   at its default (enabled)
3. Constructs `ProcessParallelController(...)`
4. Calls `controller._reps_build_prompt_extras()` and asserts the SOTA
   gap text is in the prompt
5. Runs a synthetic-failure batch through `_reps_process_batch` and
   asserts worker-pool reallocation matches the same gap

**Observed traceback (tail):**

```
tests/test_controller.py:238: in test_sota_steering_uses_same_raw_metric_for_prompt_and_reallocation
    controller = ProcessParallelController(self.config, self.eval_file, database)
reps/controller.py:256: in __init__
    self._reps_summarizer_llm = build_summarizer_llm(summarizer_cfg)
reps/program_summarizer.py:56: in build_summarizer_llm
    raise ValueError(
E   ValueError: summarizer: provider='anthropic' but no api_key set
    (neither reps.summarizer.api_key nor ANTHROPIC_API_KEY in env)
```

**Why it requires the key:** identical mechanism to Test A. The test
forgot to disable the summarizer alongside the other REPS subsystems.
The synthetic batch contains only an error result, so the summarizer
codepath at `controller.py:628-655` would never run anyway — no `result`
arrives with `turns`, no parent/child scores get computed.

**Both tests have the same root cause:** eager construction of the
summarizer LLM at controller `__init__` time, even though the LLM is
only ever *used* later (lazily) in `_reps_process_batch`.

## 3. Recommended fix

**Same fix applies to both tests, and it is a production-code change, not
a test-skip.** Defer construction of the summarizer LLM from
`__init__` to first use. Reasons:

1. **The tests' intent is correct.** Constructing a controller with REPS
   enabled should not require an Anthropic key if no summarization is
   ever requested in that controller's lifetime. The current behavior is
   a latent design bug — not "expected" — that just happens to be
   shielded in real runs because real runs always have a key.
2. **The use-site is already null-safe.** `controller.py:630` already
   checks `self._reps_summarizer_llm is not None` before invoking. So we
   already pay for the "summarizer might be unavailable" branch.
3. **It preserves the loud-failure intent.** The original eager
   construction was justified in the `build_summarizer_llm` docstring as
   "fails LOUDLY at construction time … intentional: a silent fallback
   here would hide real config mistakes." We preserve that property by
   still raising on first *use* — just not at controller construction.
   A real run that exercises REPS will hit the summarizer codepath on
   iteration 1, so config errors still surface within seconds, before
   meaningful API spend.
4. **`@pytest.mark.skipif` is the wrong tool here.** It would let the
   bug ship: any user who tries to instantiate a controller programmatically
   (e.g., a notebook walkthrough, a test in their own evaluator package)
   without a key would still see the error. We want the public API to
   work without a key when no summarization is performed.

### Why option 1 (skipif) is rejected

Skipping these tests when the key is absent would mean CI on a fresh
machine simply doesn't exercise:

- Test A — the *one* regression test for "controller does not pollute
  cwd with openevolve_output". This is a real bug class that has bitten
  this codebase before; we don't want it silently un-tested.
- Test B — the *one* regression test for the F6 SOTA-steering metric
  consistency. Same argument.

Both tests are checking pure controller logic. Skipping them on
no-key would defeat the point of having them.

## 4. Concrete changes

### Change 1 — make summarizer construction lazy in the controller

**File:** `/home/user/reps/reps/controller.py`
**Location:** lines 248-260 (the block beginning
`# Dedicated summarizer LLM (independent of worker ensemble).`)

Replace the eager construction with a stash of the config + a lazy
property/getter that builds on first access. Keep the existing field
name `self._reps_summarizer_llm` as a *cache* (`None` until first use)
and add a helper `_get_reps_summarizer_llm()`.

Sketch:

```python
# Dedicated summarizer LLM (independent of worker ensemble).
# Construction is DEFERRED to first use so importing/constructing a
# controller does not require an Anthropic/OpenAI key when the summarizer
# happens to be enabled-by-default in cfg. Failures still surface
# loudly — but at the first iteration that needs them, not at __init__.
self._reps_summarizer_llm = None  # built on first use; see _get_reps_summarizer_llm
self._reps_summarizer_cfg = getattr(reps, "summarizer", None)
```

Add a method (right after `__init__` or wherever fits the file's
layout):

```python
def _get_reps_summarizer_llm(self):
    """Lazily construct the summarizer LLM.

    Returns None if the summarizer is disabled. Raises (loudly) on first
    call if the summarizer is enabled but no api_key is resolvable —
    same loud-failure semantics as before, just deferred from __init__
    to first batch.
    """
    cfg = self._reps_summarizer_cfg
    if cfg is None or not cfg.enabled:
        return None
    if self._reps_summarizer_llm is None:
        from reps.program_summarizer import build_summarizer_llm
        self._reps_summarizer_llm = build_summarizer_llm(cfg)
        logger.info(
            f"Summarizer LLM: model={cfg.model_id} "
            f"provider={self._reps_summarizer_llm.provider}"
        )
    return self._reps_summarizer_llm
```

### Change 2 — update the use-site to call the lazy getter

**File:** `/home/user/reps/reps/controller.py`
**Location:** lines 627-648 (the block in `_reps_process_batch` that
calls `summarize_program`).

Replace direct `self._reps_summarizer_llm` reads with a call to the
getter. Sketch:

```python
summarizer_cfg = getattr(self.config.reps, "summarizer", None)
if (
    self._reps_enabled
    and (summarizer_cfg is None or summarizer_cfg.enabled)
):
    summarizer_llm = self._get_reps_summarizer_llm()
    if summarizer_llm is not None:
        try:
            from reps.program_summarizer import summarize_program

            turns_dicts = [_turn_to_dict(t) for t in result.turns]
            summary = await summarize_program(
                program_id=final_child_id,
                code=result.child_code,
                turns=turns_dicts,
                parent_score=parent_score,
                child_score=child_score,
                improved=child_score > parent_score,
                summarizer_llm=summarizer_llm,
                task_instructions=(
                    summarizer_cfg.task_instructions if summarizer_cfg else None
                ),
            )
            ...
```

The behavior delta in production: the first iteration that has both REPS
enabled and a non-None `result.turns` will pay the cost of constructing
the AnthropicLLM client (one-shot, ~ms). This is negligible relative to
the LLM call itself. After that, the cached field returns instantly.

### Change 3 — no test changes required

Once the production code is lazy, both failing tests will construct
controllers without ever touching `build_summarizer_llm`. Their
assertions don't reach the summarizer code path:

- Test A only inspects `controller.output_dir` / `_reps_metrics` /
  the file system — never runs a batch.
- Test B runs `_reps_process_batch` with a single
  `SerializableResult(error=...)` — the error branch in
  `_reps_process_batch` short-circuits before `summarize_program` is
  reached. (Verified by reading `_reps_process_batch`: the
  `summarize_program` block at line 627 sits inside the success path
  that runs *after* the child is added to the database, which only
  happens for non-error results.)

If for any reason Test B *did* reach the summarizer call site after the
fix, we would still want the test to pass without a key — at which
point monkeypatching `reps.program_summarizer.build_summarizer_llm` to
return a dummy stub in `setUp` would be the next step. We don't need
that today.

## 5. Verification

1. Apply Change 1 + Change 2 to `reps/controller.py`.
2. Run the two previously-failing tests with the key unset:

   ```bash
   unset ANTHROPIC_API_KEY
   uv run python -m pytest \
     tests/test_controller.py::TestProcessParallel::test_controller_with_reps_does_not_create_default_output_dir \
     tests/test_controller.py::TestProcessParallel::test_sota_steering_uses_same_raw_metric_for_prompt_and_reallocation \
     -v
   ```

   Expected: 2 passed.

3. Run the full controller test file:

   ```bash
   uv run python -m pytest tests/test_controller.py -v
   ```

   Expected: all tests pass (no regressions in the other 5 tests).

4. Run the full suite:

   ```bash
   uv run python -m pytest tests/ -q
   ```

   Expected: 390 passed, 0 failed (was 388 passed + 2 failed; the 2
   failures become passes; nothing else changes).

5. Spot-check that a real REPS run still surfaces a missing-key error
   loudly (just deferred by one iteration). With key unset and a config
   that has `reps.enabled = True` and default summarizer settings,
   running `reps-run` should produce a `ValueError` in the first
   iteration's logs — at the first call to `_get_reps_summarizer_llm()`
   inside `_reps_process_batch`. The error will appear in the
   per-iteration `logger.warning(...)` at controller.py:653-655, which
   currently swallows summarizer-call exceptions. Acceptable: the
   summarizer is best-effort by design, and the warning will tell the
   operator to fix their key. (If we want a *fatal* missing-key check at
   startup-ish time, that's a separate change — call out below.)

## 6. Open decisions

**Q1: Should the lazy fix preserve "fatal at startup" semantics for a
genuinely misconfigured run?**

Originally, a missing summarizer key would crash the controller before
iteration 1 — operators got immediate feedback. After the lazy fix, the
operator sees a `logger.warning(...)` once per iteration and the run
continues without summaries. That's arguably *worse* UX for the common
case (the operator wanted summaries and forgot to set the key).

Two ways to mitigate, in order of preference:

- **Recommended (do not block on this):** ship the lazy fix as
  specified. The summarizer is documented as best-effort, and a warning
  per iteration is loud enough.
- **Optional follow-up (separate PR):** add a one-shot validation in
  `Optimizer.optimize()` (or wherever the public entrypoint is) that
  resolves the summarizer config and verifies a key is present before
  the controller starts. That keeps the fail-loud-at-startup behavior
  for real runs while letting tests construct controllers freely. This
  is a UX nicety, not a correctness fix; defer.

No other open decisions.
