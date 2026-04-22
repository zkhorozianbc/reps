# Tool-Calling Worker Primitive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-LLM-call compute node in REPS with a swappable `Worker` primitive, implement three workers (SingleCall, AnthropicToolRunner, DSPyReAct), migrate the controller to asyncio-only, and verify via three independent live runs against the `circle_packing` benchmark.

**Architecture:** New `reps/workers/` package defines the `Worker` protocol, a registry, three concrete impls, and shared tools. Controller (`reps/controller.py`) rewrites to a single-process asyncio loop (`asyncio.TaskGroup` + `asyncio.Semaphore`). Evaluator (`reps/evaluator.py`) gets a new `evaluate_isolated` method using `contextvars.ContextVar` for per-call state and a per-call `env` dict for subprocess spawn (fixes a real cross-iteration race). `IterationConfig.worker_type` is renamed to `worker_name`. WorkerPool samples over named `WorkerConfig` entries; ContractSelector arms become `(worker_id, temperature?)` with `owns_temperature` axis ownership.

**Tech Stack:** Python 3.12, uv, anthropic>=0.60.0, dspy>=3.1.3, asyncio, contextvars, pytest

**Verification policy (from spec §2 and §12):** New feature code is **not** gated on new unit tests. The existing `pytest tests/` suite must still pass (regression check). Feature verification is three independent live `reps-run` executions — Task 20. Where a small unit test pins a specific invariant and is trivial to write, it may be included opportunistically.

**Worktree:** `/Users/zkhorozian/code/reps/.worktrees/feature-workers-primitive` (branch `feature/workers-primitive`), already set up with isolated `.venv` and `.env`.

---

## File Structure

### New files under `reps/workers/`

| File | Responsibility |
|------|---------------|
| `reps/workers/__init__.py` | Re-exports |
| `reps/workers/base.py` | `WorkerConfig`, `WorkerRequest`, `WorkerContext`, `WorkerResult`, `TurnRecord`, `ContentBlock`, `WorkerError`, `Worker` protocol |
| `reps/workers/registry.py` | `register()` decorator, `build_worker()` factory |
| `reps/workers/edit_serializer.py` | `serialize_diff_blocks()` |
| `reps/workers/trace_render.py` | `render_trace()` pretty-printer |
| `reps/workers/tools.py` | Shared tool implementations (view_parent, view_program, edit_file, submit_child, run_tests) |
| `reps/workers/single_call.py` | `SingleCallWorker` |
| `reps/workers/anthropic_tool_runner.py` | `AnthropicToolRunnerWorker` |
| `reps/workers/dspy_react.py` | `DSPyReActWorker` |
| `reps/workers/defaults.py` | Default preset `WorkerConfig` entries for legacy YAML shim |

### New file under `reps/`

| File | Responsibility |
|------|---------------|
| `reps/runtime.py` | `current_program_id()` contextvar helper for benchmarks |

### Modified files

| File | Change |
|------|--------|
| `reps/controller.py` | Rewrite: drop `ProcessPoolExecutor`/`mp.Event`/`pickle`; add `AsyncController`; `async def _run_iteration` calls `worker.run(...)`; `_submit_iteration` → `_spawn_iteration` with `asyncio.TaskGroup` + `asyncio.Semaphore`; delete `_create_database_snapshot` |
| `reps/evaluator.py` | Add `evaluate_isolated` + `EvaluationOutcome`; `_call_artifacts: ContextVar`; `asyncio.Semaphore`; remove `os.environ["REPS_PROGRAM_ID"]` mutation; thread per-call env dict via `run_with_timeout(env=...)`; delete `_current_program_id` |
| `reps/iteration_config.py` | Rename `worker_type` → `worker_name` (deprecated alias); add `turns: List[Dict]` field on `IterationResult` |
| `reps/worker_pool.py` | Refactor to sample over named `WorkerConfig`; crossover keyed on `role == "crossover"` |
| `reps/contract_selector.py` | Arms become `(worker_id, temperature)` with axis-ownership check |
| `reps/config.py` | `EvaluatorConfig.max_concurrent_iterations`, `DatabaseConfig.max_turns_persisted`, `EvolutionTraceConfig.include_turns`, `REPSWorkersConfig.types: List[WorkerConfig]` |
| `reps/runner.py` | No functional change (uses legacy `evaluate_program` wrapper) |
| `reps/prompt_templates/system_message_tool_runner.txt` | New template (created) |
| `experiment/benchmarks/circle_packing/evaluator.py` | `env=env` kwarg on `subprocess.Popen` at :145; `_dump_packing_markdown` reads program_id via `current_program_id()` |
| `pyproject.toml` | Bump `anthropic>=0.60.0`, add `dspy>=3.1.3` |
| `tests/test_controller.py:92,105,124,174` | Rewrite for async controller; `:206` survives |

### New files under `experiment/configs/`

| File | Purpose |
|------|---------|
| `experiment/configs/verify_single_call.yaml` | Verification Run A |
| `experiment/configs/verify_anthropic_tool_runner.yaml` | Verification Run B |
| `experiment/configs/verify_dspy_react.yaml` | Verification Run C |

---

## Task 1: Create `reps/workers/` package and `base.py` with data shapes

**Files:**
- Create: `reps/workers/__init__.py` (empty)
- Create: `reps/workers/base.py`

- [ ] **Step 1: Create the package init file**

```bash
mkdir -p reps/workers
touch reps/workers/__init__.py
```

- [ ] **Step 2: Create `reps/workers/base.py`**

```python
"""
Core data shapes for the Worker primitive.

A Worker is the swappable compute node that mutates a parent program into a child
program. Today there is one impl (single LLM call). This module defines the
contract so tool-calling agents (Anthropic tool-runner, DSPy ReAct, etc.) can
slot in as alternative workers.

See docs/superpowers/specs/2026-04-21-tool-calling-worker-primitive-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
    Union,
)

if TYPE_CHECKING:
    from reps.config import Config
    from reps.database import Program
    from reps.evaluator import Evaluator
    from reps.iteration_config import IterationConfig
    from reps.llm.base import LLMInterface
    from reps.prompt_sampler import PromptSampler

SCHEMA_VERSION = 1

ErrorKind = Literal[
    "TOOL_ERROR",
    "MAX_TURNS_HIT",
    "PARSE_ERROR",
    "REFUSED",
    "TIMEOUT",
    "INTERNAL",
]


@dataclass
class WorkerError(Exception):
    """Typed error returned by Worker.run in WorkerResult.error (non-raising)."""
    kind: ErrorKind
    detail: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" if self.detail else self.kind


@dataclass
class WorkerConfig:
    """Declarative config for a named worker instance.

    Declared in YAML under `reps.workers.types`. One WorkerConfig → one Worker
    instance constructed via the registry.
    """
    name: str                              # unique id, e.g. "anthropic_tool_runner_exploiter"
    impl: str                              # registry key: "single_call" | "anthropic_tool_runner" | "dspy_react"
    role: str = "exploiter"                # "exploiter" | "explorer" | "crossover"
    model_id: str = ""
    temperature: Optional[float] = None
    generation_mode: str = "diff"          # "diff" | "full"
    tools: List[str] = field(default_factory=list)
    max_turns: int = 1
    uses_evaluator: bool = False
    system_prompt_template: Optional[str] = None
    impl_options: Dict[str, Any] = field(default_factory=dict)
    owns_model: bool = True                # if True, ContractSelector does NOT override model
    owns_temperature: bool = False         # if True, ContractSelector does NOT override temperature
    expected_wall_clock_s: float = 15.0
    weight: float = 1.0


@dataclass
class ContentBlock:
    """One unit of turn content. Fields populated by type; others stay None.

    Maps 1:1 onto Anthropic's content-block shapes (text, thinking,
    redacted_thinking, tool_use, tool_result). DSPy and single-call workers
    produce a reduced subset (text + tool_use/tool_result).

    `signature` on a thinking block is an Anthropic signed blob that MUST be
    preserved verbatim — the API rejects subsequent turns if the signature
    is altered or stripped.
    """
    type: Literal["text", "thinking", "redacted_thinking", "tool_use", "tool_result"]

    # text / thinking
    text: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = None     # redacted_thinking base64 blob

    # tool_use
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    # tool_result
    tool_result_for_id: Optional[str] = None
    tool_result_content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_result_is_error: bool = False

    provider_extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    """One message turn inside Worker.run(). Lossless for Anthropic tool-use loops."""
    index: int                                   # 0-based within the iteration
    role: Literal["system", "user", "assistant", "tool"]
    blocks: List[ContentBlock]
    model_id: Optional[str] = None
    usage: Optional[Dict[str, int]] = None       # input_tokens, output_tokens, cache_*
    stop_reason: Optional[str] = None            # Anthropic: "end_turn" | "tool_use" | "max_tokens"
    started_at_ms: Optional[int] = None          # ms since worker.run entry
    ended_at_ms: Optional[int] = None
    worker_type: Optional[str] = None            # impl name, for renderer styling
    schema_version: int = SCHEMA_VERSION
    impl_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerRequest:
    """Structured context handed to Worker.run(). Not a pre-rendered prompt —
    workers that render their own prompts (DSPy) can read the fields directly;
    workers that use PromptSampler can pass this through."""
    parent: "Program"
    inspirations: List["Program"]
    top_programs: List["Program"]
    second_parent: Optional["Program"]
    iteration: int
    language: str
    feature_dimensions: List[str]
    generation_mode: str                      # "diff" | "full" — effective mode for this iter
    prompt_extras: Dict[str, str]             # reflection / sota_injection / dead_end_warnings
    temperature: Optional[float] = None
    model_id: Optional[str] = None


@dataclass
class WorkerContext:
    """Tool plumbing the harness hands to Worker.run(). Workers don't own harness objects —
    they receive handles."""
    prompt_sampler: "PromptSampler"
    llm_factory: Callable[[str], "LLMInterface"]
    dspy_lm_factory: Callable[["WorkerConfig"], Any]        # builds a configured dspy.LM
    evaluator: Optional["Evaluator"]                        # present iff WorkerConfig.uses_evaluator
    scratch_id_factory: Callable[[], str]                   # UUIDs for intermediate tool-call evals
    final_child_id: str                                     # assigned BEFORE worker.run
    config: "Config"
    iteration_config: "IterationConfig"


@dataclass
class WorkerResult:
    """Return value of Worker.run(). One child per iteration (multi-child deferred to v2)."""
    child_code: str
    changes_description: Optional[str] = None
    changes_summary: str = ""
    applied_edit: str = ""                    # feeds ConvergenceMonitor edit-entropy
    turns: List[TurnRecord] = field(default_factory=list)
    turn_count: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    error: Optional[WorkerError] = None


class Worker(Protocol):
    """Swappable compute-node primitive. Implementations live in reps/workers/*.py
    and register themselves via @register(impl_name) in reps/workers/registry.py."""
    config: WorkerConfig

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "Worker": ...

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult: ...
```

- [ ] **Step 3: Verify module imports cleanly**

Run: `.venv/bin/python -c "from reps.workers.base import Worker, WorkerConfig, WorkerResult, TurnRecord, ContentBlock, WorkerError, WorkerRequest, WorkerContext; print('ok')"`

Expected: `ok`

- [ ] **Step 4: Regression check — existing tests still pass**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: `111 passed`

- [ ] **Step 5: Commit**

```bash
git add reps/workers/__init__.py reps/workers/base.py
git commit -m "$(cat <<'EOF'
feat(workers): base data shapes for Worker primitive

Introduces WorkerConfig, WorkerRequest, WorkerContext, WorkerResult,
TurnRecord, ContentBlock, WorkerError, and the Worker protocol. Pure
data-shape addition — no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Worker registry

**Files:**
- Create: `reps/workers/registry.py`

- [ ] **Step 1: Create the registry**

```python
"""Registry mapping impl names to Worker classes. Each worker module decorates
its class with @register(impl_name) to self-register on import."""
from __future__ import annotations

from typing import Dict, Type

from reps.workers.base import Worker, WorkerConfig

_IMPLS: Dict[str, Type[Worker]] = {}


def register(impl_name: str):
    def deco(cls: Type[Worker]) -> Type[Worker]:
        if impl_name in _IMPLS:
            raise ValueError(f"Worker impl '{impl_name}' already registered")
        _IMPLS[impl_name] = cls
        return cls
    return deco


def build_worker(cfg: WorkerConfig) -> Worker:
    try:
        cls = _IMPLS[cfg.impl]
    except KeyError:
        known = ", ".join(sorted(_IMPLS)) or "(none registered)"
        raise ValueError(
            f"Unknown worker impl '{cfg.impl}' for config '{cfg.name}'. Known: {known}"
        ) from None
    return cls.from_config(cfg)


def known_impls() -> list[str]:
    return sorted(_IMPLS)
```

- [ ] **Step 2: Import-check**

Run: `.venv/bin/python -c "from reps.workers.registry import register, build_worker, known_impls; print(known_impls())"`

Expected: `[]` (no workers registered yet)

- [ ] **Step 3: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `111 passed`

- [ ] **Step 4: Commit**

```bash
git add reps/workers/registry.py
git commit -m "$(cat <<'EOF'
feat(workers): registry for Worker implementations

@register(impl_name) decorator + build_worker(cfg) factory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Edit serializer

**Files:**
- Create: `reps/workers/edit_serializer.py`
- Test: `tests/test_edit_serializer.py` (opportunistic trivial test — not a gate)

- [ ] **Step 1: Create serializer**

```python
"""Serialize a sequence of (search, replace) diff blocks into canonical REPS
SEARCH/REPLACE text format.

Output is the inverse of reps.utils.extract_diffs — feed it back through
extract_diffs and you get the input blocks.

Rule (from spec §5 and applied_edit research):
  - `applied_edit` for tool-calling workers in diff-mode is this serialization
    of all edit_file blocks in APPLY ORDER (effort-preserving — overrides and
    re-edits are kept, not deduped).
  - ConvergenceMonitor.classify_edit only checks length + keyword presence
    (reps/convergence_monitor.py:30-70), so a multi-block concatenation gives
    a sensible edit-entropy signal.
"""
from __future__ import annotations

from typing import Iterable, Tuple


def serialize_diff_blocks(blocks: Iterable[Tuple[str, str]]) -> str:
    """Return canonical REPS SEARCH/REPLACE text for the given blocks."""
    parts: list[str] = []
    for search, replace in blocks:
        parts.append("<<<<<<< SEARCH\n")
        parts.append(search)
        if not search.endswith("\n"):
            parts.append("\n")
        parts.append("=======\n")
        parts.append(replace)
        if not replace.endswith("\n"):
            parts.append("\n")
        parts.append(">>>>>>> REPLACE\n")
    return "".join(parts)
```

- [ ] **Step 2: Write round-trip test (opportunistic)**

```python
# tests/test_edit_serializer.py
from reps.config import Config
from reps.utils import extract_diffs
from reps.workers.edit_serializer import serialize_diff_blocks


def test_serialize_roundtrips_through_extract_diffs():
    blocks = [
        ("old_a\n", "new_a\n"),
        ("def foo():\n    pass\n", "def foo():\n    return 1\n"),
    ]
    text = serialize_diff_blocks(blocks)
    pattern = Config().diff_pattern
    extracted = extract_diffs(text, pattern)
    assert extracted == blocks


def test_serialize_handles_missing_trailing_newlines():
    blocks = [("search_no_nl", "replace_no_nl")]
    text = serialize_diff_blocks(blocks)
    assert "<<<<<<< SEARCH" in text
    assert "=======" in text
    assert ">>>>>>> REPLACE" in text
```

- [ ] **Step 3: Run new test**

Run: `.venv/bin/python -m pytest tests/test_edit_serializer.py -v`
Expected: 2 passed

- [ ] **Step 4: Full regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed` (111 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add reps/workers/edit_serializer.py tests/test_edit_serializer.py
git commit -m "$(cat <<'EOF'
feat(workers): serialize_diff_blocks canonical SEARCH/REPLACE helper

Used by SingleCallWorker and tool-calling workers (via edit_file tool) to
produce the `applied_edit` string consumed by ConvergenceMonitor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Trace renderer

**Files:**
- Create: `reps/workers/trace_render.py`

- [ ] **Step 1: Create renderer**

```python
"""Human-readable pretty-printer for List[TurnRecord]. Used by CLI inspection
and post-hoc log viewing."""
from __future__ import annotations

import json
from typing import List

from reps.workers.base import TurnRecord


def render_trace(turns: List[TurnRecord]) -> str:
    out: list[str] = []
    for t in turns:
        header = f"=== turn {t.index} [{t.role}]"
        if t.model_id:
            header += f" {t.model_id}"
        if t.stop_reason:
            header += f" stop={t.stop_reason}"
        out.append(header + " ===")
        for b in t.blocks:
            if b.type == "text":
                out.append(b.text or "")
            elif b.type == "thinking":
                sig = "[signed]" if b.signature else "[unsigned]"
                out.append(f"[thinking {sig}]\n{b.text or ''}")
            elif b.type == "redacted_thinking":
                out.append("[thinking REDACTED]")
            elif b.type == "tool_use":
                inp = json.dumps(b.tool_input, indent=2) if b.tool_input is not None else ""
                out.append(f"[tool_use {b.tool_name} id={b.tool_use_id}]\n  input: {inp}")
            elif b.type == "tool_result":
                err = " ERROR" if b.tool_result_is_error else ""
                body = (
                    b.tool_result_content
                    if isinstance(b.tool_result_content, str)
                    else json.dumps(b.tool_result_content, indent=2)
                )
                out.append(f"[tool_result for={b.tool_result_for_id}{err}]\n{body}")
        if t.usage:
            out.append(
                f"  usage: in={t.usage.get('input_tokens')} "
                f"out={t.usage.get('output_tokens')} "
                f"cache_read={t.usage.get('cache_read_input_tokens', 0)}"
            )
        out.append("")
    return "\n".join(out)
```

- [ ] **Step 2: Import-check**

Run: `.venv/bin/python -c "from reps.workers.trace_render import render_trace; from reps.workers.base import TurnRecord, ContentBlock; print(render_trace([TurnRecord(index=0, role='user', blocks=[ContentBlock(type='text', text='hi')])]))"`

Expected output contains `=== turn 0 [user] ===` and `hi`.

- [ ] **Step 3: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 4: Commit**

```bash
git add reps/workers/trace_render.py
git commit -m "$(cat <<'EOF'
feat(workers): render_trace pretty-printer for TurnRecord lists

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `reps/runtime.py` — current_program_id contextvar helper

**Files:**
- Create: `reps/runtime.py`

- [ ] **Step 1: Create runtime helper**

```python
"""Per-call program-id helper used by benchmark evaluators.

Under asyncio, multiple iterations run concurrently in one process. Each call
to Evaluator.evaluate_isolated sets the contextvar below; asyncio propagates
ContextVar state per-Task so concurrent calls see independent values.

Benchmarks that previously read os.environ["REPS_PROGRAM_ID"] should prefer
this helper. os.environ is no longer mutated by the evaluator in the asyncio
world — the env var is only set inside the subprocess child via Popen(env=...).
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

_program_id_var: ContextVar[Optional[str]] = ContextVar("reps_program_id", default=None)


def set_current_program_id(program_id: Optional[str]):
    """Set the current program_id for the calling asyncio Task. Returns a token
    the caller must pass to reset_current_program_id in a `finally` block."""
    return _program_id_var.set(program_id)


def reset_current_program_id(token) -> None:
    _program_id_var.reset(token)


def current_program_id() -> Optional[str]:
    """Return the current program_id for this Task. Falls back to
    os.environ['REPS_PROGRAM_ID'] for backwards compatibility inside
    subprocess children that inherit the env var."""
    pid = _program_id_var.get()
    if pid is not None:
        return pid
    return os.environ.get("REPS_PROGRAM_ID")
```

- [ ] **Step 2: Import-check**

Run: `.venv/bin/python -c "from reps.runtime import current_program_id, set_current_program_id, reset_current_program_id; print(current_program_id())"`
Expected: `None`

- [ ] **Step 3: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 4: Commit**

```bash
git add reps/runtime.py
git commit -m "$(cat <<'EOF'
feat(runtime): current_program_id contextvar helper

Per-asyncio-Task program_id for benchmarks that previously read
os.environ. Prerequisite for evaluator isolation and asyncio migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Evaluator — `evaluate_isolated` + contextvar artifacts

**Files:**
- Modify: `reps/evaluator.py`

- [ ] **Step 1: Read current evaluator structure**

Run: `grep -n "_pending_artifacts\|_current_program_id\|REPS_PROGRAM_ID\|def evaluate_program\|def __init__\|run_with_timeout\|evaluate_function" reps/evaluator.py | head -40`

Note the line numbers you find for later reference.

- [ ] **Step 2: Add `EvaluationOutcome` dataclass and contextvar/semaphore at top of file**

At the top of `reps/evaluator.py`, in the imports section (keep existing imports; add these), add:

```python
import asyncio
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
```

If already imported, don't duplicate. Then immediately below the existing imports but above the `Evaluator` class, add:

```python
@dataclass
class EvaluationOutcome:
    """Result of one isolated evaluation call — metrics + artifacts + id."""
    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)
    program_id: str = ""


# Per-call artifact collection scoped via asyncio Task-local contextvar.
# Each evaluate_isolated() sets a fresh dict; the collect_artifact API reads
# ContextVar-local state so concurrent calls don't share artifacts.
_call_artifacts: ContextVar[Optional[Dict[str, Union[str, bytes]]]] = ContextVar(
    "reps_evaluator_call_artifacts", default=None
)
```

- [ ] **Step 3: Add `_eval_semaphore` to `Evaluator.__init__`**

Find `Evaluator.__init__` (grep: `def __init__` near top of `Evaluator` class). After `self.task_pool = ...` or wherever concurrency primitives live, add:

```python
# Bounded concurrency for evaluate_isolated under asyncio (replaces TaskPool).
max_parallel = max(1, getattr(config, "parallel_evaluations", 1))
self._eval_semaphore = asyncio.Semaphore(max_parallel)
```

If the file is already importing `asyncio`, you're done with imports.

- [ ] **Step 4: Delete `self._current_program_id` field if present**

Find `self._current_program_id = ` inside `Evaluator.__init__` and delete the line. Also grep the rest of `evaluator.py` for any read of `self._current_program_id` and replace with `current_program_id()` imported from `reps.runtime`.

Add at top of file (near other imports):

```python
from reps.runtime import set_current_program_id, reset_current_program_id, current_program_id
```

- [ ] **Step 5: Add `evaluate_isolated` method on `Evaluator`**

Add inside the `Evaluator` class (alongside existing methods):

```python
async def evaluate_isolated(
    self,
    program_code: str,
    *,
    program_id: Optional[str] = None,
    scratch: bool = False,
    run_dir: Optional[str] = None,
) -> EvaluationOutcome:
    """Run one isolated evaluation. Safe to call concurrently from asyncio Tasks.

    - `program_id`: if None, a scratch UUID is generated.
    - `scratch`: informational flag; caller guarantees this is a throwaway eval
      (e.g., from a tool-call); artifacts are returned but not stored globally.
    - `run_dir`: optional override for REPS_RUN_DIR passed to the subprocess env.

    Returns an `EvaluationOutcome(metrics, artifacts, program_id)`.
    """
    pid = program_id or f"scratch-{uuid.uuid4().hex[:12]}"
    async with self._eval_semaphore:
        token_artifacts = _call_artifacts.set({})
        token_pid = set_current_program_id(pid)
        try:
            # Build per-call env for subprocess spawn. Does NOT mutate os.environ.
            call_env = dict(os.environ)
            call_env["REPS_PROGRAM_ID"] = pid
            if run_dir is not None:
                call_env["REPS_RUN_DIR"] = run_dir

            metrics = await self._evaluate_code_with_env(
                program_code=program_code,
                program_id=pid,
                env=call_env,
            )
            artifacts = dict(_call_artifacts.get() or {})
            return EvaluationOutcome(metrics=metrics, artifacts=artifacts, program_id=pid)
        finally:
            reset_current_program_id(token_pid)
            _call_artifacts.reset(token_artifacts)
```

Notes:
- `self._evaluate_code_with_env(...)` is a new private helper introduced in the next step; it wraps the existing retry/cascade pipeline and threads the env dict through.
- Artifacts inside the call are written via `collect_artifact(...)` (see Step 7) which looks up the contextvar.

- [ ] **Step 6: Introduce `_evaluate_code_with_env` private helper**

This is a thin wrapper around the existing eval pipeline (whatever method currently implements the retry/cascade loop inside `evaluator.py` — grep for `async def evaluate_program` or similar). Add this method and have it call the current eval logic with the env dict threaded through to `run_with_timeout` / `_direct_evaluate`.

Exact shape (adapt to match the current private method name; if the current code has `_direct_evaluate(...)` that calls `run_with_timeout(path, timeout)`, change that call to `run_with_timeout(path, timeout, env=env)`):

```python
async def _evaluate_code_with_env(
    self,
    program_code: str,
    program_id: str,
    env: Dict[str, str],
) -> Dict[str, float]:
    """Internal: run the full eval pipeline (retry/cascade) for `program_code`
    and return the final metrics dict. `env` is passed to subprocess spawns
    inside the benchmark evaluator (via run_with_timeout(env=env))."""
    # Call existing retry/cascade. Keep logic identical to evaluate_program's
    # body, but replace any os.environ mutation with passing env through.
    # If the current method signature is _direct_evaluate(code, program_id),
    # extend it to take env= and pass to run_with_timeout.
    return await self._direct_evaluate(program_code, program_id, env=env)
```

Then find the current `_direct_evaluate` (or equivalent) method. Modify its signature to accept `env: Optional[Dict[str, str]] = None`. Inside it, where `run_with_timeout(program_path, timeout_seconds)` is called — that helper either lives in the same file or in the benchmark evaluator module. Grep for `run_with_timeout` to find all references:

Run: `grep -rn "run_with_timeout" reps/ experiment/`

For each call site inside `reps/evaluator.py`, add `env=env`. If `run_with_timeout` is defined in `reps/evaluator.py`, extend its signature:

```python
def run_with_timeout(
    program_path: str,
    timeout_seconds: int,
    env: Optional[Dict[str, str]] = None,
) -> ...:
    # existing body; when spawning a subprocess via subprocess.Popen or similar,
    # pass env=env (or env=None to inherit).
    ...
```

If `run_with_timeout` lives in the benchmark evaluator (`experiment/benchmarks/circle_packing/evaluator.py`), that's handled in Task 7.

- [ ] **Step 7: Delete `os.environ["REPS_PROGRAM_ID"]` mutation**

Grep: `grep -n "REPS_PROGRAM_ID" reps/evaluator.py`

At every line that sets or unsets `os.environ["REPS_PROGRAM_ID"]` inside `Evaluator`, delete the mutation. The env var reaches benchmarks via the per-call env dict now.

- [ ] **Step 8: Refactor `_pending_artifacts` writes to use contextvar**

Grep: `grep -n "_pending_artifacts" reps/evaluator.py`

For each site that writes `self._pending_artifacts[program_id] = ...` or similar, replace with writes to the contextvar dict. Keep the legacy `_pending_artifacts` dict field — it's still populated from the contextvar inside the legacy `evaluate_program` wrapper (Step 9) so `get_pending_artifacts()` remains backward-compatible for the seed eval path in `runner.py`.

Replace patterns like:
```python
self._pending_artifacts.setdefault(program_id, {})[key] = value
```
with:
```python
artifacts = _call_artifacts.get()
if artifacts is None:
    # Not inside evaluate_isolated; fall through to legacy dict.
    self._pending_artifacts.setdefault(program_id, {})[key] = value
else:
    artifacts[key] = value
```

- [ ] **Step 9: Keep legacy `evaluate_program` as wrapper**

The existing public `async def evaluate_program(code, program_id) -> Dict[str, float]` is still called by `reps/runner.py:67` (seed eval). Make it a thin wrapper over `evaluate_isolated`:

```python
async def evaluate_program(self, program_code: str, program_id: str) -> Dict[str, float]:
    """Legacy wrapper: evaluate and stuff artifacts into the global _pending_artifacts
    dict so `get_pending_artifacts(program_id)` keeps working for seed evals."""
    outcome = await self.evaluate_isolated(program_code, program_id=program_id)
    if outcome.artifacts:
        self._pending_artifacts.setdefault(program_id, {}).update(outcome.artifacts)
    return outcome.metrics
```

- [ ] **Step 10: Regression check**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: `113 passed`. If failures appear, they're most likely from the `_direct_evaluate(env=)` signature change — trace callers and fix signatures.

- [ ] **Step 11: Commit**

```bash
git add reps/evaluator.py
git commit -m "$(cat <<'EOF'
feat(evaluator): evaluate_isolated with contextvar-scoped artifacts

- New EvaluationOutcome dataclass and evaluate_isolated(code, program_id=,
  scratch=, run_dir=) returning (metrics, artifacts, program_id).
- _call_artifacts: ContextVar propagates per asyncio Task so overlapping
  eval calls (mid-loop tool-call evals or concurrent iterations) don't
  share artifact state.
- asyncio.Semaphore(parallel_evaluations) bounds concurrency at the
  evaluator layer.
- os.environ["REPS_PROGRAM_ID"] mutation removed; env var threaded
  through a per-call env dict to subprocess spawns.
- Legacy evaluate_program kept as wrapper for the seed eval path.

Fixes a real race: circle_packing/evaluator.py:145 spawned subprocess
without env= kwarg, so concurrent iterations overwrote each other's
program_id.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Benchmark — circle_packing evaluator respects per-call env

**Files:**
- Modify: `experiment/benchmarks/circle_packing/evaluator.py`

- [ ] **Step 1: Grep current state**

Run: `grep -n "subprocess.Popen\|REPS_PROGRAM_ID\|def run_with_timeout\|def evaluate\|def _dump_packing_markdown" experiment/benchmarks/circle_packing/evaluator.py`

- [ ] **Step 2: Thread `env` kwarg into `run_with_timeout`**

Find `def run_with_timeout(program_path, timeout_seconds)`. Change signature to:

```python
def run_with_timeout(program_path, timeout_seconds, env=None):
```

Inside, find `subprocess.Popen([...], stdout=..., stderr=...)` and add `env=env`:

```python
proc = subprocess.Popen(
    [sys.executable, program_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,        # None means inherit (legacy); a dict means use these vars
)
```

- [ ] **Step 3: Thread `env` down from `evaluate` to `run_with_timeout`**

Find `def evaluate(program_path):` (or similar). Change to accept `env=None`:

```python
def evaluate(program_path, env=None):
    # existing body; at run_with_timeout call-site:
    result = run_with_timeout(program_path, timeout_seconds=..., env=env)
    ...
```

Wherever `run_with_timeout(...)` is invoked in this file, pass `env=env`.

- [ ] **Step 4: Update `_dump_packing_markdown` to read program_id via runtime helper**

Grep the file for `os.environ["REPS_PROGRAM_ID"]` or `os.environ.get("REPS_PROGRAM_ID"`. Each location reading it for the purpose of naming the markdown dump file should change to:

```python
from reps.runtime import current_program_id
...
program_id = current_program_id() or "unknown"
```

Keep `os.environ["REPS_PROGRAM_ID"]` reads inside the *subprocess child's* code path — the subprocess inherits env from `Popen(env=env)`, so its view of `os.environ` is correct.

- [ ] **Step 5: Update REPS-side caller to pass `env`**

`reps/evaluator.py` has a method that invokes the benchmark's `evaluate` (grep: `self.evaluate_function`). Since `evaluate_function` may have been loaded via `inspect`-based importing (the seed pattern), introspect its signature and pass `env` only if the function accepts it:

In `reps/evaluator.py`, inside `_direct_evaluate` (or wherever `self.evaluate_function(program_path)` is called):

```python
import inspect
sig = inspect.signature(self.evaluate_function)
kwargs = {}
if "env" in sig.parameters:
    kwargs["env"] = env
result = self.evaluate_function(program_path, **kwargs)
```

This preserves compatibility with older benchmark evaluators that don't have `env` in their signature.

- [ ] **Step 6: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

Additionally run a smoke test of the benchmark evaluator directly:

```bash
.venv/bin/python -c "
import sys
sys.path.insert(0, 'experiment/benchmarks/circle_packing')
from evaluator import evaluate
result = evaluate('experiment/benchmarks/circle_packing/initial_program.py')
print(result)
"
```

Expected: a metrics dict prints; no exception.

- [ ] **Step 7: Commit**

```bash
git add experiment/benchmarks/circle_packing/evaluator.py reps/evaluator.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): circle_packing respects per-call env dict

- run_with_timeout(env=) threaded from evaluate(env=); subprocess.Popen
  now passes env explicitly instead of inheriting os.environ at spawn
  time.
- _dump_packing_markdown reads program_id via reps.runtime.current_program_id()
  (contextvar) in the parent process; subprocess children still read
  os.environ["REPS_PROGRAM_ID"] which is now set via the per-call env dict.
- REPS evaluator detects benchmark-side env support via inspect.signature.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Rename `IterationConfig.worker_type` → `worker_name` (with alias)

**Files:**
- Modify: `reps/iteration_config.py`
- Modify: any reader of `iteration_config.worker_type` in `reps/`

- [ ] **Step 1: Grep existing usages**

Run: `grep -rn "\\.worker_type\\|worker_type=" reps/ tests/ | grep -v ".pyc"`

Note every file that reads or writes `worker_type` so you can update them.

- [ ] **Step 2: Modify `reps/iteration_config.py`**

Open `reps/iteration_config.py`. Find the `IterationConfig` dataclass. Rename the `worker_type` field to `worker_name` and add a `__post_init__` that logs deprecation on old-style construction:

```python
@dataclass
class IterationConfig:
    parent_id: Optional[str] = None

    # Worker selection — renamed from worker_type. Alias kept for one release.
    worker_name: str = "exploiter"

    model_id: Optional[str] = None
    temperature: Optional[float] = None
    prompt_extras: Dict[str, str] = field(default_factory=dict)
    second_parent_id: Optional[str] = None
    is_revisitation: bool = False
    generation_mode: str = "diff"
    target_island: Optional[int] = None

    # Backward-compat shim: accept worker_type=... at construction.
    def __post_init__(self):
        pass


def _from_dict_with_alias(cls, data: dict):
    """Helper for deserializers: rename 'worker_type' → 'worker_name' if present."""
    if "worker_type" in data and "worker_name" not in data:
        data = {**data, "worker_name": data.pop("worker_type")}
    return cls(**data)
```

Also add a read-only `worker_type` property that returns `worker_name`, so existing readers don't break:

```python
    # NB: define as an attribute via property to keep dataclass serialization stable.
    @property
    def worker_type(self) -> str:
        return self.worker_name
```

(If dataclass+property interaction is awkward, leave the `worker_type` read pattern and update all call-sites in Step 3.)

Similarly rename the `worker_type` field on `IterationResult` to `worker_name` and add a property alias.

Also add a new field on `IterationResult`:

```python
    turns: List[Dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 3: Update each call-site identified in Step 1**

For every `iter_cfg.worker_type` usage, change to `iter_cfg.worker_name`. Same for `IterationResult`. Where dict construction is `{"worker_type": ...}`, change key to `"worker_name"`. This is purely textual.

- [ ] **Step 4: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed` (properties provide back-compat; all reads still work)

- [ ] **Step 5: Commit**

```bash
git add reps/iteration_config.py reps/  # and any other touched files
git commit -m "$(cat <<'EOF'
refactor: IterationConfig/IterationResult.worker_type → worker_name

Renaming in prep for named WorkerConfig entries as the selection primitive.
Backward-compat: @property worker_type returns worker_name. Also adds
IterationResult.turns: List[Dict] for TurnRecord persistence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Controller asyncio migration — scaffolding

**Files:**
- Modify: `reps/controller.py`
- Modify: `reps/config.py` (add `max_concurrent_iterations`)

This task adds the new async infrastructure *alongside* the existing multiprocess path. The next task removes the old path. Splitting keeps each commit runnable.

- [ ] **Step 1: Add config field**

In `reps/config.py`, locate `class EvaluatorConfig`. Add:

```python
@dataclass
class EvaluatorConfig:
    # ... existing fields ...
    max_concurrent_iterations: int = 4   # asyncio-era concurrency bound
```

Keep `parallel_evaluations` as-is (used by the evaluator's semaphore in Task 6). If the YAML loader needs aliasing, add a `__post_init__` that copies `parallel_evaluations` to `max_concurrent_iterations` if the latter is left at default:

```python
    def __post_init__(self):
        # If user set only parallel_evaluations, mirror it to the iteration cap.
        if self.max_concurrent_iterations == 4 and self.parallel_evaluations not in (None, 1):
            self.max_concurrent_iterations = int(self.parallel_evaluations)
```

- [ ] **Step 2: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 3: Commit**

```bash
git add reps/config.py
git commit -m "$(cat <<'EOF'
config: add EvaluatorConfig.max_concurrent_iterations

Asyncio-era concurrency knob. Mirrors parallel_evaluations when the
latter is set and the former is left at default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Controller — async `_run_iteration` and full loop rewrite

This is the load-bearing task. Aim to do it in one sitting; the commit should leave the harness runnable.

**Files:**
- Modify: `reps/controller.py`
- Modify: `tests/test_controller.py` (lines 92, 105, 124, 174)

- [ ] **Step 1: Read the current controller**

Run: `wc -l reps/controller.py && grep -n "^def \|^    def \|^class " reps/controller.py | head -50`

Study: `_run_iteration_worker` (~L171), `ProcessParallelController` (~L460), `run_evolution` (~L651), `_submit_iteration` (~L1014), `_create_database_snapshot` (~L621), `_worker_init` (~L53).

- [ ] **Step 2: Add imports at the top of `reps/controller.py`**

At the top of `reps/controller.py`, remove:

```python
from concurrent.futures import ProcessPoolExecutor, Future
import multiprocessing as mp
import pickle
import signal
```

(Keep them if they're referenced by other helpers in the file; prune only what's safe.)

Add:

```python
import asyncio
import uuid
from contextlib import asynccontextmanager
```

- [ ] **Step 3: Write the new async iteration method**

Insert this new method on `ProcessParallelController` (we'll alias as `AsyncController` later). This replaces `_run_iteration_worker` + all its process-init dance:

```python
async def _run_iteration(
    self,
    iteration: int,
    parent_id: str,
    inspiration_ids: List[str],
    iteration_config: "IterationConfig",
) -> "SerializableResult":
    """One iteration: build WorkerRequest, dispatch Worker, evaluate child.

    Runs on the main event loop. No process fork. Shares LLMEnsemble /
    PromptSampler / Evaluator with the controller."""
    from reps.database import Program, safe_numeric_average
    from reps.iteration_config import IterationConfig
    from reps.workers.base import WorkerRequest, WorkerContext
    from reps.workers.registry import build_worker

    final_child_id = str(uuid.uuid4())

    # Look up parent and inspirations from the live database (no snapshot).
    parent = self.database.programs[parent_id]
    inspirations = [
        self.database.programs[pid] for pid in inspiration_ids
        if pid in self.database.programs
    ]

    parent_island = parent.metadata.get("island", self.database.current_island)
    island_pids = list(self.database.islands[parent_island])
    island_programs = [self.database.programs[p] for p in island_pids if p in self.database.programs]
    island_programs.sort(
        key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
        reverse=True,
    )
    programs_for_prompt = island_programs[
        : self.config.prompt.num_top_programs + self.config.prompt.num_diverse_programs
    ]
    best_programs_only = island_programs[: self.config.prompt.num_top_programs]

    second_parent = None
    if iteration_config.second_parent_id and iteration_config.second_parent_id in self.database.programs:
        second_parent = self.database.programs[iteration_config.second_parent_id]

    request = WorkerRequest(
        parent=parent,
        inspirations=inspirations,
        top_programs=best_programs_only,
        second_parent=second_parent,
        iteration=iteration,
        language=self.config.language,
        feature_dimensions=self.database.config.feature_dimensions,
        generation_mode=iteration_config.generation_mode,
        prompt_extras=dict(iteration_config.prompt_extras),
        temperature=iteration_config.temperature,
        model_id=iteration_config.model_id,
    )

    # WorkerContext construction
    worker_cfg = self.worker_pool.get_worker_config(iteration_config.worker_name)
    ctx = WorkerContext(
        prompt_sampler=self.prompt_sampler,
        llm_factory=self._llm_factory,
        dspy_lm_factory=self._dspy_lm_factory,
        evaluator=self.evaluator if worker_cfg.uses_evaluator else None,
        scratch_id_factory=lambda: f"scratch-{uuid.uuid4().hex[:12]}",
        final_child_id=final_child_id,
        config=self.config,
        iteration_config=iteration_config,
    )

    worker = build_worker(worker_cfg)

    import time
    t_start = time.time()
    result = await worker.run(request, ctx)
    iteration_time = time.time() - t_start

    if result.error is not None:
        return SerializableResult(
            error=str(result.error),
            iteration=iteration,
            iteration_time=iteration_time,
        )

    # Length check
    if len(result.child_code) > self.config.max_code_length:
        return SerializableResult(
            error=f"Generated code exceeds maximum length ({len(result.child_code)} > {self.config.max_code_length})",
            iteration=iteration,
            iteration_time=iteration_time,
        )

    # Final evaluation — the scored one whose artifacts land in the DB.
    outcome = await self.evaluator.evaluate_isolated(
        result.child_code, program_id=final_child_id
    )

    child_metadata = {
        "changes": result.changes_summary or "",
        "parent_metrics": parent.metrics,
        "island": parent_island,
        "reps_worker_name": iteration_config.worker_name,
        "reps_is_revisitation": iteration_config.is_revisitation,
    }
    if iteration_config.model_id:
        child_metadata["reps_model_id"] = iteration_config.model_id

    child = Program(
        id=final_child_id,
        code=result.child_code,
        changes_description=result.changes_description,
        language=self.config.language,
        parent_id=parent.id,
        generation=parent.generation + 1,
        metrics=outcome.metrics,
        iteration_found=iteration,
        metadata=child_metadata,
    )

    parent_score = (
        parent.metrics.get("combined_score", safe_numeric_average(parent.metrics))
        if parent.metrics
        else 0.0
    )

    # Derive legacy prompt/llm_response from turns for back-compat.
    prompt_dict = _derive_prompt_from_turns(result.turns)
    llm_response_str = _derive_llm_response_from_turns(result.turns)

    return SerializableResult(
        child_program_dict=child.to_dict(),
        parent_id=parent.id,
        iteration_time=iteration_time,
        prompt=prompt_dict,
        llm_response=llm_response_str,
        artifacts=outcome.artifacts,
        iteration=iteration,
        target_island=self.database.sampling_island if hasattr(self.database, "sampling_island") else None,
        reps_meta={
            "worker_name": iteration_config.worker_name,
            "is_revisitation": iteration_config.is_revisitation,
            "model_id": iteration_config.model_id,
            "temperature": iteration_config.temperature,
            "parent_score": parent_score,
            "diff": result.applied_edit,
            "turns": [_turn_to_dict(t) for t in result.turns],
            "tokens_in": result.usage.get("prompt_tokens", result.usage.get("input_tokens", 0)),
            "tokens_out": result.usage.get("completion_tokens", result.usage.get("output_tokens", 0)),
            "wall_clock_seconds": result.wall_clock_seconds,
        },
    )
```

And the two small helpers (put at module level):

```python
def _derive_prompt_from_turns(turns) -> Dict[str, str]:
    """Reconstruct {'system': ..., 'user': ...} from the first user/system turns."""
    system_text = ""
    user_text = ""
    for t in turns:
        if t.role == "system" and not system_text:
            system_text = "\n".join(b.text or "" for b in t.blocks if b.type == "text")
        if t.role == "user" and not user_text:
            user_text = "\n".join(b.text or "" for b in t.blocks if b.type == "text")
        if system_text and user_text:
            break
    return {"system": system_text, "user": user_text}


def _derive_llm_response_from_turns(turns) -> str:
    """Concatenate text blocks from the last assistant turn (skip thinking/tool_use)."""
    for t in reversed(turns):
        if t.role == "assistant":
            return "\n".join(b.text or "" for b in t.blocks if b.type == "text")
    return ""


def _turn_to_dict(t) -> Dict[str, Any]:
    """Dataclass-to-dict, preserving signature bytes and all provider_extras."""
    from dataclasses import asdict
    return asdict(t)
```

- [ ] **Step 4: Replace `_submit_iteration` with async `_spawn_iteration`**

Find the current `_submit_iteration(self, iteration, parent_id, reps_config)` method. Replace its body with:

```python
async def _spawn_iteration(
    self,
    iteration: int,
    parent_id: str,
    inspiration_ids: List[str],
    iteration_config: "IterationConfig",
) -> "SerializableResult":
    """Wrapper that bounds concurrency via the semaphore and catches
    per-iteration exceptions so TaskGroup doesn't cancel siblings."""
    async with self._iter_semaphore:
        try:
            return await self._run_iteration(
                iteration=iteration,
                parent_id=parent_id,
                inspiration_ids=inspiration_ids,
                iteration_config=iteration_config,
            )
        except Exception as e:
            logger.exception(f"iteration {iteration} failed in spawn")
            return SerializableResult(error=str(e), iteration=iteration)
```

- [ ] **Step 5: Rewrite `start()`, drop `_worker_init`/snapshot/executor**

In `ProcessParallelController.start()`:

- Delete everything about `ProcessPoolExecutor`, `_worker_init`, `parent_env`.
- Build one shared `LLMEnsemble`, `PromptSampler`, `Evaluator`. (These already exist as main-process attributes in the current code for reflection/evaluator usage; just wire them up consistently.)
- Initialize `self._iter_semaphore = asyncio.Semaphore(self.config.evaluator.max_concurrent_iterations)`.
- Initialize `self._shutdown = asyncio.Event()`.
- Build `self.worker_pool` — the updated version from Task 16 (we'll wire it in detail then; for now assume the existing WorkerPool with a stub `get_worker_config(name) -> WorkerConfig` is available; Task 16 replaces the stub).
- Build `self._llm_factory` and `self._dspy_lm_factory` closures.

Concretely:

```python
def start(self) -> None:
    from reps.evaluator import Evaluator
    from reps.llm.ensemble import LLMEnsemble
    from reps.prompt_sampler import PromptSampler
    from reps.workers.base import WorkerConfig

    self.llm_ensemble = LLMEnsemble(self.config.llm.models)
    self.prompt_sampler = PromptSampler(self.config.prompt)
    self.evaluator = Evaluator(
        self.config.evaluator,
        self.evaluation_file,
        LLMEnsemble(self.config.llm.evaluator_models),
        self.prompt_sampler,
        database=self.database,
        suffix=self.config.file_suffix,
    )

    self._iter_semaphore = asyncio.Semaphore(self.config.evaluator.max_concurrent_iterations)
    self._shutdown = asyncio.Event()

    def llm_factory(model_id: str):
        # Trivial model-id override path for workers that want a specific model.
        # Falls back to the ensemble's weighted sampling.
        return self.llm_ensemble

    self._llm_factory = llm_factory

    def dspy_lm_factory(wc: "WorkerConfig"):
        from reps.workers.dspy_react import make_dspy_lm   # Task 15
        return make_dspy_lm(self.config, wc)

    self._dspy_lm_factory = dspy_lm_factory

    # WorkerPool from Task 16. For now, stub if not yet migrated:
    try:
        from reps.worker_pool import WorkerPool
        self.worker_pool = WorkerPool(self.config.reps.workers, worker_types=getattr(self.config.reps.workers, "types", []))
    except TypeError:
        # Legacy constructor signature — Task 16 migrates this.
        self.worker_pool = WorkerPool(self.config.reps.workers)

    logger.info("AsyncController started (asyncio-only, semaphore=%d)", self.config.evaluator.max_concurrent_iterations)


def stop(self) -> None:
    if self._shutdown is not None:
        self._shutdown.set()
    logger.info("AsyncController stopping")


def request_shutdown(self) -> None:
    self.stop()
```

- [ ] **Step 6: Rewrite `run_evolution` main loop**

Find `async def run_evolution(...)`. Port the body so it:

1. Uses `asyncio.TaskGroup` and `self._iter_semaphore` instead of `ProcessPoolExecutor`.
2. Uses `asyncio.wait(pending, return_when=FIRST_COMPLETED, timeout=...)` instead of a busy-poll over `future.done()`.
3. Shares `self.database` directly (no snapshot).
4. Still calls the REPS feature methods (`_reps_build_iteration_config`, `_reps_process_batch`, etc.) unchanged.

Key surgery points (existing code lives at `reps/controller.py:651-1012`):

Replace:
```python
pending_futures: Dict[int, Future] = {}
# ... busy loop with future.done() ...
```
with:
```python
pending: Dict[int, asyncio.Task] = {}
```

Replace:
```python
snapshot = self._create_database_snapshot()
future = self.executor.submit(_run_iteration_worker, iteration, snapshot, ...)
pending_futures[iteration] = future
```
with:
```python
task = asyncio.create_task(
    self._spawn_iteration(
        iteration=iteration,
        parent_id=parent_id,
        inspiration_ids=inspiration_ids,
        iteration_config=iteration_config,
    )
)
pending[iteration] = task
```

Replace the busy-poll at ~L727-741 with:
```python
if pending:
    done_tasks, _ = await asyncio.wait(
        list(pending.values()),
        return_when=asyncio.FIRST_COMPLETED,
        timeout=self.config.evaluator.timeout + 30,
    )
    for task in done_tasks:
        # find the iteration key for this completed task
        for it_key, t in list(pending.items()):
            if t is task:
                result = t.result()
                del pending[it_key]
                # existing result-handling logic (database.add, reflection, etc.)
                break
```

Delete all `future.result(timeout=...)` calls — asyncio's `task.result()` is non-blocking on a completed task.

After the outer loop, add shutdown drain:
```python
if pending:
    for t in pending.values():
        t.cancel()
    await asyncio.gather(*pending.values(), return_exceptions=True)
```

- [ ] **Step 7: Delete `_run_iteration_worker`, `_worker_init`, `_lazy_init_worker_components`, `_create_database_snapshot`**

These are now dead. Remove the function bodies entirely. Leave a one-line comment pointing to the replacement if you want bystander code-readers to have a breadcrumb:

```python
# _run_iteration_worker removed — see _run_iteration (async, same process).
```

- [ ] **Step 8: Rewrite `tests/test_controller.py` sections**

Read those tests:

Run: `sed -n '85,185p' tests/test_controller.py`

Replace:
- The test at L92 that asserts `controller.executor is not None` → assert `controller.evaluator is not None` and `controller.llm_ensemble is not None`.
- The test at L105 for `_create_database_snapshot` → delete (snapshot is gone).
- The test at L124 that mocks `executor.submit` returning a `Future` → rewrite to patch `_run_iteration` (or `_spawn_iteration`) to return an awaitable (use `unittest.mock.AsyncMock`) that yields a `SerializableResult`. Drive the test via `asyncio.run(controller.run_evolution(...))`.
- The test at L174 asserting `controller.shutdown_event.is_set()` (mp.Event) → assert `controller._shutdown.is_set()`.
- The test at L206 uses `asyncio.run(run_batch())` already — leave untouched.

Paste the new tests in full (even if similar to old ones; don't reference line numbers) so the executor doesn't have to diff in their heads:

```python
def test_controller_start_stop(basic_controller):
    basic_controller.start()
    assert basic_controller.llm_ensemble is not None
    assert basic_controller.evaluator is not None
    basic_controller.stop()


def test_run_evolution_basic(basic_controller, tmp_path):
    from unittest.mock import AsyncMock, patch
    from reps.controller import SerializableResult

    basic_controller.start()
    stub_result = SerializableResult(iteration=1, child_program_dict={"id": "x", "code": "pass", "language": "python", "parent_id": None, "generation": 1, "metrics": {"combined_score": 0.1}})

    with patch.object(basic_controller, "_run_iteration", new=AsyncMock(return_value=stub_result)) as m:
        import asyncio
        asyncio.run(basic_controller.run_evolution(start_iteration=1, max_iterations=1))
        assert m.await_count == 1


def test_request_shutdown(basic_controller):
    basic_controller.start()
    basic_controller.request_shutdown()
    assert basic_controller._shutdown.is_set()
```

- [ ] **Step 9: Regression check**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: `113 passed` (some tests updated, but count should be steady or close to it).

If there are failures related to worker_pool (Task 16 hasn't happened yet), temporarily stub `self.worker_pool = ...` with the legacy pool and skip `get_worker_config`:

```python
    def get_worker_config(self, name: str) -> "WorkerConfig":
        # TEMP: legacy shim while Task 16 is pending.
        from reps.workers.base import WorkerConfig
        return WorkerConfig(name=name, impl="single_call", role=name, model_id=self.config.llm.models[0].name if self.config.llm.models else "", temperature=0.7, generation_mode="diff")
```

Remove this shim in Task 16.

- [ ] **Step 10: Commit**

```bash
git add reps/controller.py tests/test_controller.py
git commit -m "$(cat <<'EOF'
refactor(controller): asyncio-only iteration loop

- _run_iteration_worker (process-bound) → _run_iteration (async method).
- ProcessPoolExecutor / mp.Event / pickle / _create_database_snapshot gone.
- asyncio.Semaphore + asyncio.TaskGroup bound concurrency; asyncio.wait
  replaces busy-poll over Future.done().
- One shared LLMEnsemble / Evaluator / PromptSampler for all iterations.
- Tests updated for the async path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `SingleCallWorker` — port today's behavior

**Files:**
- Create: `reps/workers/single_call.py`
- Modify: `reps/workers/__init__.py` (re-export impl module to trigger registration)

- [ ] **Step 1: Create the worker**

```python
"""SingleCallWorker: port of today's one-shot LLM call path under the Worker
primitive. Uses PromptSampler for prompt construction, LLMEnsemble for the
call, and reps.utils for diff/full-rewrite parsing."""
from __future__ import annotations

import time
from typing import Dict, List

from reps.iteration_config import IterationConfig  # for type only
from reps.utils import (
    apply_diff,
    apply_diff_blocks,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
    safe_numeric_average,
    split_diffs_by_target,
)
from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    Worker,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers.edit_serializer import serialize_diff_blocks
from reps.workers.registry import register


@register("single_call")
class SingleCallWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "SingleCallWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        ensemble = ctx.llm_factory(self.config.model_id)

        # Build prompt via PromptSampler (existing path)
        changes_description_text = None
        if ctx.config.prompt.programs_as_changes_description:
            changes_description_text = (
                request.parent.changes_description
                or ctx.config.prompt.initial_changes_description
            )

        use_diff = _decide_diff_mode(request.generation_mode, ctx.config.diff_based_evolution)

        prompt = ctx.prompt_sampler.build_prompt(
            current_program=request.parent.code,
            parent_program=request.parent.code,
            program_metrics=request.parent.metrics,
            previous_programs=[p.to_dict() for p in request.top_programs],
            top_programs=[p.to_dict() for p in (request.top_programs + request.inspirations)],
            inspirations=[p.to_dict() for p in request.inspirations],
            language=request.language,
            evolution_round=request.iteration,
            diff_based_evolution=use_diff,
            program_artifacts=None,
            feature_dimensions=request.feature_dimensions,
            current_changes_description=changes_description_text,
            **request.prompt_extras,
        )

        # Append un-consumed prompt_extras (existing behavior from controller.py:287)
        for key in ("reflection", "sota_injection", "dead_end_warnings"):
            text = request.prompt_extras.get(key, "")
            if text and "{" + key + "}" not in prompt.get("user", ""):
                prompt["user"] = prompt["user"] + "\n\n" + text

        gen_kwargs = {}
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.model_id is not None:
            gen_kwargs["model"] = request.model_id

        # Make the call
        try:
            llm_response = await ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
                **gen_kwargs,
            )
        except Exception as e:
            return WorkerResult(
                child_code="",
                turns=_build_turns_for_error(prompt),
                error=WorkerError(kind="INTERNAL", detail=f"LLM call failed: {e}"),
                wall_clock_seconds=time.monotonic() - t0,
            )

        if llm_response is None:
            return WorkerResult(
                child_code="",
                turns=_build_turns_for_error(prompt),
                error=WorkerError(kind="INTERNAL", detail="LLM returned None"),
                wall_clock_seconds=time.monotonic() - t0,
            )

        # Parse
        child_code: str = ""
        changes_summary: str = ""
        changes_description_out = changes_description_text
        applied_edit: str = ""

        if use_diff:
            diff_blocks = extract_diffs(llm_response, ctx.config.diff_pattern)
            if not diff_blocks:
                return WorkerResult(
                    child_code="",
                    turns=_build_turns(prompt, llm_response),
                    error=WorkerError(kind="PARSE_ERROR", detail="No valid diffs"),
                    wall_clock_seconds=time.monotonic() - t0,
                )
            if ctx.config.prompt.programs_as_changes_description:
                code_blocks, desc_blocks, _unmatched = split_diffs_by_target(
                    diff_blocks,
                    code_text=request.parent.code,
                    changes_description_text=changes_description_text,
                )
                child_code, _ = apply_diff_blocks(request.parent.code, code_blocks)
                new_desc, desc_applied = apply_diff_blocks(changes_description_text, desc_blocks)
                if desc_applied == 0 or not new_desc.strip() or new_desc.strip() == (changes_description_text or "").strip():
                    return WorkerResult(
                        child_code="",
                        turns=_build_turns(prompt, llm_response),
                        error=WorkerError(kind="PARSE_ERROR", detail="changes_description not updated"),
                        wall_clock_seconds=time.monotonic() - t0,
                    )
                changes_description_out = new_desc
                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=ctx.config.prompt.diff_summary_max_line_len,
                    max_lines=ctx.config.prompt.diff_summary_max_lines,
                )
                applied_edit = serialize_diff_blocks(code_blocks)
            else:
                child_code = apply_diff(request.parent.code, llm_response, ctx.config.diff_pattern)
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=ctx.config.prompt.diff_summary_max_line_len,
                    max_lines=ctx.config.prompt.diff_summary_max_lines,
                )
                applied_edit = serialize_diff_blocks(diff_blocks)
        else:
            new_code = parse_full_rewrite(llm_response, request.language)
            if not new_code:
                return WorkerResult(
                    child_code="",
                    turns=_build_turns(prompt, llm_response),
                    error=WorkerError(kind="PARSE_ERROR", detail="No valid code in response"),
                    wall_clock_seconds=time.monotonic() - t0,
                )
            child_code = new_code
            changes_summary = "Full rewrite"
            applied_edit = new_code

        usage = getattr(ensemble, "last_usage", {}) or {}
        return WorkerResult(
            child_code=child_code,
            changes_description=changes_description_out,
            changes_summary=changes_summary,
            applied_edit=applied_edit,
            turns=_build_turns(prompt, llm_response),
            turn_count=2,
            usage=dict(usage),
            wall_clock_seconds=time.monotonic() - t0,
            error=None,
        )


def _decide_diff_mode(request_mode: str, global_diff_enabled: bool) -> bool:
    if not global_diff_enabled:
        return False
    if request_mode == "full":
        return False
    if request_mode == "diff":
        return True
    return global_diff_enabled


def _build_turns(prompt: Dict[str, str], llm_response: str) -> List[TurnRecord]:
    return [
        TurnRecord(
            index=0,
            role="user",
            blocks=[
                ContentBlock(type="text", text=prompt.get("system", "")),
                ContentBlock(type="text", text=prompt.get("user", "")),
            ],
            worker_type="single_call",
        ),
        TurnRecord(
            index=1,
            role="assistant",
            blocks=[ContentBlock(type="text", text=llm_response)],
            worker_type="single_call",
        ),
    ]


def _build_turns_for_error(prompt: Dict[str, str]) -> List[TurnRecord]:
    return [
        TurnRecord(
            index=0,
            role="user",
            blocks=[
                ContentBlock(type="text", text=prompt.get("system", "")),
                ContentBlock(type="text", text=prompt.get("user", "")),
            ],
            worker_type="single_call",
        )
    ]
```

- [ ] **Step 2: Re-export from package init to trigger registration**

Edit `reps/workers/__init__.py`:

```python
"""reps.workers — Worker primitive and implementations."""
from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    Worker,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)

# Importing impls here triggers @register decorators.
from reps.workers import single_call  # noqa: F401
```

- [ ] **Step 3: Verify registration**

Run: `.venv/bin/python -c "import reps.workers; from reps.workers.registry import known_impls; print(known_impls())"`
Expected: `['single_call']`

- [ ] **Step 4: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 5: Commit**

```bash
git add reps/workers/single_call.py reps/workers/__init__.py
git commit -m "$(cat <<'EOF'
feat(workers): SingleCallWorker

Ports the existing single-LLM-call code path into the Worker primitive.
Registers as impl='single_call'. applied_edit is the serialize_diff_blocks
output for diff-mode, the full rewrite verbatim for full-mode.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Shared tool implementations

**Files:**
- Create: `reps/workers/tools.py`
- Create: `reps/prompt_templates/system_message_tool_runner.txt`

- [ ] **Step 1: Create tool helpers**

```python
"""Shared tool implementations used by AnthropicToolRunnerWorker and
DSPyReActWorker. Each tool is a pair: (json_schema, async callable).

Tools take a `WorkerRequest` + `WorkerContext` closure for state. They return
strings (tool_result content) or structured dicts."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

from reps.workers.base import WorkerContext, WorkerRequest


ToolSchema = Dict[str, Any]
ToolImpl = Callable[[Dict[str, Any]], Any]  # async-or-sync returning str/dict


def submit_child_schema() -> ToolSchema:
    return {
        "name": "submit_child",
        "description": (
            "Submit the final child program. Call exactly once to end the "
            "iteration. `code` must be a complete program in the target language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Full program source."},
                "changes_description": {
                    "type": "string",
                    "description": "One-to-three-sentence summary of changes vs parent.",
                },
            },
            "required": ["code", "changes_description"],
        },
    }


def edit_file_schema() -> ToolSchema:
    return {
        "name": "edit_file",
        "description": (
            "Apply a SEARCH/REPLACE edit to the in-flight child program. The "
            "`search` substring must appear exactly once in the current child "
            "code; it will be replaced with `replace`. Multiple edit_file calls "
            "may be made before submit_child."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {"type": "string"},
                "replace": {"type": "string"},
            },
            "required": ["search", "replace"],
        },
    }


def view_parent_schema() -> ToolSchema:
    return {
        "name": "view_parent",
        "description": "Return the parent program's current source code.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


def view_program_schema() -> ToolSchema:
    return {
        "name": "view_program",
        "description": "Return the source of an inspiration or top program by id.",
        "input_schema": {
            "type": "object",
            "properties": {"program_id": {"type": "string"}},
            "required": ["program_id"],
        },
    }


def run_tests_schema() -> ToolSchema:
    return {
        "name": "run_tests",
        "description": (
            "Evaluate candidate code in an isolated scratch workspace. Returns "
            "metrics + a truncated summary. Intermediate artifacts are NOT "
            "persisted to the final child program's record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    }


def build_tool_impls(
    request: WorkerRequest,
    ctx: WorkerContext,
    tool_names: List[str],
    edit_accumulator: List[Tuple[str, str]],
    child_code_holder: List[str],          # single-element list used as mutable ref
) -> Dict[str, ToolImpl]:
    """Build a dict of {tool_name: async-callable} for the requested tool_names.

    `edit_accumulator` is the list that edit_file appends to; the worker reads
    it after submit_child to produce applied_edit.
    `child_code_holder[0]` is the current in-flight child code (starts as parent);
    edit_file mutates it, view_parent reads it.
    """
    lookup = {request.parent.id: request.parent.code}
    for p in request.inspirations:
        lookup[p.id] = p.code
    for p in request.top_programs:
        lookup[p.id] = p.code
    if request.second_parent is not None:
        lookup[request.second_parent.id] = request.second_parent.code

    async def view_parent(_args):
        return child_code_holder[0]

    async def view_program(args):
        pid = args.get("program_id", "")
        return lookup.get(pid, f"ERROR: program_id '{pid}' not available")

    async def edit_file(args):
        search = args["search"]
        replace = args["replace"]
        current = child_code_holder[0]
        if current.count(search) != 1:
            return (
                f"ERROR: search string must appear exactly once; found "
                f"{current.count(search)} occurrences."
            )
        child_code_holder[0] = current.replace(search, replace, 1)
        edit_accumulator.append((search, replace))
        return f"edit applied ({len(edit_accumulator)} total); new length={len(child_code_holder[0])}"

    async def run_tests(args):
        if ctx.evaluator is None:
            return "ERROR: run_tests not available (uses_evaluator=False)"
        scratch_id = ctx.scratch_id_factory()
        outcome = await ctx.evaluator.evaluate_isolated(
            args["code"], program_id=scratch_id, scratch=True
        )
        # Redact: return metrics + truncated artifact summary.
        art_summary = ""
        if outcome.artifacts:
            art_summary = json.dumps({k: str(v)[:200] for k, v in outcome.artifacts.items()})
            if len(art_summary) > 2000:
                art_summary = art_summary[:2000] + "...[truncated]"
        return json.dumps({
            "metrics": outcome.metrics,
            "artifacts_summary": art_summary,
        })

    async def submit_child(_args):
        # Terminal marker; worker detects submit_child via tool_name and handles
        # the code + changes_description extraction itself (avoids coupling here).
        return "OK"

    table = {
        "view_parent": view_parent,
        "view_program": view_program,
        "edit_file": edit_file,
        "run_tests": run_tests,
        "submit_child": submit_child,
    }
    return {n: table[n] for n in tool_names if n in table}


def build_tool_schemas(
    ctx: WorkerContext,
    tool_names: List[str],
) -> List[ToolSchema]:
    schemas = []
    builders = {
        "view_parent": view_parent_schema,
        "view_program": view_program_schema,
        "edit_file": edit_file_schema,
        "run_tests": run_tests_schema,
        "submit_child": submit_child_schema,
    }
    for name in tool_names:
        if name == "run_tests" and ctx.evaluator is None:
            continue
        if name in builders:
            schemas.append(builders[name]())
    return schemas
```

- [ ] **Step 2: Create system prompt template**

```bash
cat > reps/prompt_templates/system_message_tool_runner.txt <<'EOF'
You are an evolutionary program-improvement agent.

Your task: produce a CHILD program that improves on a PARENT program.

Tools available (exact names exposed to you as tool calls):
- view_parent: read the parent program's current code.
- view_program: read an inspiration/top program by id.
- edit_file: apply a SEARCH/REPLACE edit to the in-flight child.
- run_tests: (if available) evaluate a candidate in isolation; returns metrics.
- submit_child: submit the final child program and end the turn loop.

Rules:
1. Call submit_child exactly once, when you are confident in your final child program.
2. If you use edit_file, each `search` string must appear exactly once in the current child.
3. If submit_child raises a SyntaxError, you'll receive a tool_result error — you may keep iterating within the turn budget.
4. Keep edits focused on improving the metrics the evaluator reports.
EOF
```

- [ ] **Step 3: Import-check**

Run: `.venv/bin/python -c "from reps.workers.tools import build_tool_schemas, build_tool_impls, submit_child_schema; print(submit_child_schema()['name'])"`
Expected: `submit_child`

- [ ] **Step 4: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 5: Commit**

```bash
git add reps/workers/tools.py reps/prompt_templates/system_message_tool_runner.txt
git commit -m "$(cat <<'EOF'
feat(workers): shared tool implementations and tool-runner prompt

submit_child (terminal), edit_file (accumulates SEARCH/REPLACE),
view_parent, view_program, run_tests (opt-in; uses evaluate_isolated).
Edit-file maintains a child-code holder so subsequent edits see prior edits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: `AnthropicToolRunnerWorker`

**Files:**
- Create: `reps/workers/anthropic_tool_runner.py`
- Modify: `reps/workers/__init__.py` to import the module
- Modify: `pyproject.toml` — bump `anthropic>=0.60.0`

- [ ] **Step 1: Bump SDK floor**

Edit `pyproject.toml`. Find the `anthropic` dependency line. Change the version constraint:

```toml
"anthropic>=0.60.0",
```

Run: `uv pip install -e . 2>&1 | tail -5`
Expected: anthropic upgraded to >=0.60.0 successfully.

- [ ] **Step 2: Create worker**

```python
"""AnthropicToolRunnerWorker — native Anthropic tool-use loop.

Bypasses LLMInterface because we need raw content blocks (thinking, tool_use,
tool_result) and native semantics. Reuses one AsyncAnthropic client per worker
instance for connection pooling across iterations."""
from __future__ import annotations

import asyncio
import difflib
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from anthropic import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from reps.llm.anthropic import REASONING_MODEL_PATTERNS  # re-export from existing file
from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    Worker,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers.edit_serializer import serialize_diff_blocks
from reps.workers.registry import register
from reps.workers.tools import build_tool_impls, build_tool_schemas

logger = logging.getLogger(__name__)


@register("anthropic_tool_runner")
class AnthropicToolRunnerWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        api_key = config.impl_options.get("api_key")
        timeout = float(config.impl_options.get("timeout", 300.0))
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.retries = int(config.impl_options.get("retries", 3))
        self.retry_base_delay = float(config.impl_options.get("retry_base_delay", 1.0))
        self.thinking_budget = int(config.impl_options.get("thinking_budget", 8000))
        self.max_tokens = int(config.impl_options.get("max_tokens", 16000))

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "AnthropicToolRunnerWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        model = self.config.model_id or request.model_id or ""
        is_reasoning = any(p in model.lower() for p in REASONING_MODEL_PATTERNS)

        # Build the initial user message via PromptSampler using the tool-runner template.
        system_prompt, user_prompt = self._build_initial_prompt(request, ctx)

        tool_schemas = build_tool_schemas(ctx, self.config.tools)

        # In-flight child code starts as parent; edit_file mutates it.
        child_code_holder = [request.parent.code]
        edit_accumulator: List[Tuple[str, str]] = []
        tool_impls = build_tool_impls(request, ctx, self.config.tools, edit_accumulator, child_code_holder)

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        turns: List[TurnRecord] = [
            TurnRecord(
                index=0,
                role="user",
                blocks=[ContentBlock(type="text", text=user_prompt)],
                worker_type="anthropic_tool_runner",
                started_at_ms=0,
            )
        ]
        usage_total: Dict[str, int] = {}

        submitted_code: Optional[str] = None
        submitted_desc: Optional[str] = None

        for turn_idx in range(self.config.max_turns):
            try:
                response = await self._call_with_retry(
                    model=model,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tool_schemas,
                    is_reasoning=is_reasoning,
                )
            except WorkerError as we:
                return self._fail(we, turns, usage_total, t0)

            self._accumulate_usage(usage_total, response.usage)

            assistant_blocks_for_turn: List[ContentBlock] = []
            assistant_blocks_for_message: List[Dict[str, Any]] = []
            for raw in response.content:
                block_dict = self._raw_block_to_message_dict(raw)
                assistant_blocks_for_message.append(block_dict)
                assistant_blocks_for_turn.append(self._raw_block_to_content_block(raw))

            turns.append(TurnRecord(
                index=len(turns),
                role="assistant",
                blocks=assistant_blocks_for_turn,
                model_id=model,
                usage=self._snapshot_usage(response.usage),
                stop_reason=response.stop_reason,
                worker_type="anthropic_tool_runner",
            ))

            if response.stop_reason == "refusal":
                return self._fail(
                    WorkerError(kind="REFUSED", detail="model refused"),
                    turns, usage_total, t0,
                )

            if response.stop_reason != "tool_use":
                return self._fail(
                    WorkerError(kind="PARSE_ERROR", detail=f"terminal stop_reason={response.stop_reason} without submit_child"),
                    turns, usage_total, t0,
                )

            # Echo assistant turn verbatim (thinking blocks + signatures preserved).
            messages.append({"role": "assistant", "content": assistant_blocks_for_message})

            # Dispatch every tool_use block in this turn.
            tool_result_blocks: List[Dict[str, Any]] = []
            tool_result_turn_blocks: List[ContentBlock] = []
            terminated = False
            for raw in response.content:
                if getattr(raw, "type", None) != "tool_use":
                    continue
                name = raw.name
                tid = raw.id
                args = raw.input or {}

                if name == "submit_child":
                    code = args.get("code", "")
                    desc = args.get("changes_description", "")
                    try:
                        compile(code, "<submitted>", "exec")
                    except SyntaxError as se:
                        err_text = f"SyntaxError: {se}"
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": err_text,
                            "is_error": True,
                        })
                        tool_result_turn_blocks.append(ContentBlock(
                            type="tool_result",
                            tool_use_id=tid,
                            tool_result_for_id=tid,
                            tool_result_content=err_text,
                            tool_result_is_error=True,
                        ))
                        continue
                    submitted_code = code
                    submitted_desc = desc
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "accepted",
                        "is_error": False,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=tid,
                        tool_result_for_id=tid,
                        tool_result_content="accepted",
                        tool_result_is_error=False,
                    ))
                    terminated = True
                else:
                    impl = tool_impls.get(name)
                    if impl is None:
                        out = f"ERROR: unknown tool '{name}'"
                        is_err = True
                    else:
                        try:
                            out = await impl(args)
                            is_err = False
                        except Exception as e:
                            out = f"{type(e).__name__}: {e}"
                            is_err = True
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": out if isinstance(out, str) else str(out),
                        "is_error": is_err,
                    })
                    tool_result_turn_blocks.append(ContentBlock(
                        type="tool_result",
                        tool_use_id=tid,
                        tool_result_for_id=tid,
                        tool_result_content=out if isinstance(out, str) else str(out),
                        tool_result_is_error=is_err,
                    ))

            turns.append(TurnRecord(
                index=len(turns),
                role="tool",
                blocks=tool_result_turn_blocks,
                worker_type="anthropic_tool_runner",
            ))
            messages.append({"role": "user", "content": tool_result_blocks})

            if terminated and submitted_code is not None:
                # Compute applied_edit per D1 hybrid rule.
                if edit_accumulator:
                    applied = serialize_diff_blocks(edit_accumulator)
                else:
                    applied = self._compute_applied_edit(submitted_code, request)
                return WorkerResult(
                    child_code=submitted_code,
                    changes_description=submitted_desc,
                    changes_summary=(submitted_desc or "")[:120],
                    applied_edit=applied,
                    turns=turns,
                    turn_count=len(turns),
                    usage=usage_total,
                    wall_clock_seconds=time.monotonic() - t0,
                    error=None,
                )

        return self._fail(
            WorkerError(kind="MAX_TURNS_HIT", detail=f"max_turns={self.config.max_turns}"),
            turns, usage_total, t0,
        )

    # -------------------------------------------- helpers
    def _build_initial_prompt(self, request: WorkerRequest, ctx: WorkerContext) -> Tuple[str, str]:
        template_key = self.config.system_prompt_template or "system_message_tool_runner"
        sampler = ctx.prompt_sampler
        # Use PromptSampler's existing build_prompt for user-message body, then
        # override the system message with the tool-runner template.
        prompt = sampler.build_prompt(
            current_program=request.parent.code,
            parent_program=request.parent.code,
            program_metrics=request.parent.metrics,
            previous_programs=[p.to_dict() for p in request.top_programs],
            top_programs=[p.to_dict() for p in (request.top_programs + request.inspirations)],
            inspirations=[p.to_dict() for p in request.inspirations],
            language=request.language,
            evolution_round=request.iteration,
            diff_based_evolution=False,   # tool runner handles its own edit mechanics
            program_artifacts=None,
            feature_dimensions=request.feature_dimensions,
            current_changes_description=None,
            **request.prompt_extras,
        )
        # Try to load the tool-runner template; fall back to stock system message.
        try:
            system_text = sampler.load_template(template_key)
        except Exception:
            system_text = prompt.get("system", "")
        return system_text, prompt.get("user", "")

    async def _call_with_retry(self, *, model, system_prompt, messages, tools, is_reasoning):
        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
        }
        if is_reasoning:
            params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        else:
            if self.config.temperature is not None:
                params["temperature"] = self.config.temperature

        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                return await self.client.messages.create(**params)
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_exc = e
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
            except APIStatusError as e:
                if 500 <= e.status_code < 600 and attempt < self.retries:
                    last_exc = e
                    await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                    continue
                raise WorkerError(kind="INTERNAL", detail=f"{e.status_code}: {e.message}") from e
        kind = "TIMEOUT" if isinstance(last_exc, APITimeoutError) else "INTERNAL"
        raise WorkerError(kind=kind, detail=repr(last_exc))

    def _raw_block_to_message_dict(self, raw) -> Dict[str, Any]:
        """Canonicalize a response content block into the dict shape Anthropic
        expects echoed back. Preserves thinking signatures verbatim."""
        t = getattr(raw, "type", None)
        if t == "text":
            return {"type": "text", "text": raw.text}
        if t == "thinking":
            d = {"type": "thinking", "thinking": raw.thinking}
            sig = getattr(raw, "signature", None)
            if sig is not None:
                d["signature"] = sig
            return d
        if t == "redacted_thinking":
            return {"type": "redacted_thinking", "data": raw.data}
        if t == "tool_use":
            return {"type": "tool_use", "id": raw.id, "name": raw.name, "input": raw.input}
        # fallback: best-effort dict
        if hasattr(raw, "model_dump"):
            return raw.model_dump()
        return {"type": t or "unknown"}

    def _raw_block_to_content_block(self, raw) -> ContentBlock:
        t = getattr(raw, "type", None)
        if t == "text":
            return ContentBlock(type="text", text=raw.text)
        if t == "thinking":
            return ContentBlock(
                type="thinking",
                text=getattr(raw, "thinking", None),
                signature=getattr(raw, "signature", None),
            )
        if t == "redacted_thinking":
            return ContentBlock(type="redacted_thinking", data=getattr(raw, "data", None))
        if t == "tool_use":
            return ContentBlock(
                type="tool_use",
                tool_use_id=raw.id,
                tool_name=raw.name,
                tool_input=dict(raw.input) if raw.input else {},
            )
        return ContentBlock(type="text", text=str(raw))

    def _accumulate_usage(self, total: Dict[str, int], usage) -> None:
        if usage is None:
            return
        for f in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            v = getattr(usage, f, 0) or 0
            total[f] = total.get(f, 0) + int(v)

    def _snapshot_usage(self, usage) -> Optional[Dict[str, int]]:
        if usage is None:
            return None
        return {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }

    def _compute_applied_edit(self, code: str, request: WorkerRequest) -> str:
        if self.config.generation_mode == "full":
            return code
        parent = request.parent.code.splitlines(keepends=True)
        child = code.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(
            parent, child,
            fromfile=f"parent/{request.parent.id}",
            tofile="child/new",
            n=3,
        ))
        return diff or "# no textual change"

    def _fail(self, we: WorkerError, turns, usage_total, t0) -> WorkerResult:
        return WorkerResult(
            child_code="",
            turns=turns,
            turn_count=len(turns),
            usage=usage_total,
            wall_clock_seconds=time.monotonic() - t0,
            error=we,
        )
```

- [ ] **Step 3: Export `REASONING_MODEL_PATTERNS` from `reps/llm/anthropic.py`**

`REASONING_MODEL_PATTERNS` is currently an attribute of `AnthropicLLM` (`reps/llm/anthropic.py:26`). Hoist it to module scope:

Edit `reps/llm/anthropic.py` — move the tuple from the class body to right above the class:

```python
# Reasoning models reject the temperature param.
REASONING_MODEL_PATTERNS = ("opus-4-7", "opus-4-8", "opus-4-9", "opus-5")


class AnthropicLLM(LLMInterface):
    # ... (remove REASONING_MODEL_PATTERNS from here)
```

Update any in-class reference from `self.REASONING_MODEL_PATTERNS` to `REASONING_MODEL_PATTERNS`.

- [ ] **Step 4: Register**

Append to `reps/workers/__init__.py`:

```python
from reps.workers import anthropic_tool_runner  # noqa: F401
```

- [ ] **Step 5: Verify registration**

Run: `.venv/bin/python -c "import reps.workers; from reps.workers.registry import known_impls; print(known_impls())"`
Expected: `['anthropic_tool_runner', 'single_call']`

- [ ] **Step 6: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 7: Commit**

```bash
git add reps/workers/anthropic_tool_runner.py reps/workers/__init__.py reps/llm/anthropic.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(workers): AnthropicToolRunnerWorker

Native Anthropic tool-use loop. Hybrid tool set: edit_file + submit_child
+ view_parent + view_program + run_tests (opt-in). Preserves signed
thinking blocks verbatim across turns. AsyncAnthropic client reused per
worker instance for connection pooling.

Bumps anthropic>=0.60.0 floor for native thinking param + AsyncAnthropic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: `DSPyReActWorker`

**Files:**
- Create: `reps/workers/dspy_react.py`
- Modify: `reps/workers/__init__.py`
- Modify: `pyproject.toml` — add `dspy>=3.1.3`

- [ ] **Step 1: Add dspy dep**

Edit `pyproject.toml`, find dependencies, add:

```toml
"dspy>=3.1.3",
```

Run: `uv pip install -e . 2>&1 | tail -10`

Expected: dspy installed (will pull in litellm as a transitive dep). May take 30s.

- [ ] **Step 2: Create worker**

```python
"""DSPyReActWorker — DSPy ReAct program over the native Anthropic provider.

DSPy is sync; we invoke it via asyncio.to_thread(...). Uses dspy.context(lm=)
for thread-safe LM scoping (NOT dspy.configure — that's process-global and
races under concurrent workers).

IMPORTANT: cache=False on dspy.LM. Default True would collapse evolutionary
diversity by returning stale completions."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import dspy

from reps.workers.base import (
    ContentBlock,
    TurnRecord,
    Worker,
    WorkerConfig,
    WorkerContext,
    WorkerError,
    WorkerRequest,
    WorkerResult,
)
from reps.workers.registry import register

logger = logging.getLogger(__name__)


def make_dspy_lm(config, worker_config: WorkerConfig) -> dspy.LM:
    """Build a per-invocation dspy.LM via the native Anthropic LiteLLM route."""
    kwargs: Dict[str, Any] = {
        "model": f"anthropic/{worker_config.model_id}",
        "api_key": config.llm.api_key,
        "max_tokens": config.llm.max_tokens,
        "cache": False,
    }
    if worker_config.temperature is not None:
        kwargs["temperature"] = worker_config.temperature
    thinking = worker_config.impl_options.get("thinking")
    if thinking:
        kwargs["thinking"] = thinking
    return dspy.LM(**kwargs)


class EvolveProgramFull(dspy.Signature):
    """Given a parent program and evolutionary context, produce an improved child program."""
    parent_code: str = dspy.InputField(desc="current parent program source")
    language: str = dspy.InputField()
    iteration: int = dspy.InputField()
    inspirations: str = dspy.InputField(desc="formatted inspirations block")
    top_programs: str = dspy.InputField(desc="formatted top-K programs block")
    second_parent_code: str = dspy.InputField(desc="optional crossover parent, or ''")
    feature_dimensions: str = dspy.InputField(desc="comma-separated MAP-Elites dims")
    extras: str = dspy.InputField(desc="reflection/SOTA/dead-end warnings")
    child_code: str = dspy.OutputField(desc="complete rewritten child program")
    changes_description: str = dspy.OutputField(desc="1-3 sentence summary of edits")


class EvolveProgramDiff(EvolveProgramFull):
    """Emit a unified diff against parent_code."""
    child_code: str = dspy.OutputField(desc="unified diff against parent_code")


def _fmt_programs(programs) -> str:
    lines = []
    for p in programs:
        score = p.metrics.get("combined_score", 0.0) if p.metrics else 0.0
        lines.append(f"--- id={p.id} score={score:.4f}\n{p.code}\n")
    return "\n".join(lines)


def _fmt_extras(extras: Dict[str, str]) -> str:
    parts = []
    for k in ("reflection", "sota_injection", "dead_end_warnings"):
        v = extras.get(k, "")
        if v:
            parts.append(f"[{k}]\n{v}")
    return "\n\n".join(parts)


@register("dspy_react")
class DSPyReActWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    @classmethod
    def from_config(cls, config: WorkerConfig) -> "DSPyReActWorker":
        return cls(config)

    async def run(self, request: WorkerRequest, ctx: WorkerContext) -> WorkerResult:
        t0 = time.monotonic()
        lm = ctx.dspy_lm_factory(self.config)

        sig = EvolveProgramDiff if request.generation_mode == "diff" else EvolveProgramFull

        # Build tools as dspy.Tool wrappers. Bridge async ctx.evaluator via
        # asyncio.run on a captured main loop (DSPy is sync inside to_thread).
        loop = asyncio.get_running_loop()

        def make_view_parent():
            def view_parent() -> str:
                """Return the parent program's source code."""
                return request.parent.code
            return dspy.Tool(view_parent)

        def make_run_tests():
            if ctx.evaluator is None:
                return None
            def run_tests(code: str) -> str:
                """Evaluate candidate code in isolation; returns metrics JSON string."""
                scratch_id = ctx.scratch_id_factory()
                fut = asyncio.run_coroutine_threadsafe(
                    ctx.evaluator.evaluate_isolated(code, program_id=scratch_id, scratch=True),
                    loop,
                )
                outcome = fut.result(timeout=self.config.expected_wall_clock_s + 60)
                return f"score={outcome.metrics.get('combined_score', 0.0):.4f} metrics={outcome.metrics}"
            return dspy.Tool(run_tests)

        tools = []
        if "view_parent" in self.config.tools:
            tools.append(make_view_parent())
        if "run_tests" in self.config.tools:
            rt = make_run_tests()
            if rt is not None:
                tools.append(rt)

        react = dspy.ReAct(signature=sig, tools=tools, max_iters=self.config.max_turns)

        def _invoke() -> Optional[dspy.Prediction]:
            with dspy.context(lm=lm):
                return react(
                    parent_code=request.parent.code,
                    language=request.language,
                    iteration=request.iteration,
                    inspirations=_fmt_programs(request.inspirations),
                    top_programs=_fmt_programs(request.top_programs),
                    second_parent_code=(request.second_parent.code if request.second_parent else ""),
                    feature_dimensions=",".join(request.feature_dimensions),
                    extras=_fmt_extras(request.prompt_extras),
                )

        error: Optional[WorkerError] = None
        pred: Optional[dspy.Prediction] = None
        try:
            pred = await asyncio.to_thread(_invoke)
        except asyncio.TimeoutError:
            error = WorkerError(kind="TIMEOUT", detail="asyncio timeout")
        except Exception as e:
            msg = str(e).lower()
            if "parse" in msg or "adapter" in msg:
                error = WorkerError(kind="PARSE_ERROR", detail=str(e))
            elif "tool" in msg:
                error = WorkerError(kind="TOOL_ERROR", detail=str(e))
            else:
                error = WorkerError(kind="INTERNAL", detail=str(e))

        turns, turn_count = _extract_turns(pred)
        if pred is not None and turn_count >= self.config.max_turns and not _finished(pred):
            error = WorkerError(kind="MAX_TURNS_HIT", detail=f"max_iters={self.config.max_turns}")

        child_code = getattr(pred, "child_code", "") if pred is not None else ""
        changes_description = getattr(pred, "changes_description", None) if pred is not None else None
        usage = _extract_usage(lm)

        return WorkerResult(
            child_code=child_code,
            changes_description=changes_description,
            changes_summary=(changes_description or "")[:120],
            applied_edit=child_code if request.generation_mode == "full" else (child_code or ""),
            turns=turns,
            turn_count=turn_count,
            usage=usage,
            wall_clock_seconds=time.monotonic() - t0,
            error=error,
        )


def _extract_turns(pred: Optional["dspy.Prediction"]) -> tuple[List[TurnRecord], int]:
    if pred is None or not hasattr(pred, "trajectory"):
        return [], 0
    traj: Dict[str, Any] = pred.trajectory or {}
    turns: List[TurnRecord] = []
    i = 0
    while f"thought_{i}" in traj:
        blocks: List[ContentBlock] = [
            ContentBlock(type="text", text=str(traj.get(f"thought_{i}", "")))
        ]
        tool_name = traj.get(f"tool_name_{i}")
        if tool_name:
            blocks.append(ContentBlock(
                type="tool_use",
                tool_use_id=f"dspy_{i}",
                tool_name=str(tool_name),
                tool_input=dict(traj.get(f"tool_args_{i}", {})) if isinstance(traj.get(f"tool_args_{i}"), dict) else {"args": traj.get(f"tool_args_{i}")},
            ))
        turns.append(TurnRecord(
            index=len(turns),
            role="assistant",
            blocks=blocks,
            worker_type="dspy_react",
            impl_specific={"dspy_trace": dict(traj)} if i == 0 else {},
        ))
        obs = traj.get(f"observation_{i}")
        if obs is not None:
            turns.append(TurnRecord(
                index=len(turns),
                role="tool",
                blocks=[ContentBlock(
                    type="tool_result",
                    tool_use_id=f"dspy_{i}",
                    tool_result_for_id=f"dspy_{i}",
                    tool_result_content=str(obs),
                    tool_result_is_error=False,
                )],
                worker_type="dspy_react",
            ))
        i += 1
    return turns, i


def _finished(pred) -> bool:
    traj = getattr(pred, "trajectory", {}) or {}
    return any(v == "finish" for k, v in traj.items() if isinstance(k, str) and k.startswith("tool_name_"))


def _extract_usage(lm) -> Dict[str, int]:
    tin = tout = 0
    for h in getattr(lm, "history", []) or []:
        u = h.get("usage") or {}
        tin += int(u.get("prompt_tokens", 0) or u.get("input_tokens", 0) or 0)
        tout += int(u.get("completion_tokens", 0) or u.get("output_tokens", 0) or 0)
    return {"input_tokens": tin, "output_tokens": tout, "calls": len(getattr(lm, "history", []) or [])}
```

- [ ] **Step 3: Register**

Append to `reps/workers/__init__.py`:

```python
from reps.workers import dspy_react  # noqa: F401
```

- [ ] **Step 4: Verify registration**

Run: `.venv/bin/python -c "import reps.workers; from reps.workers.registry import known_impls; print(known_impls())"`
Expected: `['anthropic_tool_runner', 'dspy_react', 'single_call']`

- [ ] **Step 5: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 6: Commit**

```bash
git add reps/workers/dspy_react.py reps/workers/__init__.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(workers): DSPyReActWorker

dspy.ReAct program over native Anthropic provider. Uses dspy.context(lm=)
for thread-safe scoping (NOT dspy.configure — process-global). Forces
cache=False on dspy.LM (default True would silently collapse diversity).
Fresh dspy.LM per invocation (avoids cross-run usage accounting).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: `WorkerPool` — sample over named WorkerConfig entries

**Files:**
- Modify: `reps/worker_pool.py`
- Modify: `reps/config.py` (add `types` list to `REPSWorkersConfig`)
- Create: `reps/workers/defaults.py`

- [ ] **Step 1: Update `REPSWorkersConfig`**

In `reps/config.py`, find `@dataclass class REPSWorkersConfig`. Add a `types: List[WorkerConfig]` field:

```python
from reps.workers.base import WorkerConfig  # at top of file

@dataclass
class REPSWorkersConfig:
    """F3: Worker Type Diversity config"""
    # Legacy fields — retained for backward compat.
    types_legacy: List[str] = field(default_factory=lambda: ["exploiter", "explorer", "crossover"])
    initial_allocation: Dict[str, float] = field(
        default_factory=lambda: {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15}
    )
    exploiter_temperature: float = 0.3
    explorer_temperature: float = 1.0

    # New — preferred surface.
    types: List[WorkerConfig] = field(default_factory=list)
```

(Rename the legacy list field to `types_legacy` to free up `types` for the new list.)

Find any reader of `REPSWorkersConfig.types` that was the legacy list-of-names (in `reps/worker_pool.py`) and update the reference to `types_legacy`.

- [ ] **Step 2: Create `reps/workers/defaults.py`**

```python
"""Default preset WorkerConfig entries used by the YAML shim when a legacy
config (no `reps.workers.types` list) is loaded."""
from __future__ import annotations

from reps.workers.base import WorkerConfig


def legacy_default_configs(
    model_id: str,
    exploiter_temperature: float = 0.3,
    explorer_temperature: float = 1.0,
    allocation: dict | None = None,
) -> list[WorkerConfig]:
    alloc = allocation or {"exploiter": 0.6, "explorer": 0.25, "crossover": 0.15}
    return [
        WorkerConfig(
            name="exploiter",
            impl="single_call",
            role="exploiter",
            model_id=model_id,
            temperature=exploiter_temperature,
            generation_mode="diff",
            weight=alloc.get("exploiter", 1.0),
        ),
        WorkerConfig(
            name="explorer",
            impl="single_call",
            role="explorer",
            model_id=model_id,
            temperature=explorer_temperature,
            generation_mode="full",
            weight=alloc.get("explorer", 1.0),
        ),
        WorkerConfig(
            name="crossover",
            impl="single_call",
            role="crossover",
            model_id=model_id,
            temperature=0.7,
            generation_mode="full",
            weight=alloc.get("crossover", 1.0),
        ),
    ]
```

- [ ] **Step 3: Refactor `WorkerPool`**

Replace the body of `reps/worker_pool.py` with a version that samples over `WorkerConfig` entries:

```python
"""WorkerPool: sample among named WorkerConfig entries weighted by cfg.weight.

Crossover is a role (cfg.role == "crossover"), not a name. Any config with
role=="crossover" participates in the second-parent sampling path."""
from __future__ import annotations

import logging
import random as _random
from collections import deque
from typing import Any, Dict, List, Optional

from reps.iteration_config import IterationConfig
from reps.workers.base import WorkerConfig
from reps.workers.defaults import legacy_default_configs

logger = logging.getLogger(__name__)

_rng = _random.Random()


class WorkerPool:
    def __init__(
        self,
        workers_config,
        *,
        default_model_id: str = "",
    ):
        types: List[WorkerConfig] = list(getattr(workers_config, "types", []) or [])
        if not types:
            # legacy shim — build three presets using default_model_id
            types = legacy_default_configs(
                default_model_id,
                exploiter_temperature=getattr(workers_config, "exploiter_temperature", 0.3),
                explorer_temperature=getattr(workers_config, "explorer_temperature", 1.0),
                allocation=getattr(workers_config, "initial_allocation", None),
            )
        self._configs: Dict[str, WorkerConfig] = {c.name: c for c in types}
        total = sum(max(c.weight, 0.0) for c in types) or 1.0
        self._weights = {c.name: max(c.weight, 0.0) / total for c in types}
        self.yield_tracker: Dict[str, deque] = {c.name: deque(maxlen=100) for c in types}
        self._force_explorer_batches = 0

        seed = getattr(workers_config, "random_seed", None)
        if seed is not None:
            _rng.seed(seed)

        logger.info(
            "WorkerPool initialized: %s",
            {n: round(w, 3) for n, w in self._weights.items()},
        )

    def get_worker_config(self, name: str) -> WorkerConfig:
        return self._configs[name]

    def build_iteration_config(
        self,
        database,
        prompt_extras: Dict[str, str],
        override_name: Optional[str] = None,
        target_island: Optional[int] = None,
    ) -> IterationConfig:
        name = override_name or self._sample()
        cfg = self._configs[name]

        second_parent_id = None
        if cfg.role == "crossover" and database is not None:
            second_parent_id = self._sample_distant_parent(database, target_island)

        return IterationConfig(
            parent_id=None,
            worker_name=cfg.name,
            model_id=cfg.model_id if cfg.owns_model else None,
            temperature=cfg.temperature if cfg.owns_temperature else None,
            prompt_extras=dict(prompt_extras),
            second_parent_id=second_parent_id,
            is_revisitation=False,
            generation_mode=cfg.generation_mode,
            target_island=target_island,
        )

    def record_result(self, name: str, improved: bool):
        if name in self.yield_tracker:
            self.yield_tracker[name].append(1.0 if improved else 0.0)

    def get_yield_rate(self, name: str) -> float:
        t = self.yield_tracker.get(name)
        if not t:
            return 0.0
        return sum(t) / max(1, len(t))

    def _sample(self) -> str:
        names = list(self._weights)
        weights = [self._weights[n] for n in names]
        if self._force_explorer_batches > 0:
            self._force_explorer_batches -= 1
            explorers = [n for n in names if self._configs[n].role == "explorer"]
            if explorers and _rng.random() < 0.5:
                return _rng.choice(explorers)
        return _rng.choices(names, weights=weights, k=1)[0]

    def _sample_distant_parent(self, database, target_island: Optional[int]):
        try:
            num_islands = len(database.islands)
            if num_islands <= 1:
                return None
            current = target_island if target_island is not None else database.current_island
            other = [i for i in range(num_islands) if i != current]
            if not other:
                return None
            pick_island = _rng.choice(other)
            pids = list(database.islands[pick_island])
            if not pids:
                return None
            return _rng.choice(pids)
        except Exception as e:
            logger.debug(f"distant parent sample failed: {e}")
            return None

    # --- SOTA controller adjustments ---
    def set_allocation(self, new_allocation: Dict[str, float]):
        total = sum(max(v, 0) for v in new_allocation.values()) or 1.0
        for name, weight in new_allocation.items():
            if name in self._weights:
                self._weights[name] = max(weight, 0) / total

    def boost_explorer(self, amount: float):
        explorers = [n for n in self._weights if self._configs[n].role == "explorer"]
        if not explorers:
            return
        share = amount / len(explorers)
        for n in explorers:
            self._weights[n] += share
        # renormalize
        total = sum(self._weights.values()) or 1.0
        for n in self._weights:
            self._weights[n] /= total

    def bump_temperatures(self, delta: float):
        for cfg in self._configs.values():
            if cfg.temperature is not None:
                cfg.temperature = min(2.0, cfg.temperature + delta)

    def force_explorer_majority(self, num_batches: int):
        self._force_explorer_batches = num_batches

    def force_model_switch(self):
        logger.info("Model switch requested by convergence monitor")

    def get_alternative_worker_name(self, original: str) -> str:
        names = [n for n in self._configs if n != original]
        return _rng.choice(names) if names else original
```

- [ ] **Step 4: Update controller to pass model_id for legacy shim**

In `reps/controller.py` where `WorkerPool(...)` is constructed (`start()` from Task 10), pass the default model:

```python
primary_model = self.config.llm.models[0].name if self.config.llm.models else ""
self.worker_pool = WorkerPool(self.config.reps.workers, default_model_id=primary_model)
```

Also update any use of `self.worker_pool.build_iteration_config(override_type=...)` to `override_name=...` (rename kwarg).

- [ ] **Step 5: Remove the temporary `get_worker_config` shim from Task 10 Step 9**

If you added the shim inside the controller, delete it now.

- [ ] **Step 6: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`. Any failures should be about renamed fields or kwargs — fix locally.

- [ ] **Step 7: Commit**

```bash
git add reps/worker_pool.py reps/workers/defaults.py reps/config.py reps/controller.py
git commit -m "$(cat <<'EOF'
refactor(worker_pool): sample over named WorkerConfig entries

- REPSWorkersConfig.types: List[WorkerConfig] replaces the legacy
  list-of-names. Legacy shape still supported via
  legacy_default_configs() shim.
- WorkerPool samples by weight; crossover triggered by
  cfg.role == "crossover".
- IterationConfig built with worker_name (already renamed in Task 8).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: `ContractSelector` axis ownership

**Files:**
- Modify: `reps/contract_selector.py`

- [ ] **Step 1: Update Contract and ContractSelector**

The existing `ContractSelector` bandits over `(model_id, temperature)`. Generalize arms to be keyed by `(worker_name, model_id, temperature)` where any of the latter two may be `None` if the worker owns that axis.

Modify `reps/contract_selector.py`:

```python
@dataclass
class Contract:
    worker_name: Optional[str]
    model_id: Optional[str]
    temperature: Optional[float]
```

Update `select(...)` to return a `Contract` with fields filled from the arm, and `update(...)` to accept `worker_name=None` plus `model_id`/`temperature` to locate the arm.

Key change in how the controller applies contract overrides (in `_reps_build_iteration_config` within `controller.py`):

```python
contract = self.contract_selector.select(context=...)
if contract is not None:
    worker_cfg = self.worker_pool.get_worker_config(iteration_config.worker_name)
    if contract.model_id and not worker_cfg.owns_model:
        iteration_config.model_id = contract.model_id
    if contract.temperature is not None and not worker_cfg.owns_temperature:
        iteration_config.temperature = contract.temperature
```

Keep the Beta(alpha, beta) posterior bookkeeping intact — arms are whatever tuple the YAML declares; the keys are just triples now.

- [ ] **Step 2: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 3: Commit**

```bash
git add reps/contract_selector.py reps/controller.py
git commit -m "$(cat <<'EOF'
refactor(contracts): arm keys = (worker_name?, model_id?, temperature?)

ContractSelector now respects WorkerConfig.owns_model / owns_temperature:
overrides are applied only to axes the worker does NOT own, removing
the double-count hazard flagged in review.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Trace sidecar persistence

**Files:**
- Modify: `reps/database.py` (extend `_save_program`)
- Modify: `reps/config.py` — add `DatabaseConfig.max_turns_persisted`, `EvolutionTraceConfig.include_turns`

- [ ] **Step 1: Extend configs**

In `reps/config.py`, `DatabaseConfig`:

```python
    max_turns_persisted: Optional[int] = None   # None = unlimited
```

In `reps/config.py`, `EvolutionTraceConfig`:

```python
    include_turns: bool = False
```

- [ ] **Step 2: Extend `Database._save_program`**

Grep for `_save_program` or the method that writes `programs/<id>.json`:

Run: `grep -n "programs/.*json\|_save_program\|def log_prompt" reps/database.py`

Wherever the program JSON is serialized, also look up `turns` from `program.metadata.get("turns")` (or from a parameter passed through). If `log_prompts` is True and turns are present, write a sidecar:

```python
def _save_program(self, program_dict: dict):
    # ... existing code ...
    pid = program_dict["id"]
    program_path = os.path.join(self.programs_dir, f"{pid}.json")
    # ... existing serialization ...

    # Sidecar: trace, if present and log_prompts enabled.
    turns = program_dict.get("metadata", {}).get("turns") or []
    if turns and self.config.log_prompts:
        turns = _truncate_turns(turns, self.config.max_turns_persisted)
        trace_path = os.path.join(self.programs_dir, f"{pid}.trace.json")
        with open(trace_path, "w") as f:
            json.dump({
                "schema_version": 1,
                "worker_type": program_dict.get("metadata", {}).get("reps_worker_name"),
                "turns": turns,
            }, f, default=str)


def _truncate_turns(turns: list, max_turns: Optional[int]) -> list:
    if max_turns is None or len(turns) <= max_turns:
        return turns
    half = max_turns // 2
    kept = list(turns[:half]) + list(turns[-half:])
    dropped = len(turns) - len(kept)
    marker = {
        "index": -1, "role": "system", "blocks": [{
            "type": "text",
            "text": f"[truncated {dropped} turns]",
        }],
        "impl_specific": {"truncated": True, "truncated_count": dropped},
        "schema_version": 1,
    }
    return list(turns[:half]) + [marker] + list(turns[-half:])
```

The controller already stuffs `turns` into `reps_meta["turns"]` (Task 10 Step 3). Promote that to `program_dict["metadata"]["turns"]` — choose one location and stick to it. Preferred: keep `turns` in `metadata` of the serialized child program dict so `_save_program` sees it.

Update `controller.py`'s child construction to include:
```python
child_metadata["turns"] = [asdict(t) for t in result.turns]
```

- [ ] **Step 3: Regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed`

- [ ] **Step 4: Commit**

```bash
git add reps/database.py reps/config.py reps/controller.py
git commit -m "$(cat <<'EOF'
feat(database): persist TurnRecord traces as programs/<id>.trace.json sidecars

Gated by DatabaseConfig.log_prompts. Optional max_turns_persisted cap
with first-N/2 + truncation-marker + last-N/2 structure. Signed thinking
blocks preserved verbatim through JSON (default=str for non-JSON blobs).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Verification YAML configs

**Files:**
- Create: `experiment/configs/verify_single_call.yaml`
- Create: `experiment/configs/verify_anthropic_tool_runner.yaml`
- Create: `experiment/configs/verify_dspy_react.yaml`

- [ ] **Step 1: Copy an existing config as a baseline reference**

Run: `ls experiment/configs/ | head -20`

Pick one recent circle_packing config, e.g., `cat experiment/configs/<some_circle_packing>.yaml` to see the base fields used (api keys, database, evaluator, etc.).

- [ ] **Step 2: Create `verify_single_call.yaml`**

```yaml
# experiment/configs/verify_single_call.yaml
# Verification Run A: single-LLM-call worker (baseline for the new primitive).
max_iterations: 40
checkpoint_interval: 20
log_level: INFO
random_seed: 42
language: python
file_suffix: .py
provider: anthropic
harness: reps

llm:
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.7
  max_tokens: 8000
  timeout: 300
  retries: 3
  retry_delay: 5
  models:
    - name: claude-sonnet-4-5-20250929
      provider: anthropic
      weight: 1.0

prompt:
  template_dir: null
  num_top_programs: 3
  num_diverse_programs: 2

database:
  population_size: 200
  archive_size: 50
  num_islands: 3
  log_prompts: true
  random_seed: 42

evaluator:
  timeout: 300
  max_retries: 2
  parallel_evaluations: 3
  max_concurrent_iterations: 3
  cascade_evaluation: true

diff_based_evolution: true

reps:
  enabled: true
  batch_size: 8
  workers:
    types:
      - name: sc_exploiter
        impl: single_call
        role: exploiter
        model_id: claude-sonnet-4-5-20250929
        temperature: 0.3
        generation_mode: diff
        weight: 1.0
  reflection:
    enabled: true
    top_k: 3
    bottom_k: 2
  revisitation:
    enabled: true
  convergence:
    enabled: true
  contracts:
    enabled: false
  sota:
    enabled: false
  annotations:
    enabled: true
```

- [ ] **Step 3: Create `verify_anthropic_tool_runner.yaml`**

```yaml
# experiment/configs/verify_anthropic_tool_runner.yaml
# Verification Run B: native Anthropic tool-use worker with extended thinking.
max_iterations: 40
checkpoint_interval: 20
log_level: INFO
random_seed: 42
language: python
file_suffix: .py
provider: anthropic
harness: reps

llm:
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.7
  max_tokens: 16000
  timeout: 300
  retries: 3
  retry_delay: 5
  models:
    - name: claude-opus-4-7-20250514
      provider: anthropic
      weight: 1.0

prompt:
  num_top_programs: 3
  num_diverse_programs: 2

database:
  population_size: 200
  archive_size: 50
  num_islands: 3
  log_prompts: true
  random_seed: 42
  max_turns_persisted: 24

evaluator:
  timeout: 600
  max_retries: 2
  parallel_evaluations: 2
  max_concurrent_iterations: 2
  cascade_evaluation: true

diff_based_evolution: true

reps:
  enabled: true
  batch_size: 6
  workers:
    types:
      - name: atr_explorer
        impl: anthropic_tool_runner
        role: explorer
        model_id: claude-opus-4-7-20250514
        generation_mode: full
        tools: [view_parent, view_program, edit_file, run_tests, submit_child]
        max_turns: 10
        uses_evaluator: true
        expected_wall_clock_s: 180
        impl_options:
          thinking_budget: 8000
          retries: 3
          max_tokens: 16000
        weight: 1.0
  reflection:
    enabled: true
    top_k: 3
    bottom_k: 2
  revisitation:
    enabled: true
  convergence:
    enabled: true
  contracts:
    enabled: false
  annotations:
    enabled: true
```

- [ ] **Step 4: Create `verify_dspy_react.yaml`**

```yaml
# experiment/configs/verify_dspy_react.yaml
# Verification Run C: DSPy ReAct worker with cache=False.
max_iterations: 40
checkpoint_interval: 20
log_level: INFO
random_seed: 42
language: python
file_suffix: .py
provider: anthropic
harness: reps

llm:
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.6
  max_tokens: 8000
  timeout: 300
  retries: 3
  retry_delay: 5
  models:
    - name: claude-sonnet-4-5-20250929
      provider: anthropic
      weight: 1.0

prompt:
  num_top_programs: 3
  num_diverse_programs: 2

database:
  population_size: 200
  archive_size: 50
  num_islands: 3
  log_prompts: true
  random_seed: 42
  max_turns_persisted: 20

evaluator:
  timeout: 600
  max_retries: 2
  parallel_evaluations: 2
  max_concurrent_iterations: 2
  cascade_evaluation: true

diff_based_evolution: false

reps:
  enabled: true
  batch_size: 6
  workers:
    types:
      - name: dspy_exploiter
        impl: dspy_react
        role: exploiter
        model_id: claude-sonnet-4-5-20250929
        temperature: 0.6
        generation_mode: full
        tools: [view_parent, run_tests]
        max_turns: 6
        uses_evaluator: true
        expected_wall_clock_s: 120
        weight: 1.0
  reflection:
    enabled: true
    top_k: 3
    bottom_k: 2
  revisitation:
    enabled: true
  convergence:
    enabled: true
  contracts:
    enabled: false
  annotations:
    enabled: true
```

- [ ] **Step 5: Commit**

```bash
git add experiment/configs/verify_single_call.yaml experiment/configs/verify_anthropic_tool_runner.yaml experiment/configs/verify_dspy_react.yaml
git commit -m "$(cat <<'EOF'
chore(configs): verification YAMLs for the three worker impls

One file per verification run. Each declares exactly one WorkerConfig
entry to isolate that impl's behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Verification Run A — `SingleCallWorker`

- [ ] **Step 1: Confirm API key and worktree venv**

Run: `.env | head && ls -la .env`

Source the .env for the run:

Run: `set -a && . ./.env && set +a && env | grep ANTHROPIC_API_KEY | head`

Expected: `ANTHROPIC_API_KEY=...` shows (first few chars only if env is printed).

- [ ] **Step 2: Run**

```bash
set -a && . ./.env && set +a
.venv/bin/reps-run \
  experiment/benchmarks/circle_packing/initial_program.py \
  experiment/benchmarks/circle_packing/evaluator.py \
  --config experiment/configs/verify_single_call.yaml \
  --output /tmp/reps_verify_single_call \
  --iterations 40 2>&1 | tee /tmp/reps_verify_single_call.log
```

Expect the run to complete in ~3-8 minutes depending on API latency.

- [ ] **Step 3: Check for tracebacks**

Run: `grep -i "traceback\|error" /tmp/reps_verify_single_call.log | head -20`
Expected: no tracebacks (transient "warning" / "info" level messages are fine).

- [ ] **Step 4: Verify improvement over baseline**

Run:
```bash
.venv/bin/python -c "
import json, os, glob
output='/tmp/reps_verify_single_call'
best_by_iter = {}
for p in glob.glob(os.path.join(output, 'programs', '*.json')):
    d = json.load(open(p))
    if d.get('id', '').startswith('scratch'):
        continue
    m = d.get('metrics', {}).get('combined_score', 0.0)
    best_by_iter[d.get('iteration_found', -1)] = max(best_by_iter.get(d.get('iteration_found', -1), 0.0), m)

# Seed is iteration_found == 0 or the initial program
seed_paths = sorted(glob.glob(os.path.join(output, 'programs', '*.json')))
best_score = max((json.load(open(p)).get('metrics', {}).get('combined_score', 0.0)) for p in seed_paths)
# Compare to the seed eval log: for a rough check, load the seed program:
from reps.runner import _load_initial_program   # may not exist; fall back to evaluating manually
print('best_combined_score=', best_score)
"
```

Alternatively, read the last controller log line that prints "best score":

Run: `grep -E "best.*score|best_score" /tmp/reps_verify_single_call.log | tail`

Manual check: best score must exceed the seed's `combined_score` by ≥5% relative.

- [ ] **Step 5: Decide pass/fail**

If pass: proceed to next task.
If fail due to a bug: diagnose (check `/tmp/reps_verify_single_call.log` for tracebacks or worker errors), fix, commit the fix, and re-run.

- [ ] **Step 6: If config tweaked, commit**

```bash
git add experiment/configs/verify_single_call.yaml
git commit -m "chore(configs): tune verify_single_call.yaml after live run"
```

---

## Task 20: Verification Run B — `AnthropicToolRunnerWorker`

- [ ] **Step 1: Run**

```bash
set -a && . ./.env && set +a
.venv/bin/reps-run \
  experiment/benchmarks/circle_packing/initial_program.py \
  experiment/benchmarks/circle_packing/evaluator.py \
  --config experiment/configs/verify_anthropic_tool_runner.yaml \
  --output /tmp/reps_verify_anthropic \
  --iterations 40 2>&1 | tee /tmp/reps_verify_anthropic.log
```

Expect 10-25 minutes (tool loops are slower).

- [ ] **Step 2: Check for tracebacks**

Run: `grep -i "traceback\|error" /tmp/reps_verify_anthropic.log | head -30`
Expected: no tracebacks. Tool-result "ERROR:" messages inside the model conversation are fine (they're fed back to the model).

- [ ] **Step 3: Verify thinking blocks captured**

Run:
```bash
.venv/bin/python -c "
import json, os, glob
found = 0
for p in glob.glob(os.path.join('/tmp/reps_verify_anthropic/programs', '*.trace.json')):
    d = json.load(open(p))
    for t in d.get('turns', []):
        for b in t.get('blocks', []):
            if b.get('type') == 'thinking' and b.get('signature'):
                found += 1
                break
print(f'programs_with_signed_thinking={found}')
"
```

Expected: at least 1 (most turns for Opus 4.7 will have thinking blocks).

- [ ] **Step 4: Verify tool_use blocks**

```bash
.venv/bin/python -c "
import json, glob, os
tool_use_count = 0
for p in glob.glob(os.path.join('/tmp/reps_verify_anthropic/programs', '*.trace.json')):
    d = json.load(open(p))
    for t in d.get('turns', []):
        for b in t.get('blocks', []):
            if b.get('type') == 'tool_use':
                tool_use_count += 1
print(f'total_tool_use_blocks={tool_use_count}')
"
```

Expected: several (20+) across all iterations.

- [ ] **Step 5: Verify improvement**

Same approach as Task 19 Step 4. Best score must exceed seed by ≥5% relative.

- [ ] **Step 6: Decide pass/fail; commit any config tweaks**

---

## Task 21: Verification Run C — `DSPyReActWorker`

- [ ] **Step 1: Run**

```bash
set -a && . ./.env && set +a
.venv/bin/reps-run \
  experiment/benchmarks/circle_packing/initial_program.py \
  experiment/benchmarks/circle_packing/evaluator.py \
  --config experiment/configs/verify_dspy_react.yaml \
  --output /tmp/reps_verify_dspy \
  --iterations 40 2>&1 | tee /tmp/reps_verify_dspy.log
```

Expect 8-20 minutes.

- [ ] **Step 2: Check for tracebacks**

Run: `grep -i "traceback\|error" /tmp/reps_verify_dspy.log | head -30`
Expected: no tracebacks. DSPy parse errors on a few iterations are acceptable (they bubble up as `WorkerError(PARSE_ERROR)` and count as failed iterations, not as harness crashes).

- [ ] **Step 3: Verify `dspy_trace` captured**

Run:
```bash
.venv/bin/python -c "
import json, glob
with_trace = 0
for p in glob.glob('/tmp/reps_verify_dspy/programs/*.trace.json'):
    d = json.load(open(p))
    for t in d.get('turns', []):
        if (t.get('impl_specific') or {}).get('dspy_trace'):
            with_trace += 1
            break
print(f'programs_with_dspy_trace={with_trace}')
"
```

Expected: at least 1.

- [ ] **Step 4: Verify tool_use steps**

Run:
```bash
.venv/bin/python -c "
import json, glob
n = 0
for p in glob.glob('/tmp/reps_verify_dspy/programs/*.trace.json'):
    d = json.load(open(p))
    for t in d.get('turns', []):
        for b in t.get('blocks', []):
            if b.get('type') == 'tool_use':
                n += 1
print(f'tool_use_blocks_total={n}')
"
```

Expected: several.

- [ ] **Step 5: Verify improvement**

Same approach as above. Best score ≥5% above seed.

- [ ] **Step 6: Decide pass/fail; commit any tweaks**

---

## Task 22: Final regression pass and PR

- [ ] **Step 1: Full regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `113 passed` (or `115+` if additional small tests landed).

- [ ] **Step 2: Confirm all three verification runs passed**

Re-read notes from Tasks 19–21. If all three passed, the feature is verified.

- [ ] **Step 3: Optional — open PR**

Only if the user explicitly asks for a PR. Otherwise leave the branch and let the user review locally.

```bash
# Only if user requests:
git push -u origin feature/workers-primitive
gh pr create --title "Tool-calling Worker primitive + asyncio controller" --body "$(cat <<'EOF'
## Summary
- New `reps/workers/` package with `Worker` protocol and three impls (SingleCall, AnthropicToolRunner, DSPyReAct).
- Controller migrated to asyncio-only (drops ProcessPoolExecutor).
- Evaluator isolated via `evaluate_isolated` + contextvars (fixes real REPS_PROGRAM_ID race under concurrent evals).
- Full trace capture including Anthropic extended-thinking blocks preserved verbatim.

## Verification
All three live verification runs against circle_packing passed (see docs/superpowers/specs/2026-04-21-tool-calling-worker-primitive-design.md §12).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Rollback plan

If the migration breaks production on main:
1. `git revert` the asyncio controller commit — this restores the multiprocess path if the full chain is reverted in order.
2. Alternatively, branch `feature/workers-primitive` can be abandoned without touching main — the worktree is isolated; main still has the pre-migration `ProcessPoolExecutor` controller.

## Spec traceback

| Spec section | Covered by tasks |
|---|---|
| §5.1 Data shapes | Task 1 |
| §5.2 Registry | Task 2 |
| §6 Asyncio controller | Tasks 9, 10 |
| §7 Evaluator isolation | Tasks 6, 7 |
| §8.1 SingleCallWorker | Task 11 |
| §8.2 AnthropicToolRunnerWorker | Tasks 12, 13 |
| §8.3 DSPyReActWorker | Task 14 |
| §9.1 WorkerPool | Task 15 |
| §9.2 ContractSelector | Task 16 |
| §9.3 Controller shrinks | Task 10 |
| §9.4 Convergence monitor | No code change (consumes `applied_edit` from Task 11 onwards) |
| §10 Trace persistence | Task 17 |
| §11 YAML surface | Task 18 |
| §12 Verification plan | Tasks 19, 20, 21 |
| §13 File-level change list | Tasks 1-18 |
| §14 Implementation ordering | This document |
