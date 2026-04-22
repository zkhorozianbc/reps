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
