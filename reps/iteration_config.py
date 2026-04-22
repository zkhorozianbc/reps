"""
IterationConfig and IterationResult dataclasses.

These are the bridge between the controller process (where REPS decisions live)
and the iteration worker process (stateless). Every REPS mechanism that affects
how an iteration runs encodes its decision into IterationConfig.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationConfig:
    """Serializable config passed from controller to iteration worker process.

    The controller builds one of these per iteration dispatch. The worker process
    receives it alongside the DB snapshot and executes whatever it says.
    """

    # Parent selection
    parent_id: Optional[str] = None  # If set, use this parent; if None, sample from DB snapshot

    # Worker type (F3)
    worker_name: str = "exploiter"  # "exploiter", "explorer", "crossover"

    # Backward-compat alias. Remove in a future release.
    @property
    def worker_type(self) -> str:
        return self.worker_name

    # Model/generation params (F5 intelligence contracts)
    model_id: Optional[str] = None  # Override model selection; None = use default
    temperature: Optional[float] = None  # LLM temperature; None = use model default

    # Prompt extras (F1 reflection, F6 SOTA, F8 dead-end warnings)
    prompt_extras: Dict[str, str] = field(default_factory=dict)

    # Crossover-specific (F3)
    second_parent_id: Optional[str] = None  # For crossover worker: ID of second parent

    # Metadata
    is_revisitation: bool = False  # F2: whether this is an ε-revisitation
    generation_mode: str = "diff"  # "diff" or "full" — Explorer uses "full", Exploiter uses "diff"

    # Island targeting
    target_island: Optional[int] = None


@dataclass
class IterationResult:
    """Returned from iteration worker process to controller.

    Extends SerializableResult with REPS-specific fields for the controller
    to use in reflection, convergence monitoring, and contract updating.
    """

    # Core results
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration: int = 0
    error: Optional[str] = None
    target_island: Optional[int] = None

    # Prompt/response for logging
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None

    # REPS fields
    diff: str = ""  # Raw LLM output (for convergence monitor edit entropy)
    worker_name: str = "exploiter"
    is_revisitation: bool = False
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    parent_score: float = 0.0
    child_score: float = 0.0
    improved: bool = False

    # Turn records for tool-calling workers (Task 8+)
    turns: List[Dict[str, Any]] = field(default_factory=list)

    # Compute signature fields
    tokens_in: int = 0
    tokens_out: int = 0
    wall_clock_seconds: float = 0.0
    iteration_time: float = 0.0

    # Backward-compat alias. Remove in a future release.
    @property
    def worker_type(self) -> str:
        return self.worker_name
