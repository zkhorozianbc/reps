"""
Configuration handling for REPS (extracted from OpenEvolve)
"""

import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import dacite
import yaml

if TYPE_CHECKING:
    from reps.llm.base import LLMInterface

# Imported here (not under TYPE_CHECKING) so dacite can deserialize it from YAML.
# workers/base.py only imports Config under TYPE_CHECKING, so no circular import.
from reps.workers.base import WorkerConfig  # noqa: E402


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")  # ${VAR}
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_VALID_PROVIDERS = {"openrouter", "anthropic", "openai"}
_VALID_MODEL_PROVIDERS = _VALID_PROVIDERS | {"local"}
_PROVIDER_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_KNOWN_PROVIDER_ENV_VARS = set(_PROVIDER_ENV_VARS.values())
_VALID_HARNESSES = {"reps", "openevolve"}
_VALID_REASONING = {None, "low", "medium", "high", "xhigh", "max", "off"}
_VALID_SELECTION_STRATEGIES = {"map_elites", "pareto", "mixed"}
_VALID_DIVERSITY_METRICS = {"edit_distance", "feature_based"}
_VALID_WORKER_IMPLS = {
    "single_call",
    "anthropic_tool_runner",
    "openai_tool_runner",
    "dspy_react",
}
_VALID_WORKER_ROLES = {"exploiter", "explorer", "crossover"}
_VALID_GENERATION_MODES = {"diff", "full"}
_VALID_REVISITATION_DECAYS = {"linear", "exponential"}
_LLM_SHARED_MODEL_FIELDS = {
    "api_base",
    "api_key",
    "temperature",
    "top_p",
    "max_tokens",
    "timeout",
    "retries",
    "retry_delay",
    "random_seed",
    "reasoning_effort",
}


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    """
    Resolve ${VAR} environment variable reference in a string value.
    In current implementation pattern must match the entire string (e.g., "${OPENAI_API_KEY}"),
    not embedded within other text.

    Args:
        value: The string value that may contain ${VAR} syntax

    Returns:
        The resolved value with environment variable expanded, or original value if no match

    Raises:
        ValueError: If the environment variable is referenced but not set
    """
    if value is None:
        return None

    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return value

    var_name = match.group(1)
    env_value = os.environ.get(var_name)
    if env_value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return env_value


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None

    # Per-model provider override (e.g. "openrouter", "anthropic")
    provider: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to resolve ${VAR} env var references in api_key"""
        self.api_key = _resolve_env_var(self.api_key)


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    # Provider finalization fills this for provider-specific defaults such as
    # OpenRouter while leaving native OpenAI/Anthropic configs unset.
    api_base: str = None

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(default_factory=list)

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # YAML-facing shorthand: primary_model / secondary_model (+ weights)
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        super().__post_init__()  # Resolve ${VAR} in api_key at LLMConfig level

        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # Only validate if this looks like a user config (has some model info)
        # Don't validate during internal/default initialization
        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def sync_shared_model_params(self, keys: Optional[set[str]] = None, *, overwrite: bool = False) -> None:
        """Propagate selected shared LLM fields to concrete model entries."""
        selected = keys or _LLM_SHARED_MODEL_FIELDS
        shared_config = {key: getattr(self, key) for key in selected}
        self.update_model_params(shared_config, overwrite=overwrite)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Large-codebase mode: represent programs in prompts via compact changes descriptions
    programs_as_changes_description: bool = False
    system_message_changes_description: Optional[str] = None
    initial_changes_description: str = ""

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: Optional[int] = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: Optional[int] = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Diff summary formatting for "Previous Attempts" section
    diff_summary_max_line_len: int = 100  # Truncate lines longer than this
    diff_summary_max_lines: int = 30  # Max lines per SEARCH/REPLACE block

    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = (
        None  # Deprecated: use suggest_simplification_after_chars
    )


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    # Note: diversity_metric fixed to "edit_distance"
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # GEPA-style Pareto selection (Phase 2). When `selection_strategy` is
    # "mixed", a fraction `pareto_fraction` of parent picks comes from the
    # Pareto frontier of per_instance_scores; the rest use existing
    # MAP-Elites/exploration/exploitation logic. "pareto" = always Pareto;
    # "map_elites" = never Pareto (status quo). Inspirations are unchanged.
    selection_strategy: str = "map_elites"  # "map_elites" | "pareto" | "mixed"
    pareto_fraction: float = 0.0  # only used when selection_strategy == "mixed"
    pareto_instance_keys: Optional[List[str]] = None  # restrict frontier to these keys

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    # CRITICAL: For custom dimensions, evaluators must return RAW VALUES, not bin indices
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OpenEvolve handles all scaling and binning internally."
        },
    )
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    max_snapshot_artifacts: Optional[int] = (
        100  # Max artifacts in worker snapshots (None=unlimited)
    )

    novelty_llm: Optional["LLMInterface"] = None
    embedding_model: Optional[str] = None
    similarity_threshold: float = 0.99

    # Trace sidecar: TurnRecord persistence (Task 17)
    max_turns_persisted: Optional[int] = None  # None = unlimited


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    # Maximum concurrent in-flight iterations (bounds asyncio semaphore).
    # Defaults to parallel_evaluations when unset. Controls how many worker
    # iterations the async controller dispatches concurrently.
    max_concurrent_iterations: Optional[int] = None

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program

    # Forward-compat field for Phase 6 of the GEPA plan (per-iteration
    # minibatch sampling). Currently unused by the evaluator itself —
    # exposed so the v1 `reps.Optimizer(minibatch_size=...)` kwarg has a stable
    # landing slot until the runner consumes it.
    minibatch_size: Optional[int] = None


@dataclass
class REPSReflectionConfig:
    """F1: Reflection Engine config"""
    enabled: bool = True
    top_k: int = 3
    bottom_k: int = 2
    model: Optional[str] = None  # model to use for reflection; None = use default ensemble


@dataclass
class REPSTraceReflectionConfig:
    """Phase 3 (GEPA-inspired): per-candidate trace-grounded reflection.

    When enabled, parents with `feedback` (Phase 1.2 ASI) at least
    `min_feedback_length` chars and non-empty `per_instance_scores` get
    a one-shot LLM call to produce a `mutation_directive` that gets
    injected into the next mutation prompt for that lineage.

    Disabled by default — opt-in until live runs validate the prompt is
    pulling its weight (token spend vs score improvement).

    `lineage_depth` (Phase 5): when > 0, prepend a compact ancestral
    history of up to N programs (the parent's parent, grandparent, etc.)
    to the reflection prompt. Each line shows generation, score,
    changes_description, and the ancestor's prior directive (when set).
    0 disables (Phase 3 behavior — no lineage context).
    """
    enabled: bool = False
    model: Optional[str] = None  # None = use the worker LLM ensemble
    min_feedback_length: int = 20
    max_code_chars: int = 4000
    lineage_depth: int = 3


@dataclass
class REPSMergeConfig:
    """Phase 4 (GEPA-inspired): system-aware merge for crossover workers.

    When enabled, the second parent for a crossover iteration is chosen to
    maximize per-instance complementarity with the primary instead of the
    random distant-island pick used otherwise. Candidate pool is all
    programs on islands other than the primary's (same scope as the
    legacy random pick, just selected differently).

    Falls back to the legacy random pick when:
      - the primary has no per_instance_scores (pre-Phase-1.2 benchmarks),
      - the candidate pool from other islands is empty.

    `instance_keys`: optional restriction to a subset of per_instance_scores
    keys when computing complementary gain. None means use the union of
    keys present across primary + candidates.

    `strong_score_threshold`: scores at or above this value qualify a
    dimension as a parent's "strong" dim when rendering crossover_context.
    """
    enabled: bool = False
    instance_keys: Optional[List[str]] = None
    strong_score_threshold: float = 0.8


@dataclass
class REPSRevisitationConfig:
    """F2: epsilon-Revisitation config"""
    enabled: bool = True
    epsilon_start: float = 0.15
    epsilon_end: float = 0.05
    decay: str = "linear"
    recency_window: int = 50


@dataclass
class REPSWorkersConfig:
    """F3: Worker Type Diversity config.

    `types` is a list of explicit WorkerConfig entries declared in YAML under
    `reps.workers.types`. Empty means "no workers configured" — WorkerPool
    will raise on construction.
    """
    types: List[WorkerConfig] = field(default_factory=list)


@dataclass
class REPSConvergenceConfig:
    """F4: Convergence Monitor config"""
    enabled: bool = True
    window_size: int = 20
    entropy_threshold_mild: float = 0.5
    entropy_threshold_moderate: float = 0.3
    entropy_threshold_severe: float = 0.15


@dataclass
class REPSContractsConfig:
    """F5: Intelligence Contracts config"""
    enabled: bool = True
    models: List[str] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])


@dataclass
class REPSSOTAConfig:
    """F6: SOTA Steering config"""
    enabled: bool = False
    target_score: Optional[float] = None
    # Optional: name of the metric F6 should compare against target_score. If
    # unset, falls back to combined_score and then to a numeric-average of
    # metrics. Set this when your evaluator's primary objective lives in a
    # specific raw metric (e.g. "sum_radii") rather than combined_score.
    target_metric: Optional[str] = None


@dataclass
class REPSAnnotationsConfig:
    """F8: Enriched Annotations config"""
    enabled: bool = True
    dead_end_awareness: bool = True


@dataclass
class REPSSummarizerConfig:
    """Per-program summarizer (runs inline per iteration).

    The summarizer uses its OWN dedicated LLM client — independent of the
    worker ensemble. That's intentional: it lets a run configure a cheap
    summarizer (e.g. claude-sonnet-4-6) while workers run on a different
    provider (e.g. gpt-5.4-pro), without anyone having to add the
    summarizer model to the worker ensemble.

    `provider` and `api_key` are optional — if not set, they are derived
    from `model_id` at controller-startup time (claude-* → anthropic +
    ${ANTHROPIC_API_KEY}; gpt-* / o*-* → openai + ${OPENAI_API_KEY}).
    Set them explicitly for unusual setups (OpenRouter-hosted claude,
    custom api_base, etc.).

    General role + output rules live in the module's fixed system prompt.
    `task_instructions` is the benchmark-specific guidance appended to the
    summarizer's user message (before the trace data) — use it to correct
    common hallucinations or enforce domain-specific framing.
    """
    enabled: bool = True
    model_id: str = "claude-opus-4-7"
    task_instructions: Optional[str] = None
    # Optional overrides. When None, derived from `model_id` at init time.
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 300
    retries: int = 2
    retry_delay: int = 5

    def __post_init__(self):
        """Resolve ${VAR} in api_key early so bad env setup fails at config load."""
        self.api_key = _resolve_env_var(self.api_key)


@dataclass
class REPSConfig:
    """Master REPS configuration -- all features toggled and tuned here."""
    enabled: bool = False  # Master switch: set True to activate REPS features
    batch_size: int = 10   # REPS batch size (iterations per wave for controller-side modules)

    reflection: REPSReflectionConfig = field(default_factory=REPSReflectionConfig)
    revisitation: REPSRevisitationConfig = field(default_factory=REPSRevisitationConfig)
    workers: REPSWorkersConfig = field(default_factory=REPSWorkersConfig)
    convergence: REPSConvergenceConfig = field(default_factory=REPSConvergenceConfig)
    contracts: REPSContractsConfig = field(default_factory=REPSContractsConfig)
    sota: REPSSOTAConfig = field(default_factory=REPSSOTAConfig)
    annotations: REPSAnnotationsConfig = field(default_factory=REPSAnnotationsConfig)
    summarizer: REPSSummarizerConfig = field(default_factory=REPSSummarizerConfig)
    trace_reflection: REPSTraceReflectionConfig = field(default_factory=REPSTraceReflectionConfig)
    merge: REPSMergeConfig = field(default_factory=REPSMergeConfig)


@dataclass
class Config:
    """Master configuration for REPS / OpenEvolve"""

    # Experiment inputs — can be specified in YAML or overridden on CLI.
    # Named `..._path` to avoid collision with the `evaluator:` EvaluatorConfig section.
    initial_program: Optional[str] = None
    evaluator_path: Optional[str] = None
    output: Optional[str] = None  # base output dir; runs auto-version to <output>/run_NNN

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    file_suffix: str = ".py"

    # Top-level provider and harness selection
    provider: str = "openrouter"  # valid: "openrouter", "anthropic", "openai"
    harness: str = "reps"         # valid: "reps", "openevolve"

    # Top-level reasoning / extended-thinking toggle. Applied uniformly across
    # providers by the runner: translates to `reasoning_effort` for OpenAI and
    # OpenRouter-hosted models, and to native Anthropic `thinking` with a
    # mapped budget for `provider: anthropic`.
    reasoning: Optional[str] = None

    # Path to benchmark directory. When set, `reps-run` derives
    # `initial_program.py` and `evaluator.py` from this directory so the CLI
    # doesn't need positional args. Resolved relative to the config file.
    task: Optional[str] = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    reps: REPSConfig = field(default_factory=REPSConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    # 10k was tuned for tiny seed-style programs and silently rejects most
    # scipy.optimize-based packings (~11-12k). Production runs with capable
    # models legitimately produce 12-16k programs; reject only true outliers.
    max_code_length: int = 20000
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"

    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        config_path = Path(path).resolve()
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        config = cls.from_dict(config_dict)

        # Resolve template_dir relative to config file location
        if config.prompt.template_dir:
            template_path = Path(config.prompt.template_dir)
            if not template_path.is_absolute():
                config.prompt.template_dir = str((config_path.parent / template_path).resolve())

        for attr in ("task", "initial_program", "evaluator_path", "output"):
            value = getattr(config, attr)
            if value:
                path_value = Path(value)
                if not path_value.is_absolute():
                    setattr(config, attr, str((config_path.parent / path_value).resolve()))

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        config_dict = deepcopy(config_dict or {})
        _validate_provider_env_refs(config_dict)

        if "diff_pattern" in config_dict:
            try:
                re.compile(config_dict["diff_pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in diff_pattern: {e}")

        # Remove None values for temperature and top_p to avoid dacite type errors;
        # alternatively, pass check_types=False to dacite.from_dict, but that can hide other issues
        if "llm" in config_dict:
            if "temperature" in config_dict["llm"] and config_dict["llm"]["temperature"] is None:
                del config_dict["llm"]["temperature"]
            if "top_p" in config_dict["llm"] and config_dict["llm"]["top_p"] is None:
                del config_dict["llm"]["top_p"]

        try:
            config: Config = dacite.from_dict(
                data_class=cls,
                data=config_dict,
                config=dacite.Config(
                    cast=[List, Union],
                    forward_references={"LLMInterface": Any},
                    strict=True,
                ),
            )
        except dacite.exceptions.UnexpectedDataError as e:
            raise ValueError(f"unknown config field: {e}") from e
        except dacite.exceptions.DaciteError as e:
            raise ValueError(str(e)) from e

        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        config.finalize()

        if config.prompt.programs_as_changes_description and not config.diff_based_evolution:
            raise ValueError(
                "prompt.programs_as_changes_description=true requires diff_based_evolution=true "
                "(full rewrites cannot reliably update code and changes_description together)"
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def finalize(
        self,
        *,
        llm_shared_overrides: Optional[set[str]] = None,
        overwrite_llm_shared: bool = False,
    ) -> None:
        """Validate and complete derived config fields before use."""
        _validate_config_enums(self)
        _finalize_provider_config(self)
        if llm_shared_overrides:
            self.llm.sync_shared_model_params(llm_shared_overrides, overwrite=overwrite_llm_shared)
        else:
            self.llm.sync_shared_model_params()
        _stamp_and_validate_model_providers(self)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        # Defaults follow the package's top-level provider, currently OpenRouter.
        # OPENAI_API_BASE remains a custom OpenAI-compatible override.
        api_base = os.environ.get("OPENAI_API_BASE") or OPENROUTER_API_BASE

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})
        config.finalize()

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config


def _validate_provider_env_refs(config_dict: Dict[str, Any]) -> None:
    provider = config_dict.get("provider", Config.provider)
    llm = config_dict.get("llm") or {}
    if not isinstance(llm, dict):
        return

    _validate_api_key_env_ref(provider, llm.get("api_key"), "llm.api_key")

    for section in ("models", "evaluator_models"):
        for idx, model in enumerate(llm.get(section) or []):
            if not isinstance(model, dict):
                continue
            model_provider = model.get("provider") or provider
            _validate_api_key_env_ref(
                model_provider,
                model.get("api_key") or llm.get("api_key"),
                f"llm.{section}[{idx}].api_key",
            )

    reps = config_dict.get("reps") or {}
    summarizer = reps.get("summarizer") if isinstance(reps, dict) else None
    if isinstance(summarizer, dict):
        _validate_api_key_env_ref(
            summarizer.get("provider") or provider,
            summarizer.get("api_key"),
            "reps.summarizer.api_key",
        )


def _validate_api_key_env_ref(provider: str, value: Any, path: str) -> None:
    if not isinstance(value, str):
        return
    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return
    var_name = match.group(1)
    expected = _PROVIDER_ENV_VARS.get(provider)
    if expected and var_name in _KNOWN_PROVIDER_ENV_VARS and var_name != expected:
        raise ValueError(f"{path}: provider {provider!r} requires ${{{expected}}}, got ${{{var_name}}}")


def _validate_config_enums(config: Config) -> None:
    if config.provider not in _VALID_PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(_VALID_PROVIDERS)}, got {config.provider!r}")
    if config.harness not in _VALID_HARNESSES:
        raise ValueError(f"harness must be one of {sorted(_VALID_HARNESSES)}, got {config.harness!r}")
    if config.reasoning not in _VALID_REASONING:
        raise ValueError(f"reasoning must be one of {sorted(v for v in _VALID_REASONING if v is not None)}, got {config.reasoning!r}")
    if config.database.selection_strategy not in _VALID_SELECTION_STRATEGIES:
        raise ValueError(
            "database.selection_strategy must be one of "
            f"{sorted(_VALID_SELECTION_STRATEGIES)}, got {config.database.selection_strategy!r}"
        )
    if config.database.diversity_metric not in _VALID_DIVERSITY_METRICS:
        raise ValueError(
            "database.diversity_metric must be one of "
            f"{sorted(_VALID_DIVERSITY_METRICS)}, got {config.database.diversity_metric!r}"
        )
    if config.reps.revisitation.decay not in _VALID_REVISITATION_DECAYS:
        raise ValueError(
            "reps.revisitation.decay must be one of "
            f"{sorted(_VALID_REVISITATION_DECAYS)}, got {config.reps.revisitation.decay!r}"
        )
    for worker in config.reps.workers.types:
        if worker.impl not in _VALID_WORKER_IMPLS:
            raise ValueError(f"worker impl must be one of {sorted(_VALID_WORKER_IMPLS)}, got {worker.impl!r}")
        if worker.role not in _VALID_WORKER_ROLES:
            raise ValueError(f"worker role must be one of {sorted(_VALID_WORKER_ROLES)}, got {worker.role!r}")
        if worker.generation_mode not in _VALID_GENERATION_MODES:
            raise ValueError(
                "worker generation_mode must be one of "
                f"{sorted(_VALID_GENERATION_MODES)}, got {worker.generation_mode!r}"
            )


def _finalize_provider_config(config: Config) -> None:
    if config.provider == "openrouter" and config.llm.api_base is None:
        config.llm.api_base = OPENROUTER_API_BASE
    if config.llm.api_key is None:
        env_var = _PROVIDER_ENV_VARS.get(config.provider)
        if env_var:
            config.llm.api_key = os.environ.get(env_var)

    _validate_provider_base(config.provider, config.llm.api_base, "llm.api_base")

    summarizer = config.reps.summarizer
    if summarizer.provider is None:
        summarizer.provider = config.provider
    if summarizer.api_base is None:
        summarizer.api_base = config.llm.api_base
    if summarizer.api_key is None:
        summarizer.api_key = config.llm.api_key
    _validate_provider_base(summarizer.provider, summarizer.api_base, "reps.summarizer.api_base")


def _stamp_and_validate_model_providers(config: Config) -> None:
    for model in config.llm.models + config.llm.evaluator_models:
        if model.provider is None:
            model.provider = config.provider
        if model.provider not in _VALID_MODEL_PROVIDERS:
            raise ValueError(
                f"llm model provider must be one of {sorted(_VALID_MODEL_PROVIDERS)}, got {model.provider!r}"
            )
        if model.provider != "local":
            _validate_provider_base(model.provider, model.api_base, f"llm.models[{model.name}].api_base")


def _validate_provider_base(provider: Optional[str], api_base: Optional[str], path: str) -> None:
    if api_base is None:
        return

    base = api_base.lower()
    if provider == "openrouter" and "openrouter.ai" not in base:
        raise ValueError(f"{path} must use an openrouter.ai api_base when provider is 'openrouter'")
    if provider == "anthropic" and ("openrouter.ai" in base or "api.openai.com" in base):
        raise ValueError(f"{path} is inconsistent with provider 'anthropic'")
    if provider == "openai" and ("openrouter.ai" in base or "anthropic.com" in base):
        raise ValueError(f"{path} is inconsistent with provider 'openai'")
