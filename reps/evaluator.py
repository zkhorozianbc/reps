"""
Evaluation system for REPS (extracted from OpenEvolve)
"""

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from reps.config import EvaluatorConfig
from reps.database import ProgramDatabase
from reps.evaluation_result import EvaluationResult
from reps.llm.ensemble import LLMEnsemble
from reps.async_utils import TaskPool, run_in_executor
from reps.prompt_sampler import PromptSampler
from reps.runtime import set_current_program_id, reset_current_program_id
from reps.utils import format_metrics_safe

logger = logging.getLogger(__name__)


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


class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
        suffix: Optional[str] = ".py",
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.program_suffix = suffix
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database

        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=config.parallel_evaluations)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        # Bounded concurrency for evaluate_isolated under asyncio (replaces TaskPool).
        max_parallel = max(1, int(getattr(config, "parallel_evaluations", 1) or 1))
        self._eval_semaphore = asyncio.Semaphore(max_parallel)

        logger.info(f"Initialized evaluator with {evaluation_file}")

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")

            # Validate cascade configuration
            self._validate_cascade_configuration(module)
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    def _validate_cascade_configuration(self, module) -> None:
        """
        Validate cascade evaluation configuration and warn about potential issues

        Args:
            module: The loaded evaluation module
        """
        if self.config.cascade_evaluation:
            # Check if cascade functions exist
            has_stage1 = hasattr(module, "evaluate_stage1")
            has_stage2 = hasattr(module, "evaluate_stage2")
            has_stage3 = hasattr(module, "evaluate_stage3")

            if not has_stage1:
                logger.warning(
                    f"Configuration has 'cascade_evaluation: true' but evaluator "
                    f"'{self.evaluation_file}' does not define 'evaluate_stage1' function. "
                    f"This will fall back to direct evaluation, making the cascade setting useless. "
                    f"Consider setting 'cascade_evaluation: false' or implementing cascade functions."
                )
            elif not (has_stage2 or has_stage3):
                logger.warning(
                    f"Evaluator '{self.evaluation_file}' defines 'evaluate_stage1' but no additional "
                    f"cascade stages (evaluate_stage2, evaluate_stage3). Consider implementing "
                    f"multi-stage evaluation for better cascade benefits."
                )
            else:
                logger.debug(
                    f"Cascade evaluation properly configured with available stage functions"
                )

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
    ) -> Dict[str, float]:
        """Legacy wrapper: evaluate and stuff artifacts into _pending_artifacts so
        get_pending_artifacts(program_id) keeps working for seed evals."""
        outcome = await self.evaluate_isolated(program_code, program_id=program_id)
        if outcome.artifacts:
            self._pending_artifacts.setdefault(program_id, {}).update(outcome.artifacts)
        return outcome.metrics

    async def evaluate_isolated(
        self,
        program_code: str,
        *,
        program_id: Optional[str] = None,
        scratch: bool = False,
        run_dir: Optional[str] = None,
    ) -> "EvaluationOutcome":
        """Run one isolated evaluation. Safe to call concurrently from asyncio Tasks.

        - `program_id`: if None, a scratch UUID is generated.
        - `scratch`: informational flag; caller guarantees this is a throwaway
          (e.g., from a tool-call); artifacts are returned but not stored globally.
        - `run_dir`: optional override for REPS_RUN_DIR passed to the subprocess env.
        """
        pid = program_id or f"scratch-{uuid.uuid4().hex[:12]}"
        async with self._eval_semaphore:
            token_artifacts = _call_artifacts.set({})
            token_pid = set_current_program_id(pid)
            try:
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

    async def _evaluate_code_with_env(
        self,
        program_code: str,
        program_id: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Run the full retry/cascade evaluation pipeline.

        This is the implementation behind evaluate_isolated — it carries `env`
        through to subprocess-spawning benchmark evaluators so each concurrent
        call gets the correct REPS_PROGRAM_ID without touching os.environ.
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as temp_file:
                temp_file.write(program_code.encode("utf-8"))
                temp_file_path = temp_file.name

            try:
                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(temp_file_path, env=env)
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(temp_file_path, env=env)

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)

                # Capture timeout artifacts into contextvar-scoped dict
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    ctx_artifacts = _call_artifacts.get()
                    if ctx_artifacts is not None:
                        ctx_artifacts.update(
                            {
                                "timeout": True,
                                "timeout_duration": self.config.timeout,
                                "failure_stage": "evaluation",
                                "error_type": "timeout",
                            }
                        )
                    else:
                        self._pending_artifacts.setdefault(program_id, {}).update(
                            {
                                "timeout": True,
                                "timeout_duration": self.config.timeout,
                                "failure_stage": "evaluation",
                                "error_type": "timeout",
                            }
                        )

                # Add LLM feedback if configured
                llm_eval_result = None
                if self.config.use_llm_feedback and self.llm_ensemble:
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    # Combine metrics
                    llm_scores = []
                    for name, value in llm_eval_result.metrics.items():
                        weighted_value = value * self.config.llm_feedback_weight
                        eval_result.metrics[f"llm_{name}"] = weighted_value
                        llm_scores.append(value)  # Use unweighted value for average

                    # Add average of LLM metrics
                    if llm_scores:
                        llm_average = sum(llm_scores) / len(llm_scores)
                        eval_result.metrics["llm_average"] = (
                            llm_average * self.config.llm_feedback_weight
                        )

                        # Recalculate combined_score if it exists
                        if "combined_score" in eval_result.metrics:
                            # Original combined_score is just accuracy
                            accuracy = eval_result.metrics["combined_score"]
                            # Combine with LLM average (70% accuracy, 30% LLM quality)
                            eval_result.metrics["combined_score"] = (
                                accuracy * 0.7 + llm_average * 0.3
                            )

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    ctx_artifacts = _call_artifacts.get()

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        if ctx_artifacts is not None:
                            ctx_artifacts.update(eval_result.artifacts)
                        else:
                            self._pending_artifacts.setdefault(program_id, {}).update(
                                eval_result.artifacts
                            )
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        if ctx_artifacts is not None:
                            ctx_artifacts.update(llm_eval_result.artifacts)
                        else:
                            self._pending_artifacts.setdefault(program_id, {}).update(
                                llm_eval_result.artifacts
                            )
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Return just metrics for backward compatibility
                return eval_result.metrics

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts into contextvar-scoped dict
                if artifacts_enabled and program_id:
                    timeout_data = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }
                    ctx_artifacts = _call_artifacts.get()
                    if ctx_artifacts is not None:
                        ctx_artifacts.update(timeout_data)
                    else:
                        self._pending_artifacts[program_id] = timeout_data

                return {"error": 0.0, "timeout": True}

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()

                # Capture failure artifacts
                if artifacts_enabled and program_id:
                    failure_data = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }
                    ctx_artifacts = _call_artifacts.get()
                    if ctx_artifacts is not None:
                        ctx_artifacts.update(failure_data)
                    else:
                        self._pending_artifacts[program_id] = failure_data

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # All retries failed
        logger.error(
            f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
        )
        return {"error": 0.0}

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    async def _direct_evaluate(
        self,
        program_path: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file
            env: Per-call environment dict to pass to the benchmark evaluate function
                 (if the function accepts an `env` kwarg). Older benchmarks without
                 an `env` parameter still work via the inspect.signature gate.

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """
        # Gate: only pass env if the benchmark evaluate() accepts it.
        sig = inspect.signature(self.evaluate_function)
        kwargs: Dict[str, Any] = {}
        if "env" in sig.parameters:
            kwargs["env"] = env

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.evaluate_function(program_path, **kwargs)
            )

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Return result as-is to be processed by _process_evaluation_result
        # This supports both dict and EvaluationResult returns, just like _cascade_evaluate
        return result

    async def _cascade_evaluate(
        self,
        program_path: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file
            env: Per-call environment dict to pass to stage functions that accept it.

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        # Import the evaluation module to get cascade functions if they exist
        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                return await self._direct_evaluate(program_path, env=env)

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(program_path, env=env)

            def _make_stage_kwargs(fn) -> Dict[str, Any]:
                """Return kwargs for a stage function, gating on env support."""
                sig = inspect.signature(fn)
                return {"env": env} if "env" in sig.parameters else {}

            # Run first stage with timeout
            try:
                stage1_kwargs = _make_stage_kwargs(module.evaluate_stage1)

                async def run_stage1():
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: module.evaluate_stage1(program_path, **stage1_kwargs),
                    )

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")
                # Capture stage 1 failure with enhanced context
                error_context = self._create_cascade_error_context("stage1", e)
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        **error_context,
                    },
                )

            # Check threshold
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:
                stage2_kwargs = _make_stage_kwargs(module.evaluate_stage2)

                async def run_stage2():
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: module.evaluate_stage2(program_path, **stage2_kwargs),
                    )

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:
                stage3_kwargs = _make_stage_kwargs(module.evaluate_stage3)

                async def run_stage3():
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: module.evaluate_stage3(program_path, **stage3_kwargs),
                    )

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")
            # Return proper cascade failure result with enhanced context
            error_context = self._create_cascade_error_context("cascade_setup", e)
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    **error_context,
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            feature_dimensions = self.database.config.feature_dimensions if self.database else []
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code,
                template_key="evaluation",
                feature_dimensions=feature_dimensions,
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re

                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            traceback.print_exc()
            return {}

    def _create_cascade_error_context(self, stage: str, error: Exception) -> dict:
        """
        Create rich error context for cascade failures

        Args:
            stage: The stage where the error occurred
            error: The exception that was raised

        Returns:
            Dictionary with enhanced error context
        """
        import time

        return {
            "failure_stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "cascade_config": self.config.cascade_evaluation,
            "cascade_thresholds": getattr(self.config, "cascade_thresholds", []),
            "timeout_config": self.config.timeout,
            "evaluation_file": self.evaluation_file,
        }

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Uses 'combined_score' if available (for consistency with evolution),
        otherwise falls back to averaging all numeric metrics except 'error'

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        # Use combined_score if available - this is what evolution uses
        if "combined_score" in metrics:
            score = metrics.get("combined_score")
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        # Fallback: average all numeric metrics except 'error'
        # This maintains backward compatibility
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
