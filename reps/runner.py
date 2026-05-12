"""REPS Experiment Runner.

Entry point for running experiments with either the REPS harness or vanilla OpenEvolve.

Usage (preferred — everything in the YAML):
    uv run reps-run --config ./path/to/config.yaml

Optional CLI overrides (take precedence over YAML):
    --initial-program <path>
    --evaluator <path>
    --output <dir>
    --iterations N

Automatically loads a sibling `.env` file (via a small built-in walker) so
env-var references like `${ANTHROPIC_API_KEY}` in the YAML resolve without
manual sourcing.

Output directories are auto-versioned: <output>/run_001, run_002, etc.
"""
import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from reps.config import Config, load_config


def load_experiment_config(config_path: str) -> Config:
    """Load experiment config from YAML."""
    return Config.from_yaml(config_path)


def _next_run_dir(base: str) -> str:
    """Create a versioned run directory: base/run_001, run_002, etc."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    run_nums = []
    for entry in base_path.iterdir():
        match = re.fullmatch(r"run_(\d+)", entry.name)
        if match and entry.is_dir():
            run_nums.append(int(match.group(1)))

    next_num = max(run_nums, default=0) + 1
    while True:
        run_dir = base_path / f"run_{next_num:03d}"
        try:
            run_dir.mkdir()
            return str(run_dir)
        except FileExistsError:
            next_num += 1


async def run_reps(config: Config, initial_program: str, evaluator: str, output_dir: str):
    """Run experiment with the REPS harness."""
    from reps.controller import ProcessParallelController
    from reps.database import Program, ProgramDatabase
    from reps.llm.ensemble import LLMEnsemble

    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(level=level)

    # Also write logs to <output_dir>/run.log so runs are grep-able after the
    # fact and can be tail -f'd live from another terminal.
    log_path = Path(output_dir) / "run.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(file_handler)
    try:
        # Line-buffer so `tail -f` picks up output immediately.
        file_handler.stream.reconfigure(line_buffering=True)

        logger = logging.getLogger(__name__)
        logger.info(f"Log file: {log_path}")

        # Task-specific system prompt lives with the benchmark, not the config:
        # auto-load <evaluator_dir>/system_prompt.md when the config hasn't set
        # prompt.system_message to something custom. An explicit override in the
        # config still wins.
        task_prompt_path = Path(evaluator).parent / "system_prompt.md"
        if (
            config.prompt.system_message in ("system_message", None, "")
            and task_prompt_path.exists()
        ):
            config.prompt.system_message = task_prompt_path.read_text()
            logger.info(f"Loaded task system prompt from {task_prompt_path}")

        if config.provider == "anthropic":
            for model in (*config.llm.models, *config.llm.evaluator_models):
                if getattr(model, "provider", None) is None:
                    model.provider = "anthropic"

        # The per-iteration summarizer has its own dedicated client
        # (independent of the worker ensemble), so inherit unset provider
        # fields from the run config.
        summarizer_cfg = getattr(config.reps, "summarizer", None)
        if summarizer_cfg is not None and config.provider:
            if summarizer_cfg.provider is None:
                summarizer_cfg.provider = config.provider
            if summarizer_cfg.api_base is None:
                summarizer_cfg.api_base = config.llm.api_base
            if summarizer_cfg.api_key is None:
                summarizer_cfg.api_key = config.llm.api_key
            if (
                config.provider == "openrouter"
                and summarizer_cfg.model_id == "claude-opus-4-7"
                and config.llm.primary_model
            ):
                summarizer_cfg.model_id = config.llm.primary_model

        # Top-level `reasoning:` propagates to every model unless that model
        # explicitly set its own `reasoning_effort`. This lets configs say
        # `reasoning: high` once instead of repeating the effort on each model.
        # Accepted levels track Anthropic's effort scale (low|medium|high|xhigh|max).
        # `xhigh` and `max` are Claude-Opus-4.7-only; other providers/models may
        # reject them — the non-retryable classifier surfaces such 400s cleanly.
        if config.reasoning:
            effort = None if config.reasoning == "off" else config.reasoning
            allowed = {None, "low", "medium", "high", "xhigh", "max"}
            if effort not in allowed:
                raise ValueError(
                    f"reasoning must be one of: low, medium, high, xhigh, max, off — got {config.reasoning!r}"
                )
            for model in (*config.llm.models, *config.llm.evaluator_models):
                if model.reasoning_effort is None:
                    model.reasoning_effort = effort

        # Initialize database
        db = ProgramDatabase(config.database)

        # Load, evaluate, and add initial program
        initial_code = Path(initial_program).read_text()

        # Evaluate the seed program so it enters the database with real metrics.
        # Use evaluate_isolated so per_instance_scores and feedback (if the
        # benchmark emits them) reach the seed Program, not just evolved children.
        from reps.evaluator import Evaluator
        seed_evaluator = Evaluator(config.evaluator, evaluator)
        seed_outcome = await seed_evaluator.evaluate_isolated(initial_code, program_id="initial")
        db.metric_call_count += seed_evaluator.metric_call_count
        seed_metrics = seed_outcome.metrics
        if seed_outcome.artifacts:
            seed_evaluator._pending_artifacts.setdefault("initial", {}).update(seed_outcome.artifacts)
        logger.info(f"Seed program metrics: {seed_metrics}")
        if not seed_metrics or seed_metrics.get("combined_score", 0) == 0:
            logger.warning("Seed program scored 0 — check that it runs correctly with the evaluator")

        initial_prog = Program(
            id="initial",
            code=initial_code,
            language=config.language or "python",
            metrics=seed_metrics or {},
            per_instance_scores=seed_outcome.per_instance_scores,
            feedback=seed_outcome.feedback,
            iteration_found=0,
        )
        db.add(initial_prog)

        # Initialize controller
        controller = ProcessParallelController(
            config=config,
            evaluation_file=evaluator,
            database=db,
            output_dir=output_dir,
        )

        # Initialize reflection engine with LLM ensemble
        if config.reps.enabled:
            llm_ensemble = LLMEnsemble(config.llm.models)
            controller._reps_init_reflection_engine(llm_ensemble)

        # Run evolution
        best = None
        controller.start()
        try:
            best = await controller.run_evolution(
                start_iteration=1,
                max_iterations=config.max_iterations,
            )
            if controller.evaluator is not None:
                db.metric_call_count += controller.evaluator.metric_call_count
            if best:
                logger.info(f"Best program: {best.id}, metrics: {best.metrics}")
        finally:
            controller.stop()
            # Save all programs, prompts, and responses to disk
            db.save(output_dir)
            logger.info(f"Database saved to {output_dir}")

            # Write per-run health snapshot (annotation success rate +
            # convergence-action recovery). Emits WARNINGs at run end if either
            # signal is degraded — see MetricsLogger thresholds.
            if getattr(controller, "_reps_metrics", None) is not None:
                try:
                    controller._reps_metrics.write_health()
                except Exception as e:
                    logger.warning(f"Failed to write health.json: {e}")

            # Save best program code and visualization
            if best and best.code:
                best_code_path = Path(output_dir) / "best_program.py"
                best_code_path.write_text(best.code)
                logger.info(f"Best program saved to {best_code_path}")

                # Try to visualize via the benchmark's optional visualize.py
                try:
                    import importlib.util
                    viz_module_path = Path(evaluator).parent / "visualize.py"
                    if viz_module_path.exists():
                        spec = importlib.util.spec_from_file_location("viz", str(viz_module_path))
                        viz = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(viz)
                        viz.visualize_from_program(str(best_code_path),
                                                   save_path=str(Path(output_dir) / "packing.png"))
                except ImportError as e:
                    logger.info(
                        f"Skipping visualization ({e.name} not installed; "
                        f"`uv pip install matplotlib` to enable)."
                    )
                except Exception as e:
                    logger.warning(f"Could not generate visualization: {e}")
    finally:
        root.removeHandler(file_handler)
        file_handler.close()


def run_openevolve(config_path: str, initial_program: str, evaluator: str, output_dir: str, iterations: int):
    """Run experiment with vanilla OpenEvolve (pip-installed package)."""
    cmd = [
        sys.executable, "-m", "openevolve.cli",
        initial_program, evaluator,
        "--config", config_path,
        "--output", output_dir,
        "--iterations", str(iterations),
    ]
    subprocess.run(cmd, check=True)


def _load_dotenv(*start_dirs: Path) -> Optional[Path]:
    """Walk up from start dirs looking for .env; export non-overriding vars.

    Saves users from having to run `set -a && source .env && set +a`. Shell-exported
    vars always win over .env values (we don't overwrite os.environ).
    """
    starts = start_dirs or (Path.cwd(),)
    seen: set[Path] = set()
    first_loaded: Optional[Path] = None
    for start in starts:
        d = Path(start).resolve()
        for _ in range(6):
            if d in seen:
                break
            seen.add(d)
            p = d / ".env"
            if p.exists():
                for raw in p.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                if first_loaded is None:
                    first_loaded = p
            if d.parent == d:
                break
            d = d.parent
    return first_loaded


def _apply_overrides(config, overrides):
    """Apply `-o dotted.path=value` overrides to a nested dataclass config.

    Value is parsed as YAML so numbers, booleans, strings, lists all work
    without extra quoting: `-o reps.batch_size=10`, `-o llm.temperature=0.9`,
    `-o reps.workers.types='[exploiter, explorer]'`.
    """
    old_provider = getattr(config, "provider", None)
    old_llm_api_base = getattr(config.llm, "api_base", None)
    old_llm_api_key = getattr(config.llm, "api_key", None)
    provider_overridden = False
    explicit_provider_paths: set[str] = set()
    llm_shared_overrides: set[str] = set()

    for expr in overrides:
        if "=" not in expr:
            raise ValueError(f"--override must be key=value, got {expr!r}")
        path, _, raw = expr.partition("=")
        value = yaml.safe_load(raw)
        parts = path.strip().split(".")
        if not all(parts):
            raise ValueError(f"Unknown override path {path!r}")

        obj = config
        for p in parts[:-1]:
            obj = _get_override_child(obj, p, path)
        _set_override_value(obj, parts[-1], value, path)

        if parts[0] == "llm" and len(parts) == 2 and parts[1] in _llm_shared_fields(config):
            llm_shared_overrides.add(parts[1])
        if parts == ["provider"]:
            provider_overridden = True
        if parts[-1] == "provider":
            explicit_provider_paths.add(".".join(parts))

    if provider_overridden:
        _clear_inherited_provider_fields(
            config,
            old_provider,
            old_llm_api_base,
            old_llm_api_key,
            explicit_provider_paths,
        )

    config.finalize(
        llm_shared_overrides=llm_shared_overrides or None,
        overwrite_llm_shared=bool(llm_shared_overrides),
    )


def _clear_inherited_provider_fields(
    config: Config,
    old_provider: Optional[str],
    old_llm_api_base: Optional[str],
    old_llm_api_key: Optional[str],
    explicit_provider_paths: set[str],
) -> None:
    """Let top-level provider overrides restamp inherited provider fields."""
    for section_name in ("models", "evaluator_models"):
        for idx, model in enumerate(getattr(config.llm, section_name)):
            path = f"llm.{section_name}.{idx}.provider"
            if path not in explicit_provider_paths and model.provider == old_provider:
                model.provider = None

    summarizer_path = "reps.summarizer.provider"
    summarizer = getattr(config.reps, "summarizer", None)
    if (
        summarizer is not None
        and summarizer_path not in explicit_provider_paths
        and summarizer.provider == old_provider
    ):
        summarizer.provider = None
    if summarizer is not None:
        if summarizer.api_base == old_llm_api_base:
            summarizer.api_base = None
        if summarizer.api_key == old_llm_api_key:
            summarizer.api_key = None


def _get_override_child(obj: Any, part: str, full_path: str) -> Any:
    if isinstance(obj, list):
        try:
            return obj[int(part)]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Unknown override path {full_path!r}") from e
    if is_dataclass(obj):
        names = {field.name for field in fields(obj)}
        if part in names:
            return getattr(obj, part)
    if isinstance(obj, dict) and part in obj:
        return obj[part]
    raise ValueError(f"Unknown override path {full_path!r}")


def _set_override_value(obj: Any, part: str, value: Any, full_path: str) -> None:
    if isinstance(obj, list):
        try:
            obj[int(part)] = value
            return
        except (ValueError, IndexError) as e:
            raise ValueError(f"Unknown override path {full_path!r}") from e
    if is_dataclass(obj):
        names = {field.name for field in fields(obj)}
        if part in names:
            setattr(obj, part, value)
            return
    if isinstance(obj, dict) and part in obj:
        obj[part] = value
        return
    raise ValueError(f"Unknown override path {full_path!r}")


def _llm_shared_fields(config: Config) -> set[str]:
    model_fields = {field.name for field in fields(type(config.llm.models[0]))} if config.llm.models else set()
    return {field.name for field in fields(config.llm)} & model_fields


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{value!r} is not a positive integer") from e
    if parsed < 1:
        raise argparse.ArgumentTypeError("--iterations must be a positive integer")
    return parsed


def _resolve_run_inputs(
    config: Config,
    *,
    config_path: str,
    initial_program_arg: Optional[str],
    evaluator_arg: Optional[str],
    output_arg: Optional[str],
) -> tuple[str, str, str]:
    config_dir = Path(config_path).resolve().parent

    initial_program = (
        _resolve_cli_path(initial_program_arg)
        if initial_program_arg
        else _resolve_config_path(config.initial_program, config_dir)
    )
    evaluator = (
        _resolve_cli_path(evaluator_arg)
        if evaluator_arg
        else _resolve_config_path(config.evaluator_path, config_dir)
    )

    if (initial_program is None or evaluator is None) and config.task:
        task_dir = Path(config.task)
        if not task_dir.is_absolute():
            task_dir = (config_dir / task_dir).resolve()
        initial_program = initial_program or str((task_dir / "initial_program.py").resolve())
        evaluator = evaluator or str((task_dir / "evaluator.py").resolve())

    output = (
        _resolve_cli_path(output_arg)
        if output_arg
        else _resolve_config_path(config.output, config_dir)
    )
    if output is None:
        output = f"experiment/results/{Path(config_path).stem}"

    if not initial_program or not evaluator:
        raise ValueError(
            "Config must set `task: <benchmark_dir>` (recommended) or `initial_program:` + "
            "`evaluator_path:`. Alternatively pass them as positional args to reps-run."
        )

    return initial_program, evaluator, output


def _resolve_config_path(value: Optional[str], config_dir: Path) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (config_dir / path).resolve())


def _resolve_cli_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(Path(value).expanduser().resolve())


def main():
    parser = argparse.ArgumentParser(
        description="REPS Experiment Runner",
        epilog=(
            "Override any config field with -o / --override dotted.path=value.\n"
            "Examples: -o max_iterations=50  -o llm.temperature=0.9  -o reps.batch_size=10"
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "initial_program", nargs="?", default=None,
        help="(optional) Override config.task / config.initial_program",
    )
    parser.add_argument(
        "evaluator", nargs="?", default=None,
        help="(optional) Override config.task / config.evaluator_path",
    )
    parser.add_argument("--output", default=None,
                        help="Output base directory (defaults to experiment/results/<config-stem>/)")
    parser.add_argument("--iterations", type=_positive_int, default=None,
                        help="Shortcut for -o max_iterations=N")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config field (repeatable). Value parsed as YAML.",
    )
    args = parser.parse_args()

    _load_dotenv(Path(args.config).resolve().parent, Path.cwd())
    config = load_experiment_config(args.config)

    _apply_overrides(config, args.override)

    if args.iterations is not None:
        config.max_iterations = args.iterations

    try:
        initial_program, evaluator, output = _resolve_run_inputs(
            config,
            config_path=args.config,
            initial_program_arg=args.initial_program,
            evaluator_arg=args.evaluator,
            output_arg=args.output,
        )
    except ValueError as e:
        parser.error(str(e))

    # Auto-version the output directory
    run_dir = _next_run_dir(output)
    print(f"Output: {run_dir}")

    # Make run_dir visible to benchmark evaluators (for persisting arrays etc.)
    os.environ["REPS_RUN_DIR"] = run_dir

    if config.harness == "openevolve":
        run_openevolve(args.config, initial_program, evaluator, run_dir, config.max_iterations)
    else:
        asyncio.run(run_reps(config, initial_program, evaluator, run_dir))


if __name__ == "__main__":
    main()
