"""REPS Experiment Runner.

Entry point for running experiments with either the REPS harness or vanilla OpenEvolve.

Usage:
    reps-run <initial_program> <evaluator> --config <config.yaml> [--output <dir>] [--iterations N]

Output directories are auto-versioned: <output>/run_001, run_002, etc.
"""
import argparse
import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from reps.config import Config, load_config


def load_experiment_config(config_path: str) -> Config:
    """Load experiment config from YAML."""
    return Config.from_yaml(config_path)


def _next_run_dir(base: str) -> str:
    """Create a versioned run directory: base/run_001, run_002, etc."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    existing = sorted(base_path.glob("run_*"))
    if existing:
        last_num = max(int(d.name.split("_")[1]) for d in existing if d.name.split("_")[1].isdigit())
        next_num = last_num + 1
    else:
        next_num = 1

    run_dir = base_path / f"run_{next_num:03d}"
    run_dir.mkdir()
    return str(run_dir)


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

    # Evaluate the seed program so it enters the database with real metrics
    from reps.evaluator import Evaluator
    seed_evaluator = Evaluator(config.evaluator, evaluator)
    seed_metrics = await seed_evaluator.evaluate_program(initial_code, "initial")
    logger.info(f"Seed program metrics: {seed_metrics}")
    if not seed_metrics or seed_metrics.get("combined_score", 0) == 0:
        logger.warning("Seed program scored 0 — check that it runs correctly with the evaluator")

    initial_prog = Program(
        id="initial",
        code=initial_code,
        language=config.language or "python",
        metrics=seed_metrics or {},
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
    controller.start()
    try:
        best = await controller.run_evolution(
            start_iteration=1,
            max_iterations=config.max_iterations,
        )
        if best:
            logger.info(f"Best program: {best.id}, metrics: {best.metrics}")
    finally:
        controller.stop()
        # Save all programs, prompts, and responses to disk
        db.save(output_dir)
        logger.info(f"Database saved to {output_dir}")

        # Save best program code and visualization
        if best and best.code:
            best_code_path = Path(output_dir) / "best_program.py"
            best_code_path.write_text(best.code)
            logger.info(f"Best program saved to {best_code_path}")

            # Try to visualize the best packing
            try:
                from experiment.benchmarks.circle_packing.visualize import visualize_from_program
                viz_path = str(Path(output_dir) / "packing.png")
                visualize_from_program(str(best_code_path), save_path=viz_path)
            except Exception:
                # Visualization is optional — don't fail the run
                try:
                    import importlib.util
                    viz_module_path = Path(evaluator).parent / "visualize.py"
                    if viz_module_path.exists():
                        spec = importlib.util.spec_from_file_location("viz", str(viz_module_path))
                        viz = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(viz)
                        viz.visualize_from_program(str(best_code_path),
                                                   save_path=str(Path(output_dir) / "packing.png"))
                except Exception as e:
                    logger.warning(f"Could not generate visualization: {e}")


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


def _load_dotenv() -> Optional[Path]:
    """Walk up from CWD looking for .env; export non-overriding vars into environ.

    Saves users from having to run `set -a && source .env && set +a`. Shell-exported
    vars always win over .env values (we don't overwrite os.environ).
    """
    d = Path.cwd().resolve()
    for _ in range(6):
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
            return p
        if d.parent == d:
            break
        d = d.parent
    return None


def _apply_overrides(config, overrides):
    """Apply `-o dotted.path=value` overrides to a nested dataclass config.

    Value is parsed as YAML so numbers, booleans, strings, lists all work
    without extra quoting: `-o reps.batch_size=10`, `-o llm.temperature=0.9`,
    `-o reps.workers.types='[exploiter, explorer]'`.
    """
    for expr in overrides:
        if "=" not in expr:
            raise ValueError(f"--override must be key=value, got {expr!r}")
        path, _, raw = expr.partition("=")
        value = yaml.safe_load(raw)
        parts = path.strip().split(".")
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="REPS Experiment Runner",
        epilog=(
            "Override any config field with -o / --override dotted.path=value.\n"
            "Examples: -o max_iterations=50  -o llm.temperature=0.9  -o reps.batch_size=10"
        ),
    )
    parser.add_argument(
        "initial_program", nargs="?", default=None,
        help="Path to initial program (optional if config sets task:)",
    )
    parser.add_argument(
        "evaluator", nargs="?", default=None,
        help="Path to evaluator script (optional if config sets task:)",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="reps_output", help="Output base directory")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Shortcut for -o max_iterations=N")
    parser.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config field (repeatable). Value parsed as YAML.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)

    # Apply generic overrides first, then honor the --iterations shortcut (so
    # it still wins if both are given).
    _apply_overrides(config, args.override)
    if args.iterations:
        config.max_iterations = args.iterations

    # If positional args weren't passed, derive them from config.task.
    if args.initial_program is None or args.evaluator is None:
        if not config.task:
            parser.error(
                "Must provide `initial_program` and `evaluator` as positional args "
                "or set `task:` in the config to the benchmark directory."
            )
        task_dir = Path(config.task)
        args.initial_program = args.initial_program or str(task_dir / "initial_program.py")
        args.evaluator = args.evaluator or str(task_dir / "evaluator.py")

    # Auto-version the output directory
    run_dir = _next_run_dir(args.output)
    print(f"Output: {run_dir}")

    # Make run_dir visible to benchmark evaluators (for persisting arrays etc.)
    os.environ["REPS_RUN_DIR"] = run_dir

    if config.harness == "openevolve":
        run_openevolve(args.config, args.initial_program, args.evaluator, run_dir, config.max_iterations)
    else:
        asyncio.run(run_reps(config, args.initial_program, args.evaluator, run_dir))


if __name__ == "__main__":
    main()
