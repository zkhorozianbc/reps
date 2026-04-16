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

    logging.basicConfig(level=getattr(logging, config.log_level, logging.INFO))
    logger = logging.getLogger(__name__)

    # Configure provider on model configs
    if config.provider == "anthropic":
        for model in config.llm.models:
            if getattr(model, "provider", None) is None:
                model.provider = "anthropic"
        for model in config.llm.evaluator_models:
            if getattr(model, "provider", None) is None:
                model.provider = "anthropic"

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


def main():
    parser = argparse.ArgumentParser(description="REPS Experiment Runner")
    parser.add_argument("initial_program", help="Path to initial program")
    parser.add_argument("evaluator", help="Path to evaluator script")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", default="reps_output", help="Output base directory")
    parser.add_argument("--iterations", type=int, default=None, help="Override max_iterations")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.iterations:
        config.max_iterations = args.iterations

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
