#!/usr/bin/env python3
"""
REPS Ablation Runner

Automates running N seeds × M variants and collects results into comparison CSVs.

Usage:
    python scripts/run_ablation.py \
        --benchmark circle_packing \
        --variants baseline,+reflection,+epsilon,+reflection+epsilon \
        --seeds 42,123,456 \
        --iterations 200

Variants:
    baseline              - Unmodified OpenEvolve (reps.enabled=false)
    +reflection           - Baseline + reflection engine only
    +epsilon              - Baseline + ε-revisitation only
    +reflection+epsilon   - Both reflection + ε-revisitation
    +workers              - Add worker diversity
    +workers+monitor      - Worker diversity + convergence monitor
    +contracts            - Add intelligence contracts
    +sota                 - Add SOTA steering
    full_reps             - All REPS features enabled
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Benchmark configurations
BENCHMARKS = {
    "circle_packing": {
        "initial_program": "examples/circle_packing/initial_program.py",
        "evaluator": "examples/circle_packing/evaluator.py",
        "base_config": "examples/circle_packing/config.yaml",
        "sota_target": 2.635,
    },
    "function_minimization": {
        "initial_program": "examples/function_minimization/initial_program.py",
        "evaluator": "examples/function_minimization/evaluator.py",
        "base_config": "examples/function_minimization/config.yaml",
        "sota_target": None,
    },
    "symbolic_regression": {
        "initial_program": "examples/symbolic_regression/initial_program.py",
        "evaluator": "examples/symbolic_regression/evaluator.py",
        "base_config": "examples/symbolic_regression/config.yaml",
        "sota_target": None,
    },
}

# REPS feature flags per variant
VARIANT_CONFIGS = {
    "baseline": {
        "reps.enabled": False,
    },
    "+reflection": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": False,
        "reps.convergence.enabled": False,
        "reps.contracts.enabled": False,
        "reps.sota.enabled": False,
    },
    "+epsilon": {
        "reps.enabled": True,
        "reps.reflection.enabled": False,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": False,
        "reps.contracts.enabled": False,
        "reps.sota.enabled": False,
    },
    "+reflection+epsilon": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": False,
        "reps.contracts.enabled": False,
        "reps.sota.enabled": False,
    },
    "+workers": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": False,
        "reps.contracts.enabled": False,
        "reps.sota.enabled": False,
    },
    "+workers+monitor": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": True,
        "reps.contracts.enabled": False,
        "reps.sota.enabled": False,
    },
    "+contracts": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": True,
        "reps.contracts.enabled": True,
        "reps.sota.enabled": False,
    },
    "+sota": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": True,
        "reps.contracts.enabled": True,
        "reps.sota.enabled": True,
    },
    "full_reps": {
        "reps.enabled": True,
        "reps.reflection.enabled": True,
        "reps.revisitation.enabled": True,
        "reps.convergence.enabled": True,
        "reps.contracts.enabled": True,
        "reps.sota.enabled": True,
        "reps.annotations.enabled": True,
    },
}


def build_config_overrides(variant: str, benchmark: str, seed: int, iterations: int) -> dict:
    """Build a config dict with variant-specific overrides."""
    import yaml

    bench = BENCHMARKS[benchmark]
    base_config_path = bench["base_config"]

    # Load base config
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Set seed and iterations
    config["random_seed"] = seed
    config["max_iterations"] = iterations

    # Apply variant-specific REPS overrides
    variant_flags = VARIANT_CONFIGS.get(variant, {})

    # Ensure reps section exists
    if "reps" not in config:
        config["reps"] = {}

    for key, value in variant_flags.items():
        parts = key.split(".")
        d = config
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    # Set SOTA target if available
    if bench.get("sota_target") and config.get("reps", {}).get("sota", {}).get("enabled"):
        config.setdefault("reps", {}).setdefault("sota", {})["target_score"] = bench["sota_target"]

    return config


def run_single(benchmark: str, variant: str, seed: int, iterations: int, output_base: str) -> dict:
    """Run a single benchmark/variant/seed combination."""
    import yaml

    bench = BENCHMARKS[benchmark]
    output_dir = os.path.join(output_base, benchmark, variant, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # Build config
    config = build_config_overrides(variant, benchmark, seed, iterations)

    # Write temporary config file
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run OpenEvolve
    cmd = [
        sys.executable,
        "openevolve-run.py",
        bench["initial_program"],
        bench["evaluator"],
        "--config", config_path,
        "--iterations", str(iterations),
        "--output", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {benchmark} / {variant} / seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Collect results
    run_result = {
        "benchmark": benchmark,
        "variant": variant,
        "seed": seed,
        "iterations": iterations,
        "wall_clock_seconds": elapsed,
        "returncode": result.returncode,
    }

    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-500:]}")
        run_result["error"] = result.stderr[-500:]
    else:
        # Try to read best score from output
        best_score = _extract_best_score(output_dir)
        run_result["best_score"] = best_score
        print(f"Completed: best_score={best_score}, time={elapsed:.1f}s")

    return run_result


def _extract_best_score(output_dir: str) -> float:
    """Extract best score from an OpenEvolve output directory."""
    # Try to read from metrics CSV
    metrics_path = os.path.join(output_dir, "metrics", "score_trajectory.csv")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            best = 0.0
            for row in reader:
                score = float(row.get("best_score", 0))
                best = max(best, score)
            return best

    # Fall back to checking checkpoint metadata
    checkpoints = Path(output_dir) / "checkpoints"
    if checkpoints.exists():
        for cp_dir in sorted(checkpoints.iterdir(), reverse=True):
            meta = cp_dir / "metadata.json"
            if meta.exists():
                with open(meta) as f:
                    data = json.load(f)
                    return data.get("best_score", 0.0)

    return 0.0


def generate_comparison_table(results: list, output_path: str):
    """Generate a comparison CSV from all results."""
    import statistics

    # Group by (benchmark, variant)
    groups = {}
    for r in results:
        key = (r["benchmark"], r["variant"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Write comparison CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Benchmark", "Variant", "Seeds",
            "Best Score (mean)", "Best Score (std)",
            "Wall Clock (mean)", "Wall Clock (std)",
        ])

        for (benchmark, variant), runs in sorted(groups.items()):
            scores = [r["best_score"] for r in runs if "best_score" in r]
            times = [r["wall_clock_seconds"] for r in runs]

            if scores:
                score_mean = statistics.mean(scores)
                score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            else:
                score_mean = score_std = 0.0

            time_mean = statistics.mean(times)
            time_std = statistics.stdev(times) if len(times) > 1 else 0.0

            writer.writerow([
                benchmark, variant, len(runs),
                f"{score_mean:.6f}", f"{score_std:.6f}",
                f"{time_mean:.1f}", f"{time_std:.1f}",
            ])

    print(f"\nComparison table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="REPS Ablation Runner")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Comma-separated benchmark names: circle_packing,function_minimization,symbolic_regression",
    )
    parser.add_argument(
        "--variants",
        required=True,
        help="Comma-separated variant names: baseline,+reflection,+epsilon,full_reps",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,456",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of iterations per run",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Base output directory",
    )

    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmark.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Validate
    for b in benchmarks:
        if b not in BENCHMARKS:
            print(f"Unknown benchmark: {b}. Available: {list(BENCHMARKS.keys())}")
            sys.exit(1)
    for v in variants:
        if v not in VARIANT_CONFIGS:
            print(f"Unknown variant: {v}. Available: {list(VARIANT_CONFIGS.keys())}")
            sys.exit(1)

    total_runs = len(benchmarks) * len(variants) * len(seeds)
    print(f"REPS Ablation: {total_runs} total runs")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Variants: {variants}")
    print(f"  Seeds: {seeds}")
    print(f"  Iterations: {args.iterations}")

    all_results = []
    for benchmark in benchmarks:
        for variant in variants:
            for seed in seeds:
                result = run_single(benchmark, variant, seed, args.iterations, args.output)
                all_results.append(result)

    # Generate comparison table
    comparison_path = os.path.join(args.output, "ablation_comparison.csv")
    generate_comparison_table(all_results, comparison_path)

    # Save raw results
    raw_path = os.path.join(args.output, "ablation_raw.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()
