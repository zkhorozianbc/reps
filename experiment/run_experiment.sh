#!/bin/bash
# REPS Experiment Runner
# Runs baseline (clean OpenEvolve) vs REPS (modified OpenEvolve)
# Same config, same seeds, same iterations
#
# Requirements:
#   - OPENROUTER_API_KEY set in environment
#   - openevolve_clean/ = unmodified git clone
#   - openevolve/       = REPS-modified clone
#
# Usage: bash experiment/run_experiment.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLEAN="$ROOT/openevolve_clean"
REPS="$ROOT/openevolve"
CONFIGS="$ROOT/experiment/configs"
RESULTS="$ROOT/experiment/results"

SEEDS=(42 123 456 789 1337)
ITERATIONS=100
BENCHMARK="function_minimization"
INITIAL="examples/$BENCHMARK/initial_program.py"
EVALUATOR="examples/$BENCHMARK/evaluator.py"

# Load API key
if [ -f "$ROOT/.env" ]; then
    export $(grep -v '^#' "$ROOT/.env" | xargs)
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

echo "=========================================="
echo "REPS Experiment"
echo "=========================================="
echo "Benchmark: $BENCHMARK"
echo "Iterations: $ITERATIONS"
echo "Seeds: ${SEEDS[*]}"
echo "Baseline: $CLEAN"
echo "REPS:     $REPS"
echo ""

mkdir -p "$RESULTS"

# --- Run baseline (clean OpenEvolve) ---
for seed in "${SEEDS[@]}"; do
    OUTDIR="$RESULTS/baseline/seed_$seed"
    if [ -d "$OUTDIR/best" ]; then
        echo "SKIP baseline seed=$seed (already done)"
        continue
    fi
    echo ""
    echo ">>> BASELINE seed=$seed"

    # Create seed-specific config
    SEED_CONFIG="$RESULTS/baseline/config_seed_$seed.yaml"
    mkdir -p "$(dirname "$SEED_CONFIG")"
    cp "$CONFIGS/base.yaml" "$SEED_CONFIG"
    echo "random_seed: $seed" >> "$SEED_CONFIG"

    cd "$CLEAN"
    uv run python openevolve-run.py \
        "$INITIAL" "$EVALUATOR" \
        --config "$SEED_CONFIG" \
        --iterations "$ITERATIONS" \
        --output "$OUTDIR" 2>&1 | tee "$OUTDIR.log" | tail -5

    echo ">>> BASELINE seed=$seed DONE"
done

# --- Run REPS (modified OpenEvolve) ---
for seed in "${SEEDS[@]}"; do
    OUTDIR="$RESULTS/reps/seed_$seed"
    if [ -d "$OUTDIR/best" ]; then
        echo "SKIP reps seed=$seed (already done)"
        continue
    fi
    echo ""
    echo ">>> REPS seed=$seed"

    SEED_CONFIG="$RESULTS/reps/config_seed_$seed.yaml"
    mkdir -p "$(dirname "$SEED_CONFIG")"
    cp "$CONFIGS/reps_full.yaml" "$SEED_CONFIG"
    echo "random_seed: $seed" >> "$SEED_CONFIG"

    cd "$REPS"
    uv run python openevolve-run.py \
        "$INITIAL" "$EVALUATOR" \
        --config "$SEED_CONFIG" \
        --iterations "$ITERATIONS" \
        --output "$OUTDIR" 2>&1 | tee "$OUTDIR.log" | tail -5

    echo ">>> REPS seed=$seed DONE"
done

echo ""
echo "=========================================="
echo "All runs complete. Analyzing..."
echo "=========================================="

# --- Collect results ---
cd "$ROOT"
uv run --directory "$REPS" python3 - <<'PYTHON'
import json, os, sys, statistics

results_dir = os.environ.get("RESULTS", "experiment/results")
variants = ["baseline", "reps"]
seeds = [42, 123, 456, 789, 1337]

print(f"\n{'Variant':<12} {'Seeds':>5} {'combined_score (mean±std)':>28} {'distance_score (mean±std)':>28}")
print("-" * 80)

for variant in variants:
    scores = []
    distances = []
    for seed in seeds:
        info_path = os.path.join(results_dir, variant, f"seed_{seed}", "best", "best_program_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            metrics = info.get("metrics", {})
            scores.append(metrics.get("combined_score", 0))
            distances.append(metrics.get("distance_score", 0))

    n = len(scores)
    if n > 1:
        s_mean, s_std = statistics.mean(scores), statistics.stdev(scores)
        d_mean, d_std = statistics.mean(distances), statistics.stdev(distances)
    elif n == 1:
        s_mean, s_std = scores[0], 0.0
        d_mean, d_std = distances[0], 0.0
    else:
        s_mean = s_std = d_mean = d_std = 0.0

    print(f"{variant:<12} {n:>5} {s_mean:>12.6f} ± {s_std:<12.6f} {d_mean:>12.6f} ± {d_std:<12.6f}")

# Paired t-test if scipy available
try:
    from scipy import stats as sp_stats
    baseline_scores = []
    reps_scores = []
    for seed in seeds:
        for variant, lst in [("baseline", baseline_scores), ("reps", reps_scores)]:
            info_path = os.path.join(results_dir, variant, f"seed_{seed}", "best", "best_program_info.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                lst.append(info.get("metrics", {}).get("combined_score", 0))

    if len(baseline_scores) == len(reps_scores) and len(baseline_scores) >= 3:
        t_stat, p_value = sp_stats.ttest_rel(reps_scores, baseline_scores)
        print(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"  {'Significant at p<0.05' if p_value < 0.05 else 'NOT significant at p<0.05'}")
except ImportError:
    print("\n(install scipy for paired t-test)")
PYTHON

echo ""
echo "Done. Results in $RESULTS/"
