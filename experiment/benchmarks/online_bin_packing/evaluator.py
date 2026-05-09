"""
Online 1-D bin packing evaluator.

Datasets: OR3 (20 instances × 500 items, capacity 150) and Weibull 5k
(5 instances × 5000 items, capacity 100). Both extracted from the FunSearch
release (Romera-Paredes et al., Nature 2024).

The candidate program defines `priority(item, bins) -> np.ndarray`. For each
item we restrict attention to bins with sufficient remaining capacity, score
each via `priority`, and place the item in the highest-scoring bin.

Headline metric: per-instance excess = (bins_used - L1_lower_bound) / L1.
Lower is better. We expose 1 - mean_excess as `combined_score` (higher is
better, in line with the rest of the REPS harness).

Published targets:
- OR3:        Best Fit ~5.37%; FunSearch ~3.85%        (FunSearch Nature 2024)
- Weibull 5k: Best Fit ~3.98%; FunSearch ~0.68% (verified locally).
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from reps.evaluation_result import EvaluationResult  # type: ignore
except ImportError:  # pragma: no cover
    EvaluationResult = None  # type: ignore


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "instances" / "datasets.pkl"


# ---------------------------------------------------------------------------
# Dataset loading (cached on first call)
# ---------------------------------------------------------------------------

_DATASET_CACHE: dict | None = None


def _load_datasets() -> dict:
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        with open(DATASET_PATH, "rb") as f:
            _DATASET_CACHE = pickle.load(f)
    return _DATASET_CACHE


# ---------------------------------------------------------------------------
# Online bin packing simulator
# ---------------------------------------------------------------------------

def _online_binpack(items, capacity: int, priority_fn) -> tuple[np.ndarray, list[list[int]]]:
    """Place items online into bins of size `capacity`, scoring with `priority_fn`.

    Pre-allocates len(items) bins at full capacity so any feasible placement
    succeeds. Returns (remaining_capacities, packing) where `packing[i]` is
    the list of items in bin i.
    """
    n = len(items)
    bins = np.full(n, capacity, dtype=np.float64)
    packing: list[list[int]] = [[] for _ in range(n)]
    for item in items:
        valid_idx = np.nonzero(bins - item >= 0)[0]
        if valid_idx.size == 0:
            raise RuntimeError(
                f"No valid bin for item {item} — pre-allocation broken."
            )
        scores = priority_fn(item, bins[valid_idx])
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if scores.shape != valid_idx.shape:
            raise RuntimeError(
                f"priority() returned shape {scores.shape} but expected {valid_idx.shape}"
            )
        if not np.isfinite(scores).all():
            # NaN/inf priorities are a programmer error in the heuristic.
            raise RuntimeError("priority() returned non-finite values")
        chosen = int(valid_idx[int(np.argmax(scores))])
        bins[chosen] -= item
        packing[chosen].append(int(item))
    return bins, packing


def _is_valid_packing(packing, items, capacity) -> bool:
    flat = sorted(x for b in packing for x in b)
    if flat != sorted(int(x) for x in items):
        return False
    for b in packing:
        if sum(b) > capacity:
            return False
    return True


# ---------------------------------------------------------------------------
# Subprocess execution with timeout
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


_RUNNER_TEMPLATE = """
import sys, os, pickle, traceback
import numpy as np

sys.path.insert(0, os.path.dirname({program_path!r}))

try:
    spec = __import__('importlib.util').util.spec_from_file_location("program", {program_path!r})
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, 'priority'):
        raise RuntimeError("Program does not define priority(item, bins)")

    with open({datasets_path!r}, 'rb') as f:
        bundle = pickle.load(f)
    datasets = bundle['datasets']
    l1 = bundle['l1_lower_bounds']

    def online_binpack(items, capacity):
        n = len(items)
        bins = np.full(n, capacity, dtype=np.float64)
        packing = [[] for _ in range(n)]
        for item in items:
            valid_idx = np.nonzero(bins - item >= 0)[0]
            if valid_idx.size == 0:
                raise RuntimeError("No valid bin")
            scores = program.priority(item, bins[valid_idx])
            scores = np.asarray(scores, dtype=np.float64).ravel()
            if scores.shape != valid_idx.shape:
                raise RuntimeError(
                    f"priority() shape mismatch: got {{scores.shape}}, expected {{valid_idx.shape}}"
                )
            if not np.isfinite(scores).all():
                raise RuntimeError("priority() returned non-finite values")
            chosen = int(valid_idx[int(np.argmax(scores))])
            bins[chosen] -= item
            packing[chosen].append(int(item))
        return bins, packing

    out = {{}}
    for ds_name, instances in datasets.items():
        per_instance = []
        for inst_name, inst in instances.items():
            cap = inst['capacity']
            items = inst['items']
            bins, packing = online_binpack(items, cap)
            used = int((bins != cap).sum())
            # Validate (catches buggy heuristics that drop items somehow).
            flat = sorted(x for b in packing for x in b)
            if flat != sorted(int(x) for x in items):
                raise RuntimeError(f"Lost items in {{inst_name}}")
            for b in packing:
                if sum(b) > cap:
                    raise RuntimeError(f"Over-capacity bin in {{inst_name}}")
            per_instance.append({{'name': inst_name, 'bins_used': used,
                                  'capacity': cap, 'num_items': inst['num_items']}})
        out[ds_name] = {{'instances': per_instance, 'l1_mean': float(l1[ds_name])}}

    with open({results_path!r}, 'wb') as f:
        pickle.dump({{'results': out}}, f)
except Exception as e:
    traceback.print_exc()
    with open({results_path!r}, 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""


def _run_with_timeout(program_path: str, timeout_seconds: int = 120, env=None):
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tf:
        runner_path = tf.name
    results_path = runner_path + ".results"
    runner_src = _RUNNER_TEMPLATE.format(
        program_path=program_path,
        datasets_path=str(DATASET_PATH),
        results_path=results_path,
    )
    Path(runner_path).write_text(runner_src)
    try:
        proc = subprocess.Popen(
            [sys.executable, runner_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds}s")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Process exited {proc.returncode}; stderr: {stderr.decode(errors='replace')[:500]}"
            )
        if not os.path.exists(results_path):
            raise RuntimeError("Results file not written")
        with open(results_path, "rb") as f:
            payload = pickle.load(f)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["results"]
    finally:
        for p in (runner_path, results_path):
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _summarize_per_dataset(per_dataset: dict) -> dict:
    """Per-dataset and global stats. Every number is JSON-safe."""
    summary = {}
    all_excess = []
    for ds_name, ds in per_dataset.items():
        l1 = ds["l1_mean"]
        bins_per_instance = [inst["bins_used"] for inst in ds["instances"]]
        avg_bins = float(np.mean(bins_per_instance))
        excess = (avg_bins - l1) / l1
        summary[ds_name] = {
            "avg_bins_used": avg_bins,
            "l1_lower_bound": float(l1),
            "excess_fraction": float(excess),
            "excess_pct": float(100 * excess),
            "num_instances": len(bins_per_instance),
        }
        all_excess.append(excess)
    summary["mean_excess_fraction"] = float(np.mean(all_excess))
    summary["mean_excess_pct"] = float(100 * np.mean(all_excess))
    return summary


# ---------------------------------------------------------------------------
# Persist per-program packing summary
# ---------------------------------------------------------------------------

def _dump_summary_markdown(summary: dict, eval_time: float) -> None:
    run_dir = os.environ.get("REPS_RUN_DIR")
    try:
        from reps.runtime import current_program_id
        program_id = current_program_id() or "unknown"
    except ImportError:
        program_id = os.environ.get("REPS_PROGRAM_ID", "unknown")
    if not run_dir or not program_id or program_id == "unknown":
        return
    out_dir = Path(run_dir) / "binpack_summaries"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    lines = [f"# Bin packing: {program_id}", ""]
    for ds_name, s in summary.items():
        if not isinstance(s, dict):
            continue
        lines += [
            f"## {ds_name}",
            f"- avg_bins_used: `{s['avg_bins_used']:.4f}`",
            f"- l1_lower_bound: `{s['l1_lower_bound']:.4f}`",
            f"- excess_pct: **{s['excess_pct']:.4f}%**",
            f"- num_instances: {s['num_instances']}",
            "",
        ]
    lines += [
        f"- mean_excess_pct (across datasets): **{summary['mean_excess_pct']:.4f}%**",
        f"- eval_time: {eval_time:.3f}s",
    ]
    try:
        (out_dir / f"{program_id}.md").write_text("\n".join(lines))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public evaluate interface (REPS / OpenEvolve compatible)
# ---------------------------------------------------------------------------

def _build_result(metrics: dict, artifacts: dict):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    merged = dict(metrics)
    merged["_artifacts"] = artifacts
    return merged


def evaluate(program_path, env=None):
    """Run the candidate `priority()` over OR3 + Weibull 5k.

    Returns metrics with `combined_score = 1 - mean_excess_fraction` (higher
    is better). Best Fit seed scores ~0.953; FunSearch's published Weibull
    heuristic scores ~0.977 on the joint set.
    """
    try:
        start = time.time()
        per_dataset = _run_with_timeout(program_path, timeout_seconds=120, env=env)
        eval_time = time.time() - start

        summary = _summarize_per_dataset(per_dataset)
        mean_excess = summary["mean_excess_fraction"]
        combined_score = float(1.0 - mean_excess)

        _dump_summary_markdown(summary, eval_time)

        # Top-level metrics flat enough to survive harness float-only paths.
        metrics: dict = {
            "combined_score": combined_score,
            "mean_excess_fraction": float(mean_excess),
            "mean_excess_pct": float(summary["mean_excess_pct"]),
            "eval_time": float(eval_time),
            "validity": 1.0,
        }
        for ds_name in ("OR3", "Weibull 5k"):
            if ds_name in summary:
                key = ds_name.replace(" ", "_").lower()
                metrics[f"{key}_excess_pct"] = float(summary[ds_name]["excess_pct"])
                metrics[f"{key}_avg_bins"] = float(summary[ds_name]["avg_bins_used"])
        metrics["summary"] = summary

        artifacts = {
            "summary_json": json.dumps(summary, indent=2),
        }
        return _build_result(metrics, artifacts)

    except Exception as e:
        traceback.print_exc()
        err_metrics = {
            "combined_score": 0.0,
            "mean_excess_fraction": 1.0,
            "mean_excess_pct": 100.0,
            "eval_time": 0.0,
            "validity": 0.0,
            "error": str(e),
        }
        return _build_result(err_metrics, {"error": str(e)})


def evaluate_stage1(program_path, env=None):
    """Quick check: just run OR3 (smaller, ~10k placements) for a fast cascade."""
    try:
        start = time.time()
        per_dataset = _run_with_timeout(program_path, timeout_seconds=60, env=env)
        summary = _summarize_per_dataset(per_dataset)
        return {
            "validity": 1.0,
            "combined_score": float(1.0 - summary["mean_excess_fraction"]),
            "mean_excess_pct": float(summary["mean_excess_pct"]),
            "eval_time": time.time() - start,
        }
    except Exception as e:
        traceback.print_exc()
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path, env=None):
    return evaluate(program_path, env=env)


if __name__ == "__main__":
    # Allow standalone invocation: `python evaluator.py path/to/program.py`
    target = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "initial_program.py")
    res = evaluate(target)
    if EvaluationResult is not None and isinstance(res, EvaluationResult):
        print(json.dumps(res.metrics, indent=2, default=str))
    else:
        print(json.dumps(res, indent=2, default=str))
