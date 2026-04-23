"""
Circle packing evaluator for n=26 in unit square [0,1]x[0,1].

Validation from DeepMind's alphaevolve_repository_of_problems.
Evaluation interface compatible with OpenEvolve.
"""

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# DeepMind validation (from alphaevolve_repository_of_problems)
# ---------------------------------------------------------------------------

def _circles_overlap(centers, radii):
    """Check if any circles overlap."""
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return True
    return False


def check_construction(centers, radii, n) -> dict:
    """Evaluate circle packing for maximizing sum of radii in unit square.

    This is the DeepMind gold-standard checker. Returns -inf for any violation.
    """
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        return {"sum_of_radii": -np.inf}

    is_contained = (
        (radii[:, None] <= centers) & (centers <= 1 - radii[:, None])
    ).all(axis=1)

    if not is_contained.all():
        return {"sum_of_radii": -np.inf}

    if radii.shape != (n,) or not np.isfinite(radii).all() or not (0 <= radii).all():
        return {"sum_of_radii": -np.inf}

    if _circles_overlap(centers, radii):
        return {"sum_of_radii": -np.inf}

    return {"sum_of_radii": float(np.sum(radii))}


# ---------------------------------------------------------------------------
# Packing validation with tolerance (for optimizer outputs)
# ---------------------------------------------------------------------------

def validate_packing(centers, radii, tol=1e-6):
    """Validate packing with numerical tolerance for optimizer outputs.

    Returns True if packing is valid within tolerance.
    """
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - tol:
                return False

    return True


# ---------------------------------------------------------------------------
# Subprocess execution with timeout
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def run_with_timeout(program_path, timeout_seconds=600):
    """Run the program in a subprocess with timeout."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

sys.path.insert(0, os.path.dirname('{program_path}'))

try:
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if hasattr(program, 'run_packing'):
        centers, radii, sum_radii = program.run_packing()
    elif hasattr(program, 'construct_packing'):
        result = program.construct_packing()
        if len(result) == 3:
            centers, radii, sum_radii = result
        else:
            centers, radii = result[0], result[1]
            sum_radii = float(np.sum(radii))
    else:
        raise RuntimeError("Program has neither run_packing() nor construct_packing()")

    results = {{'centers': centers, 'radii': radii, 'sum_radii': sum_radii}}
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)

except Exception as e:
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError(f"Process exited with code {process.returncode}")

            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")
                return results["centers"], results["radii"], results["sum_radii"]
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


# ---------------------------------------------------------------------------
# Persist (centers, radii) + strict verdict to markdown for each eval
# ---------------------------------------------------------------------------

def _dump_packing_markdown(
    centers: np.ndarray,
    radii: np.ndarray,
    reported_sum: float,
    strict_sum: float,
    strict_pass: bool,
    tolerant_pass: bool,
    eval_time: float,
) -> None:
    """Write the packing arrays + verdicts to REPS_RUN_DIR/packings/<program_id>.md.

    No-op if env vars are not set (e.g., ad-hoc invocations outside a run).
    """
    run_dir = os.environ.get("REPS_RUN_DIR")
    program_id = os.environ.get("REPS_PROGRAM_ID")
    if not run_dir or not program_id:
        return

    out_dir = Path(run_dir) / "packings"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    # Full precision repr — copy-paste back into numpy verbatim.
    np_printopts = {"precision": 17, "suppress": False, "threshold": np.inf, "linewidth": 100000}
    with np.printoptions(**np_printopts):
        centers_repr = repr(np.asarray(centers))
        radii_repr = repr(np.asarray(radii))

    lines = [
        f"# Packing: {program_id}",
        "",
        f"- reported sum_radii: `{reported_sum!r}`",
        f"- strict check_construction (DeepMind, no tolerance): **{'PASS' if strict_pass else 'FAIL'}**",
        f"- strict sum_of_radii: `{strict_sum!r}`",
        f"- tolerant check (tol=1e-6): {'pass' if tolerant_pass else 'fail'}",
        f"- eval_time: {eval_time:.3f}s",
        "",
        "## centers (shape (26, 2))",
        "```python",
        centers_repr,
        "```",
        "",
        "## radii (shape (26,))",
        "```python",
        radii_repr,
        "```",
        "",
    ]
    try:
        (out_dir / f"{program_id}.md").write_text("\n".join(lines))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# OpenEvolve-compatible evaluate interface
# ---------------------------------------------------------------------------

TARGET_VALUE = 2.635  # AlphaEvolve result for n=26


def evaluate(program_path):
    """Evaluate a circle packing program. Returns full-precision metrics."""
    try:
        start_time = time.time()
        centers, radii, reported_sum = run_with_timeout(program_path, timeout_seconds=600)
        eval_time = time.time() - start_time

        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        if np.isnan(centers).any() or np.isnan(radii).any():
            return {"sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0,
                    "eval_time": eval_time, "combined_score": 0.0}

        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        tolerant_pass = shape_valid and validate_packing(centers, radii)

        strict_verdict = check_construction(centers, radii, 26) if shape_valid else {"sum_of_radii": -np.inf}
        strict_sum = strict_verdict["sum_of_radii"]
        strict_pass = np.isfinite(strict_sum)

        # Score on the strict DeepMind checker — matches the benchmark standard.
        sum_radii = float(strict_sum) if strict_pass else 0.0
        validity = 1.0 if strict_pass else 0.0
        target_ratio = sum_radii / TARGET_VALUE if strict_pass else 0.0
        combined_score = target_ratio * validity

        _dump_packing_markdown(
            centers, radii,
            reported_sum=float(reported_sum) if reported_sum is not None else float("nan"),
            strict_sum=float(strict_sum),
            strict_pass=bool(strict_pass),
            tolerant_pass=bool(tolerant_pass),
            eval_time=eval_time,
        )

        return {
            "sum_radii": sum_radii,
            "target_ratio": target_ratio,
            "validity": validity,
            "eval_time": eval_time,
            "combined_score": combined_score,
            "strict_pass": 1.0 if strict_pass else 0.0,
            "tolerant_pass": 1.0 if tolerant_pass else 0.0,
            # Keep the pre-strict-check number so logs can surface near-misses
            # ("strict FAIL but program reported 2.54"). This isn't used for
            # scoring — combined_score still comes from the strict checker.
            "reported_sum_radii": float(reported_sum) if reported_sum is not None else float("nan"),
        }

    except Exception as e:
        traceback.print_exc()
        return {"sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0,
                "eval_time": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage1(program_path):
    """Quick validation check (strict DeepMind checker)."""
    try:
        centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600)

        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            return {"validity": 0.0, "combined_score": 0.0, "error": "Invalid shapes"}

        strict = check_construction(centers, radii, 26)["sum_of_radii"]
        strict_pass = np.isfinite(strict)
        actual_sum = float(strict) if strict_pass else 0.0
        combined_score = (actual_sum / TARGET_VALUE) if strict_pass else 0.0

        return {
            "validity": 1.0 if strict_pass else 0.0,
            "sum_radii": actual_sum,
            "target_ratio": actual_sum / TARGET_VALUE if strict_pass else 0.0,
            "combined_score": combined_score,
        }

    except Exception as e:
        traceback.print_exc()
        return {"validity": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path):
    """Full evaluation."""
    return evaluate(program_path)
