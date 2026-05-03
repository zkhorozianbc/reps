"""
Circle packing evaluator for n=26 in unit square [0,1]x[0,1].

Validation from DeepMind's alphaevolve_repository_of_problems.
Evaluation interface compatible with OpenEvolve.
"""

import importlib.util
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

# EvaluationResult is part of the REPS harness; when the evaluator is invoked
# via `uv run python -c ...` outside a full run, the reps package is still
# importable (pyproject installs it editable). If import fails we fall back to
# returning a plain dict so this module stays usable as a standalone script.
try:
    from reps.evaluation_result import EvaluationResult  # type: ignore
except ImportError:  # pragma: no cover - standalone ad-hoc use
    EvaluationResult = None  # type: ignore


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


def run_with_timeout(program_path, timeout_seconds=600, env=None):
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
            env=env,
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
    try:
        from reps.runtime import current_program_id
        program_id = current_program_id() or "unknown"
    except ImportError:
        program_id = os.environ.get("REPS_PROGRAM_ID", "unknown")
    if not run_dir or not program_id or program_id == "unknown":
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
# Per-constraint diagnostics (emitted on every evaluate to give the LLM
# signal about *why* a packing failed — boundary crossings, overlaps, wrong
# count — or about how tight a valid packing is.)
# ---------------------------------------------------------------------------

_DIAGNOSTIC_CAP = 10  # cap list lengths so prompts stay small
_DIAGNOSTIC_TOL = 1e-9  # suppress float-noise "violations" at exact-touch boundaries


def _compute_constraint_diagnostics(centers: np.ndarray, radii: np.ndarray) -> dict:
    """Compute structured feedback about constraint satisfaction.

    Returns a dict with:
      - n_circles_submitted: int, actual count of circles in the submission
      - boundary_violations: list of dicts (capped) for circles crossing [0,1]^2
      - overlap_pairs: list of dicts (capped, sorted worst-first) for overlaps
      - min_pairwise_slack: float or None — tightest non-overlap margin across
        all pairs (only meaningful when there are >=2 circles).

    All numbers are native Python floats/ints (JSON-safe).
    """
    # Tolerate odd shapes — we still report what we can see.
    c_arr = np.asarray(centers, dtype=float) if centers is not None else np.zeros((0, 2))
    r_arr = np.asarray(radii, dtype=float).ravel() if radii is not None else np.zeros(0)

    if c_arr.ndim != 2 or c_arr.shape[-1] != 2:
        # Malformed centers — count what the program submitted as best we can.
        n_submitted = int(r_arr.shape[0]) if r_arr.ndim == 1 else 0
        return {
            "n_circles_submitted": n_submitted,
            "boundary_violations": [],
            "overlap_pairs": [],
            "min_pairwise_slack": None,
            "note": f"centers has malformed shape {tuple(c_arr.shape)}; expected (n, 2)",
        }

    n = int(c_arr.shape[0])
    # If radii length disagrees, use the min so pairwise math is well-defined.
    n_eff = int(min(n, r_arr.shape[0]))

    # --- boundary violations (signed slack: negative means the circle sticks
    # out of the unit square along the tightest side). ---
    boundary_violations = []
    for i in range(n_eff):
        x, y = float(c_arr[i, 0]), float(c_arr[i, 1])
        r = float(r_arr[i])
        # Slack to each side: positive = inside margin; negative = crossing.
        slacks = [x - r, (1.0 - x) - r, y - r, (1.0 - y) - r]
        min_slack = min(slacks)
        if min_slack < -_DIAGNOSTIC_TOL or not np.isfinite(min_slack):
            boundary_violations.append(
                {
                    "circle_index": i,
                    "center": [x, y],
                    "radius": r,
                    "slack": float(min_slack),
                }
            )
    # Worst (most-negative slack) first, cap. Record the uncapped total so
    # per_instance_scores can reflect true severity even when the visible
    # list is truncated for prompt size.
    boundary_violations.sort(key=lambda d: d["slack"])
    n_boundary_violations_total = len(boundary_violations)
    boundary_violations = boundary_violations[:_DIAGNOSTIC_CAP]

    # --- overlap pairs + minimum pairwise slack ---
    overlap_pairs = []
    min_slack_pairwise = None
    if n_eff >= 2:
        for i in range(n_eff):
            ri = float(r_arr[i])
            for j in range(i + 1, n_eff):
                rj = float(r_arr[j])
                d = float(np.linalg.norm(c_arr[i] - c_arr[j]))
                rsum = ri + rj
                interpen = rsum - d  # positive = overlap
                slack_ij = d - rsum  # positive = non-overlapping gap
                if min_slack_pairwise is None or slack_ij < min_slack_pairwise:
                    min_slack_pairwise = slack_ij
                if interpen > _DIAGNOSTIC_TOL:
                    overlap_pairs.append(
                        {
                            "pair": [i, j],
                            "center_distance": d,
                            "radius_sum": rsum,
                            "interpenetration": interpen,
                        }
                    )
        overlap_pairs.sort(key=lambda d: d["interpenetration"], reverse=True)
        n_overlap_pairs_total = len(overlap_pairs)
        overlap_pairs = overlap_pairs[:_DIAGNOSTIC_CAP]
    else:
        n_overlap_pairs_total = 0

    return {
        "n_circles_submitted": n,
        "boundary_violations": boundary_violations,
        "overlap_pairs": overlap_pairs,
        "n_boundary_violations_total": n_boundary_violations_total,
        "n_overlap_pairs_total": n_overlap_pairs_total,
        "min_pairwise_slack": (
            float(min_slack_pairwise) if min_slack_pairwise is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# OpenEvolve-compatible evaluate interface
# ---------------------------------------------------------------------------

TARGET_VALUE = 2.6361  # target beyond current AlphaEvolve/OpenEvolve baseline (2.635983); see algorithmicsuperintelligence/openevolve#156


def _build_per_instance_scores(
    strict_pass: bool,
    diagnostics: dict,
    target_ratio: float,
    n_expected: int = 26,
) -> dict:
    """Decompose the single circle-packing objective into independent
    sub-scores (GEPA-style "per-instance"). All keys produce values in
    [0, 1] so a Pareto front over them is well-defined.

    Keys:
        validity:    1.0 iff the strict DeepMind checker accepted the packing.
        boundary:    1.0 iff no circle crosses [0,1]^2; otherwise the fraction
                     of submitted circles that stay inside.
        overlap:     1.0 iff no pair overlaps; otherwise 1 - overlap_pairs /
                     total_pairs over submitted circles.
        sum_radii_progress: target_ratio, clamped to [0, 1.5] so a program
                     that beats the target still shows incremental signal but
                     can't dominate by an arbitrary multiple.
    """
    n_submitted = int(diagnostics.get("n_circles_submitted", 0) or 0)
    n_eff = max(1, min(n_submitted, n_expected))

    # Prefer uncapped totals — the visible *_violations / *_pairs lists are
    # truncated to _DIAGNOSTIC_CAP for prompt size and would understate
    # severity here. Fall back to len() so synthesized diags (in unit tests
    # and ad-hoc callers) still produce sane numbers.
    boundary_violations = diagnostics.get("boundary_violations") or []
    overlap_pairs = diagnostics.get("overlap_pairs") or []
    n_boundary = int(
        diagnostics.get("n_boundary_violations_total", len(boundary_violations))
    )
    n_overlap = int(diagnostics.get("n_overlap_pairs_total", len(overlap_pairs)))

    boundary_score = max(0.0, (n_eff - n_boundary) / n_eff)
    total_pairs = max(1, n_eff * (n_eff - 1) // 2)
    overlap_score = max(0.0, 1.0 - n_overlap / total_pairs)

    return {
        "validity": 1.0 if strict_pass else 0.0,
        "boundary": float(boundary_score),
        "overlap": float(overlap_score),
        "sum_radii_progress": float(min(1.5, max(0.0, target_ratio))),
    }


def _build_feedback(
    strict_pass: bool,
    diagnostics: dict,
    sum_radii: float,
    reported_sum: float,
) -> str:
    """Short ASI-style summary the reflection LLM can read directly.

    Roughly 1 line; surfaces the things a programmer would care about:
    constraint violations and how close sum_radii is to target.
    """
    boundary_violations = diagnostics.get("boundary_violations") or []
    overlap_pairs = diagnostics.get("overlap_pairs") or []
    n_boundary = int(
        diagnostics.get("n_boundary_violations_total", len(boundary_violations))
    )
    n_overlap = int(diagnostics.get("n_overlap_pairs_total", len(overlap_pairs)))
    n_submitted = diagnostics.get("n_circles_submitted")
    min_slack = diagnostics.get("min_pairwise_slack")

    if strict_pass:
        slack_part = (
            f", min pairwise slack {min_slack:.3e}" if min_slack is not None else ""
        )
        return (
            f"valid packing; sum_radii={sum_radii:.6f} "
            f"(target {TARGET_VALUE}, ratio {sum_radii / TARGET_VALUE:.4f})"
            f"{slack_part}"
        )

    parts = ["invalid"]
    if n_submitted is not None and n_submitted != 26:
        parts.append(f"got {n_submitted} circles (expected 26)")
    if n_boundary and boundary_violations:
        worst = boundary_violations[0]
        parts.append(
            f"{n_boundary} boundary violation(s)"
            f" (worst slack {worst['slack']:.3e} at circle {worst['circle_index']})"
        )
    if n_overlap and overlap_pairs:
        worst = overlap_pairs[0]
        parts.append(
            f"{n_overlap} overlap pair(s)"
            f" (worst interpen {worst['interpenetration']:.3e} between {worst['pair'][0]}"
            f" and {worst['pair'][1]})"
        )
    if reported_sum is not None and np.isfinite(reported_sum):
        parts.append(f"reported sum={reported_sum:.4f}")
    note = diagnostics.get("note")
    if note:
        parts.append(note)
    return "; ".join(parts)


def _build_result(metrics: dict, artifacts: dict, *,
                  per_instance_scores: dict | None = None,
                  feedback: str | None = None):
    """Return either an EvaluationResult (if available) or a plain dict.

    We always populate nested `constraint_diagnostics` inside metrics too so
    standalone callers (and any consumer that only inspects the top-level
    dict) can see it. The float-only stage-merge path in reps.evaluator will
    drop this nested key silently — that's fine; the artifacts channel is the
    authoritative home for the diagnostics in multi-stage pipelines.

    GEPA-style ASI: when EvaluationResult is available, also populate
    per_instance_scores (sub-objective decomposition) and feedback (free-form
    diagnostic string). When falling back to a plain dict for standalone use,
    these are merged in under reserved keys so they remain inspectable.
    """
    if EvaluationResult is not None:
        return EvaluationResult(
            metrics=metrics,
            artifacts=artifacts,
            per_instance_scores=per_instance_scores,
            feedback=feedback,
        )
    # Standalone fallback: flatten artifacts into the metrics dict so ad-hoc
    # `python -c` callers can still see everything in one place.
    merged = dict(metrics)
    merged["_artifacts"] = artifacts
    if per_instance_scores is not None:
        merged["_per_instance_scores"] = per_instance_scores
    if feedback is not None:
        merged["_feedback"] = feedback
    return merged


def evaluate(program_path, env=None):
    """Evaluate a circle packing program. Returns full-precision metrics.

    Top-level metric keys (unchanged contract):
        validity, sum_radii, target_ratio, combined_score, eval_time,
        strict_pass, tolerant_pass.

    Additional signals (new):
        - metrics["combined_score_hi_precision_str"]: "{:.12g}" formatted string
          so downstream display doesn't truncate small deltas.
          NOTE: This is intentionally a STRING, not a float, so it survives
          verbatim through any formatter. It is returned both inside the
          metrics dict (for direct-eval consumers) and as an artifact.
        - metrics["constraint_diagnostics"]: nested dict with boundary/overlap
          diagnostics (see `_compute_constraint_diagnostics`).
        - artifacts["constraint_diagnostics_json"]: same content as a JSON
          string for easy embedding in prompts.
        - artifacts["combined_score_hi_precision_str"]: see above.
    """
    try:
        start_time = time.time()
        centers, radii, reported_sum = run_with_timeout(program_path, timeout_seconds=600, env=env)
        eval_time = time.time() - start_time

        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Even on NaN we still emit diagnostics (the LLM deserves to know how
        # many circles came back and where they were — NaN in one coordinate
        # shouldn't erase the rest of the signal).
        has_nan = bool(np.isnan(centers).any() or np.isnan(radii).any())

        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        tolerant_pass = bool(shape_valid and not has_nan and validate_packing(centers, radii))

        if shape_valid and not has_nan:
            strict_verdict = check_construction(centers, radii, 26)
        else:
            strict_verdict = {"sum_of_radii": -np.inf}
        strict_sum = strict_verdict["sum_of_radii"]
        strict_pass = bool(np.isfinite(strict_sum))

        # Score on the strict DeepMind checker — matches the benchmark standard.
        sum_radii = float(strict_sum) if strict_pass else 0.0
        validity = 1.0 if strict_pass else 0.0
        target_ratio = float(sum_radii / TARGET_VALUE) if strict_pass else 0.0
        combined_score = float(target_ratio * validity)

        # Diagnostics are always computed — even on success, so the LLM can
        # see how tight the current packing is (min_pairwise_slack).
        diagnostics = _compute_constraint_diagnostics(centers, radii)
        if has_nan:
            diagnostics.setdefault(
                "note",
                "centers or radii contained NaN; treating as fully invalid",
            )

        # High-precision combined_score string. {:.12g} preserves at least 12
        # significant digits without trailing zeros, which is enough to
        # distinguish combined_score=0.9999640000 from 0.9999640001.
        hi_prec = "{:.12g}".format(combined_score)

        _dump_packing_markdown(
            centers, radii,
            reported_sum=float(reported_sum) if reported_sum is not None else float("nan"),
            strict_sum=float(strict_sum),
            strict_pass=bool(strict_pass),
            tolerant_pass=bool(tolerant_pass),
            eval_time=eval_time,
        )

        metrics = {
            "sum_radii": float(sum_radii),
            "target_ratio": float(target_ratio),
            "validity": float(validity),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
            "strict_pass": 1.0 if strict_pass else 0.0,
            "tolerant_pass": 1.0 if tolerant_pass else 0.0,
            "reported_sum_radii": float(reported_sum) if reported_sum is not None else float("nan"),
            "combined_score_hi_precision_str": hi_prec,
            "constraint_diagnostics": diagnostics,
        }
        artifacts = {
            "combined_score_hi_precision_str": hi_prec,
            "constraint_diagnostics_json": json.dumps(diagnostics, indent=2),
        }
        per_instance_scores = _build_per_instance_scores(
            strict_pass=strict_pass,
            diagnostics=diagnostics,
            target_ratio=target_ratio,
        )
        feedback = _build_feedback(
            strict_pass=strict_pass,
            diagnostics=diagnostics,
            sum_radii=sum_radii,
            reported_sum=float(reported_sum) if reported_sum is not None else float("nan"),
        )
        return _build_result(
            metrics, artifacts,
            per_instance_scores=per_instance_scores,
            feedback=feedback,
        )

    except Exception as e:
        traceback.print_exc()
        err_metrics = {
            "sum_radii": 0.0,
            "target_ratio": 0.0,
            "validity": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
            "error": str(e),
            "combined_score_hi_precision_str": "0",
        }
        return _build_result(
            err_metrics,
            {"error": str(e)},
            per_instance_scores={
                "validity": 0.0,
                "boundary": 0.0,
                "overlap": 0.0,
                "sum_radii_progress": 0.0,
            },
            feedback=f"evaluation failed: {e}",
        )


def evaluate_stage1(program_path, env=None):
    """Quick validation check (strict DeepMind checker)."""
    try:
        centers, radii, sum_radii = run_with_timeout(program_path, timeout_seconds=600, env=env)

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


def evaluate_stage2(program_path, env=None):
    """Full evaluation."""
    return evaluate(program_path, env=env)
