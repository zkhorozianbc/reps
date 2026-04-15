"""
Circle packing evaluator for n=26 in unit square [0,1]x[0,1].

Uses DeepMind's gold-standard validation from:
https://github.com/google-deepmind/alphaevolve_repository_of_problems
"""

import importlib.util
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# DeepMind validation (from alphaevolve_repository_of_problems)
# ---------------------------------------------------------------------------

def _circles_overlap(centers, radii, tol=1e-9):
    """Check if any circles overlap beyond floating-point tolerance."""
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] - dist > tol:
                return True
    return False


def check_construction(centers, radii, n) -> dict:
    """Validate a circle packing in the unit square [0,1]x[0,1].

    Returns {'sum_of_radii': float} on success, or {'sum_of_radii': -inf} on
    any constraint violation (overlap, out of bounds, bad shapes).
    """
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        return {"sum_of_radii": -np.inf}

    if radii.shape != (n,) or not np.isfinite(radii).all() or not (0 <= radii).all():
        return {"sum_of_radii": -np.inf}

    # Containment: each circle must be fully inside [0, 1] x [0, 1]
    # Allow floating-point tolerance for circles touching the boundary
    tol = 1e-9
    is_contained = (
        (radii[:, None] - tol <= centers) & (centers <= 1 - radii[:, None] + tol)
    ).all(axis=1)

    if not is_contained.all():
        return {"sum_of_radii": -np.inf}

    if _circles_overlap(centers, radii):
        return {"sum_of_radii": -np.inf}

    return {"sum_of_radii": float(np.sum(radii))}


# ---------------------------------------------------------------------------
# OpenEvolve-compatible evaluate interface
# ---------------------------------------------------------------------------

N_CIRCLES = 26


def evaluate(program_path):
    """Evaluate a circle packing program.

    Loads the program, calls construct_packing(), validates with DeepMind's
    check_construction, and returns metrics.

    Args:
        program_path: path to a .py file that defines construct_packing()

    Returns:
        dict with at least 'combined_score'
    """
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "construct_packing"):
            return {"combined_score": 0.0, "error": "missing construct_packing()"}

        result = program.construct_packing()

        if not isinstance(result, tuple) or len(result) < 2:
            return {"combined_score": 0.0, "error": "construct_packing must return (centers, radii, ...)"}

        centers = np.asarray(result[0], dtype=float)
        radii = np.asarray(result[1], dtype=float)

        validation = check_construction(centers, radii, N_CIRCLES)
        score = validation["sum_of_radii"]

        if not np.isfinite(score) or score < 0:
            return {"combined_score": 0.0, "sum_of_radii": 0.0, "error": "invalid packing"}

        return {"combined_score": score, "sum_of_radii": score}

    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
