"""Strict DeepMind verifier for circle packing best programs.

Uses the EXACT check_construction from
experiment/benchmarks/circle_packing/packing_circle_max_sum_of_radii.py
(lines 21-67) — no tolerance, no modifications.

Usage:
    python verify_strict.py <path_to_best_program.py> [<path_to_best_program.py> ...]
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# VERBATIM copy from packing_circle_max_sum_of_radii.py (DeepMind reference)
# ---------------------------------------------------------------------------

def _circles_overlap(centers, radii):
  """Protected function to compute max radii."""
  n = centers.shape[0]

  for i in range(n):
    for j in range(i + 1, n):
      dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
      if radii[i] + radii[j] > dist:
        return True

  return False

def check_construction(centers, radii, n) -> dict:
  """Evaluates circle packing for maximizing sum of radii in unit square."""

  # General checks for the whole array
  if centers.shape != (n, 2) or not np.isfinite(centers).all():
    print(
        "Error: The 'centers' array has an invalid shape or non-finite values."
    )
    return {'sum_of_radii': -np.inf}

  # --- Start of the modified geometric check ---

  # 1. Check each circle individually to see if it's contained
  is_contained = (
      (radii[:, None] <= centers) & (centers <= 1 - radii[:, None])
  ).all(axis=1)

  # 2. If not all of them are contained...
  if not is_contained.all():
    return {'sum_of_radii': -np.inf}

  if (
      radii.shape != (n,)
      or not np.isfinite(radii).all()
      or not (0 <= radii).all()
  ):
    print('radii bad shape')
    return {'sum_of_radii': -np.inf}

  if _circles_overlap(centers, radii):
    print('circles overlap')
    return {'sum_of_radii': -np.inf}

  print("The circles are disjoint and lie inside the unit square.")
  return {'sum_of_radii': float(np.sum(radii))}


# ---------------------------------------------------------------------------
# Load a best_program.py, run it, feed (centers, radii) into check_construction
# ---------------------------------------------------------------------------

def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_program(mod):
    if hasattr(mod, "run_packing"):
        return mod.run_packing()
    if hasattr(mod, "construct_packing"):
        return mod.construct_packing()
    raise RuntimeError("Program exposes neither run_packing() nor construct_packing()")


def verify_one(path: Path) -> None:
    print(f"\n=== {path} ===")
    mod = _load_module(path)
    result = _call_program(mod)

    if len(result) == 3:
        centers, radii, reported = result
    elif len(result) >= 2:
        centers, radii = result[0], result[1]
        reported = float(np.sum(np.asarray(radii)))
    else:
        raise RuntimeError(f"Unexpected return shape: {len(result)}")

    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    print(f"shapes: centers={centers.shape}, radii={radii.shape}")
    print(f"reported sum_radii: {reported!r}")

    verdict = check_construction(centers, radii, n=26)
    strict_sum = verdict["sum_of_radii"]

    if np.isfinite(strict_sum):
        print(f"STRICT PASS. sum_of_radii = {strict_sum!r}")
        diff = reported - strict_sum
        print(f"reported - strict = {diff!r}")
    else:
        print("STRICT FAIL. (See diagnostics above.)")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        verify_one(Path(arg).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
