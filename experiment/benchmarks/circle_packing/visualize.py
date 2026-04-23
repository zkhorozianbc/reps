"""
Circle packing visualization.

Adapted from DeepMind's alphaevolve_repository_of_problems notebook.
Can be used standalone or called from the runner after each experiment.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib

# Force a non-GUI backend — we only save PNGs, never open a window. This
# avoids a hard failure on systems where Tk/Qt aren't available (e.g. stock
# uv-managed Python).
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_packing(
    centers: np.ndarray,
    radii: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
    title: str = "Circle Packing",
    save_path: str = None,
):
    """Plot circle packing within a rectangle.

    Args:
        centers: (n, 2) array of circle centers.
        radii: (n,) array of circle radii.
        width: Rectangle width.
        height: Rectangle height.
        title: Plot title.
        save_path: If provided, save to this path instead of showing.
    """
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")

    rect = patches.Rectangle(
        (0, 0), width, height,
        linewidth=2, edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)

    for i in range(len(radii)):
        circle = patches.Circle(
            (centers[i, 0], centers[i, 1]), radii[i],
            edgecolor="royalblue", facecolor="skyblue", alpha=0.6,
        )
        ax.add_patch(circle)
        ax.text(
            centers[i, 0], centers[i, 1], str(i),
            ha="center", va="center", fontsize=8, color="navy",
        )

    margin = max(width, height) * 0.05
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(-margin, height + margin)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved packing visualization to {save_path}")
    else:
        plt.show()


def visualize_from_program(program_path: str, save_path: str = None):
    """Load a program file and visualize its packing."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if hasattr(program, "run_packing"):
        centers, radii, sum_radii = program.run_packing()
    elif hasattr(program, "construct_packing"):
        result = program.construct_packing()
        centers, radii = result[0], result[1]
        sum_radii = float(np.sum(radii))
    else:
        raise RuntimeError("Program has neither run_packing() nor construct_packing()")

    title = f"Circle Packing n={len(radii)}, sum_radii={sum_radii}"
    visualize_packing(centers, radii, title=title, save_path=save_path)
    return sum_radii


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <program.py> [output.png]")
        sys.exit(1)

    program_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    score = visualize_from_program(program_path, save_path)
    print(f"sum_radii = {score}")
