"""Visualize circle packing result."""
import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def load_program(path):
    spec = importlib.util.spec_from_file_location("program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def visualize(centers, radii, title=None, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw unit square
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # Draw circles
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(radii)))
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, facecolor=colors[i], edgecolor='black',
                       linewidth=0.5, alpha=0.7)
        ax.add_patch(circle)
        # Label with index and radius
        fontsize = max(5, min(10, radius * 80))
        ax.text(center[0], center[1], f'{i}', ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    sum_radii = np.sum(radii)
    title_text = title or f'Circle Packing (n={len(radii)})'
    ax.set_title(f'{title_text}\nsum_radii = {sum_radii:.10f}', fontsize=14)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    plt.show()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'experiment/results/circle_sonnet_reps/best_program.py'
    save = sys.argv[2] if len(sys.argv) > 2 else None

    mod = load_program(path)
    centers, radii, sum_radii = mod.construct_packing()
    print(f'n = {len(radii)}')
    print(f'sum_radii = {sum_radii:.14f}')
    print(f'min radius = {np.min(radii):.6f}')
    print(f'max radius = {np.max(radii):.6f}')

    visualize(centers, radii, title='REPS Circle Packing (n=26)', save_path=save)
