"""
Compare SVD across different methods (baseline, c, cs) at the same epoch.
Only plots Per-Image SVD (Log scale).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse


def load_images(folder, max_images=500):
    images = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])[:max_images]
    for f in files:
        img_path = os.path.join(folder, f)
        img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)


def svd_per_image_avg(images):
    N, H, W, C = images.shape
    all_svs = []
    for i in range(N):
        img = images[i]
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        U, S, Vh = np.linalg.svd(gray, full_matrices=False)
        all_svs.append(S)
    all_svs = np.array(all_svs)
    return np.mean(all_svs, axis=0), np.std(all_svs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--steps', type=str, default='60000,80000,100000,120000,140000,160000')
    args = parser.parse_args()

    # Three methods
    methods = {
        'baseline': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543',
        'c': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_c/20260101_160525',
        'cs': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_cs/20260101_161820',
    }

    # Colors and styles for methods
    method_styles = {
        'baseline': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Baseline'},
        'c': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Curriculum (C)'},
        'cs': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'label': 'Curriculum+Sparsity (CS)'},
    }

    steps = [int(s) for s in args.steps.split(',')]
    print(f"Steps to compare: {steps}")

    # Create output directory
    os.makedirs('analysis_outputs', exist_ok=True)

    # For each step, create a comparison plot
    for step in steps:
        print(f"\n=== Processing {step//1000}k ===")

        fig, ax = plt.subplots(figsize=(10, 6))

        for method_name, base_dir in methods.items():
            folder = os.path.join(base_dir, f'{step}_ema')
            if not os.path.exists(folder):
                print(f"  {method_name}: {folder} not found, skipping")
                continue

            print(f"  Loading {method_name}...")
            images = load_images(folder, args.max_images)
            avg_svs, std_svs = svd_per_image_avg(images)

            k = np.arange(1, len(avg_svs) + 1)
            style = method_styles[method_name]
            ax.semilogy(k, avg_svs,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       marker=style['marker'],
                       markevery=8,
                       markersize=6,
                       linewidth=2.5,
                       label=style['label'])

        ax.set_xlabel('k (singular value index)', fontsize=14)
        ax.set_ylabel('Singular Value (log scale)', fontsize=14)
        ax.set_title(f'Per-Image SVD Comparison @ {step//1000}k steps', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        output_path = f'analysis_outputs/svd_compare_{step//1000}k.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {output_path}")

    # Also create a combined figure with all steps
    n_steps = len(steps)
    n_cols = 3
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, step in enumerate(steps):
        ax = axes[idx]

        for method_name, base_dir in methods.items():
            folder = os.path.join(base_dir, f'{step}_ema')
            if not os.path.exists(folder):
                continue

            images = load_images(folder, args.max_images)
            avg_svs, _ = svd_per_image_avg(images)

            k = np.arange(1, len(avg_svs) + 1)
            style = method_styles[method_name]
            ax.semilogy(k, avg_svs,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       linewidth=2,
                       label=style['label'])

        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('SV (log)', fontsize=10)
        ax.set_title(f'{step//1000}k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide empty subplots
    for idx in range(len(steps), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Per-Image SVD: Method Comparison Across Training Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = 'analysis_outputs/svd_compare_all_steps.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined figure to {output_path}")


if __name__ == '__main__':
    main()
