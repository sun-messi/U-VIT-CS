"""
SVD Analysis - Clean version with fewer checkpoints.
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


def svd_gram_matrix(images):
    N, H, W, C = images.shape
    X = images.reshape(N, -1)
    X_centered = X - X.mean(axis=0, keepdims=True)
    if N < X_centered.shape[1]:
        gram = X_centered @ X_centered.T
        eigenvalues, _ = np.linalg.eigh(gram)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))[::-1]
    else:
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        singular_values = S
    return singular_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='analysis_outputs/svd_clean.png')
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--steps', type=str, default='20000,50000,80000,120000,150000')
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(',')]
    print(f"Selected checkpoints: {steps}")

    # Colors and linestyles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    results_per_image = {}
    results_gram = {}

    for step in tqdm(steps, desc='Loading'):
        folder = os.path.join(args.base_dir, f'{step}_ema')
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue
        images = load_images(folder, args.max_images)
        print(f"  {step}_ema: {len(images)} images")
        results_per_image[step] = svd_per_image_avg(images)
        results_gram[step] = svd_gram_matrix(images)

    # Create clean figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Per-image SVD (linear)
    ax1 = axes[0, 0]
    for i, step in enumerate(steps):
        if step in results_per_image:
            avg_svs, _ = results_per_image[step]
            k = np.arange(1, len(avg_svs) + 1)
            ax1.plot(k, avg_svs, color=colors[i], linestyle=linestyles[i],
                    label=f'{step//1000}k', linewidth=2.5, marker=markers[i],
                    markevery=8, markersize=6)
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title('Per-Image SVD (Linear)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-image SVD (log)
    ax2 = axes[0, 1]
    for i, step in enumerate(steps):
        if step in results_per_image:
            avg_svs, _ = results_per_image[step]
            k = np.arange(1, len(avg_svs) + 1)
            ax2.semilogy(k, avg_svs, color=colors[i], linestyle=linestyles[i],
                        label=f'{step//1000}k', linewidth=2.5, marker=markers[i],
                        markevery=8, markersize=6)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Singular Value (log)', fontsize=12)
    ax2.set_title('Per-Image SVD (Log)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gram Matrix SVD (linear, zoomed)
    ax3 = axes[1, 0]
    for i, step in enumerate(steps):
        if step in results_gram:
            svs = results_gram[step]
            k = np.arange(1, min(100, len(svs)) + 1)
            ax3.plot(k, svs[:100], color=colors[i], linestyle=linestyles[i],
                    label=f'{step//1000}k', linewidth=2.5, marker=markers[i],
                    markevery=10, markersize=6)
    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('Singular Value', fontsize=12)
    ax3.set_title('Gram Matrix SVD (Top 100)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gram Matrix SVD (log)
    ax4 = axes[1, 1]
    for i, step in enumerate(steps):
        if step in results_gram:
            svs = results_gram[step]
            k = np.arange(1, len(svs) + 1)
            ax4.semilogy(k, svs, color=colors[i], linestyle=linestyles[i],
                        label=f'{step//1000}k', linewidth=2.5)
    ax4.set_xlabel('k', fontsize=12)
    ax4.set_ylabel('Singular Value (log)', fontsize=12)
    ax4.set_title('Gram Matrix SVD (Log)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'SVD Analysis\n{os.path.basename(args.base_dir)}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
