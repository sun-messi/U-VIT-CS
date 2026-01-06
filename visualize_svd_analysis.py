"""
SVD Analysis for Generated Images across Checkpoints.
Compares eigenvalue distributions to understand feature complexity evolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from tqdm import tqdm


def load_images(folder, max_images=500):
    """Load images from a checkpoint folder."""
    images = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])[:max_images]
    for f in files:
        img_path = os.path.join(folder, f)
        img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)  # [N, H, W, C]


def svd_per_image_avg(images):
    """
    Method 1: Per-image SVD with averaging.
    For each image, compute SVD and get singular values.
    Average across all images.

    Args:
        images: [N, H, W, C] array

    Returns:
        avg_singular_values: [min(H,W)] array of averaged singular values
    """
    N, H, W, C = images.shape
    all_svs = []

    for i in range(N):
        img = images[i]  # [H, W, C]
        # Convert to grayscale for cleaner SVD
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        # SVD
        U, S, Vh = np.linalg.svd(gray, full_matrices=False)
        all_svs.append(S)

    all_svs = np.array(all_svs)  # [N, min(H,W)]
    avg_svs = np.mean(all_svs, axis=0)
    std_svs = np.std(all_svs, axis=0)

    return avg_svs, std_svs


def svd_gram_matrix(images):
    """
    Method 2: Gram Matrix SVD.
    Flatten images and compute SVD on the data matrix.

    Args:
        images: [N, H, W, C] array

    Returns:
        singular_values: singular values of the data matrix
    """
    N, H, W, C = images.shape
    # Flatten each image
    X = images.reshape(N, -1)  # [N, H*W*C]

    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # SVD (use truncated for efficiency)
    # For [N, D] matrix where N << D, we compute SVD efficiently
    if N < X_centered.shape[1]:
        # Compute X @ X.T which is [N, N]
        gram = X_centered @ X_centered.T
        eigenvalues, _ = np.linalg.eigh(gram)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))[::-1]
    else:
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        singular_values = S

    return singular_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543')
    parser.add_argument('--output', type=str, default='analysis_outputs/svd_analysis.png')
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--start_step', type=int, default=20000)
    parser.add_argument('--end_step', type=int, default=150000)
    parser.add_argument('--step_interval', type=int, default=10000)
    args = parser.parse_args()

    # Generate checkpoint steps
    steps = list(range(args.start_step, args.end_step + args.step_interval, args.step_interval))
    print(f"Checkpoints: {steps}")

    # Colormap for different checkpoints
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

    # Store results
    results_per_image = {}
    results_gram = {}

    for step in tqdm(steps, desc='Loading checkpoints'):
        folder = os.path.join(args.base_dir, f'{step}_ema')
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue

        images = load_images(folder, args.max_images)
        print(f"  {step}_ema: loaded {len(images)} images")

        # Method 1: Per-image SVD
        avg_svs, std_svs = svd_per_image_avg(images)
        results_per_image[step] = (avg_svs, std_svs)

        # Method 2: Gram Matrix SVD
        gram_svs = svd_gram_matrix(images)
        results_gram[step] = gram_svs

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Per-image SVD (linear scale)
    ax1 = axes[0, 0]
    for i, step in enumerate(steps):
        if step in results_per_image:
            avg_svs, std_svs = results_per_image[step]
            k = np.arange(1, len(avg_svs) + 1)
            ax1.plot(k, avg_svs, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax1.set_xlabel('k (singular value index)', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title('Per-Image SVD (Averaged) - Linear Scale', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-image SVD (log scale)
    ax2 = axes[0, 1]
    for i, step in enumerate(steps):
        if step in results_per_image:
            avg_svs, std_svs = results_per_image[step]
            k = np.arange(1, len(avg_svs) + 1)
            ax2.semilogy(k, avg_svs, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax2.set_xlabel('k (singular value index)', fontsize=12)
    ax2.set_ylabel('Singular Value (log)', fontsize=12)
    ax2.set_title('Per-Image SVD (Averaged) - Log Scale', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gram Matrix SVD (linear scale)
    ax3 = axes[1, 0]
    for i, step in enumerate(steps):
        if step in results_gram:
            svs = results_gram[step]
            k = np.arange(1, len(svs) + 1)
            ax3.plot(k, svs, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax3.set_xlabel('k (singular value index)', fontsize=12)
    ax3.set_ylabel('Singular Value', fontsize=12)
    ax3.set_title('Gram Matrix SVD - Linear Scale', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gram Matrix SVD (log scale)
    ax4 = axes[1, 1]
    for i, step in enumerate(steps):
        if step in results_gram:
            svs = results_gram[step]
            k = np.arange(1, len(svs) + 1)
            ax4.semilogy(k, svs, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax4.set_xlabel('k (singular value index)', fontsize=12)
    ax4.set_ylabel('Singular Value (log)', fontsize=12)
    ax4.set_title('Gram Matrix SVD - Log Scale', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', ncol=2, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'SVD Analysis: Eigenvalue Distribution across Checkpoints\n{os.path.basename(args.base_dir)}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {args.output}")

    # Additional plot: Normalized singular values (to compare shape rather than magnitude)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes2[0]
    for i, step in enumerate(steps):
        if step in results_per_image:
            avg_svs, _ = results_per_image[step]
            # Normalize by sum
            normalized = avg_svs / avg_svs.sum()
            k = np.arange(1, len(normalized) + 1)
            ax1.plot(k, normalized, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax1.set_xlabel('k', fontsize=12)
    ax1.set_ylabel('Normalized Singular Value', fontsize=12)
    ax1.set_title('Per-Image SVD (Normalized)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes2[1]
    for i, step in enumerate(steps):
        if step in results_gram:
            svs = results_gram[step]
            normalized = svs / svs.sum()
            k = np.arange(1, len(normalized) + 1)
            ax2.plot(k, normalized, color=colors[i], label=f'{step//1000}k', linewidth=1.5)
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('Normalized Singular Value', fontsize=12)
    ax2.set_title('Gram Matrix SVD (Normalized)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Normalized SVD Analysis (Shape Comparison)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output2 = args.output.replace('.png', '_normalized.png')
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output2}")


if __name__ == '__main__':
    main()
