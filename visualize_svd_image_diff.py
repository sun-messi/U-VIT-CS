"""
SVD on Image Differences between consecutive checkpoints.
1. Compute diff_image = image_high - image_low
2. Compute SVD on diff_image
3. Average singular values across all images
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
    filenames = []
    for f in files:
        img_path = os.path.join(folder, f)
        img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        images.append(img)
        filenames.append(f)
    return np.array(images), filenames


def svd_per_image_avg(images):
    """Compute SVD for each image and average singular values."""
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
    parser.add_argument('--step_pairs', type=str, default='60-50,80-70,100-90,120-110,140-130,160-150')
    args = parser.parse_args()

    # Three methods
    methods = {
        'baseline': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543',
        'c': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_c/20260101_160525',
        'cs': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_cs/20260101_161820',
    }

    method_styles = {
        'baseline': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Baseline'},
        'c': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'label': 'Curriculum (C)'},
        'cs': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'label': 'Curriculum+Sparsity (CS)'},
    }

    # Parse step pairs
    step_pairs = []
    for pair in args.step_pairs.split(','):
        high, low = pair.split('-')
        step_pairs.append((int(high) * 1000, int(low) * 1000))

    print(f"Step pairs: {step_pairs}")
    os.makedirs('analysis_outputs', exist_ok=True)

    # For each step pair, create comparison plot
    for high_step, low_step in step_pairs:
        print(f"\n=== Processing {high_step//1000}k - {low_step//1000}k ===")

        fig, ax = plt.subplots(figsize=(10, 6))

        for method_name, base_dir in methods.items():
            folder_high = os.path.join(base_dir, f'{high_step}_ema')
            folder_low = os.path.join(base_dir, f'{low_step}_ema')

            if not os.path.exists(folder_high) or not os.path.exists(folder_low):
                print(f"  {method_name}: folders not found, skipping")
                continue

            print(f"  Loading {method_name}...")
            images_high, files_high = load_images(folder_high, args.max_images)
            images_low, files_low = load_images(folder_low, args.max_images)

            # Match images by filename
            common_files = set(files_high) & set(files_low)
            if len(common_files) == 0:
                print(f"  {method_name}: no common files, skipping")
                continue

            # Build diff images
            diff_images = []
            high_dict = {f: img for f, img in zip(files_high, images_high)}
            low_dict = {f: img for f, img in zip(files_low, images_low)}

            for f in sorted(common_files)[:args.max_images]:
                diff = high_dict[f] - low_dict[f]
                diff_images.append(diff)

            diff_images = np.array(diff_images)
            print(f"    {len(diff_images)} diff images")

            # SVD on diff images
            avg_svs, std_svs = svd_per_image_avg(np.abs(diff_images))  # Use abs for cleaner SVD

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
        ax.set_title(f'SVD of Image Difference: {high_step//1000}k - {low_step//1000}k', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        output_path = f'analysis_outputs/svd_imgdiff_{high_step//1000}k-{low_step//1000}k.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {output_path}")

    # Combined figure
    n_pairs = len(step_pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, (high_step, low_step) in enumerate(step_pairs):
        ax = axes[idx]

        for method_name, base_dir in methods.items():
            folder_high = os.path.join(base_dir, f'{high_step}_ema')
            folder_low = os.path.join(base_dir, f'{low_step}_ema')

            if not os.path.exists(folder_high) or not os.path.exists(folder_low):
                continue

            images_high, files_high = load_images(folder_high, args.max_images)
            images_low, files_low = load_images(folder_low, args.max_images)

            common_files = set(files_high) & set(files_low)
            if len(common_files) == 0:
                continue

            diff_images = []
            high_dict = {f: img for f, img in zip(files_high, images_high)}
            low_dict = {f: img for f, img in zip(files_low, images_low)}

            for f in sorted(common_files)[:args.max_images]:
                diff = high_dict[f] - low_dict[f]
                diff_images.append(diff)

            diff_images = np.array(diff_images)
            avg_svs, _ = svd_per_image_avg(np.abs(diff_images))

            k = np.arange(1, len(avg_svs) + 1)
            style = method_styles[method_name]
            ax.semilogy(k, avg_svs,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       linewidth=2,
                       label=style['label'])

        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('SV (log)', fontsize=10)
        ax.set_title(f'{high_step//1000}k - {low_step//1000}k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    for idx in range(len(step_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('SVD of Image Differences Across Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = 'analysis_outputs/svd_imgdiff_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined figure to {output_path}")


if __name__ == '__main__':
    main()
