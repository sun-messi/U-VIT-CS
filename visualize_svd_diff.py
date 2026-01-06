"""
Compare SVD differences between consecutive checkpoints.
Shows how singular value distribution changes between epochs.
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

    # Parse step pairs (e.g., "60-50" means 60k - 50k)
    step_pairs = []
    for pair in args.step_pairs.split(','):
        high, low = pair.split('-')
        step_pairs.append((int(high) * 1000, int(low) * 1000))

    print(f"Step pairs to compare: {step_pairs}")

    os.makedirs('analysis_outputs', exist_ok=True)

    # Cache SVD results
    svd_cache = {}

    def get_svd(method_name, step):
        key = (method_name, step)
        if key not in svd_cache:
            folder = os.path.join(methods[method_name], f'{step}_ema')
            if not os.path.exists(folder):
                return None
            images = load_images(folder, args.max_images)
            avg_svs, _ = svd_per_image_avg(images)
            svd_cache[key] = avg_svs
        return svd_cache[key]

    # Create individual plots for each step pair
    for high_step, low_step in step_pairs:
        print(f"\n=== Processing {high_step//1000}k - {low_step//1000}k ===")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Absolute difference
        ax1 = axes[0]
        # Right plot: Relative difference (percentage)
        ax2 = axes[1]

        for method_name in methods.keys():
            svs_high = get_svd(method_name, high_step)
            svs_low = get_svd(method_name, low_step)

            if svs_high is None or svs_low is None:
                print(f"  {method_name}: missing data, skipping")
                continue

            print(f"  {method_name}: loaded")

            diff = svs_high - svs_low
            rel_diff = (svs_high - svs_low) / (svs_low + 1e-10) * 100  # percentage

            k = np.arange(1, len(diff) + 1)
            style = method_styles[method_name]

            ax1.plot(k, diff,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markevery=8,
                    markersize=5,
                    linewidth=2,
                    label=style['label'])

            ax2.plot(k, rel_diff,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markevery=8,
                    markersize=5,
                    linewidth=2,
                    label=style['label'])

        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('k (singular value index)', fontsize=12)
        ax1.set_ylabel('Δ Singular Value', fontsize=12)
        ax1.set_title(f'Absolute Difference: {high_step//1000}k - {low_step//1000}k', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('k (singular value index)', fontsize=12)
        ax2.set_ylabel('Δ Singular Value (%)', fontsize=12)
        ax2.set_title(f'Relative Difference: {high_step//1000}k - {low_step//1000}k', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f'analysis_outputs/svd_diff_{high_step//1000}k-{low_step//1000}k.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {output_path}")

    # Create combined figure showing all differences
    n_pairs = len(step_pairs)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, (high_step, low_step) in enumerate(step_pairs):
        ax = axes[idx]

        for method_name in methods.keys():
            svs_high = get_svd(method_name, high_step)
            svs_low = get_svd(method_name, low_step)

            if svs_high is None or svs_low is None:
                continue

            diff = svs_high - svs_low
            k = np.arange(1, len(diff) + 1)
            style = method_styles[method_name]

            ax.plot(k, diff,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=2,
                   label=style['label'])

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Δ SV', fontsize=10)
        ax.set_title(f'{high_step//1000}k - {low_step//1000}k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide empty subplots
    for idx in range(len(step_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('SVD Difference Between Consecutive Checkpoints', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = 'analysis_outputs/svd_diff_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined figure to {output_path}")

    # Create a summary showing L2 norm of differences
    print("\n=== Summary: L2 Norm of SVD Differences ===")
    summary_data = {method: [] for method in methods.keys()}
    pair_labels = []

    for high_step, low_step in step_pairs:
        pair_labels.append(f'{high_step//1000}k-{low_step//1000}k')
        for method_name in methods.keys():
            svs_high = get_svd(method_name, high_step)
            svs_low = get_svd(method_name, low_step)

            if svs_high is None or svs_low is None:
                summary_data[method_name].append(np.nan)
            else:
                diff = svs_high - svs_low
                l2_norm = np.linalg.norm(diff)
                summary_data[method_name].append(l2_norm)
                print(f"  {pair_labels[-1]} {method_name}: L2 = {l2_norm:.4f}")

    # Plot L2 norm summary
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pair_labels))
    width = 0.25

    for i, (method_name, values) in enumerate(summary_data.items()):
        style = method_styles[method_name]
        ax.bar(x + i*width, values, width,
               label=style['label'], color=style['color'], alpha=0.8)

    ax.set_xlabel('Checkpoint Pair', fontsize=12)
    ax.set_ylabel('L2 Norm of Δ Singular Values', fontsize=12)
    ax.set_title('Change Magnitude Between Consecutive Checkpoints', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = 'analysis_outputs/svd_diff_l2_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved L2 summary to {output_path}")


if __name__ == '__main__':
    main()
