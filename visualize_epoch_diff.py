"""
Visualize the difference between consecutive checkpoints.
Shows what each training epoch learned by computing image differences.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def load_images(folder, image_ids, num_images=10):
    """Load first N images from a checkpoint folder."""
    images = []
    for i in image_ids[:num_images]:
        img_path = os.path.join(folder, f'{i}.png')
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
            images.append(img)
        else:
            print(f"Warning: {img_path} not found")
            images.append(None)
    return images


def compute_diff(img1, img2):
    """Compute absolute difference between two images."""
    if img1 is None or img2 is None:
        return None
    diff = np.abs(img2 - img1)
    return diff


def normalize_diff(diff, scale=2.0):
    """Normalize difference for better visualization."""
    if diff is None:
        return None
    # Scale up the difference for visibility
    diff_normalized = np.clip(diff * scale, 0, 255).astype(np.uint8)
    return diff_normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543')
    parser.add_argument('--output', type=str, default='analysis_outputs/epoch_diff_visualization.png')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--start_step', type=int, default=20000)
    parser.add_argument('--end_step', type=int, default=150000)
    parser.add_argument('--step_interval', type=int, default=10000)
    parser.add_argument('--diff_scale', type=float, default=3.0, help='Scale factor for difference visualization')
    args = parser.parse_args()

    # Generate checkpoint steps
    steps = list(range(args.start_step, args.end_step + args.step_interval, args.step_interval))
    print(f"Checkpoints: {steps}")

    # Image IDs to use
    image_ids = list(range(args.num_images))

    # Load all images from all checkpoints
    checkpoint_images = {}
    for step in steps:
        folder = os.path.join(args.base_dir, f'{step}_ema')
        if os.path.exists(folder):
            checkpoint_images[step] = load_images(folder, image_ids, args.num_images)
            print(f"Loaded {step}_ema")
        else:
            print(f"Warning: {folder} not found")
            checkpoint_images[step] = [None] * args.num_images

    # Compute differences between consecutive checkpoints
    diff_pairs = []
    for i in range(1, len(steps)):
        prev_step = steps[i-1]
        curr_step = steps[i]
        diff_pairs.append((prev_step, curr_step))

    print(f"\nDifference pairs: {diff_pairs}")

    # Create visualization
    num_rows = len(diff_pairs)
    num_cols = args.num_images * 3  # prev, curr, diff for each image

    # Actually, let's do: each row is one diff pair, columns are the 10 images (showing diff only)
    # Plus one column for labels

    fig, axes = plt.subplots(num_rows, args.num_images + 1,
                             figsize=(2 * (args.num_images + 1), 2 * num_rows))

    for row_idx, (prev_step, curr_step) in enumerate(diff_pairs):
        prev_images = checkpoint_images[prev_step]
        curr_images = checkpoint_images[curr_step]

        # Label column
        axes[row_idx, 0].text(0.5, 0.5, f'{curr_step//1000}k-{prev_step//1000}k',
                              ha='center', va='center', fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Difference images
        for col_idx in range(args.num_images):
            ax = axes[row_idx, col_idx + 1]

            diff = compute_diff(prev_images[col_idx], curr_images[col_idx])
            diff_vis = normalize_diff(diff, scale=args.diff_scale)

            if diff_vis is not None:
                ax.imshow(diff_vis)
            ax.axis('off')

    plt.suptitle(f'Epoch Differences (scale={args.diff_scale}x)\nBrighter = More Change',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {args.output}")

    # Also create a version showing actual images side by side
    fig2, axes2 = plt.subplots(num_rows, args.num_images * 2 + 1,
                               figsize=(1.5 * (args.num_images * 2 + 1), 1.5 * num_rows))

    for row_idx, (prev_step, curr_step) in enumerate(diff_pairs):
        prev_images = checkpoint_images[prev_step]
        curr_images = checkpoint_images[curr_step]

        # Label column
        axes2[row_idx, 0].text(0.5, 0.5, f'{curr_step//1000}k-{prev_step//1000}k',
                               ha='center', va='center', fontsize=8, fontweight='bold')
        axes2[row_idx, 0].axis('off')

        for col_idx in range(args.num_images):
            # Current image
            ax_curr = axes2[row_idx, col_idx * 2 + 1]
            if curr_images[col_idx] is not None:
                ax_curr.imshow(curr_images[col_idx].astype(np.uint8))
            ax_curr.axis('off')
            if row_idx == 0:
                ax_curr.set_title(f'#{col_idx}', fontsize=8)

            # Difference
            ax_diff = axes2[row_idx, col_idx * 2 + 2]
            diff = compute_diff(prev_images[col_idx], curr_images[col_idx])
            diff_vis = normalize_diff(diff, scale=args.diff_scale)
            if diff_vis is not None:
                ax_diff.imshow(diff_vis)
            ax_diff.axis('off')

    plt.suptitle('Current Image | Difference (from previous checkpoint)', fontsize=12)
    plt.tight_layout()
    output2 = args.output.replace('.png', '_with_images.png')
    plt.savefig(output2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output2}")


if __name__ == '__main__':
    main()
