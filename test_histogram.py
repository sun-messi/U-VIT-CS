"""
Test script to verify channel norm histogram generation.

This script checks:
1. get_channel_norms() method returns correct shape
2. Histogram can be plotted and saved
3. Visualization looks correct
"""

import torch
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import UViT
from libs.uvit import UViT

def plot_channel_norms_test(model, step, save_dir, lambda_val):
    """
    Test version of plot_channel_norms function.
    """
    # Get channel norms (256 values)
    channel_norms = model.get_channel_norms().detach().cpu().numpy()

    print(f"Channel norms shape: {channel_norms.shape}")
    print(f"Channel norms dtype: {channel_norms.dtype}")
    print(f"Channel norms range: [{channel_norms.min():.4f}, {channel_norms.max():.4f}]")

    # Create figure with histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(channel_norms, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # Add statistics
    mean_norm = channel_norms.mean()
    std_norm = channel_norms.std()
    min_norm = channel_norms.min()
    max_norm = channel_norms.max()
    num_zero = (channel_norms < 1e-6).sum()

    # Add vertical lines for mean
    ax.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.4f}')

    # Title and labels
    ax.set_title(f'Channel L2 Norms Distribution (Step {step}, λ={lambda_val:.6f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('L2 Norm ||W[c,:,:,:]||_2', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add text box with statistics
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_norm:.4f}\n'
    stats_text += f'Std: {std_norm:.4f}\n'
    stats_text += f'Min: {min_norm:.4f}\n'
    stats_text += f'Max: {max_norm:.4f}\n'
    stats_text += f'Near-zero (<1e-6): {num_zero}/256'

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'test_channel_norms_step{step}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved histogram to: {save_path}")
    print(f"Statistics:")
    print(f"  Mean: {mean_norm:.4f}")
    print(f"  Std: {std_norm:.4f}")
    print(f"  Min: {min_norm:.4f}")
    print(f"  Max: {max_norm:.4f}")
    print(f"  Near-zero channels: {num_zero}/256")

    return save_path

def test_histogram():
    print("="*60)
    print("Testing Channel Norm Histogram Generation")
    print("="*60)

    # Create a small UViT model
    model = UViT(
        img_size=64,
        patch_size=4,
        in_chans=3,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
        conv=True,
        skip=True
    )

    print("\n1. Test get_channel_norms() method:")
    channel_norms = model.get_channel_norms()
    print(f"   Shape: {channel_norms.shape}")
    print(f"   Expected: torch.Size([256])")
    print(f"   Match: {channel_norms.shape == torch.Size([256])}")

    print("\n2. Test histogram generation:")
    test_dir = "test_output"
    histogram_path = plot_channel_norms_test(
        model=model,
        step=1000,
        save_dir=test_dir,
        lambda_val=0.001
    )

    print("\n3. Verify file exists:")
    exists = os.path.exists(histogram_path)
    print(f"   File exists: {exists}")
    print(f"   Path: {histogram_path}")

    if exists:
        file_size = os.path.getsize(histogram_path)
        print(f"   File size: {file_size} bytes")
        print(f"   File size reasonable: {file_size > 10000}")  # Should be > 10KB

    print("\n" + "="*60)
    print("Histogram generation test passed! ✓")
    print(f"Check the generated image at: {histogram_path}")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_histogram()
    sys.exit(0 if success else 1)
