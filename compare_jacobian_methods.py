"""
Compare Jacobian Eigenvectors across different training methods.
Baseline vs Curriculum (C) vs Curriculum+Sparsity (CS)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/sunj11/Documents/U-ViT-fresh')
from libs.uvit import UViT


def load_uvit_checkpoint(ckpt_path, config, device):
    """Load U-ViT model from checkpoint."""
    nnet = UViT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        mlp_time_embed=config['mlp_time_embed'],
        num_classes=config['num_classes'],
    )

    ema_path = os.path.join(ckpt_path, 'nnet_ema.pth')
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location='cpu', weights_only=True)
        nnet.load_state_dict(state_dict)
    else:
        nnet_path = os.path.join(ckpt_path, 'nnet.pth')
        state_dict = torch.load(nnet_path, map_location='cpu', weights_only=True)
        nnet.load_state_dict(state_dict)

    nnet = nnet.to(device)
    nnet.eval()
    return nnet


def rgb_to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def compute_jacobian(img, nnet, timestep, device, work_size=32):
    """Compute Jacobian of denoiser at given image."""
    H, W = work_size, work_size
    N = H * W
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    # Resize image to work_size
    if img.shape[2] != work_size:
        img_small = torch.nn.functional.interpolate(img, size=(work_size, work_size), mode='bilinear')
    else:
        img_small = img

    J = torch.zeros(N, N, device=device)

    for i in tqdm(range(N), desc="Jacobian", leave=False):
        input_img = img_small.clone().detach().requires_grad_(True)
        input_64 = torch.nn.functional.interpolate(input_img, size=(64, 64), mode='bilinear')
        rgb_input = gray_to_rgb(input_64)

        noise_pred = nnet(rgb_input, t)
        gray_output = rgb_to_gray(noise_pred)
        output_small = torch.nn.functional.interpolate(gray_output, size=(work_size, work_size), mode='bilinear')
        output_flat = output_small.view(-1)

        grad = torch.autograd.grad(output_flat[i], input_img, retain_graph=False)[0]
        J[i, :] = grad.view(-1)

    return J


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = 'analysis_outputs/jacobian_compare'
    os.makedirs(output_dir, exist_ok=True)

    # Model config
    config = {
        'img_size': 64,
        'patch_size': 4,
        'embed_dim': 256,
        'depth': 12,
        'num_heads': 8,
        'mlp_ratio': 4,
        'qkv_bias': False,
        'mlp_time_embed': False,
        'num_classes': -1,
    }

    # Three methods
    methods = {
        'baseline': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small/default_20260101_030900/ckpts/100000.ckpt',
        'C': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small_c/default/ckpts/100000.ckpt',
        'CS': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small_cs/default_20260101_073037/ckpts/100000.ckpt',
    }

    # Use a sample image
    img_path = '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543/100000_ema/0.png'
    img = Image.open(img_path).convert('L')
    img = img.resize((32, 32))
    img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    img = img.unsqueeze(0).unsqueeze(0).to(device)

    timestep = 500
    work_size = 32
    N = work_size * work_size

    results = {}

    for method_name, ckpt_path in methods.items():
        print(f"\n=== Processing {method_name} ===")

        nnet = load_uvit_checkpoint(ckpt_path, config, device)
        J = compute_jacobian(img, nnet, timestep, device, work_size)

        I = torch.eye(N, device=device)
        I_minus_J = I - J
        U, S, V = torch.linalg.svd(I_minus_J)

        results[method_name] = {
            'U': U.cpu(),
            'S': S.cpu(),
            'J': J.cpu()
        }

        print(f"  Top 5 singular values: {S[:5].cpu().numpy()}")
        print(f"  Bottom 5 singular values: {S[-5:].cpu().numpy()}")

        del nnet
        torch.cuda.empty_cache()

    # Create comparison plots
    print("\nCreating comparison plots...")

    # 1. Eigenvalue spectrum comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (method_name, data) in enumerate(results.items()):
        ax.semilogy(data['S'].numpy(), color=colors_list[i], linewidth=2, label=method_name)
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Singular Value (log)', fontsize=14)
    ax.set_title('Eigenvalue Spectrum Comparison (I - J)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrum_comparison.png'), dpi=150)
    plt.close()

    # 2. Eigenvector visualization for each method
    n_show = 15
    skip = max(1, N // 50)
    ids = list(range(5, skip * n_show + 5, skip))[:n_show]
    n_cols = 5
    n_rows = 3

    fig, axes = plt.subplots(len(methods), n_cols, figsize=(n_cols * 3, len(methods) * 3.5))
    method_names_list = list(results.keys())
    for row_idx, (method_name, data) in enumerate(results.items()):
        U = data['U']
        S = data['S']
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            j = ids[col_idx]
            if j < U.shape[1]:
                eigvec = U[:, j].numpy().reshape(work_size, work_size)
                ax.imshow(-eigvec, cmap='RdBu', norm=colors.CenteredNorm())
                if row_idx == 0:
                    ax.set_title(f'λ_{j}={S[j].item():.3f}', fontsize=10)
            ax.axis('off')
        # Add row label on the left
        axes[row_idx, 0].annotate(method_name, xy=(-0.3, 0.5), xycoords='axes fraction',
                                   fontsize=14, fontweight='bold', ha='right', va='center')

    plt.suptitle('Eigenvectors Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvectors_comparison.png'), dpi=150)
    plt.close()

    # 3. Effective rank comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (method_name, data) in enumerate(results.items()):
        S = data['S'].numpy()
        # Normalized singular values
        S_norm = S / S.sum()
        # Cumulative energy
        cumsum = np.cumsum(S_norm)
        ax.plot(cumsum, color=colors_list[i], linewidth=2, label=method_name)

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% energy')
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='99% energy')
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Cumulative Energy', fontsize=14)
    ax.set_title('Cumulative Energy of Singular Values', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_energy.png'), dpi=150)
    plt.close()

    # 4. Summary statistics
    print("\n=== Summary Statistics ===")
    for method_name, data in results.items():
        S = data['S'].numpy()
        S_norm = S / S.sum()

        # Effective rank (entropy-based)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
        effective_rank = np.exp(entropy)

        # 90% energy rank
        cumsum = np.cumsum(S_norm)
        rank_90 = np.searchsorted(cumsum, 0.9) + 1

        print(f"{method_name}:")
        print(f"  Max SV: {S[0]:.4f}, Min SV: {S[-1]:.6f}")
        print(f"  Effective Rank: {effective_rank:.2f}")
        print(f"  90% Energy Rank: {rank_90}")
        print(f"  Condition Number: {S[0]/S[-1]:.2f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
