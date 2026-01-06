"""
Compare Jacobian Eigenvectors across epochs and methods.
Shows λ_1, λ_5, λ_15, λ_25, λ_45, λ_65, λ_85 for each method at each epoch.
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


def compute_jacobian(img, nnet, timestep, device, work_size=32, chunk_size=64):
    """Compute Jacobian of denoiser using chunked computation to save memory."""
    H, W = work_size, work_size
    N = H * W
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    if img.shape[2] != work_size:
        img_small = torch.nn.functional.interpolate(img, size=(work_size, work_size), mode='bilinear')
    else:
        img_small = img

    J = torch.zeros(N, N, device=device)

    # Process in chunks to balance speed and memory
    n_chunks = (N + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc="Jacobian chunks", leave=False):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, N)

        for i in range(start_idx, end_idx):
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

    output_dir = 'analysis_outputs/jacobian_epochs'
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

    # Methods and their checkpoint paths
    method_paths = {
        'baseline': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small/default_20260101_030900/ckpts',
        'C': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small_c/default/ckpts',
        'CS': '/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small_cs/default_20260101_073037/ckpts',
    }

    # Epochs to compare
    epochs = [60000, 80000, 100000, 120000, 140000, 160000]

    # Eigenvector indices to show
    eigvec_indices = [1, 5, 15, 25, 45, 65, 85]

    # Use a sample image
    img_path = '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543/100000_ema/0.png'
    img = Image.open(img_path).convert('L')
    img = img.resize((32, 32))
    img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    img = img.unsqueeze(0).unsqueeze(0).to(device)

    timestep = 500
    work_size = 32
    N = work_size * work_size

    # Store all results
    all_results = {}

    for epoch in epochs:
        print(f"\n{'='*50}")
        print(f"Processing Epoch {epoch//1000}k")
        print(f"{'='*50}")

        all_results[epoch] = {}

        for method_name, base_path in method_paths.items():
            ckpt_path = os.path.join(base_path, f'{epoch}.ckpt')
            if not os.path.exists(ckpt_path):
                print(f"  {method_name}: {ckpt_path} not found, skipping")
                continue

            print(f"\n  Loading {method_name}...")
            nnet = load_uvit_checkpoint(ckpt_path, config, device)
            J = compute_jacobian(img, nnet, timestep, device, work_size)

            I = torch.eye(N, device=device)
            I_minus_J = I - J
            U, S, V = torch.linalg.svd(I_minus_J)

            all_results[epoch][method_name] = {
                'U': U.cpu(),
                'S': S.cpu(),
            }

            print(f"    Top 5 SVs: {S[:5].cpu().numpy()}")

            del nnet
            torch.cuda.empty_cache()

    # Create comparison figure for each epoch
    print("\n\nCreating comparison figures...")

    n_methods = 3
    n_eigvecs = len(eigvec_indices)

    for epoch in epochs:
        if epoch not in all_results or len(all_results[epoch]) == 0:
            continue

        fig, axes = plt.subplots(n_methods, n_eigvecs, figsize=(n_eigvecs * 2.5, n_methods * 2.8))

        method_names = ['baseline', 'C', 'CS']

        for row_idx, method_name in enumerate(method_names):
            if method_name not in all_results[epoch]:
                for col_idx in range(n_eigvecs):
                    axes[row_idx, col_idx].axis('off')
                    axes[row_idx, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
                continue

            data = all_results[epoch][method_name]
            U = data['U']
            S = data['S']

            for col_idx, eigvec_idx in enumerate(eigvec_indices):
                ax = axes[row_idx, col_idx]

                if eigvec_idx < U.shape[1]:
                    eigvec = U[:, eigvec_idx].numpy().reshape(work_size, work_size)
                    ax.imshow(-eigvec, cmap='RdBu', norm=colors.CenteredNorm())

                    # Title with method name and λ value
                    sv_val = S[eigvec_idx].item()
                    ax.set_title(f'{method_name}\nλ_{eigvec_idx}={sv_val:.3f}', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

                ax.axis('off')

        plt.suptitle(f'Eigenvectors Comparison @ {epoch//1000}k steps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'eigenvectors_{epoch//1000}k.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved eigenvectors_{epoch//1000}k.png")

    # Create combined figures for multiple λ values
    lambda_indices = [1, 5, 10, 20, 30, 50, 70]
    method_names = ['baseline', 'C', 'CS']

    for eigvec_idx in lambda_indices:
        print(f"\nCreating combined epoch comparison for λ_{eigvec_idx}...")

        fig, axes = plt.subplots(n_methods, len(epochs), figsize=(len(epochs) * 2.5, n_methods * 2.8))

        for col_idx, epoch in enumerate(epochs):
            for row_idx, method_name in enumerate(method_names):
                ax = axes[row_idx, col_idx]

                if epoch not in all_results or method_name not in all_results[epoch]:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                    continue

                data = all_results[epoch][method_name]
                U = data['U']
                S = data['S']

                if eigvec_idx < U.shape[1]:
                    eigvec = U[:, eigvec_idx].numpy().reshape(work_size, work_size)
                    ax.imshow(-eigvec, cmap='RdBu', norm=colors.CenteredNorm())

                    sv_val = S[eigvec_idx].item()
                    if row_idx == 0:
                        ax.set_title(f'{epoch//1000}k\nλ_{eigvec_idx}={sv_val:.3f}', fontsize=10)
                    else:
                        ax.set_title(f'λ_{eigvec_idx}={sv_val:.3f}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

                ax.axis('off')

                if col_idx == 0:
                    ax.annotate(method_name, xy=(-0.15, 0.5), xycoords='axes fraction',
                               fontsize=12, fontweight='bold', ha='right', va='center')

        plt.suptitle(f'λ_{eigvec_idx} Eigenvector Evolution Across Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'eigenvector_lambda{eigvec_idx}_epochs.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved eigenvector_lambda{eigvec_idx}_epochs.png")

    # Create eigenvalue spectrum comparison across epochs
    print("\nCreating eigenvalue spectrum comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors_list = {'baseline': '#1f77b4', 'C': '#ff7f0e', 'CS': '#2ca02c'}

    for idx, epoch in enumerate(epochs):
        ax = axes[idx]

        if epoch not in all_results:
            ax.set_title(f'{epoch//1000}k - No data')
            continue

        for method_name in ['baseline', 'C', 'CS']:
            if method_name in all_results[epoch]:
                S = all_results[epoch][method_name]['S'].numpy()
                ax.semilogy(S, color=colors_list[method_name], linewidth=1.5, label=method_name)

        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('Singular Value (log)', fontsize=10)
        ax.set_title(f'{epoch//1000}k steps', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Eigenvalue Spectrum Comparison Across Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrum_epochs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved spectrum_epochs.png")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == '__main__':
    main()
