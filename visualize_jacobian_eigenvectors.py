"""
Jacobian Eigenvector Visualization for U-ViT Denoiser.
Similar to the memorization/generalization paper analysis.

Computes (I - J) where J is the Jacobian of the denoiser,
then visualizes the eigenvectors as images.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import argparse
from tqdm import tqdm

# Add project root to path
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

    # Load EMA weights
    ema_path = os.path.join(ckpt_path, 'nnet_ema.pth')
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location='cpu')
        nnet.load_state_dict(state_dict)
        print(f"Loaded EMA weights from {ema_path}")
    else:
        nnet_path = os.path.join(ckpt_path, 'nnet.pth')
        state_dict = torch.load(nnet_path, map_location='cpu')
        nnet.load_state_dict(state_dict)
        print(f"Loaded weights from {nnet_path}")

    nnet = nnet.to(device)
    nnet.eval()
    return nnet


def rgb_to_gray(x):
    """Convert RGB tensor to grayscale. x: [B, 3, H, W] -> [B, 1, H, W]"""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gray_to_rgb(x):
    """Convert grayscale tensor to RGB. x: [B, 1, H, W] -> [B, 3, H, W]"""
    return x.repeat(1, 3, 1, 1)


def create_denoiser_fn(nnet, timestep, device):
    """Create a function that maps grayscale image to grayscale denoised output."""
    t = torch.tensor([timestep], device=device)

    def denoiser(gray_img):
        """gray_img: [1, 1, H, W] -> [1, 1, H, W]"""
        # Convert to RGB
        rgb_img = gray_to_rgb(gray_img)
        # Run through U-ViT (predicts noise)
        with torch.no_grad():
            noise_pred = nnet(rgb_img, t)
        # Denoised = noisy - predicted_noise (simplified for analysis)
        # For Jacobian analysis, we use the noise prediction directly
        # Convert output to grayscale
        gray_output = rgb_to_gray(noise_pred)
        return gray_output

    return denoiser


def calc_jacobian(input_img, nnet, timestep, device):
    """
    Compute Jacobian of denoiser at input_img.

    Args:
        input_img: [1, 1, H, W] grayscale image
        nnet: U-ViT model
        timestep: diffusion timestep
        device: torch device

    Returns:
        J: [H*W, H*W] Jacobian matrix
    """
    H, W = input_img.shape[2], input_img.shape[3]
    N = H * W

    t = torch.tensor([timestep], device=device, dtype=torch.long)

    # We need to compute ∂output[i] / ∂input[j] for all i, j
    # This requires N forward passes with autograd

    input_flat = input_img.view(1, -1).clone()  # [1, N]
    input_flat.requires_grad_(True)

    J = torch.zeros(N, N, device=device)

    for i in tqdm(range(N), desc="Computing Jacobian"):
        # Reconstruct image from flat
        if input_flat.grad is not None:
            input_flat.grad.zero_()

        img = input_flat.view(1, 1, H, W)
        rgb_img = gray_to_rgb(img)

        # Forward pass
        noise_pred = nnet(rgb_img, t)
        gray_output = rgb_to_gray(noise_pred)
        output_flat = gray_output.view(-1)

        # Compute gradient of output[i] w.r.t. input
        grad = torch.autograd.grad(output_flat[i], input_flat, retain_graph=True)[0]
        J[i, :] = grad.view(-1)

    return J


def calc_jacobian_efficient(input_img, nnet, timestep, device, batch_size=64):
    """
    More efficient Jacobian computation using batched backward passes.
    Uses the identity: J = dL/dx where L = sum(output * e_i) for each unit vector e_i
    """
    H, W = input_img.shape[2], input_img.shape[3]
    N = H * W

    t = torch.tensor([timestep], device=device, dtype=torch.long)

    J = torch.zeros(N, N, device=device)

    for start_idx in tqdm(range(0, N, batch_size), desc="Computing Jacobian"):
        end_idx = min(start_idx + batch_size, N)
        batch_size_actual = end_idx - start_idx

        # Create input that requires grad
        input_batch = input_img.clone().requires_grad_(True)
        rgb_img = gray_to_rgb(input_batch)

        # Forward
        noise_pred = nnet(rgb_img, t)
        gray_output = rgb_to_gray(noise_pred)
        output_flat = gray_output.view(-1)

        # Compute gradients for this batch of outputs
        for i in range(batch_size_actual):
            idx = start_idx + i
            if input_batch.grad is not None:
                input_batch.grad.zero_()

            grad = torch.autograd.grad(output_flat[idx], input_batch, retain_graph=True)[0]
            J[idx, :] = grad.view(-1)

    return J


def visualize_eigenvectors(U, S, img_size, output_path, n_show=21, skip=12):
    """
    Visualize eigenvectors as images.

    Args:
        U: [N, N] matrix of eigenvectors (columns)
        S: [N] singular values
        img_size: (H, W) tuple
        output_path: path to save figure
        n_show: number of eigenvectors to show
        skip: show every skip-th eigenvector
    """
    H, W = img_size

    # Select eigenvector indices (similar to paper: range(5, 12*21, 12))
    ids = list(range(5, skip * n_show + 5, skip))[:n_show]

    n_cols = 7
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    plt.tight_layout()
    axs = axs.ravel()

    for i in range(n_show):
        j = ids[i]
        if j >= U.shape[1]:
            axs[i].axis('off')
            continue

        # Reshape eigenvector to image
        eigvec = U[:, j].cpu().numpy().reshape(H, W)

        # Plot with centered colormap (red-blue)
        im = axs[i].imshow(-eigvec, cmap='RdBu', norm=colors.CenteredNorm())
        axs[i].set_title(f'λ_{j} = {S[j].cpu().item():.3f}', fontsize=14)
        axs[i].axis('off')

    # Hide remaining axes
    for i in range(n_show, len(axs)):
        axs[i].axis('off')

    plt.suptitle('Eigenvectors of (I - J)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvector visualization to {output_path}")


def visualize_projection_and_eigenvalues(U, S, img, output_path):
    """
    Visualize image projection onto eigenvectors and eigenvalues.
    Similar to paper's dual-axis plot.
    """
    img_flat = img.view(-1).cpu()

    # Compute projections <x, e_k>
    projections = []
    for j in range(U.shape[1]):
        proj = torch.dot(U[:, j].cpu(), img_flat).item()
        projections.append(abs(proj))

    projections = np.array(projections)
    eigenvalues = S.cpu().numpy()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Left axis: projections
    color1 = 'tab:blue'
    ax1.set_xlabel('k', fontsize=14)
    ax1.plot(projections, '*', color=color1, alpha=0.7, markersize=3)
    ax1.set_ylabel(r'$|\langle x, e_k \rangle|$', color=color1, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right axis: eigenvalues
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(r'$\lambda_k$', color=color2, fontsize=14)
    ax2.plot(eigenvalues, '.', color=color2, alpha=0.7, markersize=3)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Image Projection and Eigenvalues', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved projection plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/sunj11/Documents/U-ViT-fresh/workdir/celeba64_uvit_small/default_20260101_030900/ckpts/100000.ckpt')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs/jacobian')
    parser.add_argument('--timestep', type=int, default=500, help='Diffusion timestep (0-1000)')
    parser.add_argument('--noise_level', type=float, default=0.5, help='Noise level for test image')
    parser.add_argument('--image_path', type=str, default=None, help='Optional: path to test image')
    parser.add_argument('--img_size', type=int, default=32, help='Image size for Jacobian (smaller = faster)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Model config for celeba64_uvit_small
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

    print("Loading model...")
    nnet = load_uvit_checkpoint(args.ckpt_path, config, device)

    # Create or load test image
    if args.image_path and os.path.exists(args.image_path):
        img = Image.open(args.image_path).convert('L')  # Grayscale
        img = img.resize((args.img_size, args.img_size))
        img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img = img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    else:
        # Create a simple test image (e.g., from noise or a pattern)
        print(f"Creating test image of size {args.img_size}x{args.img_size}")
        # Use a face-like pattern or random
        torch.manual_seed(42)
        clean = torch.randn(1, 1, args.img_size, args.img_size, device=device) * 0.3
        noise = torch.randn_like(clean) * args.noise_level
        img = clean + noise
        img = img.clamp(-1, 1)

    # For U-ViT which expects 64x64, we need to resize
    if args.img_size != 64:
        print(f"Note: U-ViT expects 64x64. Using {args.img_size}x{args.img_size} for Jacobian analysis.")
        # We'll compute Jacobian at the specified size, then show results
        # But for actual model forward, we need to resize

    # For simplicity, let's work with a smaller size for Jacobian computation
    # Then scale up for visualization

    print(f"\nComputing Jacobian at timestep {args.timestep}...")
    print(f"Image size: {args.img_size}x{args.img_size} = {args.img_size**2} dimensions")
    print(f"Jacobian size: {args.img_size**2}x{args.img_size**2} = {args.img_size**4} elements")

    # For a 64x64 image, Jacobian would be 4096x4096 = 16M elements
    # For a 32x32 image, Jacobian would be 1024x1024 = 1M elements
    # For a 16x16 image, Jacobian would be 256x256 = 65K elements

    # Use smaller size for efficiency
    work_size = min(args.img_size, 32)
    if args.img_size > 32:
        print(f"Resizing to {work_size}x{work_size} for efficient Jacobian computation")
        img_small = torch.nn.functional.interpolate(img, size=(work_size, work_size), mode='bilinear')
    else:
        img_small = img

    # We need to modify the approach since U-ViT expects 64x64
    # Let's create a wrapper that handles resizing

    H, W = work_size, work_size
    N = H * W
    t = torch.tensor([args.timestep], device=device, dtype=torch.long)

    print(f"\nActual computation size: {work_size}x{work_size}")

    # Compute Jacobian
    J = torch.zeros(N, N, device=device)

    for i in tqdm(range(N), desc="Computing Jacobian rows"):
        # Create input with gradient tracking
        input_img = img_small.clone().detach().requires_grad_(True)

        # Resize to 64x64 for U-ViT
        input_64 = torch.nn.functional.interpolate(input_img, size=(64, 64), mode='bilinear')
        rgb_input = gray_to_rgb(input_64)

        # Forward
        noise_pred = nnet(rgb_input, t)
        gray_output = rgb_to_gray(noise_pred)

        # Resize output back to work_size
        output_small = torch.nn.functional.interpolate(gray_output, size=(work_size, work_size), mode='bilinear')
        output_flat = output_small.view(-1)

        # Compute gradient of output[i] w.r.t input
        if i < N:
            grad = torch.autograd.grad(output_flat[i], input_img, retain_graph=False)[0]
            J[i, :] = grad.view(-1)

    print("\nComputing SVD of (I - J)...")
    I = torch.eye(N, device=device)
    I_minus_J = I - J

    U, S, V = torch.linalg.svd(I_minus_J)

    print(f"Top 10 singular values: {S[:10].cpu().numpy()}")
    print(f"Bottom 10 singular values: {S[-10:].cpu().numpy()}")

    # Visualize eigenvectors
    step_name = os.path.basename(args.ckpt_path).replace('.ckpt', '')

    eigvec_path = os.path.join(args.output_dir, f'eigenvectors_{step_name}_t{args.timestep}.png')
    visualize_eigenvectors(U, S, (work_size, work_size), eigvec_path, n_show=21, skip=max(1, N//100))

    proj_path = os.path.join(args.output_dir, f'projection_{step_name}_t{args.timestep}.png')
    visualize_projection_and_eigenvalues(U, S, img_small, proj_path)

    # Save eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(S.cpu().numpy(), 'b-', linewidth=1)
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Singular Value (log)', fontsize=14)
    ax.set_title(f'Eigenvalue Spectrum of (I - J)\nStep: {step_name}, Timestep: {args.timestep}', fontsize=14)
    ax.grid(True, alpha=0.3)
    spectrum_path = os.path.join(args.output_dir, f'spectrum_{step_name}_t{args.timestep}.png')
    plt.savefig(spectrum_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrum to {spectrum_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
