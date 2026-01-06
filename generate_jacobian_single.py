"""
批量生成单独的 Jacobian 特征向量图片。
支持缓存机制，避免重复计算 SVD。

输出结构：
- jacobian_cache/{method}/{epoch}/U.pt, S.pt  # SVD 缓存
- jacobian_single/{method}/{epoch}/lambda_XX.png  # 单独图片
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


# ========== 配置区 ==========
METHODS = {
    'baseline': 'workdir/celeba64_uvit_small/default_20260101_030900/ckpts',
    'C': 'workdir/celeba64_uvit_small_c/default/ckpts',
    'CS': 'workdir/celeba64_uvit_small_cs/default_20260101_073037/ckpts',
}

EPOCHS = list(range(10000, 210000, 10000))  # 10k-200k
LAMBDA_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80]

CACHE_DIR = 'analysis_outputs/jacobian_cache'
OUTPUT_DIR = 'analysis_outputs/jacobian_single'

# Model config
MODEL_CONFIG = {
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

WORK_SIZE = 32
TIMESTEP = 500


# ========== 工具函数 ==========
def rgb_to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


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


def compute_jacobian(img, nnet, timestep, device, work_size=32):
    """Compute Jacobian of denoiser."""
    N = work_size * work_size
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    if img.shape[2] != work_size:
        img_small = torch.nn.functional.interpolate(img, size=(work_size, work_size), mode='bilinear')
    else:
        img_small = img

    J = torch.zeros(N, N, device=device)

    for i in tqdm(range(N), desc="Computing Jacobian", leave=False):
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


def cache_exists(cache_path):
    """Check if cache exists."""
    return os.path.exists(os.path.join(cache_path, 'U.pt')) and \
           os.path.exists(os.path.join(cache_path, 'S.pt'))


def load_cache(cache_path):
    """Load cached U and S."""
    U = torch.load(os.path.join(cache_path, 'U.pt'), weights_only=True)
    S = torch.load(os.path.join(cache_path, 'S.pt'), weights_only=True)
    return U, S


def save_cache(cache_path, U, S):
    """Save U and S to cache."""
    os.makedirs(cache_path, exist_ok=True)
    torch.save(U.cpu(), os.path.join(cache_path, 'U.pt'))
    torch.save(S.cpu(), os.path.join(cache_path, 'S.pt'))


def save_eigenvector_image(eigvec, work_size, output_path):
    """Save eigenvector as 32x32 image without title."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    eigvec_2d = eigvec.reshape(work_size, work_size)

    # Create figure with exact pixel size
    fig, ax = plt.subplots(figsize=(1, 1), dpi=32)
    ax.imshow(-eigvec_2d, cmap='RdBu', norm=colors.CenteredNorm())
    ax.axis('off')

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(output_path, dpi=32, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load sample image
    img_path = '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543/100000_ema/0.png'
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('L')
        img = img.resize((WORK_SIZE, WORK_SIZE))
        img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img = img.unsqueeze(0).unsqueeze(0).to(device)
    else:
        print(f"Warning: Sample image not found at {img_path}")
        print("Using random noise as input")
        img = torch.randn(1, 1, WORK_SIZE, WORK_SIZE, device=device)

    N = WORK_SIZE * WORK_SIZE

    # Count total work
    total_models = 0
    for method, ckpt_base in METHODS.items():
        for epoch in EPOCHS:
            ckpt_path = os.path.join(ckpt_base, f'{epoch}.ckpt')
            if os.path.exists(ckpt_path):
                total_models += 1

    print(f"\nFound {total_models} checkpoints to process")
    print(f"Lambda indices: {LAMBDA_INDICES}")
    print(f"Total images to generate: {total_models * len(LAMBDA_INDICES)}")

    processed = 0
    for method, ckpt_base in METHODS.items():
        print(f"\n{'='*50}")
        print(f"Processing method: {method}")
        print(f"{'='*50}")

        for epoch in EPOCHS:
            ckpt_path = os.path.join(ckpt_base, f'{epoch}.ckpt')
            if not os.path.exists(ckpt_path):
                continue

            epoch_str = f"{epoch // 1000:03d}k"
            cache_path = os.path.join(CACHE_DIR, method, epoch_str)
            output_base = os.path.join(OUTPUT_DIR, method, epoch_str)

            print(f"\n  [{method}] {epoch_str}:")

            # Check/load cache
            if cache_exists(cache_path):
                print(f"    Loading from cache...")
                U, S = load_cache(cache_path)
            else:
                print(f"    Computing Jacobian + SVD...")
                nnet = load_uvit_checkpoint(ckpt_path, MODEL_CONFIG, device)

                J = compute_jacobian(img, nnet, TIMESTEP, device, WORK_SIZE)

                I = torch.eye(N, device=device)
                I_minus_J = I - J
                U, S, _ = torch.linalg.svd(I_minus_J)

                # Save cache
                save_cache(cache_path, U, S)
                print(f"    Cached to {cache_path}")

                # Cleanup
                del nnet, J, I_minus_J
                torch.cuda.empty_cache()

                U = U.cpu()
                S = S.cpu()

            # Generate images
            print(f"    Generating {len(LAMBDA_INDICES)} images...")
            for lambda_idx in LAMBDA_INDICES:
                if lambda_idx < U.shape[1]:
                    output_path = os.path.join(output_base, f'lambda_{lambda_idx:02d}.png')
                    eigvec = U[:, lambda_idx].numpy()
                    save_eigenvector_image(eigvec, WORK_SIZE, output_path)

            processed += 1
            print(f"    Done ({processed}/{total_models})")

    print(f"\n{'='*50}")
    print(f"All done! Generated images in: {OUTPUT_DIR}")
    print(f"Cache stored in: {CACHE_DIR}")


if __name__ == '__main__':
    main()
