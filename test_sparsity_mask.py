"""
Test script to verify EmbeddingSparsityMask works correctly.
"""
import torch
import sys
sys.path.insert(0, '.')

from train_c import EmbeddingSparsityMask

def test_sparsity_mask():
    print("=" * 60)
    print("Testing EmbeddingSparsityMask")
    print("=" * 60)

    embed_dim = 512
    batch_size = 2
    num_patches = 256

    # Create test input
    x = torch.randn(batch_size, num_patches, embed_dim)
    print(f"\nInput shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")

    # Create mask
    mask = EmbeddingSparsityMask(embed_dim=embed_dim)

    # Test different sparsity levels
    sparsity_levels = [0.0, 0.1, 0.5, 0.8, 0.9]

    for sparsity in sparsity_levels:
        mask.set_sparsity(sparsity)
        active_dims = int(embed_dim * (1.0 - sparsity))

        # Apply mask
        x_masked = mask(x)

        # Check active dimensions
        x_active = x_masked[..., :active_dims]
        x_frozen = x_masked[..., active_dims:]

        print(f"\n--- Sparsity = {sparsity:.1f} ---")
        print(f"Active dims: {active_dims}/{embed_dim}")
        print(f"Active part - mean: {x_active.mean():.4f}, std: {x_active.std():.4f}")
        print(f"Frozen part - mean: {x_frozen.mean():.4f}, std: {x_frozen.std():.4f}")
        print(f"Frozen part all zeros: {torch.allclose(x_frozen, torch.zeros_like(x_frozen))}")

        # Verify gradient behavior
        x_test = torch.randn(1, 1, embed_dim, requires_grad=True)
        mask.set_sparsity(sparsity)
        out = mask(x_test)
        loss = out.sum()
        loss.backward()

        # Check gradient
        grad_active = x_test.grad[..., :active_dims]
        grad_frozen = x_test.grad[..., active_dims:]
        print(f"Gradient active - mean: {grad_active.abs().mean():.4f}")
        print(f"Gradient frozen - all zeros: {torch.allclose(grad_frozen, torch.zeros_like(grad_frozen))}")

    # Test inference mode
    print(f"\n--- Inference Mode ---")
    mask.set_sparsity(0.8)
    mask.set_inference_mode(True)
    x_inference = mask(x)
    print(f"Output mean: {x_inference.mean():.4f}, std: {x_inference.std():.4f}")
    print(f"Inference mode bypasses mask: {torch.allclose(x_inference, x)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_sparsity_mask()
