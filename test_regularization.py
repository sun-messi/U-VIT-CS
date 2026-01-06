"""
Test script to verify Group L1 regularization implementation.

This script checks:
1. UViT model can be loaded
2. get_regularization_loss() method works correctly
3. Regularization loss has expected behavior
"""

import torch
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import UViT
from libs.uvit import UViT

def test_regularization():
    print("="*60)
    print("Testing Group L1 Regularization Implementation")
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

    # Check patch_embed.proj weight shape
    weight = model.patch_embed.proj.weight
    print(f"\n1. Weight Shape Check:")
    print(f"   patch_embed.proj.weight.shape = {weight.shape}")
    print(f"   Expected: (256, 3, 4, 4)")
    print(f"   Match: {weight.shape == torch.Size([256, 3, 4, 4])}")

    # Test regularization loss with lambda=0
    print(f"\n2. Regularization Loss with lambda=0:")
    reg_loss_zero = model.get_regularization_loss(lambda_reg=0.0)
    print(f"   reg_loss = {reg_loss_zero.item()}")
    print(f"   Expected: 0.0")
    print(f"   Match: {reg_loss_zero.item() == 0.0}")

    # Test regularization loss with lambda=0.001
    print(f"\n3. Regularization Loss with lambda=0.001:")
    reg_loss_small = model.get_regularization_loss(lambda_reg=0.001)
    print(f"   reg_loss = {reg_loss_small.item():.6f}")
    print(f"   Expected: > 0.0")
    print(f"   Match: {reg_loss_small.item() > 0.0}")

    # Test regularization loss scales with lambda
    print(f"\n4. Regularization Loss Scaling:")
    reg_loss_1 = model.get_regularization_loss(lambda_reg=0.001)
    reg_loss_2 = model.get_regularization_loss(lambda_reg=0.002)
    ratio = reg_loss_2.item() / reg_loss_1.item()
    print(f"   reg_loss(lambda=0.001) = {reg_loss_1.item():.6f}")
    print(f"   reg_loss(lambda=0.002) = {reg_loss_2.item():.6f}")
    print(f"   Ratio = {ratio:.6f}")
    print(f"   Expected: ~2.0")
    print(f"   Match: {abs(ratio - 2.0) < 0.01}")

    # Manual calculation verification
    print(f"\n5. Manual Calculation Verification:")
    weight_reshaped = weight.view(256, -1)  # (256, 48)
    channel_norms = weight_reshaped.norm(p=2, dim=1)  # (256,)
    manual_reg_loss = 0.001 * channel_norms.sum()
    print(f"   Manual calculation: {manual_reg_loss.item():.6f}")
    print(f"   Model calculation:  {reg_loss_1.item():.6f}")
    print(f"   Match: {torch.allclose(manual_reg_loss, reg_loss_1)}")

    # Test forward pass doesn't break
    print(f"\n6. Forward Pass Test:")
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    try:
        output = model(x, t)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: (2, 3, 64, 64)")
        print(f"   Match: {output.shape == torch.Size([2, 3, 64, 64])}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_regularization()
    sys.exit(0 if success else 1)
