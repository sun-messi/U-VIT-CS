import ml_collections

# accelerate launch --multi_gpu --num_processes 6 --mixed_precision fp16 train_c.py --config=configs/celeba64_uvit_small_cr.py


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.sparsity_enabled = False  # Disabled for CR mode
    config.regularization_enabled = True  # Enable Group L1 regularization

    # === Curriculum Learning Configuration (CR Mode: Curriculum + Regularization) ===
    # Stages define progressive training from high noise (easy) to low noise (hard)
    # t_min=1.0: pure noise, t_min=0.0: clean image
    # lambda_reg: Group L1 regularization strength (applied to patch_embed.proj)
    # Each stage has independent n_steps (not cumulative)
    config.curriculum = d(
        enabled=True,
        stages=[
            # Stage 1: High noise + Strong regularization
            d(t_min=0.3, t_max=1.0, n_steps=10000, sparsity=0.0, lambda_reg=0.1, name="stage1_coarse"),
            # Stage 2: Expand to medium noise + Moderate regularization
            d(t_min=0.2, t_max=1.0, n_steps=10000, sparsity=0.0, lambda_reg=0.01, name="stage2_coarse"),
            # Stage 3: Add fine details + Decreasing regularization
            d(t_min=0.1, t_max=1.0, n_steps=10000, sparsity=0.0, lambda_reg=0.001, name="stage3_fine"),
            # Stage 4: Finer details
            d(t_min=0.07, t_max=1.0, n_steps=10000, sparsity=0.0, lambda_reg=0.0003, name="stage4_finer"),
            # Stage 5: Very fine details + Weak regularization
            d(t_min=0.05, t_max=1.0, n_steps=10000, sparsity=0.0, lambda_reg=0.0001, name="stage5_finer"),
            # # Stage 6: Minimal regularization
            # d(t_min=0.03, t_max=1.0, n_steps=20000, sparsity=0.0, lambda_reg=0.00005, name="stage6_finer"),
            # # Stage 7: Tiny regularization
            # d(t_min=0.01, t_max=1.0, n_steps=20000, sparsity=0.0, lambda_reg=0.00001, name="stage7_finer"),
            # # Stage 8: Full range + No regularization
            # d(t_min=0.0, t_max=1.0, n_steps=100000, sparsity=0.0, lambda_reg=0.0, name="stage8_full"),
        ]
    )

    # Total training steps = sum of all stage n_steps
    total_steps = sum(s['n_steps'] for s in config.curriculum.stages)

    config.train = d(
        n_steps=total_steps,  # 200000
        batch_size=126*2,  # 126 = 6 GPU × 21 per GPU
        mode='uncond',
        log_interval=100,
        eval_interval=5000,
        save_interval=10000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=64,
        patch_size=4,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celeba',
        path='assets/datasets/celeba',
        resolution=64,
    )

    config.sample = d(
        sample_steps=50,  # ODE sampling steps
        n_samples=50,  # Number of samples for FID calculation
        mini_batch_size=50,  # Batch size per GPU
        algorithm='dpm_solver',  # Use ODE sampling
        path=''
    )

    return config
