import sde
import ml_collections
import torch
import torch.nn as nn
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb


class EmbeddingSparsityMask(nn.Module):
    """
    Apply progressive sparsity mask to embeddings.

    Masks are applied at two locations:
    1. After PatchEmbed: (B, 256, 512) -> mask last dims to 0
    2. Before decoder: (B, 256+extras, 512) -> mask last dims to 0

    This creates hierarchical feature learning:
    - Early stages (high noise, high sparsity): Only first K dims active
    - Later stages (low noise, low sparsity): More dims activated
    - Inference: All dims active (no mask)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.current_sparsity = 0.0
        self.is_inference = False

    def set_sparsity(self, sparsity):
        """Update current sparsity level."""
        self.current_sparsity = float(sparsity)

    def set_inference_mode(self, is_inference):
        """Set inference mode (disables masking)."""
        self.is_inference = is_inference

    def forward(self, x):
        """
        Apply sparsity mask to embeddings.

        Args:
            x: [..., embed_dim] embeddings

        Returns:
            Masked embeddings with same shape
        """
        # No masking during inference or when sparsity=0
        if self.is_inference or self.current_sparsity == 0.0:
            return x

        # Calculate active dimensions
        active_dims = int(self.embed_dim * (1.0 - self.current_sparsity))
        active_dims = max(1, min(active_dims, self.embed_dim))

        # Create and apply mask
        mask = torch.zeros(self.embed_dim, device=x.device, dtype=x.dtype)
        mask[:active_dims] = 1.0

        return x * mask  # Zero out inactive dimensions


def LSimple_curriculum(score_model, x0, pred='noise_pred', t_min=0.0, t_max=1.0, **kwargs):
    """
    Curriculum Learning version of LSimple loss.

    Samples timesteps uniformly from [t_min, t_max] instead of [0, 1].
    When t_min=0.0 and t_max=1.0, this is equivalent to the original LSimple.

    Args:
        score_model: ScoreModel instance
        x0: clean images [B, C, H, W]
        pred: prediction type ('noise_pred' or 'x0_pred')
        t_min: minimum time ratio (0.0 = clean, 1.0 = noise)
        t_max: maximum time ratio
        **kwargs: additional arguments (e.g., y for conditional training)

    Returns:
        loss: per-sample loss tensor [B]
    """
    # Sample timesteps uniformly from [t_min, t_max]
    t = torch.rand(x0.shape[0], device=x0.device) * (t_max - t_min) + t_min

    # Forward diffusion: q(x_t | x_0)
    mean, std = score_model.sde.marginal_prob(x0, t)
    eps = torch.randn_like(x0)
    xt = mean + sde.stp(std, eps)

    # Compute loss based on prediction type
    if pred == 'noise_pred':
        noise_pred = score_model.noise_pred(xt, t, **kwargs)
        return sde.mos(eps - noise_pred)
    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, **kwargs)
        return sde.mos(x0 - x0_pred)
    else:
        raise NotImplementedError(pred)


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision

    # === Read Curriculum Config BEFORE Freezing ===
    curriculum_config = config.get('curriculum', None)
    curriculum_enabled = curriculum_config and curriculum_config.get('enabled', False)

    # Build step-to-stage mapping
    step_to_stage = {}
    current_t_min = 0.0
    current_t_max = 1.0
    current_sparsity = 0.0

    if curriculum_enabled:
        cumulative_steps = 0
        for stage_idx, stage in enumerate(curriculum_config['stages']):
            stage_steps = stage['n_steps']
            for s in range(stage_steps):
                step_to_stage[cumulative_steps + s] = stage_idx
            cumulative_steps += stage_steps

    # Remove curriculum from config before freezing to avoid FrozenConfigDict validation error
    if 'curriculum' in config:
        del config['curriculum']

    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    # assert os.path.exists(dataset.fid_stat)  # Skip FID stat check
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # Log curriculum configuration
    if curriculum_enabled and accelerator.is_main_process:
        logging.info(f"[Curriculum] Enabled with {len(curriculum_config['stages'])} stages")
        for i, stage in enumerate(curriculum_config['stages']):
            sparsity = stage.get('sparsity', 0.0)
            active_dims = int(config.nnet.embed_dim * (1.0 - sparsity))
            logging.info(f"  Stage {i+1}: t_min={stage.get('t_min', 0.0):.2f}, "
                       f"t_max={stage.get('t_max', 1.0):.2f}, "
                       f"n_steps={stage.get('n_steps', 0)}, "
                       f"sparsity={sparsity:.1f} (active={active_dims}/{config.nnet.embed_dim}), "
                       f"name={stage.get('name', 'unnamed')}")
    elif accelerator.is_main_process:
        logging.info("[Curriculum] Disabled (using standard training)")

    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    # === Sparsity Mask Setup ===
    embed_mask = EmbeddingSparsityMask(embed_dim=config.nnet.embed_dim).to(device)
    embed_mask_ema = EmbeddingSparsityMask(embed_dim=config.nnet.embed_dim).to(device)

    # Hook function to apply mask after PatchEmbed
    def patch_embed_hook(module, input, output):
        """Apply mask after patch embedding."""
        if hasattr(module, '_sparsity_mask'):
            masked_output = module._sparsity_mask(output)

            # Debug: Print verification on first batch of first step
            if hasattr(module, '_debug_printed') and not module._debug_printed:
                sparsity = module._sparsity_mask.current_sparsity
                if sparsity > 0:
                    # Check how many zeros in last dimension
                    B, L, D = masked_output.shape
                    # Count zeros per embedding dimension across all patches
                    zero_counts = (masked_output == 0).float().mean(dim=(0, 1))  # [D]
                    num_zero_dims = (zero_counts > 0.99).sum().item()
                    active_dims = int(D * (1.0 - sparsity))

                    if accelerator.is_main_process:
                        logging.info(f"\n{'='*60}")
                        logging.info(f"[Sparsity Verification] PatchEmbed Output Shape: {masked_output.shape}")
                        logging.info(f"  Expected active dims: {active_dims}/{D}")
                        logging.info(f"  Actual zero dims: {num_zero_dims}/{D}")
                        logging.info(f"  First 5 active dims mean: {masked_output[0, 0, :5].tolist()}")
                        logging.info(f"  Last 5 frozen dims (should be 0): {masked_output[0, 0, -5:].tolist()}")
                        logging.info(f"{'='*60}\n")
                    module._debug_printed = True

            return masked_output
        return output

    # Hook function to apply mask before decoder
    def norm_hook(module, input, output):
        """Apply mask before decoder (after final norm)."""
        if hasattr(module, '_sparsity_mask'):
            return module._sparsity_mask(output)
        return output

    # Register hooks and attach masks
    nnet.patch_embed.register_forward_hook(patch_embed_hook)
    nnet.patch_embed._sparsity_mask = embed_mask
    nnet.patch_embed._debug_printed = False  # For debug printing
    nnet.norm.register_forward_hook(norm_hook)
    nnet.norm._sparsity_mask = embed_mask

    nnet_ema.patch_embed.register_forward_hook(patch_embed_hook)
    nnet_ema.patch_embed._sparsity_mask = embed_mask_ema
    nnet_ema.patch_embed._debug_printed = False  # For debug printing
    nnet_ema.norm.register_forward_hook(norm_hook)
    nnet_ema.norm._sparsity_mask = embed_mask_ema

    if accelerator.is_main_process:
        logging.info(f"[Sparsity] Mask registered at PatchEmbed and norm (embed_dim={config.nnet.embed_dim})")


    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        # Use curriculum loss function with current time range
        if config.train.mode == 'uncond':
            loss = LSimple_curriculum(score_model, _batch, pred=config.pred,
                                    t_min=current_t_min, t_max=current_t_max)
        elif config.train.mode == 'cond':
            loss = LSimple_curriculum(score_model, _batch[0], pred=config.pred, y=_batch[1],
                                    t_min=current_t_min, t_max=current_t_max)
        else:
            raise NotImplementedError(config.train.mode)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    def eval_step(n_samples, sample_steps, algorithm):
        # Disable sparsity mask during inference
        embed_mask_ema.set_inference_mode(True)

        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError

            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(
                    _x_init,
                    steps=sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        # Re-enable training mode for sparsity mask
        embed_mask_ema.set_inference_mode(False)

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    current_stage_idx = -1
    metrics = {}  # Initialize to avoid UnboundLocalError when resuming from completed checkpoint

    while train_state.step < config.train.n_steps:
        # === Curriculum Stage Update ===
        if curriculum_enabled:
            stage_idx = step_to_stage.get(train_state.step, -1)
            if stage_idx >= 0 and stage_idx != current_stage_idx:
                # Stage transition detected
                current_stage_idx = stage_idx
                stage = curriculum_config['stages'][stage_idx]
                current_t_min = stage.get('t_min', 0.0)
                current_t_max = stage.get('t_max', 1.0)
                current_sparsity = stage.get('sparsity', 0.0)

                # Update sparsity masks
                embed_mask.set_sparsity(current_sparsity)
                embed_mask_ema.set_sparsity(current_sparsity)

                # Reset debug flag to print verification for new stage
                nnet.patch_embed._debug_printed = False
                nnet_ema.patch_embed._debug_printed = False

                # Calculate active dimensions
                active_dims = int(config.nnet.embed_dim * (1.0 - current_sparsity))

                if accelerator.is_main_process:
                    logging.info(f"\n{'='*60}")
                    logging.info(f"[Curriculum] Stage {stage_idx+1}/{len(curriculum_config['stages'])}: "
                               f"t_range=[{current_t_min:.2f}, {current_t_max:.2f}], "
                               f"sparsity={current_sparsity:.1f} (active_dims={active_dims}/{config.nnet.embed_dim}), "
                               f"name={stage.get('name', 'unnamed')}")
                    logging.info(f"{'='*60}\n")

                    # Stage checkpoint saving disabled - only regular interval checkpoints will be saved
                    # if stage_idx > 0:
                    #     stage_ckpt_path = os.path.join(config.ckpt_root, f'stage{stage_idx}_step{train_state.step}.ckpt')
                    #     train_state.save(stage_ckpt_path)
                    #     logging.info(f"[Curriculum] Saved stage checkpoint: {stage_ckpt_path}")

        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            logging.info('Save a grid of images...')
            # Disable sparsity mask during sampling
            embed_mask_ema.set_inference_mode(True)

            x_init = torch.randn(100, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=x_init, sample_steps=50)
            elif config.train.mode == 'cond':
                y = einops.repeat(torch.arange(10, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=x_init, sample_steps=50, y=y)
            else:
                raise NotImplementedError
            samples = make_grid(dataset.unpreprocess(samples), 10)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)

            # Re-enable training mode
            embed_mask_ema.set_inference_mode(False)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            # Skip FID evaluation (no reference statistics available)
            # fid = eval_step(n_samples=10000, sample_steps=50, algorithm='dpm_solver')
            # step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    # Skip FID-based checkpoint selection (no FID evaluation)
    # logging.info(f'step_fid: {step_fid}')
    # step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    # logging.info(f'step_best: {step_best}')
    # train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    # eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps, algorithm=config.sample.algorithm)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
