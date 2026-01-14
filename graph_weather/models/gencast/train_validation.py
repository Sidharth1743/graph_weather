"""
Quick Validation Training for GenCast - Testing Bug Fixes

This config trains a SMALL model to validate that the noise fix works.
Should complete in 6-12 hours on 2x RTX 3060 12GB.

After validation succeeds, use train.py for full production model.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Both RTX 3060s
# Enable async error handling for NCCL timeouts to work properly
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Set matplotlib backend before importing pyplot (headless mode)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import lightning as L  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datetime import timedelta  # noqa: E402
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from graph_weather.data.gencast_dataloader import GenCastDataset  # noqa: E402
from graph_weather.models.gencast import Denoiser, Sampler, WeightedMSELoss  # noqa: E402

torch.set_float32_matmul_precision("high")

############################################## VALIDATION SETTINGS ############################################

# SMALL MODEL - Quick validation of bug fix (fits in 12GB VRAM)
# Training settings
NUM_EPOCHS = 10
NUM_DEVICES = 2  # 2x RTX 3060
NUM_ACC_GRAD = 1  # No gradient accumulation
INITIAL_LR = 5e-4  # Slightly lower LR for small model
BATCH_SIZE = 1  # Per GPU - effective batch = 2 (2 GPU × 1 batch) - MINIMAL to prevent sync issues
WARMUP = 500

# Dataloader settings
NUM_WORKERS = 4  # Increase to prevent data loading bottleneck
PREFETCH_FACTOR = 4  # Prefetch more batches
PERSISTENT_WORKERS = True

# Model configs - VERY SMALL for 12GB VRAM
CFG = {
    "hidden_dims": [128, 128],  # Reduced from [256, 256] to save memory
    "num_blocks": 4,  # Reduced from 6 to save memory
    "num_heads": 4,
    "splits": 4,
    "num_hops": 4,  # Reduced from 6 to save memory
    "sparse": False,  # False for smaller grids (faster)
    "use_edges_features": False,
    "scale_factor": 1.0,
}

# Dataset configs - MINIMAL FEATURES for quick validation
atmospheric_features = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
single_features = [
    "2m_temperature",
    "mean_sea_level_pressure",
]
static_features = [
    "geopotential_at_surface",
    "land_sea_mask",
]

# RECOMMENDED: Stream data directly from Google Cloud (no download needed!)
# Requires: gcsfs (install with: pixi add gcsfs or pip install gcsfs)
OBS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"

# Alternative: Use local dataset if you downloaded it
# OBS_PATH = "dataset_128x64.zarr"

CHECKPOINT_DIR = "checkpoints_validation/"
WANDB_PROJECT = "gencast-validation-bugfix"

#################################################################################################


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Scheduler with Warmup"""

    def __init__(self, optimizer, warmup, max_iters):
        """Initialize the scheduler"""
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        """Return the learning rates"""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """Return the scaling factor for the learning rate at a given iteration"""
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class LitModel(L.LightningModule):
    """Lightning wrapper for Gencast"""

    def __init__(
        self,
        warmup,
        learning_rate,
        cosine_t_max,
        pressure_levels,
        grid_lon,
        grid_lat,
        input_features_dim,
        output_features_dim,
        hidden_dims,
        num_blocks,
        num_heads,
        splits,
        num_hops,
        sparse,
        use_edges_features,
        scale_factor=1.0,
    ):
        """Initialize the module"""
        super().__init__()

        self.model = Denoiser(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            splits=splits,
            num_hops=num_hops,
            device=self.device,
            sparse=sparse,
            use_edges_features=use_edges_features,
            scale_factor=scale_factor,
        )

        self.criterion = WeightedMSELoss(
            grid_lat=torch.tensor(grid_lat).to(self.device),
            pressure_levels=torch.tensor(pressure_levels).to(self.device),
            num_atmospheric_features=len(atmospheric_features),
            single_features_weights=torch.tensor([1.0, 0.1]).to(self.device),
        )

        self.learning_rate = learning_rate
        self.cosine_t_max = cosine_t_max
        self.warmup = warmup

    def forward(self, corrupted_targets, prev_inputs, noise_levels):
        """Compute forward pass"""
        return self.model(corrupted_targets, prev_inputs, noise_levels)

    def training_step(self, batch):
        """Single training step"""
        corrupted_targets, prev_inputs, noise_levels, target_residuals = batch

        preds = self.model(
            corrupted_targets=corrupted_targets,
            prev_inputs=prev_inputs,
            noise_levels=noise_levels,
        )
        loss = self.criterion(preds, noise_levels, target_residuals)
        self.log("train_loss", loss, prog_bar=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Initialize the optimizer"""
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.95)
        )
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.cosine_t_max)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }

    def plot_sample(self, prev_inputs, target_residuals):
        """Plot sample predictions"""
        prev_inputs = prev_inputs[:1, :, :, :]
        target = target_residuals[:1, :, :, :]
        sampler = Sampler(num_steps=10)  # Use 10 steps for validation
        preds = sampler.sample(self.model, prev_inputs)

        # Plot 2m_temperature (last feature in atmospheric + single)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Prediction
        ax[0].imshow(preds[0, :, :, -2].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[0].set_title("Prediction (2m temp)")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # Target
        ax[1].imshow(target[0, :, :, -2].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[1].set_title("Ground Truth (2m temp)")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()
        return fig


class SamplingCallback(Callback):
    """Callback for sampling when a new epoch starts"""

    def __init__(self, data):
        """Initialize the callback"""
        _, prev_inputs, _, target_residuals = data
        self.prev_inputs = torch.tensor(prev_inputs).unsqueeze(0)
        self.target_residuals = torch.tensor(target_residuals).unsqueeze(0)

    def on_train_epoch_start(self, trainer, pl_module):
        """Sample and log predictions"""
        print(f"\n{'='*60}")
        print(f"Epoch {trainer.current_epoch} starting")
        print(f"{'='*60}")

        # Only log images if using W&B logger
        if hasattr(trainer.logger, 'log_image'):
            fig = pl_module.plot_sample(
                self.prev_inputs.to(pl_module.device),
                self.target_residuals.to(pl_module.device)
            )
            trainer.logger.log_image(
                key="validation_samples", images=[fig], caption=[f"Epoch {trainer.current_epoch}"]
            )
            plt.close(fig)
            print("Sample uploaded to W&B")
        else:
            print("Skipping sample generation (W&B not configured)")


class ValidationMetricsCallback(Callback):
    """Track validation metrics every N steps"""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log metrics periodically"""
        if trainer.global_step % self.log_every_n_steps == 0:
            print(f"Step {trainer.global_step}: Loss = {outputs['loss']:.6f}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENCAST QUICK VALIDATION TRAINING - BUG FIX VERIFICATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {CFG['num_blocks']} blocks, {CFG['hidden_dims']} hidden dims")
    print(f"  Batch size: {BATCH_SIZE} per GPU x {NUM_DEVICES} GPUs x {NUM_ACC_GRAD} accum = {BATCH_SIZE * NUM_DEVICES * NUM_ACC_GRAD}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Dataset: {OBS_PATH}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print("="*80 + "\n")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Define dataloader
    print("Loading dataset...")
    dataset = GenCastDataset(
        obs_path=OBS_PATH,
        atmospheric_features=atmospheric_features,
        single_features=single_features,
        static_features=static_features,
        max_year=2018,
        time_step=2,
    )

    print(f"Dataset loaded:")
    print(f"  Grid shape: {len(dataset.grid_lon)}x{len(dataset.grid_lat)}")
    print(f"  Input features: {dataset.input_features_dim}")
    print(f"  Output features: {dataset.output_features_dim}")
    print(f"  Samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        multiprocessing_context="forkserver",
    )

    # Define model
    num_steps = NUM_EPOCHS * len(dataloader) // (NUM_DEVICES * NUM_ACC_GRAD)
    print(f"\nTotal training steps: {num_steps}")

    denoiser = LitModel(
        warmup=WARMUP,
        learning_rate=INITIAL_LR,
        cosine_t_max=num_steps,
        pressure_levels=dataset.pressure_levels,
        grid_lon=dataset.grid_lon,
        grid_lat=dataset.grid_lat,
        input_features_dim=dataset.input_features_dim,
        output_features_dim=dataset.output_features_dim,
        **CFG,
    )

    # Define trainer
    # W&B logger (optional - comment out if not using wandb)
    try:
        wandb_logger = WandbLogger(project=WANDB_PROJECT, name="validation-128x64-6blocks")
    except Exception as e:
        print(f"W&B not configured: {e}")
        print("Continuing without W&B logging. To enable:")
        print("  pixi run pip install wandb && pixi run wandb login")
        wandb_logger = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="gencast-validation-{epoch:02d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    sampling_callback = SamplingCallback(data=dataset[0])
    metrics_callback = ValidationMetricsCallback(log_every_n_steps=50)

    # Check if GPU is available
    import torch as torch_check
    if torch_check.cuda.is_available():
        accelerator = "gpu"
        devices = NUM_DEVICES
        # Configure DDP strategy with extended timeout for slow data streaming
        strategy = DDPStrategy(
            timeout=timedelta(hours=2),  # 2 hour timeout for GCS streaming
            find_unused_parameters=False,
        )
        precision = "16-mixed"
        print(f"Using GPU training with {NUM_DEVICES} GPUs (2 hour NCCL timeout)")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
        precision = "32"
        print("WARNING: CUDA not available, using CPU (will be VERY slow!)")
        print("To enable GPU: Install PyTorch with CUDA support")

    trainer = L.Trainer(
        accumulate_grad_batches=NUM_ACC_GRAD,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, sampling_callback, lr_monitor, metrics_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Prevent gradient explosion
    )

    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    print("Monitor progress at: https://wandb.ai")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    print("\n")

    trainer.fit(model=denoiser, train_dataloaders=dataloader)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Last checkpoint: {checkpoint_callback.last_model_path}")
    print("\nNext steps:")
    print("1. Test the trained model with: python test_ocf_sampler_mse_progression.py")
    print("2. If MSE decreases with more steps → BUG FIX VERIFIED! ✅")
    print("3. Then train full model with: python graph_weather/models/gencast/train.py")
