import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from jepa.dataset import JEPA_Dataset
from jepa.model import JEPA
from jepa.jepa_config import config

import wandb
from dotenv import load_dotenv
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger



def main():


    load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="jepa-mountaincar", name=f"jepa-mountaincar_{config.embed_dim}")
    wandb.config.update(config)
    wandb_logger = WandbLogger(experiment=wandb.run)
   

    # Set random seed for reproducibility
    pl.seed_everything(42, workers = True)

    # Enable mixed precision and optimize matmul
    torch.set_float32_matmul_precision('medium')

    train_dataset = JEPA_Dataset(split="train", val_ratio=config.val_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=15)

    val_dataset = JEPA_Dataset(split="val", val_ratio=config.val_ratio)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=15)

    model = JEPA(
        obs_dim=2,
        action_dim=1,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        lr=config.lr,
        ema_tau=config.ema_tau,
        lr_scheduler=config.lr_scheduler,
        cosine_T_max=config.cosine_T_max,
        cosine_eta_min=config.cosine_eta_min,
        plateau_factor=config.plateau_factor,
        plateau_patience=config.plateau_patience,
        plateau_min_lr=config.plateau_min_lr,
        dropout=config.dropout,
        weight_decay=config.weight_decay,
        obs0_loss_weight=config.obs0_loss_weight,
    )

    # Directory for checkpoints specific to this embed_dim
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        "lightning_logs",
        f"embed_dim_{config.embed_dim}",
        "checkpoints",
    )

    # Callback: save best checkpoint by val_loss (and last). Use enough precision
    # in filename so different val_loss values don't collide (e.g. 2.9e-6 vs 3.1e-6).
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="jepa-epoch={epoch:04d}-val_loss={val_loss:.6e}",
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,  # we already include val_loss in filename
        ),
        pl.callbacks.LearningRateMonitor(logging_interval = 'step'),
        pl.callbacks.EarlyStopping(
            monitor = 'val_loss',
            mode = 'min',
            patience = 15
        ),
        pl.callbacks.TQDMProgressBar(refresh_rate=10)
    ]

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision='bf16-mixed',
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the model to wandDB as an artifact
    lightning_root = os.path.join(os.path.dirname(__file__), "lightning_logs")
    embed_checkpoint_dir = os.path.join(
        lightning_root, f"embed_dim_{config.embed_dim}", "checkpoints"
    )

    # Prefer the new embed_dim-based folder; fall back to legacy version_*
    artifact_dir = None
    artifact_name_suffix = None

    if os.path.isdir(embed_checkpoint_dir):
        artifact_dir = embed_checkpoint_dir
        artifact_name_suffix = f"embed_dim_{config.embed_dim}"
    else:
        version_dirs = [
            d
            for d in os.listdir(lightning_root)
            if d.startswith("version_")
            and os.path.isdir(os.path.join(lightning_root, d))
        ]

        if version_dirs:
            version_dirs.sort(key=lambda d: int(d.split("_")[1]))
            latest_version = version_dirs[-1]
            legacy_checkpoint_dir = os.path.join(
                lightning_root, latest_version, "checkpoints"
            )
            if os.path.isdir(legacy_checkpoint_dir):
                artifact_dir = legacy_checkpoint_dir
                artifact_name_suffix = latest_version

    if artifact_dir is not None:
        artifact = wandb.Artifact(
            name=f"jepa-checkpoints-{artifact_name_suffix}-embed_dim_{config.embed_dim}",
            type="model",
            description=f"JEPA checkpoints from {artifact_name_suffix}",
        )
        artifact.add_dir(artifact_dir)
        wandb.log_artifact(artifact)

    return model
    


if __name__ == "__main__":
    main()