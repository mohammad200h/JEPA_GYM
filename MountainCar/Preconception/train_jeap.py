import argparse
import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from jepa.dataset import JEPA_Dataset
from jepa.model import JEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["cosine", "plateau"],
        default="cosine",
        help="LR schedule: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau on val_loss)",
    )
    parser.add_argument("--cosine_T_max", type=int, default=1_000_000, help="Max steps for cosine schedule")
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6, help="Min LR for cosine schedule")
    parser.add_argument("--plateau_factor", type=float, default=0.5, help="LR reduction factor for plateau")
    parser.add_argument("--plateau_patience", type=int, default=10, help="Epochs without improvement before reducing LR")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6, help="Min LR for plateau schedule")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of data for validation (proper train/val split)")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout in encoder/predictor (0 to disable)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Max epochs")
    args = parser.parse_args()
    # Set random seed for reproducibility
    pl.seed_everything(42, workers = True)

    # Enable mixed precision and optimize matmul
    torch.set_float32_matmul_precision('medium')

    train_dataset = JEPA_Dataset(split="train", val_ratio=args.val_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=15)

    val_dataset = JEPA_Dataset(split="val", val_ratio=args.val_ratio)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=15)

    model = JEPA(
        obs_dim=2,
        action_dim=1,
        lr_scheduler=args.lr_scheduler,
        cosine_T_max=args.cosine_T_max,
        cosine_eta_min=args.cosine_eta_min,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_min_lr=args.plateau_min_lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )



    # Callback: save best checkpoint by val_loss (and last). Use enough precision
    # in filename so different val_loss values don't collide (e.g. 2.9e-6 vs 3.1e-6).
    callbacks = [
        pl.callbacks.ModelCheckpoint(
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
        max_epochs=args.max_epochs,
        precision='bf16-mixed',
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


if __name__ == "__main__":
    main()