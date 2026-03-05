import os
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from jepa.dataset import JEPA_Dataset
from jepa.model import JEPA


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42, workers = True)

    # Enable mixed precision and optimize matmul
    torch.set_float32_matmul_precision('medium')

    train_dataset = JEPA_Dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=15)

    val_dataset = JEPA_Dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=15)

    model = JEPA(obs_dim=2, action_dim=1)


    # Callback
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor = 'val_loss',
            mode = 'min',
            save_top_k = 3,
            save_last = True,
            filename = 'jepa{epoch:02d}-{val_loss:.3f}',
            save_on_train_epoch_end = False
        ),
        pl.callbacks.LearningRateMonitor(logging_interval = 'step'),
        pl.callbacks.EarlyStopping(
            monitor = 'val_loss',
            mode = 'min',
            patience = 15
        ),
        pl.callbacks.TQDMProgressBar(refresh_rate=10)
    ]

    trainer = pl.Trainer(max_epochs=100, precision='bf16-mixed')
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


if __name__ == "__main__":
    main()