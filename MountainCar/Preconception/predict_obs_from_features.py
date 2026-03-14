#!/usr/bin/env python3
"""
Load dataset, JEPA model, feed obs to predictor to get features,
then use a decoder to predict observations from features.
"""

import os
import sys
import wandb 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jepa.model import JEPA
from jepa.dataset import JEPA_Dataset
from jepa.decorder import ObsDecoder


def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "..", "RL_Policy", "dataset", "dataset.h5")
    checkpoint_path = os.path.join(
        project_root,
        "lightning_logs",
        "version_7",
        "checkpoints",
        "jepa-epoch=0072-val_loss=3.946645e-06.ckpt",
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    print("Loading dataset...")
    dataset = JEPA_Dataset(h5_path=os.path.abspath(dataset_path))
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # 2. Load JEPA model
    print("Loading JEPA model...")
    jepa = JEPA.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    jepa.eval()

    # Get dimensions from JEPA hyperparams
    obs_dim = jepa.hparams.obs_dim
    action_dim = jepa.hparams.action_dim
    embed_dim = jepa.hparams.embed_dim

    # 3. Create and train decoder: z_pred -> obs
    decoder = ObsDecoder(
        embed_dim=embed_dim,
        obs_dim=obs_dim,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    print("Training decoder (z_pred -> obs)...")
    n_epochs = 20
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            obs_tm1, obs_t, action_t, obs_tp1 = [b.to(device) for b in batch]

            with torch.no_grad():
                z_pred = jepa(obs_tm1, obs_t, action_t)

            obs_pred = decoder(z_pred)
            loss = F.mse_loss(obs_pred, obs_tp1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}/{n_epochs}  MSE: {avg_loss:.6e}")

    # 4. Evaluate
    decoder.eval()
    with torch.no_grad():
        total_mse = 0.0
        n_samples = 0
        for batch in loader:
            obs_tm1, obs_t, action_t, obs_tp1 = [b.to(device) for b in batch]
            z_pred = jepa(obs_tm1, obs_t, action_t)
            obs_pred = decoder(z_pred)
            total_mse += F.mse_loss(obs_pred, obs_tp1, reduction="sum").item()
            n_samples += obs_tp1.shape[0]

        avg_mse = total_mse / n_samples
        print(f"\nFinal reconstruction MSE: {avg_mse:.6e}")

    # Sample a few predictions
    print("\nSample predictions (obs_tp1 vs pred):")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            obs_tm1, obs_t, action_t, obs_tp1 = [b.to(device) for b in batch]
            z_pred = jepa(obs_tm1, obs_t, action_t)
            obs_pred = decoder(z_pred)
            for j in range(min(3, obs_tp1.shape[0])):
                print(
                    f"  {j + 1}. true: {obs_tp1[j].cpu().numpy().round(4)}  pred: {obs_pred[j].cpu().numpy().round(4)}"
                )


if __name__ == "__main__":
    main()
