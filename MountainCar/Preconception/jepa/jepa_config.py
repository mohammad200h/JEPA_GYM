from ml_collections import config_dict

config = config_dict.ConfigDict()

# Scheduler and training-related hyperparameters for JEPA training.
config.embed_dim = 2        # Embedding dimension
config.lr_scheduler = "cosine"   # "cosine" or "plateau"
config.cosine_T_max = 1_000_000  # Max steps for cosine schedule
config.cosine_eta_min = 1e-6     # Min LR for cosine schedule
config.plateau_factor = 0.5      # LR reduction factor for plateau
config.plateau_patience = 10     # Epochs without improvement before reducing LR
config.plateau_min_lr = 1e-6     # Min LR for plateau schedule
config.val_ratio = 0.1           # Fraction of data for validation (proper train/val split)
config.dropout = 0.0             # Dropout in encoder/predictor (0 to disable)
config.weight_decay = 1e-5       # Weight decay for AdamW optimizer
config.max_epochs = 2000         # Max epochs
config.hidden_dim = 256          # Hidden dimension
config.lr = 3e-4                 # Learning rate
config.ema_tau = 0.99            # EMA coefficient for target encoder update
config.obs0_loss_weight = 1.0    # Weight for auxiliary loss predicting obs_{t+1}[0] from z_pred
config.seed = 42                 # Seed for reproducibility
config.batch_size = 128          # Batch size