import os
from typing import Literal, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class JEPA_Dataset(Dataset):
    """
    Dataset for JEPA training on MountainCar rollouts stored in HDF5.

    Exposes tuples:
        (obs_{t-1}, obs_t, action_t, obs_{t+1})
    suitable for `JEPA.training_step`.
    """

    def __init__(
        self,
        h5_path: Optional[str] = None,
        episode_obs_key: str = "obs",
        episode_action_key: str = "action",
        dtype: torch.dtype = torch.float32,
        max_episodes_per_policy: Optional[int] = None,
        normalize_obs: bool = False,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        normalize_action: bool = False,
        action_min: Optional[float] = None,
        action_max: Optional[float] = None,
        split: Optional[Literal["train", "val"]] = None,
        val_ratio: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        """
        Loads (obs_{t-1}, obs_t, action_t, obs_{t+1}) from multiple episodes per
        policy. If normalize_obs is True, mean and std are computed from all
        loaded observations (pooled across policies and episodes) and applied.

        Args:
            h5_path: Path to `dataset.h5`. If None, uses the default path
                under `MountainCar/RL_Policy/dataset/dataset.h5` relative
                to this file.
            episode_obs_key: Name of the observation dataset inside each
                episode group (default: \"obs\").
            episode_action_key: Name of the action dataset inside each
                episode group (default: \"action\").
            dtype: Torch dtype for returned tensors.
            max_episodes_per_policy: If set, load at most this many episodes per
                policy (episodes are taken in sorted order). If None, load all
                episodes from each policy.
            normalize_obs: If True, observations are normalized (x - mean) / std.
                Mean and std are computed from all loaded observations (multiple
                episodes per policy, pooled).
            obs_mean: Observation mean for normalization. If None and normalize_obs
                is True, computed from all loaded observations.
            obs_std: Observation std for normalization. If None and normalize_obs
                is True, computed from all loaded observations (1e-8 floor for stability).
            normalize_action: If True, actions are scaled to [0, 1] via
                (action - action_min) / (action_max - action_min).
            action_min: Min action value. If None and normalize_action is True,
                computed from all loaded actions.
            action_max: Max action value. If None and normalize_action is True,
                computed from all loaded actions.
            split: If "train" or "val", use only that subset of the data (for a proper
                train/val split). If None, use all data (legacy behavior).
            val_ratio: Fraction of data to use for validation when split is set (e.g. 0.1 = 10% val).
            split_seed: Random seed for the train/val split so train and val datasets are reproducible.
        """
        super().__init__()

        if h5_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            h5_path = os.path.join(project_root, "RL_Policy", "dataset", "dataset.h5")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset not found at: {h5_path}")

        self.h5_path = h5_path
        self.episode_obs_key = episode_obs_key
        self.episode_action_key = episode_action_key
        self.dtype = dtype
        self.max_episodes_per_policy = max_episodes_per_policy

        # Collect (obs_{t-1}, obs_t, action_t, obs_{t+1}) from multiple episodes
        # per policy; then compute mean/std from this pooled data for normalization.
        obs_tm1_list = []
        obs_t_list = []
        obs_tp1_list = []
        action_t_list = []

        with h5py.File(self.h5_path, "r") as f:
            if not len(f.keys()):
                raise ValueError("HDF5 file is empty; expected policy groups.")

            for policy_name in f.keys():
                policy_grp = f[policy_name]
                # Each policy group contains episode groups: eps_0, eps_1, ...
                eps_names = sorted(policy_grp.keys())
                if max_episodes_per_policy is not None:
                    eps_names = eps_names[:max_episodes_per_policy]
                for eps_name in eps_names:
                    eps_grp = policy_grp[eps_name]

                    if (
                        self.episode_obs_key not in eps_grp
                        or self.episode_action_key not in eps_grp
                    ):
                        # Skip groups that don't contain the required datasets.
                        continue

                    episode_obs = eps_grp[self.episode_obs_key][:]
                    episode_actions = eps_grp[self.episode_action_key][:]
                    T = episode_obs.shape[0]

                    if T != episode_actions.shape[0]:
                        raise ValueError(
                            f"Observation and action time dimensions must match in "
                            f"{policy_name}/{eps_name}, got "
                            f"{episode_obs.shape[0]} and {episode_actions.shape[0]}"
                        )

                    if T < 3:
                        # Not enough timesteps in this episode to form tuples.
                        continue

                    # Valid t indices are 1 .. T-2 inclusive so that t-1 and t+1 exist.
                    T = episode_obs.shape[0]
                    for t in range(1, T - 1):
                        obs_tm1_list.append(episode_obs[t - 1])
                        obs_t_list.append(episode_obs[t])
                        obs_tp1_list.append(episode_obs[t + 1])
                        action_t_list.append(episode_actions[t])

        if not obs_t_list:
            raise ValueError(
                "No valid (obs_{t-1}, obs_t, action_t, obs_{t+1}) tuples could be "
                "constructed from the HDF5 file."
            )

        self.obs_tm1 = np.asarray(obs_tm1_list, dtype=np.float32)
        self.obs_t = np.asarray(obs_t_list, dtype=np.float32)
        self.obs_tp1 = np.asarray(obs_tp1_list, dtype=np.float32)
        self.actions_t = np.asarray(action_t_list, dtype=np.float32)

        self.length = self.obs_t.shape[0]
        self.split = split
        self.val_ratio = val_ratio
        self.split_seed = split_seed

        # Train/val split: deterministic shuffle then split so val measures generalization.
        if split is not None:
            rng = np.random.default_rng(split_seed)
            indices = np.arange(self.length, dtype=np.int64)
            rng.shuffle(indices)
            n_val = int(self.length * val_ratio)
            n_train = self.length - n_val
            if split == "train":
                self._indices = indices[:n_train]
            else:
                self._indices = indices[n_train:]
        else:
            self._indices = None

        # Normalization: mean/std from all loaded observations (multiple episodes
        # per policy, pooled). Use these to normalize in __getitem__.
        self.normalize_obs = normalize_obs
        if normalize_obs:
            if obs_mean is not None and obs_std is not None:
                _mean = np.asarray(obs_mean, dtype=np.float32)
                _std = np.asarray(obs_std, dtype=np.float32)
            else:
                all_obs = np.concatenate(
                    [self.obs_tm1, self.obs_t, self.obs_tp1], axis=0
                )
                _mean = all_obs.mean(axis=0).astype(np.float32)
                _std = all_obs.std(axis=0).astype(np.float32)
                _std = np.where(_std < 1e-8, 1.0, _std).astype(np.float32)
            self._obs_mean = torch.from_numpy(_mean)
            self._obs_std = torch.from_numpy(_std)
        else:
            self._obs_mean = None
            self._obs_std = None

        self.normalize_action = normalize_action
        if normalize_action:
            if action_min is not None and action_max is not None:
                self.action_min = float(action_min)
                self.action_max = float(action_max)
            else:
                self.action_min = float(self.actions_t.min())
                self.action_max = float(self.actions_t.max())
            self.action_scale = self.action_max - self.action_min
            if self.action_scale < 1e-8:
                self.action_scale = 1.0
        else:
            self.action_min = None
            self.action_max = None
            self.action_scale = None

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else self.length

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs_tm1: observation at t-1
            obs_t:   observation at t
            action_t: action at t
            obs_tp1: observation at t+1
        """
        n = len(self._indices) if self._indices is not None else self.length
        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for length {n}")
        real_idx = self._indices[idx] if self._indices is not None else idx

        obs_tm1 = torch.from_numpy(self.obs_tm1[real_idx].copy()).to(dtype=self.dtype)
        obs_t = torch.from_numpy(self.obs_t[real_idx].copy()).to(dtype=self.dtype)
        obs_tp1 = torch.from_numpy(self.obs_tp1[real_idx].copy()).to(dtype=self.dtype)
        action_t = torch.from_numpy(self.actions_t[real_idx].copy()).to(dtype=self.dtype)

        if self.normalize_obs:
            obs_tm1 = (obs_tm1 - self._obs_mean) / self._obs_std
            obs_t = (obs_t - self._obs_mean) / self._obs_std
            obs_tp1 = (obs_tp1 - self._obs_mean) / self._obs_std
        if self.normalize_action:
            action_t = (action_t - self.action_min) / self.action_scale

        return obs_tm1, obs_t, action_t, obs_tp1
