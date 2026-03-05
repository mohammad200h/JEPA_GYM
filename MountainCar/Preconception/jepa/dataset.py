import os
from typing import Optional, Tuple

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
    ) -> None:
        """
        Args:
            h5_path: Path to `dataset.h5`. If None, uses the default path
                under `MountainCar/RL_Policy/dataset/dataset.h5` relative
                to this file.
            episode_obs_key: Name of the observation dataset inside each
                episode group (default: \"obs\").
            episode_action_key: Name of the action dataset inside each
                episode group (default: \"action\").
            dtype: Torch dtype for returned tensors.
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

        # Collect all (obs_{t-1}, obs_t, action_t, obs_{t+1}) tuples from
        # every episode of every policy into contiguous numpy arrays.
        obs_tm1_list = []
        obs_t_list = []
        obs_tp1_list = []
        action_t_list = []

        with h5py.File(self.h5_path, "r") as f:
            if not len(f.keys()):
                raise ValueError("HDF5 file is empty; expected policy groups.")

            for policy_name in f.keys():
                policy_grp = f[policy_name]
                # Each policy group is expected to contain episode groups: eps_0, eps_1, ...
                for eps_name in policy_grp.keys():
                    eps_grp = policy_grp[eps_name]

                    if (
                        self.episode_obs_key not in eps_grp
                        or self.episode_action_key not in eps_grp
                    ):
                        # Skip groups that don't contain the required datasets.
                        continue

                    episode_obs = eps_grp[self.episode_obs_key][:]
                    episode_actions = eps_grp[self.episode_action_key][:]

                    if episode_obs.shape[0] != episode_actions.shape[0]:
                        raise ValueError(
                            f"Observation and action time dimensions must match in "
                            f"{policy_name}/{eps_name}, got "
                            f"{episode_obs.shape[0]} and {episode_actions.shape[0]}"
                        )

                    if episode_obs.shape[0] < 3:
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

    def __len__(self) -> int:
        return self.length

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
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for length {self.length}")

        obs_tm1 = torch.from_numpy(self.obs_tm1[idx]).to(dtype=self.dtype)
        obs_t = torch.from_numpy(self.obs_t[idx]).to(dtype=self.dtype)
        obs_tp1 = torch.from_numpy(self.obs_tp1[idx]).to(dtype=self.dtype)
        action_t = torch.from_numpy(self.actions_t[idx]).to(dtype=self.dtype)

        return obs_tm1, obs_t, action_t, obs_tp1
