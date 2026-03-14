import os

import torch
import h5py
import numpy as np
from jepa.model import JEPA

import gymnasium as gym


def load_goal_obs_from_policy(policy_name: str, episode_index: int = 0):
    """Load goal_obs and goal_obs_tm1 into memory (file closed after read)."""
    mountaincar_root = os.path.dirname(os.path.dirname(__file__))  # MountainCar/
    dataset_path = os.path.join(mountaincar_root, "RL_Policy", "dataset", "dataset.h5")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Policy dataset not found: {dataset_path}\n"
            "Generate it by running (from MountainCar/RL_Policy):\n"
            "  python data_geenrator.py  # after training PPO and having policy_checkpoints/*.zip"
        )
    with h5py.File(dataset_path, "r") as f:
        obs = np.array(f[policy_name][f"eps_{episode_index}"]["obs"])
    goal_obs = obs[-1]
    goal_obs_tm1 = obs[-2]
    goal_obs_tm2 = obs[-3]
    return goal_obs, goal_obs_tm1, goal_obs_tm2



class MountainCarEnvInRepresentationSpace(gym.Env):
    """Gymnasium-compliant env that operates in JEPA representation space."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
   

    def __init__(
        self,
        checkpoint_path: str | None = None,
        render_mode: str | None = None,
        reward_type: str = "gym",
        done_type:str = "gym",
        embed_dim: int = None

    ):
        super().__init__()
        model_path = os.path.join(
            os.path.dirname(__file__),
            f"lightning_logs/embed_dim_{embed_dim}/checkpoints/",
        )
        # Find lowest val_loss checkpoint
        lowest_val_loss = float("inf")
        checkpoint_with_lowest_val_loss = None
        for file in os.listdir(model_path):
            if file.endswith(".ckpt") and "val_loss=" in file:
                val_loss = float(file.split("val_loss=")[1].split(".ckpt")[0])
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    checkpoint_with_lowest_val_loss = os.path.join(model_path, file)
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.done_type = done_type

        self.world_model = self._load_model(checkpoint_path or checkpoint_with_lowest_val_loss)
        goal_obs, goal_obs_tm1, goal_obs_tm2 = load_goal_obs_from_policy("policy_90000_4")
        self.goal_obs_z = self._goal_obs_z(goal_obs, goal_obs_tm1)
        self.goal_obs_tm1_z = self._goal_obsm1_z(goal_obs_tm1, goal_obs_tm2)
        self.success_threshold = self._work_out_threshold(self.goal_obs_tm1_z, self.goal_obs_z)

        render_mode_inner = "rgb_array" if render_mode else None
        self.gym_env = gym.make("MountainCarContinuous-v0", render_mode=render_mode_inner)

        self.action_space = self.gym_env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(embed_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obs_z = self._reset_world_model(seed)
        obs = self._to_numpy(self.obs_z)
        info = {"ground_truth": self.ground_truth}
        return obs, info

    def step(self, action):
        obs_z, reward, terminated, truncated = self._step_world_model(self.obs_z, action)
        ground_truth = self._step_gym(action)
        self.obs_z = obs_z
        obs = self._to_numpy(obs_z)
        info = {"ground_truth": ground_truth}

        if self.reward_type == "gym":
            reward = float(ground_truth["reward"])
        
        if self.done_type == "gym":
            terminated = ground_truth["terminated"]
            truncated = ground_truth["truncated"]
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def _reward_representaion_distance(self, obs_z_t):
        return -1 * self._representation_dist(obs_z_t, self.goal_obs_z)

    
    def _terminated(self, obs_z_t):
        """Terminate when representation is close enough to goal representation."""
        # feed obs_z_t to predict obs_t[0]
        obs0_pred = self.world_model.predict_obs0(obs_z_t)
        return obs0_pred >= 0.45

    def _step_world_model(self, obs_z, action):
        obs_z_next = self._predict_next_z(obs_z, action)
        reward = self._reward_representaion_distance(obs_z_next)
        terminated = self._terminated(obs_z_next)
        truncated = False  # no truncation in representation space; add TimeLimit wrapper if needed
        return obs_z_next, reward, terminated, truncated

    def _step_gym(self,action):
        self.obs, reward, terminated, truncated, info = self.gym_env.step(action)
        ground_truth = {
            "obs": self.obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
        return ground_truth

    def _reset_world_model(self, seed=None):
        self.ground_truth = self._reset_gym(seed)
        self.obs = self.ground_truth["obs"]
        o = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            self.obs_z = self.world_model.encode_context(o, o).squeeze(0)
        return self.obs_z

    def _reset_gym(self, seed=None):
        obs, info = self.gym_env.reset(seed=seed)
        info = {
            "obs": obs,
            "info": info
        }
        return info

    def _representation_dist(self, obs_z_t, goal_z):
        d = torch.norm(obs_z_t - goal_z, p=2)
        return d.item() if isinstance(d, torch.Tensor) else float(d)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    def _predict_next_z(self, obs_z, action):
        """Predict next representation: z_next = predictor(concat(z, action))."""
        z = torch.as_tensor(obs_z, dtype=torch.float32) if not isinstance(obs_z, torch.Tensor) else obs_z
        a = torch.as_tensor(action, dtype=torch.float32) if not isinstance(action, torch.Tensor) else action
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        with torch.no_grad():
            z_in = torch.cat([z, a], dim=-1)
            z_next = self.world_model.predictor(z_in)
        return z_next.squeeze(0) if z_next.shape[0] == 1 else z_next

    def _load_model(self, checkpoint_path: str) -> JEPA:
        model = JEPA.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            strict=True,
        )
        model.eval()
        return model

    def _goal_obs_z(self, goal_obs: np.ndarray, goal_obs_tm1: np.ndarray):
        g = torch.as_tensor(goal_obs, dtype=torch.float32).unsqueeze(0)
        g_tm1 = torch.as_tensor(goal_obs_tm1, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            goal_obs_z = self.world_model.encode_context(g_tm1, g).squeeze(0)
        return goal_obs_z
    
    def _goal_obsm1_z(self, goal_obs_tm1: np.ndarray, goal_obs_tm2: np.ndarray):
        g_tm1 = torch.as_tensor(goal_obs_tm1, dtype=torch.float32).unsqueeze(0)
        g_tm2 = torch.as_tensor(goal_obs_tm2, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            goal_obs_tm1_z = self.world_model.encode_context(g_tm2, g_tm1).squeeze(0)
        return goal_obs_tm1_z

    def _work_out_threshold(self,goal_obs_tm1_z: torch.Tensor, goal_obs_z: torch.Tensor):
        threshold = self._representation_dist(goal_obs_z, goal_obs_tm1_z)
        return threshold



# Register so gym.make("MountainCarRepSpace-v0") works.
# Import this module before calling gym.make(), e.g.:
#   from env import MountainCarEnvInRepresentationSpace  # registers on import
#   env = gym.make("MountainCarRepSpace-v0")
gym.register(
    id="MountainCarRepSpace-v0",
    entry_point=__name__ + ":MountainCarEnvInRepresentationSpace"
)