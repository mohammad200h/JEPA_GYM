from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import env  # registers MountainCarRepSpace-v0 (needed for SubprocVecEnv workers)
from statistic import PolicyComparisonCallback

from dotenv import load_dotenv
from pathlib import Path
import os
import wandb
import shutil


# https://medium.com/@emikea03/the-power-of-ppo-how-proximal-policy-optimization-solves-a-range-of-rl-problems-10076d9da34e


class EvalCallbackWithEarlyStopping(EvalCallback):
    """EvalCallback that stops training when mean_reward > threshold for n_consecutive evals."""

    def __init__(
        self,
        *args,
        reward_threshold: float = 80.0,
        n_consecutive: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_threshold = reward_threshold
        self.n_consecutive = n_consecutive
        self.consecutive_above_threshold = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            return False
        # Check after each evaluation (parent evaluates when n_calls % eval_freq == 0)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward > self.reward_threshold:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.n_consecutive:
                    if self.verbose >= 1:
                        print(
                            f"Early stopping: mean_reward {self.last_mean_reward:.1f} > "
                            f"{self.reward_threshold} for {self.n_consecutive} consecutive evals"
                        )
                    return False
            else:
                self.consecutive_above_threshold = 0
        return True


class EntropyDecayCallback(BaseCallback):
    """Decay ent_coef linearly from initial to final value over decay_steps, starting after start_after_timesteps."""

    def __init__(
        self,
        initial_ent_coef: float,
        final_ent_coef: float,
        decay_steps: int,
        start_after_timesteps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps
        self.start_after_timesteps = start_after_timesteps

    def _on_step(self) -> bool:
        total = self.num_timesteps
        if total < self.start_after_timesteps:
            return True
        elapsed = total - self.start_after_timesteps
        if elapsed >= self.decay_steps:
            self.model.ent_coef = self.final_ent_coef
        else:
            alpha = elapsed / self.decay_steps
            self.model.ent_coef = self.initial_ent_coef + alpha * (self.final_ent_coef - self.initial_ent_coef)
        return True


N_ENVS = 4
MAX_EPISODE_STEPS = 1000
LOG_DIR = "logs/"
reward_type = "gym"
done_type = "gym"

def make_env(embed_dim: int ):
    env = gym.make("MountainCarRepSpace-v0", 
                    max_episode_steps=MAX_EPISODE_STEPS, 
                    embed_dim=embed_dim, reward_type=reward_type)
    env = Monitor(env, LOG_DIR)  # needed so SB3 can log rollout/ep_rew_mean and rollout/ep_len_mean
    return env
    


def make_eval_env(embed_dim: int):
    env = gym.make("MountainCarRepSpace-v0", render_mode="rgb_array",
                    max_episode_steps=MAX_EPISODE_STEPS, 
                    embed_dim=embed_dim, reward_type=reward_type)
    env = Monitor(env, LOG_DIR)
    env = RecordVideo(
            env,
            video_folder="videos",
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix="mountaincar",
        )
    return env


def train(embed_dim: int = None):
    load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="rl_world_model_mountain_car",
        name=f"rl_edim:{embed_dim}_rtype:{reward_type}_dtype:{done_type}",
        sync_tensorboard=True,
    )

    train_env = None
    try:
        train_env = SubprocVecEnv([lambda: make_env(
                    embed_dim=embed_dim) for _ in range(N_ENVS)])
        eval_env = make_eval_env(embed_dim=embed_dim)

        eval_callback = EvalCallbackWithEarlyStopping(
            eval_env,
            best_model_save_path="./best_model/",
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            reward_threshold=80,
            n_consecutive=4,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            name_prefix="ppo_mountain_car_rep_space",
            save_path="./checkpoints/",
        )

        policy_compare_callback = PolicyComparisonCallback(
            make_eval_env=lambda: make_eval_env(embed_dim=embed_dim),
            save_path="./policy_checkpoints/",
            name_prefix="policy",
            compare_freq=10000,
            steps_per_collect=500,
            deterministic=True,
            mmd_threshold=0.05,
            verbose=1,
        )
        initial_ent_coef = 0.3  # Initial entropy coefficient
        final_ent_coef = 0.001  # Final entropy coefficient
        decay_steps = 100000    # Number of steps over which to decay entropy
        start_after_timesteps = 100000

        entropy_decay_callback = EntropyDecayCallback(initial_ent_coef=initial_ent_coef,
                                     final_ent_coef=final_ent_coef,
                                     decay_steps=decay_steps,
                                    start_after_timesteps=start_after_timesteps)

        callback = CallbackList([
            eval_callback,
            checkpoint_callback,
            policy_compare_callback,
            entropy_decay_callback,
        ])

        run_name = f"PPO_embed_dim_{embed_dim}"
        model = PPO("MlpPolicy", train_env, verbose=2, n_steps=2048, seed=37,
            tensorboard_log=os.path.join(LOG_DIR, run_name), ent_coef=initial_ent_coef)
        model.learn(total_timesteps=1e6, callback=callback)
        model.save("ppo_mountain_car_rep_space")
        # Save best_model as wandb artifact at end of training
        best_model_path = Path("./best_model")
        if best_model_path.exists():
            artifact = wandb.Artifact(name=f"best_model_embed_dim_{embed_dim}_{reward_type}", type="model")
            artifact.add_dir(str(best_model_path))
            wandb.log_artifact(artifact)
        # Save videos as wandb artifact at end of training
        if os.path.exists("videos"):
            artifact = wandb.Artifact(name=f"videos_embed_dim_{embed_dim}_{reward_type}", type="videos")
            artifact.add_dir("videos")
            wandb.log_artifact(artifact)
    finally:
        if train_env is not None:
            train_env.close()
        wandb.finish()
    return model


if __name__ == "__main__":
    for embed_dim in [2, 4, 8, 16, 32, 64, 128]:
        # embty videos folder
        if os.path.exists("videos"):
            shutil.rmtree("videos")
        
        train(embed_dim=embed_dim)