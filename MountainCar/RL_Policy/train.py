from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from statistic import PolicyComparisonCallback

# https://medium.com/@emikea03/the-power-of-ppo-how-proximal-policy-optimization-solves-a-range-of-rl-problems-10076d9da34e


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


def make_env(record_video: bool = False):
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env, LOG_DIR)  # needed so SB3 can log rollout/ep_rew_mean and rollout/ep_len_mean
    return env
    


def make_eval_env():
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env, LOG_DIR)
    env = RecordVideo(
            env,
            video_folder="videos",
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix="mountaincar",
        )
    return env


def train():
    train_env = SubprocVecEnv([lambda: make_env(record_video=False) for _ in range(N_ENVS)])
    eval_env = make_eval_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        name_prefix="ppo_mountain_car",
        save_path="./checkpoints/",
    )

    policy_compare_callback = PolicyComparisonCallback(
        make_eval_env=make_eval_env,
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


    model = PPO("MlpPolicy", train_env, verbose=2, n_steps = 2048, seed=37, tensorboard_log=LOG_DIR, ent_coef=initial_ent_coef)
    model.learn(total_timesteps=1e6, callback=callback)
    model.save("ppo_mountain_car")
    train_env.close()
    return model


if __name__ == "__main__":
    train()