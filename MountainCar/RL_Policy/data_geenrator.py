from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np


def main(num_episodes: int = 10):
    # Load model
    model = PPO.load("ppo_mountain_car")

    # Environment with render_mode="rgb_array" to get frames
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    # All episodes' recordings
    all_episodes = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        records = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()  # RGB array (H, W, 3), uint8
            records.append({
                "obs": obs.copy(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": dict(info),
                "frame": frame.copy(),
            })
            done = terminated or truncated

        all_episodes.append(records)
        print(f"Episode {episode + 1}/{num_episodes} done, {len(records)} steps")

    env.close()

    # Optional: save all_episodes (list of lists of step dicts) to disk
    return all_episodes


if __name__ == "__main__":
    import sys
    num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(num_episodes)