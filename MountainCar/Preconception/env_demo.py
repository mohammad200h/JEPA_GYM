import gymnasium as gym
from env import MountainCarEnvInRepresentationSpace  # registers on import


def main():
    # Use gym.make() - the env is registered as "MountainCarRepSpace-v0"
    env = gym.make("MountainCarRepSpace-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"obs shape: {obs}, reward: {reward:.4f}, terminated: {terminated}, truncated: {truncated}",
            f"ground_truth obs: {info['ground_truth']['obs']}",
        )
        if done:
            obs, info = env.reset()
        frame = env.render()
    env.close()


if __name__ == "__main__":
    main()
