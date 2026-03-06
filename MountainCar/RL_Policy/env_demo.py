import gymnasium as gym

# https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
def main():
    env = gym.make("MountainCarContinuous-v0", 
        render_mode="rgb_array", goal_velocity=0.1
    )
    obs, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            obs, info = env.reset()
        env.render()
    env.close() 


if __name__ == "__main__":
    main()