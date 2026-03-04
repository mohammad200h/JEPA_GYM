"""
Visualize and compare (state, action) distributions across saved policy checkpoints.
Run each policy, collect transitions, and plot them to see how distributions change over training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym


def collect_transitions(model_path: str, n_steps: int = 1000, deterministic: bool = True) -> np.ndarray:
    """Load policy, run in env, return (N, 4) array of [pos, vel, action]."""
    model = PPO.load(model_path)
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=1000)

    obs_list, action_list = [], []
    obs, _ = env.reset()

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs_list.append(np.asarray(obs).flatten())
        action_list.append(np.asarray(action).flatten())
        next_obs, _, terminated, truncated, _ = env.step(action)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    obs_arr = np.array(obs_list)
    action_arr = np.array(action_list)
    return np.hstack([obs_arr, action_arr])  # (N, 3): pos, vel, action


def load_all_checkpoints(checkpoint_dir: str) -> list[tuple[str, str]]:
    """Return list of (name, path) for policy_*.zip files, sorted by training step."""
    if not os.path.isdir(checkpoint_dir):
        return []
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    # Sort by step number from filename, e.g. policy_10000_0 -> 10000
    def key(f: str) -> int:
        try:
            return int(f.split("_")[1])
        except (IndexError, ValueError):
            return 0
    files.sort(key=key)
    return [(os.path.splitext(f)[0], os.path.join(checkpoint_dir, f)) for f in files]


def plot_distributions(policies_dir: str = "./policy_checkpoints/", n_steps: int = 1500):
    """Collect data from all policies and create comparison visualizations."""
    checkpoints = load_all_checkpoints(policies_dir)
    if not checkpoints:
        print(f"No policy checkpoints found in {policies_dir}")
        return

    # Collect (pos, vel, action) for each policy
    data_by_policy: dict[str, np.ndarray] = {}
    for name, path in checkpoints:
        print(f"Collecting from {name}...")
        data_by_policy[name] = collect_transitions(path, n_steps=n_steps)

    n_policies = len(data_by_policy)
    names = list(data_by_policy.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. State visitation (pos vs vel) - one subplot per policy in a grid
    n_cols = min(4, n_policies)
    n_rows = (n_policies + n_cols - 1) // n_cols
    fig1, ax1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_policies == 1:
        ax1 = np.array([[ax1]])
    elif n_rows == 1:
        ax1 = ax1.reshape(1, -1)

    for i, name in enumerate(names):
        row, col = i // n_cols, i % n_cols
        ax = ax1[row, col]
        d = data_by_policy[name]
        pos, vel, action = d[:, 0], d[:, 1], d[:, 2]
        scatter = ax.scatter(pos, vel, c=action, s=5, alpha=0.6, cmap="viridis")
        ax.set_title(f"{name}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        plt.colorbar(scatter, ax=ax, label="Action")

    for j in range(i + 1, n_rows * n_cols):
        row, col = j // n_cols, j % n_cols
        ax1[row, col].set_visible(False)

    fig1.suptitle("State visitation (position vs velocity) colored by action")
    fig1.tight_layout()
    fig1.savefig(os.path.join(policies_dir, "state_visitation_by_policy.png"), dpi=120)
    plt.close(fig1)
    print(f"Saved: {policies_dir}state_visitation_by_policy.png")

    # 2. Action distributions (histograms) - all policies overlaid
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    for name, d in data_by_policy.items():
        pos, vel, action = d[:, 0], d[:, 1], d[:, 2]
        axes2[0].hist(pos, bins=40, alpha=0.4, label=name, density=True)
        axes2[1].hist(vel, bins=40, alpha=0.4, label=name, density=True)
        axes2[2].hist(action, bins=40, alpha=0.4, label=name, density=True)

    axes2[0].set_title("Position distribution")
    axes2[0].set_xlabel("Position")
    axes2[0].legend(bbox_to_anchor=(1.02, 1), fontsize=7)

    axes2[1].set_title("Velocity distribution")
    axes2[1].set_xlabel("Velocity")
    axes2[1].legend(bbox_to_anchor=(1.02, 1), fontsize=7)

    axes2[2].set_title("Action distribution")
    axes2[2].set_xlabel("Action")
    axes2[2].legend(bbox_to_anchor=(1.02, 1), fontsize=7)

    fig2.suptitle("Marginal distributions across policies")
    fig2.tight_layout()
    fig2.savefig(os.path.join(policies_dir, "marginal_distributions.png"), dpi=120)
    plt.close(fig2)
    print(f"Saved: {policies_dir}marginal_distributions.png")

    # 3. 2D heatmaps of position-velocity density per policy
    n_cols = min(4, n_policies)
    n_rows = (n_policies + n_cols - 1) // n_cols
    fig3, ax3 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_policies == 1:
        ax3 = np.array([[ax3]])
    elif n_rows == 1:
        ax3 = ax3.reshape(1, -1)

    for i, name in enumerate(names):
        row, col = i // n_cols, i % n_cols
        ax = ax3[row, col]
        d = data_by_policy[name]
        pos, vel = d[:, 0], d[:, 1]
        h, xe, ye = np.histogram2d(pos, vel, bins=30)
        ax.imshow(h.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto", cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

    for j in range(i + 1, n_rows * n_cols):
        row, col = j // n_cols, j % n_cols
        ax3[row, col].set_visible(False)

    fig3.suptitle("State-space density (position vs velocity)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(policies_dir, "state_density_heatmaps.png"), dpi=120)
    plt.close(fig3)
    print(f"Saved: {policies_dir}state_density_heatmaps.png")

    # 4. Action vs position (how action changes with state) - overlaid
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for name, d in data_by_policy.items():
        pos, action = d[:, 0], d[:, 2]
        # Sort by position for cleaner line
        idx = np.argsort(pos)
        ax4.scatter(pos[idx], action[idx], s=3, alpha=0.5, label=name)
    ax4.set_xlabel("Position")
    ax4.set_ylabel("Action")
    ax4.set_title("Action vs Position (how policy responds to state)")
    ax4.legend(bbox_to_anchor=(1.02, 1), fontsize=7)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(os.path.join(policies_dir, "action_vs_position.png"), dpi=120)
    plt.close(fig4)
    print(f"Saved: {policies_dir}action_vs_position.png")

    print("Done. Check the policy_checkpoints folder for PNG files.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./policy_checkpoints/", help="Directory with policy checkpoints")
    parser.add_argument("--steps", type=int, default=1500, help="Transitions to collect per policy")
    args = parser.parse_args()
    plot_distributions(policies_dir=args.dir, n_steps=args.steps)
