import os
import numpy as np
from collections.abc import Callable
from typing import Any

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


def _copy_obs(obs: Any) -> np.ndarray:
    if hasattr(obs, "copy"):
        return np.copy(obs)
    return np.array(obs, copy=True)


def _transitions_to_sa(transitions: list[dict[str, Any]], max_samples: int | None = None) -> np.ndarray:
    """Stack (obs, action) from transitions into (N, obs_dim + action_dim)."""
    obs_list = [np.asarray(t["obs"]).flatten() for t in transitions]
    action_list = [np.asarray(t["action"]).flatten() for t in transitions]
    n = len(obs_list)
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        obs_list = [obs_list[i] for i in idx]
        action_list = [action_list[i] for i in idx]
    obs = np.array(obs_list)
    action = np.array(action_list)
    return np.hstack([obs, action])


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel between two samples.
    Larger value => distributions more different.
    """
    def rbf(x: np.ndarray, y: np.ndarray, g: float) -> np.ndarray:
        d = np.sum(x**2, axis=1, keepdims=True) + np.sum(y**2, axis=1) - 2 * x @ y.T
        return np.exp(-g * np.maximum(d, 0))

    n, m = len(X), len(Y)
    Kxx = rbf(X, X, gamma)
    Kyy = rbf(Y, Y, gamma)
    Kxy = rbf(X, Y, gamma)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return float(np.sqrt(np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)))


def transition_distribution_distance(
    transitions_a: list[dict[str, Any]],
    transitions_b: list[dict[str, Any]],
    max_samples: int = 2000,
    gamma: float = 1.0,
) -> float:
    """
    Compare two transition batches in terms of (s, a) distribution via MMD.
    Returns MMD distance; 0 = same, larger = more different.
    """
    sa_a = _transitions_to_sa(transitions_a, max_samples)
    sa_b = _transitions_to_sa(transitions_b, max_samples)
    scale = np.std(sa_a) + np.std(sa_b) + 1e-8
    gamma_scaled = gamma / (scale**2)
    return mmd_rbf(sa_a, sa_b, gamma_scaled)


class TransitionCollectorCallback(BaseCallback):
    """
    Callback that collects transitions by running the eval env with the current
    policy (model.predict). Stores (obs, action, reward, next_obs, done, info).
    """

    def __init__(
        self,
        make_eval_env: Callable[[], gym.Env],
        max_transitions: int = 100_000,
        collect_freq: int = 5000,
        steps_per_collect: int = 500,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_eval_env = make_eval_env
        self.max_transitions = max_transitions
        self.collect_freq = collect_freq
        self.steps_per_collect = steps_per_collect
        self.deterministic = deterministic
        self.transitions: list[dict[str, Any]] = []
        self._eval_env: gym.Env | None = None

    def _on_training_start(self) -> None:
        self.transitions = []
        self._eval_env = self.make_eval_env()

    def _on_step(self) -> bool:
        if len(self.transitions) >= self.max_transitions:
            return True

        if self.n_calls % self.collect_freq != 0:
            return True

        if self._eval_env is None:
            return True

        n_to_collect = min(
            self.steps_per_collect,
            self.max_transitions - len(self.transitions),
        )
        collected = 0

        obs, _ = self._eval_env.reset()
        for _ in range(n_to_collect):
            if len(self.transitions) >= self.max_transitions:
                break
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            next_obs, reward, terminated, truncated, info = self._eval_env.step(action)
            done = terminated or truncated

            transition = {
                "obs": _copy_obs(obs),
                "action": np.copy(action) if np.ndim(action) > 0 else np.array(action, copy=True),
                "reward": float(reward),
                "next_obs": _copy_obs(next_obs),
                "done": done,
                "info": dict(info),
            }
            self.transitions.append(transition)
            collected += 1
            obs = next_obs

            if done:
                obs, _ = self._eval_env.reset()

        if self.verbose >= 1 and collected > 0:
            print(f"TransitionCollector: collected {collected} transitions (total {len(self.transitions)})")

        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None


class PolicyComparisonCallback(BaseCallback):
    """
    Every compare_freq steps, collects transitions from the current policy and
    compares their (s, a) distribution to every previously stored policy using MMD.
    If the current policy is different from all stored ones (MMD > mmd_threshold
    for each), saves the model and adds its transitions to the stored set.
    """

    def __init__(
        self,
        make_eval_env: Callable[[], gym.Env],
        save_path: str = "./policy_checkpoints/",
        name_prefix: str = "policy",
        compare_freq: int = 10_000,
        steps_per_collect: int = 500,
        deterministic: bool = True,
        mmd_threshold: float = 0.05,
        max_samples: int = 2000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_eval_env = make_eval_env
        self.save_path = save_path.rstrip("/")
        self.name_prefix = name_prefix
        self.compare_freq = compare_freq
        self.steps_per_collect = steps_per_collect
        self.deterministic = deterministic
        self.mmd_threshold = mmd_threshold
        self.max_samples = max_samples
        self._eval_env: gym.Env | None = None
        self._stored_transitions: list[list[dict[str, Any]]] = []
        self._num_saved = 0

    def _on_training_start(self) -> None:
        self._eval_env = self.make_eval_env()
        self._stored_transitions = []
        self._num_saved = 0

    def _collect_transitions(self) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        if self._eval_env is None:
            return transitions
        obs, _ = self._eval_env.reset()
        for _ in range(self.steps_per_collect):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            next_obs, reward, terminated, truncated, info = self._eval_env.step(action)
            done = terminated or truncated
            transition = {
                "obs": _copy_obs(obs),
                "action": np.copy(action) if np.ndim(action) > 0 else np.array(action, copy=True),
                "reward": float(reward),
                "next_obs": _copy_obs(next_obs),
                "done": done,
                "info": dict(info),
            }
            transitions.append(transition)
            obs = next_obs
            if done:
                obs, _ = self._eval_env.reset()
        return transitions

    def _on_step(self) -> bool:
        if self.n_calls % self.compare_freq != 0:
            return True
        if self._eval_env is None:
            return True

        current = self._collect_transitions()
        if len(current) == 0:
            return True

        mmds: list[float] = []
        for ref_batch in self._stored_transitions:
            mmd = transition_distribution_distance(
                ref_batch,
                current,
                max_samples=self.max_samples,
            )
            mmds.append(mmd)

        different_from_all = all(mmd > self.mmd_threshold for mmd in mmds)
        min_mmd = min(mmds) if mmds else float("inf")

        if self.logger is not None:
            self.logger.record("policy_compare/min_mmd", min_mmd)
            self.logger.record("policy_compare/num_stored", len(self._stored_transitions))

        if different_from_all:
            os.makedirs(self.save_path, exist_ok=True)
            path = f"{self.save_path}/{self.name_prefix}_{self.n_calls}_{self._num_saved}"
            self.model.save(path)
            self._num_saved += 1
            self._stored_transitions.append(current)
            if self.logger is not None:
                self.logger.record("policy_compare/saved", 1)
            if self.verbose >= 1:
                print(f"PolicyComparison (step {self.n_calls}): different from all {len(self._stored_transitions)-1} stored → saved to {path}")
        else:
            if self.logger is not None:
                self.logger.record("policy_compare/saved", 0)
            if self.verbose >= 1:
                print(f"PolicyComparison (step {self.n_calls}): min_mmd = {min_mmd:.4f} (threshold {self.mmd_threshold}), not saving")

        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None