"""Training, checkpoint, and evaluation callbacks."""

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """Count circuit simulations and log them with the training metrics.

    A TT-only step simulates one netlist. With Skip-on-Fail (SoF), a step that
    fails at TT simulates one netlist and a step that passes simulates every corner.
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.num_corners = env.num_corners
        self.SoF = env.SoF
        self.tt_only = getattr(env, "tt_only", False)
        self.tt_sim = self.full_sim = self.total_sim = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if self.tt_only or (self.SoF and not info.get("tt_done")):
            self.tt_sim += 1
            self.total_sim += 1
        else:
            self.full_sim += 1
            self.total_sim += self.num_corners
        for key, value in {
            "simulation/tt_sim": self.tt_sim,
            "simulation/full_sim": self.full_sim,
            "simulation/total_sim": self.total_sim,
            "simulation/corner_norm_std": info.get("ep_corner_norm_std", 1.0),
        }.items():
            self.logger.record(key, value)
        return True

    def _on_training_start(self) -> None:
        self.tt_sim = self.full_sim = self.total_sim = 0

    def _on_training_end(self) -> None:
        self.logger.dump(self.num_timesteps)


class SaveCallback(BaseCallback):
    """Save the model every log_interval steps and at the end (only when save=True)."""

    def __init__(self, log_interval, log_path, save=False, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.log_path = log_path
        self.save = save

    def _on_step(self) -> bool:
        if self.save and self.num_timesteps % self.log_interval == 0:
            self.model.save(os.path.join(self.log_path, f"model_{self.num_timesteps}"))
        return True

    def _on_training_end(self) -> None:
        if self.save:
            self.model.save(os.path.join(self.log_path, "model_last"))


class EvaluationCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        n_eval_episodes,
        eval_freq,
        traj_len,
        initial_log_timesteps=0,
        verbose=1,
    ):
        super().__init__(verbose)
        self.initial_log_timesteps = initial_log_timesteps
        self.last_logged_timesteps = initial_log_timesteps
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.traj_len = traj_len

    def _on_step(self):
        if (
            self.eval_freq <= 0
            or self.num_timesteps < self.last_logged_timesteps + self.eval_freq
        ):
            return True
        self.last_logged_timesteps = self.num_timesteps
        rewards, lengths, corner_stds = [], [], []
        successes = 0
        self.eval_env.reset_idx()
        for episode in range(self.n_eval_episodes):
            observation, reset_info = self.eval_env.reset()
            total_reward = 0.0
            for step in range(self.traj_len):
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                total_reward += reward
                if terminated:
                    successes += 1
                    corner_stds.append(info["ep_corner_norm_std"])
                if terminated or truncated or step + 1 == self.traj_len:
                    rewards.append(total_reward)
                    lengths.append(step + 1)
                    if self.verbose:
                        print(f"eval task {episode + 1}: success={terminated}")
                        print("target specs:", reset_info["target_specs"])
                        print("cur specs:", info["cur_specs"])
                        print("params:", info["params"])
                    break
        for key, value in {
            "eval/mean_reward": np.mean(rewards),
            "eval/mean_ep_length": np.mean(lengths),
            "eval/corner_norm_std": np.mean(corner_stds) if corner_stds else 1.0,
            "eval/success_rate": successes / self.n_eval_episodes,
        }.items():
            self.logger.record(key, value)
        self.logger.dump(self.num_timesteps)
        return True

    def _on_training_start(self):
        self.last_logged_timesteps = self.initial_log_timesteps
