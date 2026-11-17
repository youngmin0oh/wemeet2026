"""Training, checkpoint, and evaluation callbacks."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    def __init__(
        self,
        num_corners,
        env,
        verbose=0,
        obj_style="softmax",
        temp=10.0,
        pareto_freq=0,
    ):
        super().__init__(verbose)
        self.tt_sim = 0
        self.full_sim = 0
        self.total_sim = 0
        self.num_corners = num_corners
        self.env = env
        self.obs, _ = env.reset()
        self.SoF = env.SoF
        self.done_in_goal = getattr(env, "done_in_goal", False)
        self.obj_style = obj_style
        self.temp = temp
        self.pareto_freq = pareto_freq

    def _on_step(self) -> bool:
        if self.SoF:
            if self.locals["infos"][0].get("tt_done"):
                self.total_sim += self.num_corners
                self.full_sim += 1
            else:
                self.total_sim += 1
                self.tt_sim += 1
        else:
            self.total_sim += self.num_corners
            self.full_sim += 1
        ep_corner_norm_std = self.locals["infos"][0].get("ep_corner_norm_std", 1.0)
        pareto_buffer_size = self.locals["infos"][0].get("pareto_buffer_size", 0)
        for key, value in {
            "simulation/tt_sim": self.tt_sim,
            "simulation/full_sim": self.full_sim,
            "simulation/total_sim": self.total_sim,
            "simulation/corner_norm_std": ep_corner_norm_std,
            "goals/pareto_buffer_size": pareto_buffer_size,
        }.items():
            self.logger.record(key, value)

        # Pareto Dominant Goal Sampling
        if self.pareto_freq > 0:
            if self.locals["infos"][0].get("done") or self.locals["infos"][0].get(
                "truncated"
            ):
                obs = self.obs
                obs_tensor_dict = {}
                goals_norm = self.env.get_goals_norm()
                goals_norm_tensor = torch.tensor(
                    goals_norm, dtype=torch.float32, device="cpu"
                )
                num_goals = goals_norm_tensor.shape[0]
                if self.done_in_goal:
                    done_padding = torch.ones(num_goals, 2, device="cpu")
                    goals_norm_tensor = torch.cat(
                        [goals_norm_tensor, done_padding], dim=1
                    )

                for key, value in obs.items():
                    value_tensor = torch.tensor(
                        value, dtype=torch.float32, device="cpu"
                    ).unsqueeze(0)
                    value_tensor = value_tensor.repeat(num_goals, 1)
                    obs_tensor_dict[key] = value_tensor
                obs_tensor_dict["desired_goal"] = goals_norm_tensor

                with torch.no_grad():
                    action_batch = self.model.predict(
                        obs_tensor_dict, deterministic=False
                    )[0]
                    action_batch = torch.tensor(
                        action_batch, dtype=torch.float32, device=self.model.device
                    )
                    obs_tensor_dict = {
                        key: value.to(self.model.device)
                        for key, value in obs_tensor_dict.items()
                    }
                    q1_batch, q2_batch = self.model.critic(
                        obs_tensor_dict, action_batch
                    )

                    if self.obj_style == "max":
                        mean_vals = (-(q1_batch + q2_batch) / 2).squeeze()
                        sampled_goal_idx = mean_vals.argmax().cpu().item()

                    elif self.obj_style == "softmax":
                        mean_vals = (-(q1_batch + q2_batch) / 2).squeeze()
                        probs = F.softmax(mean_vals / self.temp, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        sampled_goal_idx = dist.sample().cpu().item()

                    else:  # random sampling
                        sampled_goal_idx = 0

                self.env.set_goal_idx(sampled_goal_idx)
        return True

    def _on_training_start(self) -> None:
        self.tt_sim = 0
        self.full_sim = 0
        self.total_sim = 0

    def _on_training_end(self) -> None:
        self.logger.dump(self.num_timesteps)


class SaveCallback(BaseCallback):
    def __init__(self, log_interval, log_path, algo, save=False, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.log_path = log_path
        self.algo = algo
        self.save = save

    def _on_step(self) -> bool:
        if self.save:
            log_path = os.path.join(self.log_path, "model_" + str(self.num_timesteps))
            if self.num_timesteps % self.log_interval == 0:
                self.model.save(log_path)
                if self.algo == "HER":
                    self.model.save_replay_buffer(log_path)
        return True

    def _on_training_end(self) -> None:
        if self.save:
            log_path = os.path.join(self.log_path, "model_last")
            self.model.save(log_path)
            if self.algo == "HER":
                self.model.save_replay_buffer(log_path)


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
