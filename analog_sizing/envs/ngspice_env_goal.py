"""Goal-conditioned observations and batched hindsight rewards."""

import math

from gymnasium import spaces
import numpy as np
import torch

from analog_sizing.envs.ngspice_env import ngspice_env_cont


class ngspice_env_goal(ngspice_env_cont):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.goal_in_obs = env_config.get("goal_in_obs", True)
        self.done_in_goal = env_config.get("done_in_goal", True)
        self.relabel_goal = env_config.get("relabel_goal", True)

        base_space = self.observation_space
        param_count = len(self.params_id)
        if not self.goal_in_obs:
            obs_space = spaces.Box(
                -np.inf, np.inf, shape=(param_count,), dtype=np.float64
            )
        elif self.done_in_goal:
            obs_space = spaces.Box(
                low=np.concatenate(
                    (
                        base_space.low[:-param_count],
                        [0.0, 0.0],
                        base_space.low[-param_count:],
                    )
                ),
                high=np.concatenate(
                    (
                        base_space.high[:-param_count],
                        [1.0, 1.0],
                        base_space.high[-param_count:],
                    )
                ),
                dtype=np.float64,
            )
        else:
            obs_space = base_space
        flags = 2 if self.done_in_goal else 0
        goal_space = spaces.Box(
            low=np.array([-np.inf] * len(self.specs_id) + [0.0] * flags),
            high=np.array([np.inf] * len(self.specs_id) + [1.0] * flags),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "achieved_goal": goal_space,
                "desired_goal": spaces.Box(
                    goal_space.low.copy(), goal_space.high.copy(), dtype=np.float64
                ),
            }
        )
        self.get_from_info = True

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.gc_obs_init = self.translate_obs(obs, info)
        return self.gc_obs_init, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        obs = self.translate_obs(obs, info)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        return obs, reward, done, truncated, info

    def get_obs(self):
        return self.gc_obs_init

    def compute_reward(self, achieved_goal, desired_goal, infos):
        if isinstance(infos, dict):
            return infos["reward"]

        # virtual transition (batch)
        device = getattr(self, "device", None) or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ag = torch.as_tensor(achieved_goal, device=device, dtype=torch.float32)
        dg = torch.as_tensor(desired_goal, device=device, dtype=torch.float32)

        if not self.relabel_goal:
            base = [info["reward"] for info in infos]
            return np.asarray(base, dtype=np.float32)

        if self.done_in_goal:
            cur_norm = ag[:, :-2]
            goal_norm = dg[:, :-2]
            done = ag[:, -2]
            tt_done = ag[:, -1]
        else:
            cur_norm, goal_norm = ag, dg
            done = torch.as_tensor([info["done"] for info in infos], device=device)
            tt_done = torch.as_tensor(
                [info["tt_done"] for info in infos], device=device
            )

        cur_specs = self.unlookup(cur_norm, self.global_g)
        goal_specs = self.unlookup(goal_norm, self.global_g)

        all_specs = torch.stack(
            [
                torch.as_tensor(info["all_specs"], device=device, dtype=torch.float32)
                for info in infos
            ],
            dim=0,
        )

        base_r = self.reward_batch(cur_specs, goal_specs)
        if self.tt_only:
            # With only TT simulated, the achieved goal is the complete result,
            # so success is decided again for the relabeled goal (as in step).
            reward = torch.where(
                base_r >= 0,
                torch.full_like(base_r, 30.0),
                -self.min_threshold * (base_r / len(self.specs_id)),
            )
            return reward.cpu().numpy()
        if all_specs.shape[1] > 1:
            corner_std = (
                torch.sqrt(
                    torch.mean(
                        (all_specs[:, 1:] / all_specs[:, [0]] - 1.0).square(),
                        dim=(1, 2),
                    )
                )
                .clamp(0.0, 1.0)
                .nan_to_num(nan=1.0)
            )
        else:
            corner_std = torch.zeros_like(base_r)
        done_mask = done > 0.5
        tt_mask = tt_done > 0.5

        reward = torch.where(
            done_mask,
            torch.full_like(base_r, 30.0),
            torch.where(
                tt_mask,
                (0.0 - self.tt_threshold_2) * (base_r / len(self.specs_id))
                - self.alpha * corner_std,
                self.tt_threshold_2
                + (self.tt_threshold_2 - self.min_threshold)
                * (base_r / len(self.specs_id))
                - self.alpha,
            ),
        )

        return reward.cpu().numpy()

    def unlookup(self, norm_spec: torch.Tensor, goal_spec):
        # Ensure goal_spec is a torch Tensor on the same device/dtype
        if not isinstance(goal_spec, torch.Tensor):
            goal_spec = torch.as_tensor(
                goal_spec, device=norm_spec.device, dtype=norm_spec.dtype
            )

        if self.lookup_style == "normd":
            return -(norm_spec + 1.0) * goal_spec / (norm_spec - 1.0)

        elif self.lookup_style == "tanh":
            scale = getattr(self, "scale_factor", 10.0)
            f = math.tanh(1.0 / scale)
            x = torch.clamp(norm_spec * f, -0.999, 0.999)
            atanh_x = 0.5 * (torch.log1p(x) - torch.log1p(-x))
            return goal_spec + goal_spec.abs() * scale * atanh_x

        else:
            raise ValueError(f"Unknown lookup_style {self.lookup_style!r}")

    def reward_batch(self, spec_batch: torch.Tensor, goal_spec_batch: torch.Tensor):
        device = spec_batch.device
        epsilon = 1e-9

        spec = spec_batch
        goal = goal_spec_batch
        # avoid division by zero
        goal_safe = torch.where(goal == 0.0, torch.full_like(goal, epsilon), goal)

        if self.lookup_style == "normd":
            delta = spec - goal_safe
            abs_delta = delta.abs() + epsilon
            denom = goal_safe.abs() + spec.abs() + epsilon
            rel_specs = torch.sign(delta) * torch.exp(
                torch.log(abs_delta) - torch.log(denom)
            )

        elif self.lookup_style == "tanh":
            scale = getattr(self, "scale_factor", 10.0)
            raw = (spec - goal_safe) / (scale * goal_safe.abs())
            rel_specs = torch.tanh(raw) / math.tanh(1.0 / scale)

        else:
            raise ValueError(f"Unknown lookup_style {self.lookup_style!r}")

        # --- Adjustment factors for "max" specs ---
        m = rel_specs.size(1)
        adj = torch.ones((m,), device=device)
        for i, sid in enumerate(self.specs_id):
            if sid.endswith("max"):
                adj[i] = -1.0
        adjusted = rel_specs * adj.unsqueeze(0)

        # --- Sign-mismatch penalty: a negative value against a non-negative goal ---
        sign_mismatch = (spec < 0.0) & (goal >= 0.0)
        adjusted = torch.where(sign_mismatch, torch.full_like(adjusted, -1.0), adjusted)

        negative_mask = adjusted < 0.0
        summed = torch.sum(
            torch.where(negative_mask, adjusted, torch.zeros_like(adjusted)), dim=1
        )

        rewards = torch.where(summed < -0.02, summed, torch.full_like(summed, 10.0))

        return rewards

    def translate_obs(self, obs, info):
        params = obs[-len(self.params_id) :]
        achieved_goal = self.lookup(info["cur_specs"], self.global_g).astype(np.float64)
        desired_goal = self.specs_ideal_norm.astype(np.float64)
        if self.done_in_goal:
            flags = np.array([info["done"], info["tt_done"]], dtype=np.float64)
            obs = np.concatenate((obs[: -len(self.params_id)], flags, params))
            achieved_goal = np.concatenate((achieved_goal, flags))
            desired_goal = np.concatenate((desired_goal, [1.0, 1.0]))
        return {
            "observation": obs if self.goal_in_obs else params,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
