"""Gymnasium environment that sizes a circuit by simulating it with ngspice.

Observation = normalized current specs + normalized target specs + parameters.
Each step changes the parameters, simulates the netlists (TT first, then the
other PVT corners when TT passes), and scores the worst value of every
specification against the target.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import pickle

import gymnasium
from gymnasium import spaces
import numpy as np

from analog_sizing.config import (
    PROJECT_ROOT,
    benchmark_config,
    load_config,
    resolve_path,
    resolve_spec_path,
)
from eval_engines.ngspice.CircuitClass import CircuitClass


class ngspice_env(gymnasium.Env, ABC):
    metadata = {"render_modes": []}

    ACT_LOW = -1
    ACT_HIGH = 1

    # obtains yaml file
    path = str(PROJECT_ROOT)
    CIR_YAML = benchmark_config("TSA")["CIR_YAML"]

    def __init__(self, env_config):
        self.generalize = env_config.get("generalize", True)
        self.episode_len = env_config.get("episode_len", 30)
        self.valid = env_config.get("run_valid", False)
        self.CIR_YAML = str(resolve_path(env_config.get("CIR_YAML", self.CIR_YAML)))
        self.spec_path = env_config.get(
            "spec_path", benchmark_config("TSA")["spec_path"]
        )
        self.SoF = env_config.get("SoF", True)
        # Simulate only the nominal TT netlist (dsn_netlist[0]) and ignore PVT corners.
        self.tt_only = env_config.get("tt_only", False)
        self.lookup_style = env_config.get("lookup_style", "normd")
        self.min_threshold = env_config.get("min_threshold", -6.0)
        self.tt_threshold = env_config.get("tt_threshold", -1.0)
        self.concat_all_specs = env_config.get("concat_all_specs", False)
        self.alpha = env_config.get("alpha", 0.1)
        self.online_goal = env_config.get("online_goal", False)
        yaml_data = load_config(self.CIR_YAML)
        self.yaml_data = yaml_data

        # Targets: target_specs ranges (one fixed target, or sampled every episode)
        # or a pre-generated list of targets (spec_path).
        if self.generalize and not self.online_goal:
            with open(resolve_spec_path(self.spec_path), "rb") as f:
                specs = pickle.load(f)
        else:
            specs = yaml_data["target_specs"]
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))

        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.num_os = len(list(self.specs.values())[0])
        self.cur_steps = 0
        self.full_sim = 0
        self.tt_sim = 0
        self.num_corners = 1 if self.tt_only else len(yaml_data["dsn_netlist"])

        # param array
        params = yaml_data["params"]
        self.params_id = list(params.keys())
        self.params_val = list(params.values())

        self.episode_corner_norm_std = 1.0

        # initialize sim environment
        tt_design_netlist = [yaml_data["dsn_netlist"][0]]
        corner_design_netlists = yaml_data["dsn_netlist"][1:]
        self.tt_sim_env = CircuitClass(
            yaml_path=self.CIR_YAML,
            path=ngspice_env.path,
            design_netlists=tt_design_netlist,
        )
        if self.tt_only:
            # Every simulation set of a TT-only environment is the nominal netlist.
            self.corner_sim_env = None
            self.full_sim_env = self.tt_sim_env
        else:
            self.corner_sim_env = CircuitClass(
                yaml_path=self.CIR_YAML,
                path=ngspice_env.path,
                design_netlists=corner_design_netlists,
            )
            self.full_sim_env = CircuitClass(
                yaml_path=self.CIR_YAML, path=ngspice_env.path
            )

        if env_config.get("action_type") == "discrete":
            self.action_meaning = [-1, 0, 2]
            self.action_space = spaces.MultiDiscrete(
                [len(self.action_meaning)] * len(self.params_id)
            )
        elif env_config.get("action_type") == "continuous":
            self.action_space = spaces.Box(
                low=np.array([ngspice_env.ACT_LOW] * len(self.params_id)),
                high=np.array([ngspice_env.ACT_HIGH] * len(self.params_id)),
            )

        if self.concat_all_specs:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * (self.num_corners + 1) * len(self.specs_id)
                    + len(self.params_id) * [-np.inf]
                ),
                high=np.array(
                    [np.inf] * (self.num_corners + 1) * len(self.specs_id)
                    + len(self.params_id) * [np.inf]
                ),
                dtype=np.float64,
            )
            self.cur_specs = np.zeros(
                len(self.specs_id) * self.num_corners, dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array(
                    [-np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [-np.inf]
                ),
                high=np.array(
                    [np.inf] * 2 * len(self.specs_id) + len(self.params_id) * [np.inf]
                ),
                dtype=np.float64,
            )
            self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float64)

        if env_config.get("action_type") == "discrete":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.int32)
        elif env_config.get("action_type") == "continuous":
            self.cur_params = np.zeros(len(self.params_id), dtype=np.float64)

        # Get the g* (overall design spec) you want to reach
        self.g_star = np.array(yaml_data["normalize"])
        self.global_g = np.array(yaml_data["normalize"])

        # Next pre-generated target used by evaluation (run_valid=True)
        self.obj_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if not self.generalize:
            self.specs_ideal = self.g_star  # one fixed target
        elif self.online_goal:
            # A new target from the target_specs ranges every episode
            self.specs_ideal = self.sample_goal_uniform(list(self.specs.values()))[0]
        else:
            # A pre-generated target: in order for evaluation, at random for training
            if self.valid:
                idx = self.obj_idx % self.num_os
                self.obj_idx = idx + 1
            else:
                idx = int(self.np_random.integers(self.num_os))
            self.specs_ideal = np.array([spec[idx] for spec in self.specs.values()])

        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        self.cur_steps = 0
        self.cur_params = self.init_params()
        self.full_sim = 0
        self.tt_sim = 0
        cur_spec_norm, reward, tt_done, all_specs, _, _ = self._evaluate_params()
        done = bool(reward >= 0)
        truncated = False

        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "done": done,
            "truncated": truncated,
            "tt_done": tt_done,
        }
        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params]
        )
        return self.ob, info

    def step(self, action):
        action = np.asarray(action).reshape(-1)
        self.cur_params = self.update_params(action)
        (cur_spec_norm, reward, tt_done, all_specs, worst_idx, corner_norm_std) = (
            self._evaluate_params()
        )
        if self.tt_only or (self.SoF and not tt_done):
            self.tt_sim += 1
        else:
            self.full_sim += 1

        # add deviation penalty
        reward = reward - self.alpha * corner_norm_std

        self.cur_steps += 1
        done = reward >= 0
        truncated = self.cur_steps >= self.episode_len
        if done or truncated:
            self.episode_corner_norm_std = corner_norm_std

        self.ob = np.concatenate(
            [cur_spec_norm, self.specs_ideal_norm, self.cur_params]
        )
        info = {
            "reward": reward,
            "params": self.cur_params,
            "target_specs": self.specs_ideal,
            "cur_specs": self.cur_specs,
            "all_specs": all_specs,
            "worst_index": worst_idx,
            "done": done,
            "tt_done": tt_done,
            "truncated": truncated,
            "full_sim": self.full_sim,
            "tt_sim": self.tt_sim,
            "ep_corner_norm_std": self.episode_corner_norm_std,
        }
        return self.ob, float(reward), bool(done), bool(truncated), info

    def _evaluate_params(self):
        """Simulate corners and select each specification's worst value.

        Reset and step share the simulation and base reward calculation. Step
        applies the corner variation penalty and increments simulation counts.
        """
        tt_done = False
        skipped_corners = False
        if self.tt_only:
            all_specs = np.asarray(self.update(self.cur_params, self.tt_sim_env))
        elif self.SoF:
            nominal = self.update(self.cur_params, self.tt_sim_env)[0]
            nominal_diff = self.aggregate(nominal, self.specs_ideal)
            if nominal_diff < 0:
                skipped_corners = True
                all_specs = np.repeat([nominal], self.num_corners, axis=0)
            else:
                tt_done = True
                all_specs = np.asarray(
                    [nominal] + self.update(self.cur_params, self.corner_sim_env)
                )
        else:
            all_specs = np.asarray(self.update(self.cur_params, self.full_sim_env))

        norms = self.lookup(all_specs, self.global_g)
        directions = np.array(
            [-1.0 if sid.endswith("max") else 1.0 for sid in self.specs_id]
        )
        worst_idx = np.argmin(norms * directions, axis=0)
        columns = np.arange(len(self.specs_id))
        self.cur_specs = all_specs[worst_idx, columns]
        cur_spec_norm = (
            norms.flatten() if self.concat_all_specs else norms[worst_idx, columns]
        )
        spec_diff = self.aggregate(self.cur_specs, self.specs_ideal)
        if self.tt_only:
            # The nominal corner is the whole evaluation, so passing TT is success.
            tt_done = bool(spec_diff >= 0)

        if skipped_corners:
            reward = self.tt_threshold + (self.tt_threshold - self.min_threshold) * (
                spec_diff / len(self.specs_id)
            )
            corner_norm_std = 1.0
        else:
            # Without a TT stage before the corners, use the full reward range.
            threshold = (
                self.tt_threshold if self.SoF and not self.tt_only else self.min_threshold
            )
            reward = (
                -threshold * spec_diff / len(self.specs_id) if spec_diff < 0 else 30.0
            )
            if len(all_specs) > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    deviation = np.sqrt(
                        np.mean((all_specs[1:] / all_specs[0] - 1) ** 2)
                    )
                corner_norm_std = float(
                    np.nan_to_num(np.clip(deviation, 0, 1), nan=1.0)
                )
            else:
                corner_norm_std = 0.0
        return (
            cur_spec_norm,
            float(reward),
            tt_done,
            all_specs,
            worst_idx,
            corner_norm_std,
        )

    def lookup(self, spec, goal_spec):
        spec = np.asarray(spec, dtype=np.float32)
        goal_spec = np.asarray(goal_spec, dtype=np.float32)
        epsilon = 1e-9
        goal_spec = np.where(goal_spec == 0, epsilon, goal_spec)
        if self.lookup_style == "normd":
            delta = spec - goal_spec
            abs_delta = np.abs(delta) + epsilon  # Avoid log(0)
            # |goal| keeps the sign of the comparison right for negative goals.
            denom = np.abs(goal_spec) + np.abs(spec) + epsilon  # Avoid log(0)
            norm_spec = np.sign(delta) * np.exp(np.log(abs_delta) - np.log(denom))

        elif self.lookup_style == "tanh":
            scale_factor = 10.0
            norm_spec = np.tanh(
                (spec - goal_spec) / (scale_factor * np.abs(goal_spec))
            ) / np.tanh(1 / scale_factor)
        else:
            raise ValueError(f"Unknown lookup_style {self.lookup_style!r}")
        return norm_spec

    def aggregate(self, spec, goal_spec):
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if self.specs_id[i][-3:] == "max":
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec
        return reward if reward < -0.02 else 10.0

    def update(self, cur_params, sim_env):
        params = self.translate_params(cur_params)
        param_val = [OrderedDict(list(zip(self.params_id, params)))]
        _, specs, _ = sim_env.run(param_val[0])
        cur_specs = []
        for spec in specs:
            cur_spec = OrderedDict(sorted(spec.items(), key=lambda k: k[0]))
            cur_specs.append(np.array(list(cur_spec.values())))
        return cur_specs

    def sample_goal_uniform(self, specs_range_vals, num_goals_to_sample=1):
        sampled_goals = []
        while len(sampled_goals) < num_goals_to_sample:
            specs_valid = []
            for spec in specs_range_vals:
                if isinstance(spec[0], int):
                    specs_valid.append(
                        self.np_random.integers(int(spec[0]), int(spec[1]) + 1)
                    )
                else:
                    specs_valid.append(
                        self.np_random.uniform(float(spec[0]), float(spec[1]))
                    )
            sampled_goals.append(specs_valid)
        return np.array(sampled_goals)

    def reset_idx(self):
        self.obj_idx = 0

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def update_params(self, action):
        pass

    @abstractmethod
    def translate_params(self, cur_params):
        pass
