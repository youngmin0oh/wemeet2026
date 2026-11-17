"""Discrete and continuous parameter updates for the circuit environment."""

import numpy as np

from analog_sizing.envs.ngspice_env_base import ngspice_env


class ngspice_env_disc(ngspice_env):
    def __init__(self, env_config):
        super().__init__({**env_config, "action_type": "discrete"})
        self.params = []
        bins = int(env_config.get("discrete_bins", 101))
        if bins < 2:
            raise ValueError("discrete_bins must be at least 2")
        for value in self.params_val:
            if len(value) == 3:
                param_vec = np.arange(*value)
            else:
                param_vec = np.linspace(value[0], value[1], bins)
            self.params.append(param_vec)
        self.initial_vals = self.yaml_data["init_params"]

    def init_params(self):
        return np.array(
            [
                np.argmin(np.abs(values - initial))
                for values, initial in zip(self.params, self.initial_vals)
            ],
            dtype=np.int64,
        )

    def update_params(self, action):
        self.cur_params = self.cur_params + np.array(
            [self.action_meaning[a] for a in action]
        )
        self.cur_params = np.clip(
            self.cur_params,
            [0] * len(self.params_id),
            [(len(param_vec) - 1) for param_vec in self.params],
        )
        return self.cur_params

    def translate_params(self, cur_params):
        return [self.params[i][cur_params[i]] for i in range(len(self.params_id))]


class ngspice_env_cont(ngspice_env):
    def __init__(self, env_config):
        super().__init__({**env_config, "action_type": "continuous"})
        self.params = []
        for value in self.params_val:
            param_vec = np.array([value[0], value[1]], dtype=np.float64)
            self.params.append(param_vec)
        self.params = np.array(self.params, dtype=np.float64)
        self.initial_vals = self.yaml_data["init_params"]
        self.prec_params = self.yaml_data["prec_params"]

    def init_params(self):
        init_params = np.array(self.initial_vals)
        return init_params

    def update_params(self, action):
        scale = 1.0 * self.params[:, 1] / self.episode_len
        self.cur_params = self.cur_params + scale * np.array(action, dtype=np.float64)
        precision = self.prec_params
        self.cur_params = np.array(
            [
                np.round(self.cur_params[i], precision[i])
                for i in range(len(self.cur_params))
            ]
        )
        self.cur_params = np.clip(self.cur_params, self.params[:, 0], self.params[:, 1])
        return self.cur_params

    def translate_params(self, cur_params):
        return cur_params
