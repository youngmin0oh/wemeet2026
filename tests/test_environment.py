"""Regression coverage for reward stages and Gymnasium observation variants."""

import pickle

import numpy as np
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env

from analog_sizing.config import BENCHMARKS, benchmark_config, load_config
from analog_sizing.envs import ngspice_env_base
from analog_sizing.envs.ngspice_env import ngspice_env_cont, ngspice_env_disc
from analog_sizing.envs.ngspice_env_goal import ngspice_env_goal
from analog_sizing.gen_specs import gen_data


@pytest.fixture
def circuit(tmp_path, monkeypatch):
    rows = {"nominal": [120.0, 8.0], "hot": [80.0, 15.0], "cold": [110.0, 9.0]}
    config = {
        "dsn_netlist": list(rows),
        "params": {"width": [0.5, 10.0]},
        "init_params": [2.0],
        "prec_params": [3],
        "normalize": [100.0, 10.0],
        "target_specs": {"gain_min": [90, 110], "power_max": [8.0, 12.0]},
    }
    yaml_path = tmp_path / "circuit.yaml"
    yaml_path.write_text(yaml.safe_dump(config))

    class Circuit:
        def __init__(self, yaml_path, path, design_netlists=None):
            self.names = (
                config["dsn_netlist"] if design_netlists is None else design_netlists
            )

        def run(self, state):
            specs = [
                dict(zip(("gain_min", "power_max"), rows[name])) for name in self.names
            ]
            return (), specs, ()

    monkeypatch.setattr(ngspice_env_base, "CircuitClass", Circuit)
    return {"CIR_YAML": str(yaml_path), "generalize": False, "episode_len": 2}, rows


def test_nominal_failure_skips_corners_and_preserves_reward(circuit):
    config, rows = circuit
    rows["nominal"] = [90.0, 12.0]
    env = ngspice_env_cont(config)
    obs, info = env.reset(seed=0)
    assert info["reward"] == pytest.approx(-1.3588517)
    obs, reward, terminated, truncated, info = env.step(np.zeros(1))
    assert reward == pytest.approx(-1.4588517)
    assert (terminated, truncated, info["tt_sim"], info["full_sim"]) == (
        False,
        False,
        1,
        0,
    )
    assert env.observation_space.contains(obs)
    assert (type(reward), type(terminated), type(truncated)) == (float, bool, bool)


@pytest.mark.parametrize("skip_on_fail", [False, True])
@pytest.mark.parametrize("concat", [False, True])
def test_worst_corners_and_observation_shape(circuit, skip_on_fail, concat):
    config, _ = circuit
    env = ngspice_env_cont({**config, "SoF": skip_on_fail, "concat_all_specs": concat})
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    np.testing.assert_array_equal(info["cur_specs"], [80.0, 15.0])
    assert info["all_specs"].shape == (3, 2)
    obs, reward, terminated, truncated, info = env.step(np.zeros(1))
    assert env.observation_space.contains(obs)
    assert reward == pytest.approx(-0.20297141 if skip_on_fail else -0.98074914)
    assert info["full_sim"] == 1
    assert not terminated
    assert env.step(np.zeros(1))[3]


def test_tt_only_ignores_corners(circuit):
    config, rows = circuit
    env = ngspice_env_cont({**config, "tt_only": True})
    assert env.num_corners == 1
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    # The hot corner (80, 15) would fail; only nominal (120, 8) is simulated.
    np.testing.assert_array_equal(info["cur_specs"], rows["nominal"])
    assert info["all_specs"].shape == (1, 2)
    obs, reward, terminated, truncated, info = env.step(np.zeros(1))
    assert (reward, terminated, info["tt_done"]) == (30.0, True, True)
    assert (info["tt_sim"], info["full_sim"]) == (1, 0)


def test_tt_only_failure_uses_full_reward_range(circuit):
    config, rows = circuit
    rows["nominal"] = [90.0, 12.0]
    env = ngspice_env_cont({**config, "tt_only": True})
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(np.zeros(1))
    spec_diff = env.aggregate(info["cur_specs"], env.specs_ideal)
    assert reward == pytest.approx(-env.min_threshold * spec_diff / 2)
    assert env.min_threshold < reward < 0
    assert (terminated, info["tt_done"], info["tt_sim"]) == (False, False, 1)


@pytest.mark.parametrize("nominal", [[120.0, 8.0], [90.0, 12.0]])
def test_tt_only_hindsight_reward_matches_step_reward(circuit, nominal):
    config, rows = circuit
    rows["nominal"] = nominal
    env = ngspice_env_goal({**config, "tt_only": True})
    env.reset(seed=0)
    obs, reward, _, _, info = env.step(np.zeros(1))
    batch = {key: np.stack([value]) for key, value in obs.items()}
    relabeled = env.compute_reward(batch["achieved_goal"], batch["desired_goal"], [info])
    np.testing.assert_allclose(relabeled, [reward], rtol=1e-4)


@pytest.mark.parametrize("env_type", [ngspice_env_cont, ngspice_env_disc])
def test_seeded_reset_and_framework_contract(circuit, env_type):
    config, _ = circuit
    config.update(generalize=True, online_goal=True)
    before = config.copy()
    env = env_type(config)
    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first_info["target_specs"], second_info["target_specs"]
    )
    assert config == before
    check_env(env, warn=True)
    params = env.translate_params(env.cur_params)
    assert 0.5 <= params[0] <= 10.0


@pytest.mark.parametrize("done_in_goal", [False, True])
@pytest.mark.parametrize("goal_in_obs", [False, True])
@pytest.mark.parametrize("concat", [False, True])
def test_goal_observations_and_batched_rewards(
    circuit, done_in_goal, goal_in_obs, concat
):
    config, _ = circuit
    env = ngspice_env_goal(
        {
            **config,
            "SoF": False,
            "done_in_goal": done_in_goal,
            "goal_in_obs": goal_in_obs,
            "concat_all_specs": concat,
        }
    )
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, _, _, _, info = env.step(np.zeros(1))
    assert env.observation_space.contains(obs)
    batch = {key: np.stack([value, value]) for key, value in obs.items()}
    rewards = env.compute_reward(
        batch["achieved_goal"], batch["desired_goal"], [info, info]
    )
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (2,)
    assert np.isfinite(rewards).all()
    env.relabel_goal = False
    np.testing.assert_allclose(
        env.compute_reward(batch["achieved_goal"], batch["desired_goal"], [info, info]),
        [info["reward"], info["reward"]],
    )


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_targets_are_valid_reproducible_and_loadable(tmp_path, name):
    config = benchmark_config(name)
    source = load_config(config["CIR_YAML"])
    output = tmp_path / "targets.pkl"
    gen_data(config["CIR_YAML"], "test", 5, seed=42, output=output)
    first = output.read_bytes()
    gen_data(config["CIR_YAML"], "test", 5, seed=42, output=output)
    assert output.read_bytes() == first
    targets = pickle.loads(first)
    assert list(targets) == list(source["target_specs"])
    for key, values in targets.items():
        low, high = source["target_specs"][key]
        assert len(values) == 5
        assert all(low <= value <= high for value in values)


@pytest.mark.parametrize("done_in_goal", [False, True])
def test_goal_sampling_uses_the_raw_environment(circuit, monkeypatch, done_in_goal):
    import torch
    from stable_baselines3.common.logger import configure
    from analog_sizing.callbacks import TrainingCallback

    config, _ = circuit
    env = ngspice_env_goal(
        {**config, "online_goal": True, "done_in_goal": done_in_goal}
    )
    env.reset(seed=0)
    env.step(np.zeros(1))
    _, _, _, _, info = env.step(np.zeros(1))
    chosen = []
    monkeypatch.setattr(env, "set_goal_idx", chosen.append)
    callback = TrainingCallback(env.num_corners, env, pareto_freq=1)

    class Model:
        device = torch.device("cpu")

        def predict(self, observations, deterministic):
            goals = observations["desired_goal"]
            assert goals.shape[1] == len(env.specs_id) + (2 if done_in_goal else 0)
            return np.zeros((goals.shape[0], len(env.params_id))), None

        def critic(self, observations, actions):
            values = torch.zeros((actions.shape[0], 1))
            return values, values

    callback.model = Model()
    callback.model.logger = configure(format_strings=[])
    callback.locals = {"infos": [info]}
    assert callback._on_step()
    assert len(chosen) == 1
    assert 0 <= chosen[0] < 16
    assert callback.logger.name_to_value["simulation/total_sim"] == 3
    assert callback.logger.name_to_value["simulation/full_sim"] == 1


def test_final_training_metrics_are_flushed(circuit, tmp_path):
    import csv
    from types import SimpleNamespace
    from stable_baselines3.common.logger import configure
    from analog_sizing.callbacks import TrainingCallback

    config, _ = circuit
    env = ngspice_env_cont(config)
    callback = TrainingCallback(env.num_corners, env)
    logger = configure(str(tmp_path / "training"), ["csv"])
    callback.model = SimpleNamespace(logger=logger)
    callback.num_timesteps = 1
    callback.locals = {"infos": [env.step(np.zeros(1))[4]]}
    assert callback._on_step()
    callback._on_training_end()
    logger.close()
    with (tmp_path / "training/progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["simulation/total_sim"]) == 3
    assert float(rows[0]["simulation/full_sim"]) == 1


def test_evaluation_metrics_are_written_to_the_local_logger(circuit, tmp_path):
    import csv
    from types import SimpleNamespace
    from stable_baselines3.common.logger import configure
    from analog_sizing.callbacks import EvaluationCallback

    config, _ = circuit
    env = ngspice_env_cont(config)
    logger = configure(str(tmp_path / "evaluation"), ["csv"])
    callback = EvaluationCallback(
        env, n_eval_episodes=2, eval_freq=1, traj_len=2, verbose=0
    )
    callback.model = SimpleNamespace(
        logger=logger,
        predict=lambda observation, deterministic: (np.zeros(1), None),
    )
    callback.num_timesteps = 1
    assert callback._on_step()
    logger.close()
    with (tmp_path / "evaluation/progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["eval/success_rate"]) == 0
    assert float(rows[0]["eval/mean_ep_length"]) == 2
    assert float(rows[0]["eval/mean_reward"]) == pytest.approx(-0.40594282)
