"""Regression coverage for reward stages and Gymnasium observation variants."""

import pickle

import numpy as np
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env

from analog_sizing.config import BENCHMARKS, benchmark_config, load_config
from analog_sizing.envs import ngspice_env_base
from analog_sizing.envs.ngspice_env import ngspice_env_cont, ngspice_env_disc
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


@pytest.mark.parametrize("style", ["tanh", "normd"])
def test_negative_goals_keep_the_inequality_direction(circuit, style):
    config, _ = circuit
    env = ngspice_env_cont({**config, "lookup_style": style})
    goal = np.array([-3.0, -3.0])                 # gain_min >= -3, power_max <= -3
    assert env.aggregate(np.array([0.5, -4.0]), goal) == 10.0
    assert env.aggregate(np.array([-4.0, 0.5]), goal) < -0.02
    # Above the goal is positive and below is negative, also for negative goals.
    assert (np.sign(env.lookup(np.array([0.5, -4.0]), goal)) == [1, -1]).all()


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


def test_final_training_metrics_are_flushed(circuit, tmp_path):
    import csv
    from types import SimpleNamespace
    from stable_baselines3.common.logger import configure
    from analog_sizing.callbacks import TrainingCallback

    config, _ = circuit
    env = ngspice_env_cont(config)
    env.reset(seed=0)
    callback = TrainingCallback(env)
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


def test_cli_options_reach_the_environment_under_their_config_names():
    # click lowercases derived names (--SoF would arrive as "sof" and be ignored).
    from analog_sizing.train import main

    names = {param.name for param in main.params}
    assert {"SoF", "tt_only", "online_goal", "CIR_YAML", "eval_CIR_YAML"} <= names
