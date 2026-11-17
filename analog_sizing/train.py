"""Train analog circuit sizing policies with Stable-Baselines3."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import click
from stable_baselines3 import SAC, HerReplayBuffer, PPO, DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_sizing.config import BENCHMARKS, PROJECT_ROOT, benchmark_config
from analog_sizing.envs.ngspice_env_goal import ngspice_env_goal
from analog_sizing.envs.ngspice_env import ngspice_env_cont


@click.command()
# Stable baseline options.
@click.option(
    "--algo",
    help="Algorithm to use (PPO, HER, DDPG, SAC)",
    type=click.Choice(["HER", "PPO", "DDPG", "SAC"]),
    default="HER",
    required=False,
)
@click.option(
    "--learning_starts",
    help="Number of steps before starting training",
    type=int,
    default=1000,
    required=False,
)
@click.option(
    "--lr", help="Initial learning rate", type=float, default=3e-3, required=False
)
@click.option(
    "--n_goal",
    "n_sample_goal",
    help="Number of sampled goals",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--goal",
    "goal_selection_strategy",
    help="Goal selection strategy",
    type=click.Choice(["final", "episode", "future"]),
    default="future",
    required=False,
)
@click.option(
    "--gamma", help="Discount factor", type=float, default=0.9, required=False
)
@click.option(
    "--tau",
    help="Target smoothing coefficient",
    type=float,
    default=0.005,
    required=False,
)
@click.option("--batch_size", help="Batch size", type=int, default=256, required=False)
@click.option(
    "--total_timesteps",
    help="Total timesteps for training",
    type=int,
    default=12000,
    required=False,
)
@click.option(
    "--seed", help="Random seed (optional)", type=int, default=50, required=False
)

# Run options
@click.option(
    "--name",
    help="Name of the run",
    type=str,
    default="default_run_name",
    required=False,
)
@click.option(
    "--env_name",
    help="Environment to use (TSA, CMA, COMP, LDO)",
    type=click.Choice(list(BENCHMARKS)),
    default="COMP",
    required=False,
)
# logging & evaluation options
@click.option(
    "--log_interval",
    help="Logging(save model) interval",
    type=int,
    default=6000,
    required=False,
)
@click.option(
    "--eval_yaml",
    "eval_CIR_YAML",
    help="Path to the yaml file for evaluation",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--eval_spec_path",
    help="Path to the pre-fixed target specs file for evaluation",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--eval_freq", help="Evaluation frequency", type=int, default=6000, required=False
)
@click.option(
    "--num_eval",
    help="Number of evaluation episodes",
    type=int,
    default=20,
    required=False,
)

# environment options
@click.option(
    "--yaml",
    "CIR_YAML",
    help="Path to the yaml file",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--spec_path",
    help="Path to the pre-fixed target specs file for training",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--generalize",
    help="Generalize the environment",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--episode_len", help="Episode length", type=int, default=30, required=False
)
@click.option(
    "--goal_in_obs",
    help="Goal in observation",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--done_in_goal", help="Done in goal", type=bool, default=True, required=False
)
@click.option(
    "--SoF",
    help="Skip on Fail",
    type=bool,
    default=True,
    required=False,
)
@click.option(
    "--lookup_style",
    help="Lookup style",
    type=click.Choice(["normd", "tanh"]),
    default="tanh",
    required=False,
)
@click.option(
    "--min_threshold",
    help="Minimum threshold",
    type=float,
    default=-6.0,
    required=False,
)
@click.option(
    "--tt_threshold", help="TT threshold", type=float, default=-1.0, required=False
)
@click.option(
    "--tt_threshold_2",
    help="Conservative TT threshold",
    type=float,
    default=-3.0,
    required=False,
)
@click.option(
    "--concat_all_specs",
    help="Concat all specs",
    type=bool,
    default=False,
    required=False,
)
@click.option(
    "--alpha",
    help="regularization coefficient to minimize corner variation",
    type=float,
    default=0.1,
    required=False,
)
@click.option(
    "--online_goal",
    help="Sample goal from target range every episode",
    type=bool,
    default=True,
    required=False,
)

# PGDS options
@click.option(
    "--pareto_freq",
    help="Pareto goal sampling frequency",
    type=int,
    default=1,
    required=False,
)
@click.option(
    "--obj_style",
    help="PGDS sampling strategy",
    type=str,
    default="softmax",
    required=False,
)
@click.option(
    "--temp", help="sampling temperature", type=float, default=10.0, required=False
)
@click.option("--n_warmup", help="warmup size", type=int, default=2, required=False)
def main(**kwargs):
    from analog_sizing.callbacks import (
        EvaluationCallback,
        SaveCallback,
        TrainingCallback,
    )

    defaults = benchmark_config(kwargs["env_name"])
    kwargs["CIR_YAML"] = kwargs["CIR_YAML"] or defaults["CIR_YAML"]
    kwargs["eval_CIR_YAML"] = kwargs["eval_CIR_YAML"] or kwargs["CIR_YAML"]
    kwargs["spec_path"] = kwargs["spec_path"] or defaults["spec_path"]
    kwargs["eval_spec_path"] = kwargs["eval_spec_path"] or defaults["spec_path"]
    if kwargs["algo"] != "HER":
        kwargs["pareto_freq"] = 0
    cfg = SimpleNamespace(**kwargs)
    env_config = {key: value for key, value in kwargs.items() if value is not None}
    env_config["run_valid"] = False
    env_config["relabel_goal"] = cfg.algo == "HER"
    eval_env_config = env_config.copy()
    eval_env_config["generalize"] = True
    eval_env_config["run_valid"] = True
    eval_env_config["spec_path"] = cfg.eval_spec_path
    eval_env_config["CIR_YAML"] = cfg.eval_CIR_YAML
    eval_env_config["online_goal"] = False

    sim_env = ngspice_env_goal if cfg.algo == "HER" else ngspice_env_cont

    envs = sim_env(env_config=env_config)
    eval_envs = sim_env(env_config=eval_env_config)

    try:
        seed = cfg.seed
        envs.action_space.seed(seed)
        set_random_seed(seed)

        envs.tt_sim = 0
        envs.full_sim = 0

        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256, 256], qf=[256, 256, 128]))

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

        model_kwargs = dict(
            learning_rate=cfg.lr,
            gamma=cfg.gamma,
            tensorboard_log=str(log_dir / cfg.algo.lower()),
            verbose=1,
            seed=seed,
        )
        if cfg.algo == "PPO":
            model = PPO(
                "MlpPolicy",
                envs,
                n_steps=360,
                batch_size=120,
                gae_lambda=0.85,
                **model_kwargs,
            )
        else:
            model_kwargs.update(
                policy_kwargs=policy_kwargs,
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
            )
            if cfg.algo == "HER":
                model_kwargs.update(
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                        n_sampled_goal=cfg.n_sample_goal,
                        goal_selection_strategy=cfg.goal_selection_strategy,
                        copy_info_dict=True,
                    ),
                )
            model_class = DDPG if cfg.algo == "DDPG" else SAC
            policy = "MultiInputPolicy" if cfg.algo == "HER" else "MlpPolicy"
            model = model_class(policy, envs, **model_kwargs)

        tb_log_base = f"{cfg.algo}_{cfg.env_name}_{cfg.name}"
        tb_log_name = tb_log_base + "_" + str(cfg.total_timesteps)
        model_dir = str(PROJECT_ROOT / "models" / tb_log_base)

        os.makedirs(model_dir, exist_ok=True)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
            callback=CallbackList(
                [
                    TrainingCallback(
                        num_corners=envs.num_corners,
                        env=envs,
                        obj_style=cfg.obj_style,
                        temp=cfg.temp,
                        pareto_freq=cfg.pareto_freq,
                    ),
                    EvaluationCallback(
                        eval_env=eval_envs,
                        n_eval_episodes=cfg.num_eval,
                        eval_freq=cfg.eval_freq,
                        traj_len=cfg.episode_len,
                        initial_log_timesteps=0,
                    ),
                    SaveCallback(cfg.log_interval, model_dir, cfg.algo),
                ]
            ),
        )

    finally:
        envs.close()
        eval_envs.close()
    print("Finished training:", tb_log_name)


if __name__ == "__main__":
    main()
