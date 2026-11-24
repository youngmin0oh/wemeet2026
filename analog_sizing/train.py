"""Train analog circuit sizing policies with Stable-Baselines3 (SAC, PPO, DDPG)."""

import sys
from pathlib import Path

import click
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_sizing.callbacks import EvaluationCallback, SaveCallback, TrainingCallback
from analog_sizing.config import BENCHMARKS, PROJECT_ROOT, benchmark_config
from analog_sizing.envs.ngspice_env import ngspice_env_cont


@click.command()
# Algorithm
@click.option("--algo", type=click.Choice(["SAC", "PPO", "DDPG"]), default="SAC", help="Stable-Baselines3 algorithm")
@click.option("--learning_starts", type=int, default=1000, help="Random steps before SAC/DDPG updates start")
@click.option("--lr", type=float, default=3e-3, help="Learning rate")
@click.option("--gamma", type=float, default=0.9, help="Discount factor")
@click.option("--tau", type=float, default=0.005, help="Target network smoothing coefficient (SAC/DDPG)")
@click.option("--batch_size", type=int, default=256, help="Minibatch size (SAC/DDPG)")
@click.option("--total_timesteps", type=int, default=12000, help="Training steps")
@click.option("--seed", type=int, default=50, help="Random seed")
# Run
@click.option("--name", default="default_run_name", help="Run name for the log and model directories")
@click.option("--env_name", type=click.Choice(list(BENCHMARKS)), default="COMP", help="Benchmark circuit")
@click.option("--log_interval", type=int, default=6000, help="Checkpoint interval in steps")
# Evaluation
@click.option("--eval_yaml", "eval_CIR_YAML", default=None, help="Circuit YAML for evaluation (default: --yaml)")
@click.option("--eval_spec_path", default=None, help="Evaluation targets (default: the benchmark's file)")
@click.option("--eval_freq", type=int, default=6000, help="Evaluate every N steps")
@click.option("--num_eval", type=int, default=20, help="Evaluation episodes (targets)")
# Environment
@click.option("--yaml", "CIR_YAML", default=None, help="Circuit YAML (default: the benchmark's file)")
@click.option("--spec_path", default=None, help="Training targets, used with --online_goal False")
@click.option("--generalize", type=bool, default=True, help="Train on many targets (False: one fixed target)")
@click.option("--episode_len", type=int, default=30, help="Steps per episode")
@click.option("--SoF", "SoF", type=bool, default=True, help="Skip on Fail: simulate the other corners only after TT passes")
@click.option("--tt_only", type=bool, default=False, help="Simulate only the nominal TT corner (training and evaluation)")
@click.option("--lookup_style", type=click.Choice(["normd", "tanh"]), default="tanh", help="Spec normalization")
@click.option("--min_threshold", type=float, default=-6.0, help="Lowest reward of a failing design")
@click.option("--tt_threshold", type=float, default=-1.0, help="Reward boundary between failing and passing TT")
@click.option("--concat_all_specs", type=bool, default=False, help="Observe every corner's specs, not only the worst")
@click.option("--alpha", type=float, default=0.1, help="Penalty weight for corner-to-corner variation")
@click.option("--online_goal", type=bool, default=True, help="Sample a new target from the target_specs ranges every episode")
def main(**kwargs):
    defaults = benchmark_config(kwargs["env_name"])
    kwargs["CIR_YAML"] = kwargs["CIR_YAML"] or defaults["CIR_YAML"]
    kwargs["eval_CIR_YAML"] = kwargs["eval_CIR_YAML"] or kwargs["CIR_YAML"]
    kwargs["spec_path"] = kwargs["spec_path"] or defaults["spec_path"]
    kwargs["eval_spec_path"] = kwargs["eval_spec_path"] or defaults["spec_path"]
    algo = kwargs["algo"]

    env_config = {**kwargs, "run_valid": False}
    # Evaluation goes through the fixed target list in order.
    eval_config = {
        **env_config,
        "generalize": True,
        "online_goal": False,
        "run_valid": True,
        "spec_path": kwargs["eval_spec_path"],
        "CIR_YAML": kwargs["eval_CIR_YAML"],
    }
    env = ngspice_env_cont(env_config)
    eval_env = ngspice_env_cont(eval_config)

    try:
        seed = kwargs["seed"]
        env.action_space.seed(seed)
        set_random_seed(seed)

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        common = dict(
            learning_rate=kwargs["lr"],
            gamma=kwargs["gamma"],
            tensorboard_log=str(log_dir / algo.lower()),
            verbose=1,
            seed=seed,
        )
        if algo == "PPO":
            model = PPO("MlpPolicy", env, n_steps=360, batch_size=120, gae_lambda=0.85, **common)
        else:
            model = (SAC if algo == "SAC" else DDPG)(
                "MlpPolicy",
                env,
                learning_starts=kwargs["learning_starts"],
                batch_size=kwargs["batch_size"],
                tau=kwargs["tau"],
                policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256, 256], qf=[256, 256, 128])),
                **common,
            )

        run_name = f"{algo}_{kwargs['env_name']}_{kwargs['name']}"
        model_dir = PROJECT_ROOT / "models" / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        tb_log_name = f"{run_name}_{kwargs['total_timesteps']}"
        model.learn(
            total_timesteps=kwargs["total_timesteps"],
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
            callback=CallbackList(
                [
                    TrainingCallback(env),
                    EvaluationCallback(
                        eval_env,
                        n_eval_episodes=kwargs["num_eval"],
                        eval_freq=kwargs["eval_freq"],
                        traj_len=kwargs["episode_len"],
                    ),
                    SaveCallback(kwargs["log_interval"], str(model_dir)),
                ]
            ),
        )
    finally:
        env.close()
        eval_env.close()
    print("Finished training:", tb_log_name)


if __name__ == "__main__":
    main()
