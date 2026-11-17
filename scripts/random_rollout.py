"""Run a few Gymnasium steps using randomly sampled actions."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_sizing.config import BENCHMARKS, benchmark_config
from analog_sizing.envs.ngspice_env import ngspice_env_cont, ngspice_env_disc


def print_state(env, observation, info):
    """Show every observation value alongside its physical meaning."""
    count = len(env.specs_id)
    name_width = max(24, *(len(name) for name in env.specs_id + env.params_id))
    if env.SoF and not info["tt_done"]:
        print(
            f"  Simulation: nominal TT only; {env.num_corners - 1} other corners skipped"
        )
    else:
        print(
            f"  Simulation: all {env.num_corners} corners; worst values per specification"
        )
    print(
        f"  {'Specification':<{name_width}} {'Current':>12} {'Target':>15} {'Obs.current':>12} {'Obs.target':>12}"
    )
    for i, name in enumerate(env.specs_id):
        relation = "<=" if name.endswith("max") else ">="
        target = f"{relation} {info['target_specs'][i]:.6g}"
        print(
            f"  {name:<{name_width}} {info['cur_specs'][i]:>12.6g} {target:>15} "
            f"{observation[i]:>12.6g} {observation[count + i]:>12.6g}"
        )
    print(f"  {'Parameter':<{name_width}} {'State':>12} {'Physical value':>15}")
    physical_params = env.translate_params(info["params"])
    for name, state, physical in zip(
        env.params_id, observation[2 * count :], physical_params
    ):
        print(f"  {name:<{name_width}} {state:>12.6g} {physical:>15.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=BENCHMARKS, default="TSA")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-space", choices=("continuous", "discrete"), default="continuous"
    )
    parser.add_argument(
        "--full-corners",
        action="store_true",
        help="Simulate every PVT corner at each step",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be positive")
    config = {
        **benchmark_config(args.env),
        "episode_len": args.steps,
        "SoF": not args.full_corners,
    }
    env_type = ngspice_env_disc if args.action_space == "discrete" else ngspice_env_cont
    env = env_type(config)
    env.action_space.seed(args.seed)
    try:
        specs = len(env.specs_id)
        params = len(env.params_id)
        print(
            f"Environment: {args.env} | actions={args.action_space} | seed={args.seed}"
        )
        print(
            f"Observation: {env.observation_space.shape[0]} values "
            f"({specs} normalized current specs + {specs} normalized target specs + {params} parameter states)"
        )
        print("Action order: " + ", ".join(env.params_id))
        if args.action_space == "discrete":
            print(
                "Discrete actions: 0 -> -1 grid step, 1 -> no change, 2 -> +2 grid steps"
            )
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(f"\nEpisode {episode + 1} reset")
            print_state(env, observation, info)
            total_reward = 0.0
            for step in range(args.steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"\nStep {step + 1}/{args.steps}")
                action_text = np.array2string(
                    action,
                    precision=6,
                    separator=", ",
                    threshold=action.size,
                    max_line_width=200,
                )
                print(f"  Action: {action_text}")
                print(
                    f"  Reward: {reward:.6f} | terminated={terminated} | truncated={truncated}"
                )
                print_state(env, observation, info)
                if terminated or truncated:
                    break
            reason = "success threshold reached" if terminated else "step limit reached"
            print(
                f"\nEpisode {episode + 1} finished: {reason}; steps={step + 1}; return={total_reward:.6f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
