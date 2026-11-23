# Analog Circuit Sizing

Gymnasium environments for sizing analog circuits with ngspice, with continuous
or discrete actions and optional goal-conditioned observations. Training supports Stable-Baselines3
PPO, SAC, DDPG, and SAC with hindsight experience replay (HER).

| Benchmark | Circuit | Process |
| --- | --- | --- |
| TSA | Two-stage amplifier | GF180 |
| CMA | Cascode Miller-compensated amplifier | GF180 |
| LDO | Low-dropout regulator | SKY130 |
| COMP | Comparator | GF180 |

## Setup

> 처음 설치한다면 단계별 한국어 매뉴얼 [docs/MANUAL.md](docs/MANUAL.md) 를 따라 하세요
> (ngspice 설치, GPU 별 torch, 학습, 연습 예제, ALIGN 자동 레이아웃까지).

```bash
conda env create -f environment.yml
conda activate analog_sizing
conda install -c conda-forge ngspice   # or: sudo apt-get install ngspice
```

Run commands from the repository root so ngspice picks up the bundled `.spiceinit`
(`set num_threads=1`). Without it, ngspice builds that default to several OpenMP
threads oversubscribe the CPU when 16 corners run in parallel (179 s vs 0.27 s per
TSA corner sweep on our server).

Model paths are resolved from the source netlists when simulations run. No input
file rewriting is required. Keep the bundled PDK files in `eval_engines/pdk/`.
Training uses Gymnasium 0.29.1 with Stable-Baselines3 2.3.2, as pinned above.

## Try the environment

Run a short rollout with randomly sampled actions:

```bash
python scripts/random_rollout.py --env TSA --episodes 1 --steps 2 --seed 0
python scripts/random_rollout.py --env TSA --action-space discrete --steps 2
python scripts/random_rollout.py --env LDO --steps 2 --tt-only
```

The continuous command above produces the following complete output with the
bundled TSA netlist and seed 0. It includes reset and both sampled steps.

```text
Environment: TSA | actions=continuous | seed=0
Observation: 19 values (6 normalized current specs + 6 normalized target specs + 7 parameter states)
Action order: wp1, wn1, wp3, wn3, wn4, wn5, cc

Episode 1 reset
  Simulation: nominal TT only; 16 other corners skipped
  Specification                 Current          Target  Obs.current   Obs.target
  gain_min                       543.46          >= 229     0.369849   -0.0438413
  ibias_max                  0.00055581    <= 0.0078434    -0.687379     0.446668
  phm_min                       29.2142           >= 60    -0.345078            0
  t_settle_max              1.65635e-06   <= 6.4717e-06    -0.656977    -0.105668
  ugbw_min                  2.59279e+07   >= 1.6008e+07     0.129069    -0.110864
  vswing_min                    1.02069     >= 0.284231     0.606512     0.064076
  Parameter                       State  Physical value
  wp1                         1.733e-05       1.733e-05
  wn1                         8.876e-05       8.876e-05
  wp3                          7.21e-05        7.21e-05
  wn3                         4.621e-05       4.621e-05
  wn4                         1.964e-05       1.964e-05
  wn5                             7e-05           7e-05
  cc                           3.48e-12        3.48e-12

Step 1/2
  Action: [ 0.273923, -0.460427, -0.918053, -0.966945,  0.62654 ,  0.825511,  0.213272]
  Reward: -4.526812 | terminated=False | truncated=False
  Simulation: nominal TT only; 16 other corners skipped
  Specification                 Current          Target  Obs.current   Obs.target
  gain_min                      0.50197          >= 229    -0.995993   -0.0438413
  ibias_max                 0.000151171    <= 0.0078434    -0.904054     0.446668
  phm_min                          -180           >= 60           -1            0
  t_settle_max              8.20897e-06   <= 6.4717e-06    0.0129531    -0.105668
  ugbw_min                            1   >= 1.6008e+07           -1    -0.110864
  vswing_min                0.000262992     >= 0.284231    -0.997898     0.064076
  Parameter                       State  Physical value
  wp1                        3.1026e-05      3.1026e-05
  wn1                        6.5739e-05      6.5739e-05
  wp3                        2.6197e-05      2.6197e-05
  wn3                             5e-07           5e-07
  wn4                        5.0967e-05      5.0967e-05
  wn5                            0.0001          0.0001
  cc                         4.5464e-12      4.5464e-12

Step 2/2
  Action: [ 0.458993,  0.08725 ,  0.870145,  0.631707, -0.994523,  0.714809, -0.932829]
  Reward: -3.639559 | terminated=False | truncated=True
  Simulation: nominal TT only; 16 other corners skipped
  Specification                 Current          Target  Obs.current   Obs.target
  gain_min                       7.8088          >= 229    -0.939422   -0.0438413
  ibias_max                  0.00505058    <= 0.0078434     0.254713     0.446668
  phm_min                       39.2427           >= 60    -0.209157            0
  t_settle_max                0.0421819   <= 6.4717e-06     0.999621    -0.105668
  ugbw_min                  3.11565e+08   >= 1.6008e+07      0.87936    -0.110864
  vswing_min                  0.0142416     >= 0.284231    -0.892208     0.064076
  Parameter                       State  Physical value
  wp1                        5.3976e-05      5.3976e-05
  wn1                        7.0101e-05      7.0101e-05
  wp3                        6.9704e-05      6.9704e-05
  wn3                        3.2085e-05      3.2085e-05
  wn4                         1.241e-06       1.241e-06
  wn5                            0.0001          0.0001
  cc                              1e-13           1e-13

Episode 1 finished: step limit reached; steps=2; return=-8.166371
```

Each specification row shows the current circuit value, its target inequality,
and the two normalized values included in the observation. The parameter rows
complete the 19 observation values. `State` is the physical value for continuous
actions and the grid index for discrete actions; `Physical value` is the value
passed to the circuit netlist.

`terminated=True` means the environment's success threshold was reached.
`truncated=True` means the episode reached its step limit. The episode return is
the sum of step rewards; this example ends after two steps with return -8.166371.
The simulation line identifies the corners that were actually evaluated. Here
only nominal TT runs because its goal checks fail. Add `--full-corners` to
evaluate all configured corners at every reset and step, or `--tt-only` to
simulate only the nominal TT netlist (the first `dsn_netlist` entry).

The environment follows the Gymnasium reset/step API:

```python
from analog_sizing.config import benchmark_config
from analog_sizing.envs.ngspice_env import ngspice_env_cont

env = ngspice_env_cont({**benchmark_config("TSA"), "episode_len": 30})
try:
    observation, info = env.reset(seed=0)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

## Train

```bash
python analog_sizing/train.py --env_name TSA --algo HER --total_timesteps 12000
bash scripts/TSA.sh
```

Launchers are also available for `CMA`, `LDO`, and `COMP`. Their arguments can be
overridden by appending options, for example `bash scripts/TSA.sh --seed 42`.
Both direct scripts and `python -m analog_sizing.train` are supported.

By default, success requires the worst value over all PVT corners to meet every
target. Add `--tt_only True` to train and evaluate on the nominal TT netlist only
(one simulation per step), for example `bash scripts/LDO.sh --tt_only True`.
Designs found this way are not checked across corners.

Training and evaluation metrics are recorded locally using the SB3 logger.
TensorBoard logs are written under `logs/<algorithm>/` and model directories
under `models/`. Use `--name` to distinguish runs.

```bash
tensorboard --logdir logs
```

```bash
python analog_sizing/train.py --help
```

## Generate evaluation targets

```bash
python analog_sizing/gen_specs.py --env TSA --num-specs 100 --seed 0
```

Use `--yaml` to choose another benchmark YAML or `--output` to set the pickle
output path. Bare spec filenames are loaded from `analog_sizing/gen_specs/`;
other relative configuration and output paths are resolved from the repository
root. Built-in evaluation data is preserved separately from newly generated data.

## Practice examples and automatic layout

`ex01`–`ex05` introduce actor-critic, SAC, and PPO on Gymnasium tasks. `ex06`–`ex08`
apply SAC to the SKY130 LDO from start to finish:

```bash
python ex06_ldo_tt_sac.py              # TT corner only, then check the design on 9 PVT corners
python ex07_ldo_pvt_sac.py             # 9 PVT corners with estimated layout parasitics
python ex08_ldo_layout_postlayout.py   # ALIGN layout, extraction, post-layout PVT check
```

`ex08` and `layout/run_ldo_layout.sh` need ALIGN. Installation, expected output, and
the example specification are described in [docs/MANUAL.md](docs/MANUAL.md)
(sections 8 and 9).

## Repository layout

```text
analog_sizing/
  config.py            # Benchmark names, YAML loading, and repository paths
  envs/                # Base, action-space, and goal-conditioned environments
  callbacks.py         # Simulation accounting, evaluation, and checkpoints
  train.py             # Stable-Baselines3 training CLI
  gen_specs.py         # Seeded evaluation-target generator
  gen_specs/           # Bundled evaluation targets
eval_engines/
  ngspice/             # Simulation wrappers, circuit measurements, and netlists
  pdk/                 # GF180 and SKY130 models
scripts/               # Circuit launchers and random-rollout example
tests/                 # Environment and simulation-wrapper regression checks
layout/                # ALIGN layout and post-layout scripts for the LDO
ex01_*.py ... ex08_*.py  # Practice examples (ex_ldo_common.py is shared by ex06-ex08)
docs/MANUAL.md         # Step-by-step manual (Korean)
```

Run the regression tests with `python -m pytest -q tests` after installing `pytest`
in the project environment.

## Attribution

The original implementation is by [Seunggeun Kim](https://seunggeunkimkr.github.io),
Ziyi Wang, Sungyoung Lee, Youngmin Oh, Hanqing Zhu, Doyun Kim, and David Z. Pan.
See the [upstream research paper (ICCAD 2025)](https://arxiv.org/abs/2507.17003).
Parts of this codebase were adapted from [AutoCkt](https://github.com/ksettaluri6/AutoCkt).
