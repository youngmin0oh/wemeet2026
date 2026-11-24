# Analog Circuit Sizing (아날로그 회로 크기 정하기)

ngspice 로 시뮬레이션하는 아날로그 회로의 크기(트랜지스터 폭 · 길이, 커패시터 등)를 강화학습으로 찾는
Gymnasium 환경입니다. 행동은 연속 또는 이산으로 고를 수 있고, 학습은 Stable-Baselines3 의
SAC(기본), PPO, DDPG 를 씁니다.

| 벤치마크 | 회로 | 공정 |
| --- | --- | --- |
| TSA | 2단 증폭기 (Two-stage amplifier) | GF180 |
| CMA | 캐스코드 밀러 보상 증폭기 (Cascode Miller-compensated amplifier) | GF180 |
| LDO | 저전압 강하 레귤레이터 (Low-dropout regulator) | SKY130 |
| COMP | 비교기 (Comparator) | GF180 |

> 처음이라면 단계별 매뉴얼 [docs/MANUAL.md](docs/MANUAL.md) 를 따라 하세요.
> ngspice 설치, GPU 별 torch, 학습, 연습 예제 ex00–ex10, ALIGN 자동 레이아웃까지 다룹니다.
> 강의 자료는 [slides/rl_ex01_10_lecture.pptx](slides/rl_ex01_10_lecture.pptx) 에 있습니다.

## 설치

```bash
conda env create -f environment.yml
conda activate analog_sizing
conda install -c conda-forge ngspice   # 또는: sudo apt-get install ngspice
```

명령은 저장소 맨 위 폴더에서 실행하세요. 그래야 ngspice 가 함께 들어 있는 `.spiceinit`
(`set num_threads=1`) 을 읽습니다. 이 파일이 없으면 OpenMP 스레드를 여러 개 쓰도록 빌드된 ngspice 가
corner 16개를 동시에 돌릴 때 CPU 를 서로 차지하려고 다퉈서 크게 느려집니다
(이 서버에서 TSA 의 corner 전체를 한 번 돌리는 데 179초 vs 0.27초).

회로 모델(PDK) 경로는 시뮬레이션할 때 원본 넷리스트에서 찾으므로 입력 파일을 고칠 필요가 없습니다.
함께 들어 있는 PDK 파일은 `eval_engines/pdk/` 에 그대로 두세요.
학습에는 위 `environment.yml` 에 고정된 Gymnasium 0.29.1 과 Stable-Baselines3 2.3.2 를 씁니다.

## 환경 직접 만져 보기

가장 쉬운 방법은 LDO 회로 환경을 둘러보는 예제입니다 (학습 없음, 약 5초).
관측 · 행동 · 보상이 회로에서 무엇인지, 같은 크기가 9개 PVT corner 에서 어떻게 달라지는지 한국어로 보여 줍니다.

```bash
python ex00_circuit_env.py
```

무작위 행동으로 몇 step 돌려 보며 숫자를 모두 출력할 수도 있습니다:

```bash
python scripts/random_rollout.py --env TSA --episodes 1 --steps 2 --seed 0
python scripts/random_rollout.py --env TSA --action-space discrete --steps 2
python scripts/random_rollout.py --env LDO --steps 2 --tt-only
```

첫 번째 명령(TSA, 연속 행동, seed 0)은 아래를 그대로 출력합니다. reset 과 무작위 step 두 번이 모두 들어 있습니다.
(프로그램이 실제로 찍는 내용이라 영어 그대로 둡니다.)

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
  wn4                        1.241e-06       1.241e-06
  wn5                            0.0001          0.0001
  cc                              1e-13           1e-13

Episode 1 finished: step limit reached; steps=2; return=-8.166371
```

출력 읽는 법:

- `Specification` 줄: 스펙(성능)마다 지금 회로의 값(`Current`), 목표 부등호(`Target`), 관측에 들어가는
  정규화된 값 두 개(`Obs.current`, `Obs.target`). 이름이 `_min` 이면 그 값 **이상**, `_max` 면 **이하** 여야 합격입니다.
- `Parameter` 줄: 관측 19개 중 나머지 7개. `State` 는 연속 행동일 때 물리값 그대로, 이산 행동일 때 격자 번호이고,
  `Physical value` 는 실제로 넷리스트에 들어가는 값입니다.
- `terminated=True` 는 목표를 달성해 끝났다는 뜻, `truncated=True` 는 step 수 제한에 걸려 끊겼다는 뜻입니다.
  에피소드 return 은 step 보상의 합이고, 이 예에서는 두 step 뒤 -8.166371 입니다.
- `Simulation` 줄은 실제로 시뮬레이션한 corner 를 알려 줍니다. 여기서는 기본 조건(TT)에서 목표를 못 넘어서
  TT 만 돌렸습니다 (SoF: Skip on Fail). `--full-corners` 를 붙이면 reset 과 step 마다 모든 corner 를 돌리고,
  `--tt-only` 를 붙이면 TT 넷리스트(`dsn_netlist` 의 첫 번째)만 돌립니다.

환경은 Gymnasium 의 reset/step 방식을 그대로 따릅니다 (ex01 과 같음):

```python
from analog_sizing.config import benchmark_config
from analog_sizing.envs.ngspice_env import ngspice_env_cont

env = ngspice_env_cont({**benchmark_config("TSA"), "episode_len": 30})
try:
    observation, info = env.reset(seed=0)                     # 새 게임 시작: 목표를 정하고 처음 크기로 시뮬레이션
    action = env.action_space.sample()                        # 크기 7개를 얼마나 바꿀지, 각각 [-1, 1]
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

## 학습

```bash
python analog_sizing/train.py --env_name TSA --algo SAC --total_timesteps 12000
bash scripts/TSA.sh
```

`CMA`, `LDO`, `COMP` 실행 스크립트도 있습니다. 옵션을 뒤에 붙이면 스크립트의 값을 바꿀 수 있습니다
(예: `bash scripts/TSA.sh --seed 42`). `python -m analog_sizing.train` 으로 실행해도 됩니다.

기본값에서는 모든 PVT corner 의 최악값이 모든 목표를 넘어야 성공입니다. `--tt_only True` 를 붙이면
TT 넷리스트 하나만으로 학습하고 평가합니다 (step 당 시뮬레이션 1번, 예: `bash scripts/LDO.sh --tt_only True`).
이렇게 찾은 설계는 다른 corner 에서 확인되지 않은 것입니다 (ex08 참고).

학습 · 평가 기록은 SB3 logger 로 저장됩니다. TensorBoard 로그는 `logs/<알고리즘>/`, 모델은 `models/` 에 생깁니다.
실행을 구분하려면 `--name` 을 쓰세요.

```bash
tensorboard --logdir logs
```

```bash
python analog_sizing/train.py --help
```

## 평가용 목표 만들기

```bash
python analog_sizing/gen_specs.py --env TSA --num-specs 100 --seed 0
```

다른 벤치마크 YAML 은 `--yaml`, 저장할 pickle 경로는 `--output` 으로 고릅니다. 경로 없이 파일 이름만 주면
`analog_sizing/gen_specs/` 에서 읽고, 그 밖의 상대 경로는 저장소 맨 위 폴더 기준입니다.
함께 들어 있는 평가 목표 파일은 새로 만든 것과 따로 보관됩니다.

## 연습 예제와 자동 레이아웃

- `ex00` 은 학습 없이 회로 환경(LDO)을 둘러봅니다.
- `ex01`–`ex05` 는 Gymnasium 기본 게임으로 Actor-Critic, SAC, PPO 를 직접 만듭니다.
- `ex06`, `ex07` 은 같은 Pendulum 문제를 Stable-Baselines3 의 SAC, PPO 로 풉니다.
- `ex08`–`ex10` 은 Stable-Baselines3 SAC 를 SKY130 LDO 에 처음부터 끝까지 적용합니다:

```bash
python ex00_circuit_env.py             # 회로 환경 둘러보기: 관측 · 행동 · 보상, 9개 PVT corner (학습 없음)
python ex06_sac_sb3.py                 # ex04 를 Stable-Baselines3 로
python ex07_ppo_sb3.py                 # ex05 를 Stable-Baselines3 로
python ex08_ldo_tt_sac.py              # TT corner 하나만 보고 학습 -> 찾은 설계를 9개 PVT corner 에서 확인
python ex09_ldo_pvt_sac.py             # 9개 PVT corner + 레이아웃 기생 추정을 보고 학습
python ex10_ldo_layout_postlayout.py   # ALIGN 레이아웃 -> 기생 추출 -> 레이아웃 후 9개 PVT 확인
```

`ex10` 과 `layout/run_ldo_layout.sh` 에는 ALIGN 이 필요합니다. 설치 방법, 예상 출력, 예제 스펙은
[docs/MANUAL.md](docs/MANUAL.md) 8장과 9장에 있습니다.

## 저장소 구성

```text
analog_sizing/
  config.py            # 벤치마크 이름, YAML 읽기, 저장소 경로
  envs/                # 환경: 공통 뼈대(ngspice_env_base.py)와 연속 · 이산 행동(ngspice_env.py)
  callbacks.py         # 시뮬레이션 횟수 집계, 평가, 모델 저장
  train.py             # Stable-Baselines3 학습 명령
  gen_specs.py         # seed 로 평가용 목표 만들기
  gen_specs/           # 함께 들어 있는 평가용 목표
eval_engines/
  ngspice/             # 시뮬레이션 실행, 회로 측정, 넷리스트
  pdk/                 # GF180, SKY130 회로 모델
scripts/               # 회로별 실행 스크립트, 무작위 rollout 예제
tests/                 # 환경과 시뮬레이션 회귀 테스트
layout/                # LDO 용 ALIGN 레이아웃 · post-layout 스크립트
ex00_*.py ... ex10_*.py  # 연습 예제 (ex_ldo_common.py 는 ex00, ex08-ex10 이 같이 씀)
docs/MANUAL.md         # 단계별 매뉴얼
slides/                # 강의 · 발표 자료
```

회귀 테스트는 프로젝트 환경에 `pytest` 를 설치한 뒤 `python -m pytest -q tests` 로 돌립니다.

## 만든 사람

원래 구현은 [Seunggeun Kim](https://seunggeunkimkr.github.io), Ziyi Wang, Sungyoung Lee, Youngmin Oh,
Hanqing Zhu, Doyun Kim, David Z. Pan 이 했습니다. [원 논문 (ICCAD 2025)](https://arxiv.org/abs/2507.17003) 을 참고하세요.
코드 일부는 [AutoCkt](https://github.com/ksettaluri6/AutoCkt) 에서 가져와 고쳤습니다.
