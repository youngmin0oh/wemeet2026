"""
ex08 · ex09 · ex10 이 같이 쓰는 도구 (LDO, SKY130)

    SPEC                  예제 스펙 (레이아웃 후에도 9개 PVT corner 에서 지켜야 하는 값)
    make_env()            Gymnasium 환경 (TT 만 / 9개 PVT, 기생 추정 포함 여부)
    DesignLog             Stable-Baselines3 callback: 학습 중 목표를 만족한 설계를 기록
    Simulator             설계 하나를 9개 corner 에서 시뮬레이션
    print_pvt_table()     corner 별 최악값을 스펙과 나란히 출력

예제 스펙을 원래 벤치마크(ldo_sky130.yaml 의 target_specs)보다 낮춘 이유
    ALIGN 으로 그린 LDO 레이아웃(docs/MANUAL.md 9-5)에서 기생을 하나씩 넣어 보면
      - 증폭기 출력 -> 패스 게이트 배선 C (패스 소자 폭 1um 당 약 0.22 fF, 접지로 추출)
            1MHz 이상 PSRR -3.3 dB, 위상 여유 -4.6 도
      - VDD 배선 R (약 3.5 ohm, 10mA 에서 35mV 강하)
            10kHz PSRR -4.5 dB
    가 떨어진다. 이 두 기생을 넣고 설계 공간을 탐색해 보면 "1MHz 이상 PSRR >= 0 dB",
    "최대 부하 10kHz PSRR >= 20 dB" 를 만족하는 설계를 찾지 못했다. 그래서 이 레이아웃 흐름으로
    실제 만들 수 있는 수준(-3 dB, 14 dB 등)으로 예제 스펙을 정했다. 실제 설계라면 이 단계에서
    스펙을 조정하거나, 레이아웃(넓은 전원선, 게이트 배선 차폐)을 바꾸게 된다.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# 신경망의 CPU 연산은 스레드 1개로: 시뮬레이션(ngspice 9개)과 CPU 를 나눠 쓰고, 작은 신경망은 이쪽이 더 빠르다
torch.set_num_threads(1)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "layout"))

from analog_sizing.config import benchmark_config, load_config  # noqa: E402
from analog_sizing.envs.ngspice_env import ngspice_env_cont  # noqa: E402

RESULTS = REPO / "results"
LDO_YAML = benchmark_config("LDO")["CIR_YAML"]

# ---------------- 예제 스펙 ----------------
# (환경의 스펙 이름, 부등호, 값, 단위, 설명)  순서 = 환경의 specs_id (이름 알파벳 순)
SPEC = [
    ("Vdrop_max",                          "<=", 250.0, "mV", "드롭아웃 전압 (최대 부하 10mA)"),
    ("mPSRR_maxload_worst_10kHz_min",      ">=",  14.0, "dB", "PSRR 최대 부하, 10kHz"),
    ("mPSRR_maxload_worst_1MHz_min",       ">=",   9.0, "dB", "PSRR 최대 부하, 1MHz"),
    ("mPSRR_maxload_worst_above_1MHz_min", ">=",  -3.0, "dB", "PSRR 최대 부하, 1MHz 이상 최악"),
    ("mPSRR_minload_worst_10kHz_min",      ">=",  10.0, "dB", "PSRR 최소 부하, 10kHz"),
    ("mPSRR_minload_worst_1MHz_min",       ">=",  10.0, "dB", "PSRR 최소 부하, 1MHz"),
    ("mPSRR_minload_worst_above_1MHz_min", ">=",  12.0, "dB", "PSRR 최소 부하, 1MHz 이상 최악"),
    ("phm_maxload_min",                    ">=",  60.0, "deg", "위상 여유, 최대 부하"),
    ("phm_minload_min",                    ">=",  60.0, "deg", "위상 여유, 최소 부하"),
]
SPEC_IDS = [s[0] for s in SPEC]
SPEC_VALUES = np.array([s[2] for s in SPEC])
SIGN = np.array([-1.0 if s[1] == "<=" else 1.0 for s in SPEC])   # (값 - 스펙) * SIGN >= 0 이면 통과
# 여유(margin)를 잴 때의 단위: 10mV, 1dB, 1도 를 "1" 로 본다
MARGIN_UNIT = np.array([10.0 if s[3] == "mV" else 1.0 for s in SPEC])

# 레이아웃 기생 추정값 (docs/MANUAL.md 9-5: 기본 예제 LDO 의 ALIGN 추출 결과에서)
C_NAMP_FF_PER_UM = 0.22   # 패스 게이트 배선 C = 0.22 fF x (W_pass x M_pass [um])  (1571 fF / 7148 um)
R_VDD_OHM = 3.5           # VDD 배선 R


# ==================================================
# 환경
# ==================================================
class NormalizeParams(gym.ObservationWrapper):
    """관측의 마지막 부분(회로 파라미터 13개)은 물리값 그대로라 크기가 1e-12 ~ 2000 으로 제각각이다.
    신경망이 다루기 쉽게 각 파라미터 범위를 [-1, 1] 로 바꾼다 (환경 안의 실제 값은 그대로)."""

    def __init__(self, env):
        super().__init__(env)
        self.lo, self.hi = env.params[:, 0], env.params[:, 1]
        n = len(self.lo)
        self.n = n
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=env.observation_space.shape,
                                                dtype=np.float64)

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float64)
        obs[-self.n:] = 2 * (obs[-self.n:] - self.lo) / (self.hi - self.lo) - 1
        return obs


def make_env(tt_only=False, target=SPEC_VALUES, cir_yaml=LDO_YAML, episode_len=30):
    """목표 하나(target)를 맞추는 LDO 크기 찾기 환경.

    tt_only=True  : TT corner 하나만 시뮬레이션 (step 당 1번)
    tt_only=False : 9개 PVT corner. 먼저 TT 를 보고, TT 를 통과하면 나머지 8개도 돌린다 (SoF)
    성공(terminated) = 시뮬레이션한 corner 들의 최악값이 target 을 만족 (합계 위반 2% 이내 허용)
    """
    env = ngspice_env_cont({
        "CIR_YAML": str(cir_yaml),
        "spec_path": benchmark_config("LDO")["spec_path"],
        "generalize": False,          # 목표를 매 에피소드 바꾸지 않고 하나로 고정
        "tt_only": tt_only,
        "SoF": True,
        "episode_len": episode_len,
        "lookup_style": "tanh",       # scripts/LDO.sh 와 같은 보상 설정
        "alpha": 0.0,
        "tt_threshold": -1.0,
        "min_threshold": -6.0,
    })
    assert env.specs_id == SPEC_IDS, env.specs_id
    env.g_star = np.asarray(target, dtype=float)      # generalize=False 일 때 reset 이 쓰는 목표
    return NormalizeParams(env)


def corner_names(cir_yaml=LDO_YAML):
    """ldo_cont_sky130_ss_2.0_m40.cir -> 'ss 2.0V -40C'"""
    names = []
    for path in load_config(cir_yaml)["dsn_netlist"]:
        m = re.search(r"_(tt|ss|ff|sf|fs)_([\d.]+)_(m?\d+)", Path(path).stem)
        names.append(f"{m.group(1)} {m.group(2)}V {m.group(3).replace('m', '-')}C" if m else Path(path).stem)
    return names


def parasitic_aware_yaml(out_dir, c_namp_ff_per_um=C_NAMP_FF_PER_UM, r_vdd=R_VDD_OHM):
    """9개 corner 넷리스트에 레이아웃 기생 추정값을 넣은 복사본을 만들고 그 yaml 경로를 돌려준다.
       - 패스 게이트(N_AMP) 배선 C : 패스 소자 총 폭에 비례 -> 넷리스트 식 {k * W_pass * M_pass}
       - VDD 배선 R               : 고정값
    (layout/postlayout_ldo.py 가 실제 추출값으로 만드는 post-layout 넷리스트와 같은 구조)"""
    from postlayout_ldo import make_postlayout_netlists
    out_dir = Path(out_dir)
    new_yaml = make_postlayout_netlists(LDO_YAML, {"VDD": (r_vdd, 0.0), "N_AMP": (0.0, 1.0)}, out_dir)
    half = c_namp_ff_per_um / 2 * 1e-15
    for f in out_dir.glob("*_post.cir"):
        t = f.read_text()
        t = re.sub(r"^Cpx_namp1 n_amp_a 0 .*$", f"Cpx_namp1 n_amp_a 0 {{{half:.4g}*W_pass*M_pass}}", t, flags=re.M)
        t = re.sub(r"^Cpx_namp2 n_amp_g 0 .*$", f"Cpx_namp2 n_amp_g 0 {{{half:.4g}*W_pass*M_pass}}", t, flags=re.M)
        f.write_text(t)
    return new_yaml


def layout_widths(params_id, params):
    """ALIGN 이 실제로 그릴 폭으로 바꾼 파라미터 (layout/ldo_to_align.py 와 같은 반올림, L 은 그대로).
    ALIGN SKY130 PDK 는 폭을 fin(0.21um) 단위로만 그리므로 작은 소자는 오차가 커진다 (예: 4.66um -> 5.04um)."""
    from ldo_to_align import FIN_UM, W_MAX_UM, quantize_w
    v = dict(zip(params_id, params))

    def q(w, w_max=None):
        nfin, nf, m = quantize_w(w, w_max=w_max)
        return FIN_UM * nfin * nf * m

    for k in ("W_M1", "W_M3", "W_M5"):
        v[k] = q(v[k])
    # 패스 소자는 총 폭(W_pass x M_pass)으로 그린다. 병렬 하나의 폭은 모델 최대 폭을 넘지 않게
    v["W_pass"] = q(v["W_pass"] * v["M_pass"], W_MAX_UM * v["M_pass"]) / v["M_pass"]
    return [v[k] for k in params_id]


# ==================================================
# 시뮬레이션 · 판정
# ==================================================
class Simulator:
    """설계 하나(파라미터 13개)를 yaml 의 모든 corner 에서 시뮬레이션한다."""

    def __init__(self, cir_yaml=LDO_YAML):
        self.cir_yaml = str(cir_yaml)
        self.env = ngspice_env_cont({"CIR_YAML": self.cir_yaml, "generalize": False,
                                     "spec_path": benchmark_config("LDO")["spec_path"]})
        self.env.reset(seed=0)
        self.corners = corner_names(cir_yaml)
        self.params_id = self.env.params_id

    def run(self, params):
        """-> (corner 수 x 스펙 9개) 배열"""
        return np.asarray(self.env.update(np.asarray(params, dtype=float), self.env.full_sim_env))

    def close(self):
        self.env.close()


def worst(table):
    """corner 별 값 -> 스펙별 최악값, 최악 corner 번호"""
    table = np.atleast_2d(table)
    idx = np.where(SIGN > 0, table.argmin(0), table.argmax(0))
    return table[idx, np.arange(table.shape[1])], idx


def passes(values, spec=SPEC_VALUES):
    return (np.asarray(values) - spec) * SIGN >= 0


def min_margin(values, spec=SPEC_VALUES):
    """가장 빠듯한 스펙의 여유 (10mV, 1dB, 1도 단위). 0 이상이면 모두 통과"""
    return float(((np.asarray(values) - spec) * SIGN / MARGIN_UNIT).min())


def pad(text, width, right=False):
    """한글처럼 화면에서 두 칸을 차지하는 글자를 세어 폭을 맞춘다."""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))
    fill = " " * max(0, width - shown)
    return fill + str(text) if right else str(text) + fill


def print_pvt_table(title, table, corners, extra=None):
    """스펙별 최악값과 그 corner. extra={"열 이름": 스펙 9개 값} 은 오른쪽에 같이 보여 줄 다른 결과."""
    w, idx = worst(table)
    ok = passes(w)
    print(f"\n[{title}]  {len(corners)}개 corner 중 최악값")
    head = f"  {pad('스펙', 34)} {pad('기준', 13, True)} {pad('최악값', 9, True)}  {pad('최악 corner', 13)}   "
    if extra:
        head += "".join(f" {pad(k, 12, True)}" for k in extra)
    print(head)
    for i, (sid, op, val, unit, desc) in enumerate(SPEC):
        line = (f"  {pad(desc, 34)} {op}{val:7.1f} {unit:3s} {w[i]:9.2f}  {corners[idx[i]]:13s}"
                f" {'OK' if ok[i] else 'X '}")
        if extra:
            line += "".join(f" {v[i]:12.2f}" for v in extra.values())
        print(line)
    verdict = "모두 통과" if ok.all() else f"미달 {int((~ok).sum())}개"
    print(f"  => {len(corners)}개 corner {verdict} (가장 빠듯한 여유 {min_margin(w):+.2f})")
    return bool(ok.all())


# ==================================================
# 학습 기록 (Stable-Baselines3 callback)
# ==================================================
class DesignLog(BaseCallback):
    """model.learn(callback=DesignLog(...)) 로 붙인다. 매 step 불려서
      - 진행바를 보여 주고
      - 시뮬레이션 수를 센다 (TT 만 본 step = 1번, TT 를 통과해 나머지 corner 까지 본 step = corner 수)
      - 목표를 만족한(terminated) 설계를 모두 기록한다 (found)
    stop_after 개를 모으면 False 를 돌려준다 -> SB3 가 학습을 멈춘다."""

    def __init__(self, env, total_steps, stop_after=None, desc="SAC"):
        super().__init__()
        self.base = env.unwrapped
        self.total_steps, self.stop_after, self.desc = total_steps, stop_after, desc
        self.found, self.sims = [], 0

    def _on_training_start(self):
        self.bar = tqdm(total=self.total_steps, desc=self.desc, unit="step", dynamic_ncols=True)

    def _on_step(self):
        info = self.locals["infos"][0]
        self.bar.update(1)
        self.sims += self.base.num_corners if (info.get("tt_done") and not self.base.tt_only) else 1
        if info.get("done"):                       # 이 step 에서 목표 달성
            self.found.append({"step": int(self.num_timesteps),
                               "params": [float(x) for x in info["params"]],
                               "specs": [float(x) for x in info["cur_specs"]]})
            self.bar.set_postfix(found=len(self.found), first=self.found[0]["step"])
        return not (self.stop_after and len(self.found) >= self.stop_after)

    def _on_training_end(self):
        self.bar.close()


def rollout(model, env):
    """학습된 정책(평균 행동)으로 처음 크기부터 한 에피소드 설계해 본다. 성공하면 그 설계를 돌려준다."""
    obs, _ = env.reset(seed=12345)
    for t in range(env.unwrapped.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated:
            return {"step": "policy", "steps_in_episode": t + 1,
                    "params": [float(x) for x in info["params"]],
                    "specs": [float(x) for x in info["cur_specs"]]}
        if truncated:
            break
    return None


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def params_text(params_id, params):
    return "  ".join(f"{k}={v:.4g}" for k, v in zip(params_id, params))
