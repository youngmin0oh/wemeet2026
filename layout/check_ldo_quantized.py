"""ALIGN 격자에 맞춰 반올림한 LDO 폭(W)이 여전히 목표 스펙을 만족하는지 ngspice로 확인한다.

    python layout/check_ldo_quantized.py --align_dir layout/work/ldo_rl

원래 RL 크기와 반올림된(레이아웃) 폭을 TT + 8개 corner 전부에서 시뮬레이션하고,
스펙별 최악값(worst corner)을 목표와 나란히 보여준다.
L 은 원래 값을 그대로 쓴다. ALIGN 의 stack 근사(0.15um x N)는 SKY130 5V 소자의 최소 L(0.5um)
모델 범위 밖이 될 수 있어서 시뮬레이션할 수 없고, 실제 칩에서는 L 을 그대로 그려야 하기 때문이다.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analog_sizing.config import benchmark_config  # noqa: E402
from analog_sizing.envs.ngspice_env import ngspice_env_cont  # noqa: E402

# 기본 예제 설계(layout/ldo_to_align.py 의 DEFAULT)를 찾을 때 쓰인 목표 스펙 (HER seed42 eval task 25)
DEFAULT_TARGET = [255.318994, 21.3834079, 14.0359173, 2.83812654e-05,
                  10.5887289, 10.0905644, 15.2552415, 60.0052552, 60.0097432]


def worst_specs(env, params):
    all_specs = np.asarray(env.update(np.asarray(params, dtype=float), env.full_sim_env))
    # max 스펙은 최댓값, min 스펙은 최솟값이 최악
    worst = [all_specs[:, i].max() if sid.endswith("max") else all_specs[:, i].min()
             for i, sid in enumerate(env.specs_id)]
    return np.array(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align_dir", default="layout/work/ldo_rl", help="ldo_to_align.py 의 --out 폴더")
    ap.add_argument("--target", help="목표 스펙 9개 (공백 구분, 기본=예제 설계의 목표)")
    a = ap.parse_args()

    q = json.load(open(os.path.join(a.align_dir, "quantized.json")))
    rl = q["rl_params"]
    names = list(rl.keys())
    orig = [rl[n] for n in names]
    lay = dict(rl)
    lay["W_M1"] = q["layout_W_um"]["M1"]
    lay["W_M3"] = q["layout_W_um"]["M3"]
    lay["W_M5"] = q["layout_W_um"]["M5"]
    lay["W_pass"] = q["layout_W_um"]["M6"] / rl["M_pass"]          # 총 폭 / 병렬 개수
    quant = [lay[n] for n in names]

    target = np.array([float(x) for x in a.target.split()]) if a.target else np.array(DEFAULT_TARGET)
    env = ngspice_env_cont({**benchmark_config("LDO"), "episode_len": 30})
    try:
        env.reset(seed=0)
        env.specs_ideal = target
        w_orig, w_quant = worst_specs(env, orig), worst_specs(env, quant)
        print(f"{'spec (worst of 9 corners)':36s} {'target':>11s} {'RL 원래':>11s} {'레이아웃용':>11s}")
        for i, sid in enumerate(env.specs_id):
            op = ">=" if sid.endswith("min") else "<="
            ok = (w_quant[i] >= target[i]) if op == ">=" else (w_quant[i] <= target[i])
            print(f"{sid:36s} {op}{target[i]:10.4g} {w_orig[i]:11.4g} {w_quant[i]:11.4g}  {'OK' if ok else 'X'}")
        for label, w in (("RL 원래", w_orig), ("레이아웃용", w_quant)):
            score = env.aggregate(w, target)
            print(f"{label}: {'목표 달성' if score > 0 else '목표 미달'} (aggregate={score:.3f}, 0 이상이면 통과)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
