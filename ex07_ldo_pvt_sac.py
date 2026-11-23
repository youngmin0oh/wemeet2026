"""
ex07 — LDO(SKY130) 크기 찾기 ②: 9개 PVT corner 를 다 보고 SAC 로 찾기 (레이아웃 기생 추정 포함)
conda activate analog_sizing ; python ex07_ldo_pvt_sac.py        (약 30–60분)

ex06 과 달라진 점 세 가지
  1. PVT corner 9개 (공정 tt/ss/ff/sf/fs x 온도 -40/27/125도) 의 "최악값" 으로 성공을 판단한다.
     매 step TT 를 먼저 시뮬레이션하고, TT 를 통과했을 때만 나머지 8개를 돌린다 (Skip-on-Fail, 시뮬레이션 절약).
  2. 레이아웃 기생 추정값을 넣은 넷리스트로 학습한다 (parasitic-aware).
     ALIGN 레이아웃에서 성능을 가장 많이 떨어뜨린 두 기생 (docs/MANUAL.md 9-5):
        - 패스 게이트 배선 C = 0.22 fF x 패스 소자 총 폭(um)     -> 위상 여유, 1MHz 이상 PSRR
        - VDD 배선 R = 3.5 ohm                                    -> 10kHz PSRR
     이걸 모르고 학습하면 목표에 딱 붙은 설계가 나와서 레이아웃 후 시뮬레이션에서 떨어진다.
  3. 학습 목표 = 예제 스펙 + 여유(TRAIN_MARGIN). 기생 추정이 실제와 조금 달라도 버티도록.

학습 중 성공한 설계(후보)들을 ALIGN 이 실제로 그릴 폭(fin 0.21um 격자로 반올림)으로 바꿔 기생 추정 넷리스트에서
다시 시뮬레이션하고, 여유가 가장 큰 설계를 고른다. 그 설계를 (a) 기생 없는 회로도 (b) 레이아웃 폭 + 기생 추정
두 경우로 9개 corner 에서 확인하고 저장한다.
다음: python ex08_ldo_layout_postlayout.py   (이 설계로 ALIGN 레이아웃 -> 실제 기생으로 다시 확인)

결과: results/ex07_ldo_pvt_sac.json
"""
import argparse
import time

import numpy as np

from tqdm import tqdm

from ex_ldo_common import (C_NAMP_FF_PER_UM, LDO_YAML, R_VDD_OHM, RESULTS, SIGN, SPEC, SPEC_VALUES, Simulator,
                           layout_widths, make_env, min_margin, pad, parasitic_aware_yaml, params_text,
                           print_pvt_table, rollout, save_json, train_sac, worst)

# 학습 목표의 여유: 스펙보다 이만큼 더 좋게 (Vdrop 은 5mV 낮게, PSRR 은 1dB, 위상 여유는 1.5도 높게)
TRAIN_MARGIN = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000, help="최대 학습 step")
    ap.add_argument("--stop_after", type=int, default=30, help="성공한 설계를 이만큼 모으면 학습을 멈춤")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_parasitic", action="store_true", help="기생 추정 없이 회로도만으로 학습 (비교용)")
    ap.add_argument("--no_margin", action="store_true", help="여유 없이 예제 스펙 그대로를 목표로 학습 (비교용)")
    ap.add_argument("--out", default=str(RESULTS / "ex07_ldo_pvt_sac.json"), help="결과 json 경로")
    a = ap.parse_args()

    target = SPEC_VALUES + (0 if a.no_margin else SIGN * TRAIN_MARGIN)
    est_yaml = parasitic_aware_yaml(RESULTS / "ex07_netlists")      # 기생 추정을 넣은 9개 corner 넷리스트
    train_yaml = LDO_YAML if a.no_parasitic else est_yaml
    print(f"학습 넷리스트: {'회로도 (기생 없음)' if a.no_parasitic else f'기생 추정 포함 ({est_yaml.parent})'}")
    print(f"  게이트 배선 C = {C_NAMP_FF_PER_UM} fF/um x 패스 소자 총 폭, VDD 배선 R = {R_VDD_OHM} ohm")
    print("학습 목표 (예제 스펙 -> 여유를 더한 목표):")
    for (sid, op, val, unit, desc), t in zip(SPEC, target):
        print(f"  {pad(desc, 34)} {op} {val:g} {unit:3s} -> {op} {t:g}")

    env = make_env(tt_only=False, target=target, cir_yaml=train_yaml)    # ① 9개 PVT corner 환경
    params_id = env.unwrapped.params_id
    t0 = time.time()
    model, found, sims = train_sac(env, a.steps, seed=a.seed, stop_after=a.stop_after, desc="SAC (PVT)")
    learned = rollout(model, env)                                         # ② 학습된 정책으로 한 번 더
    env.close()
    print(f"\n학습 {time.time() - t0:.0f}초, 시뮬레이션 {sims}번 (corner 1개 = 1번) | 성공한 설계 {len(found)}개"
          + (f" (첫 성공: {found[0]['step']} step)" if found else "")
          + f" | 학습된 정책 단독 설계: {'성공' if learned else '실패'}")
    candidates = found + ([learned] if learned else [])
    if not candidates:
        raise SystemExit("목표를 만족한 설계가 없습니다. --steps 를 늘리거나 --seed 를 바꿔 보세요.")

    # ③ 후보 고르기: ALIGN 이 그릴 폭(fin 격자로 반올림)으로 바꾼 뒤 기생 추정 넷리스트로 9개 corner 를 다시
    #    시뮬레이션해서, 가장 빠듯한 스펙의 여유가 가장 큰 설계를 고른다 (ex08 에서 실제로 그려질 크기로 미리 확인)
    sim = Simulator(est_yaml)
    try:
        for d in tqdm(candidates, desc="후보 확인 (레이아웃 폭 + 기생 추정)", unit="설계"):
            d["layout_params"] = layout_widths(params_id, d["params"])
            d["layout_margin"] = min_margin(worst(sim.run(d["layout_params"]))[0])
    finally:
        sim.close()
    best = max(candidates, key=lambda d: d["layout_margin"])
    print(f"고른 설계 ({best['step']}, 레이아웃 폭 + 기생 추정에서 가장 빠듯한 여유 {best['layout_margin']:+.2f}):"
          f"\n  {params_text(params_id, best['params'])}")

    # ④ 9개 PVT corner 에서 확인: 회로도 / 레이아웃 폭 + 기생 추정
    checks = {"회로도 (원래 크기, 기생 없음)": (LDO_YAML, best["params"]),
              "레이아웃 폭 + 기생 추정 (ex08 의 예상)": (est_yaml, best["layout_params"])}
    tables = {}
    for label, (cir_yaml, p) in checks.items():
        sim = Simulator(cir_yaml)
        try:
            tables[label] = sim.run(p)
            corners = sim.corners
        finally:
            sim.close()
    ok = {label: print_pvt_table(f"{label} -> 9개 PVT corner", tab, corners) for label, tab in tables.items()}
    if all(ok.values()):
        print("\n두 경우 모두 9개 PVT corner 통과. 다음: python ex08_ldo_layout_postlayout.py"
              + ("" if a.out == str(RESULTS / "ex07_ldo_pvt_sac.json") else f" --result {a.out}"))

    path = save_json(a.out, {
        "example": "ex07: 9 PVT corners + SAC" + ("" if a.no_parasitic else " + parasitic estimate"),
        "spec": {s[0]: [s[1], s[2]] for s in SPEC},
        "train_target": dict(zip([s[0] for s in SPEC], target.tolist())),
        "parasitic_estimate": None if a.no_parasitic else {"C_namp_fF_per_um": C_NAMP_FF_PER_UM,
                                                            "R_vdd_ohm": R_VDD_OHM},
        "params": dict(zip(params_id, best["params"])),
        "chosen_from": best["step"],
        "found": len(found),
        "simulations": int(sims),
        "pvt_corners": corners,
        "layout_params": dict(zip(params_id, best["layout_params"])),
        "pvt_schematic": tables["회로도 (원래 크기, 기생 없음)"].tolist(),
        "pvt_layout_estimate": tables["레이아웃 폭 + 기생 추정 (ex08 의 예상)"].tolist(),
        "pvt_pass": ok,
    })
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
