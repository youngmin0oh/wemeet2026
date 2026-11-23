"""
ex08 — ex07 에서 찾은 LDO 크기로 ALIGN 레이아웃 -> 레이아웃 후(post-layout) 9개 PVT corner 확인
conda activate analog_sizing ; python ex08_ldo_layout_postlayout.py        (약 5분)
필요: ALIGN 설치 (docs/MANUAL.md 9-1, conda 환경 align + ALIGN-pdk-sky130)

단계
  1. results/ex07_ldo_pvt_sac.json 의 파라미터 13개를 읽는다
  2. layout/run_ldo_layout.sh 로 ALIGN 자동 배치·배선 -> GDS, PNG, 배선 기생 추출(.cir)
       (트랜지스터 폭은 ALIGN 격자(fin 0.21um)에 맞춰 반올림, 보상 R/C 와 부하 C 는 레이아웃 밖)
  3. 추출 결과에서 net 마다 배선 R/C 를 구한다 (layout/postlayout_ldo.py)
     -> ex07 이 학습에 쓴 기생 추정값(게이트 배선 C, VDD R)과 비교
  4. 9개 corner 넷리스트에 실제 기생을 넣은 post-layout 넷리스트를 만든다
  5. pre-layout(레이아웃에 쓴 폭, 기생 없음) 과 post-layout 을 9개 corner 에서 시뮬레이션해
     예제 스펙(ex_ldo_common.SPEC)을 모두 지키는지 판정한다

    python ex08_ldo_layout_postlayout.py --skip_layout      # 레이아웃은 그대로 두고 4-5 단계만 다시
    python ex08_ldo_layout_postlayout.py --result results/ex06_ldo_tt_sac.json   # ex06 설계로 해 보기

결과: results/ex08_ldo_postlayout.json, 레이아웃 그림 layout/work/ex08_ldo/ex08_ldo.png
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ex_ldo_common import (C_NAMP_FF_PER_UM, LDO_YAML, R_VDD_OHM, REPO, RESULTS, SPEC, Simulator,
                           params_text, passes, print_pvt_table, save_json, worst)
from postlayout_ldo import make_postlayout_netlists, net_resistances, parse_pex   # layout/


def run_align(params, name):
    """layout/run_ldo_layout.sh = PDK 준비 -> ALIGN 넷리스트 -> 배치·배선(+기생 추출) -> PNG"""
    cmd = ["bash", str(REPO / "layout" / "run_ldo_layout.sh"), "--params", " ".join(f"{v:.6g}" for v in params)]
    env = {**os.environ, "NAME": name}
    # ALIGN 은 별도 conda 환경(align)에 있다. 지정이 없으면 지금 환경(analog_sizing)의 옆 폴더를 쓴다.
    sibling = Path(sys.prefix).parent / "align"
    if "ALIGN_ENV" not in env and (sibling / "bin" / "python").exists():
        env["ALIGN_ENV"] = str(sibling)
    print(f"$ NAME={name} ALIGN_ENV={env.get('ALIGN_ENV', '(conda 기본 위치의 align)')} "
          f"bash layout/run_ldo_layout.sh --params \"...\"", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        raise SystemExit("레이아웃 단계가 실패했습니다. ALIGN 설치(docs/MANUAL.md 9-1)와 위 메시지를 확인하세요.")
    print(f"(레이아웃 {time.time() - t0:.0f}초)")


def layout_params(adir):
    """quantized.json -> 시뮬레이션에 쓸 파라미터: 폭은 레이아웃에 그린 값(반올림), L 은 원래 값.
    (ALIGN 은 L 을 0.15um 게이트 여러 개로 근사해서 그리는데, 이건 SKY130 5V 모델 범위 밖이라 원래 L 로 시뮬레이션)"""
    q = json.load(open(adir / "quantized.json"))
    rl = q["rl_params"]
    lay = dict(rl)
    for k, dev in (("W_M1", "M1"), ("W_M3", "M3"), ("W_M5", "M5")):
        lay[k] = q["layout_W_um"][dev]
    lay["W_pass"] = q["layout_W_um"]["M6"] / rl["M_pass"]
    return list(rl), [lay[k] for k in rl], q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(RESULTS / "ex07_ldo_pvt_sac.json"), help="ex06/ex07 결과 json")
    ap.add_argument("--name", default="ex08_ldo", help="레이아웃 폴더 이름 (layout/work/<name>)")
    ap.add_argument("--skip_layout", action="store_true", help="이미 만든 레이아웃을 다시 쓴다")
    a = ap.parse_args()

    # 1. 파라미터
    res = json.load(open(a.result))
    names, params = list(res["params"]), list(res["params"].values())
    print(f"[1] 설계 ({a.result}):\n    {params_text(names, params)}")

    # 2. ALIGN 레이아웃
    adir = REPO / "layout" / "work" / a.name
    print("\n[2] ALIGN 레이아웃")
    if a.skip_layout and (adir / "quantized.json").exists():
        print(f"    --skip_layout: {adir} 를 그대로 씁니다")
    else:
        run_align(params, a.name)
    cir = sorted((adir / "run").glob("*.cir"))
    if not cir:
        raise SystemExit(f"{adir}/run/*.cir (기생 추출 결과) 가 없습니다.")
    names, lay_params, q = layout_params(adir)
    if not np.allclose([q["rl_params"][k] for k in names], params, rtol=1e-4):
        raise SystemExit(f"{adir} 의 레이아웃이 이 설계로 만든 것이 아닙니다. --skip_layout 없이 다시 돌리세요.")

    # 3. 배선 기생
    nets = parse_pex(cir[0])
    par = {}
    print(f"\n[3] 배선 기생 (ALIGN 추출 {cir[0].name}, net 마다 C 합과 R(분산))")
    for n in sorted(nets, key=lambda k: -nets[k]["C"]):
        r_dist = net_resistances(nets[n])[1]
        par[n] = (r_dist, nets[n]["C"])
        print(f"    {n:8s} C {nets[n]['C']:8.1f} fF   R {r_dist:7.2f} ohm")
    w_pass_total = q["layout_W_um"]["M6"]
    c_per_um = par.get("N_AMP", (0, 0))[1] / w_pass_total
    print(f"    ex07 의 추정과 비교: 게이트 배선 C {c_per_um:.3f} fF/um (추정 {C_NAMP_FF_PER_UM}), "
          f"VDD R {par.get('VDD', (0, 0))[0]:.2f} ohm (추정 {R_VDD_OHM})")

    # 4. post-layout 넷리스트
    post_yaml = make_postlayout_netlists(LDO_YAML, par, adir / "postlayout")
    print(f"\n[4] post-layout 넷리스트 9개: {post_yaml.parent}")

    # 5. 9개 PVT corner: pre-layout vs post-layout
    tables = {}
    for label, y in (("pre-layout", LDO_YAML), ("post-layout", post_yaml)):
        sim = Simulator(y)
        try:
            tables[label] = sim.run(lay_params)
            corners = sim.corners
        finally:
            sim.close()
    print("\n[5] 9개 PVT corner 시뮬레이션 (트랜지스터 폭 = 레이아웃에 그린 값)")
    pre_ok = print_pvt_table("pre-layout (기생 없음)", tables["pre-layout"], corners)
    post_ok = print_pvt_table("post-layout (ALIGN 추출 배선 기생 포함)", tables["post-layout"], corners,
                              extra={"pre-layout": worst(tables["pre-layout"])[0]})

    per_corner = np.array([passes(row).all() for row in tables["post-layout"]])
    print("\n    corner 별 post-layout 판정: "
          + ", ".join(f"{c} {'OK' if ok else 'X'}" for c, ok in zip(corners, per_corner)))
    if post_ok:
        print("\n=> 레이아웃 후에도 9개 PVT corner 에서 예제 스펙을 모두 지킵니다.")
    else:
        bad = [SPEC[i][4] for i in np.where(~passes(worst(tables["post-layout"])[0]))[0]]
        print(f"\n=> 레이아웃 후 미달: {', '.join(bad)}"
              "\n   ex07 의 TRAIN_MARGIN 이나 기생 추정값(ex_ldo_common.C_NAMP_FF_PER_UM, R_VDD_OHM)을 키워 다시 학습하세요.")

    png = adir / f"{a.name}.png"
    path = save_json(RESULTS / "ex08_ldo_postlayout.json", {
        "example": "ex08: ALIGN layout + post-layout PVT check",
        "design_from": a.result,
        "params": dict(zip(names, params)),
        "layout_params": dict(zip(names, lay_params)),
        "gds": str(next((adir / "run").glob("*_0.gds"), "")),
        "png": str(png),
        "parasitics": {n: {"R_ohm": v[0], "C_fF": v[1]} for n, v in par.items()},
        "gate_C_fF_per_um": c_per_um,
        "pvt_corners": corners,
        "pre_layout": tables["pre-layout"].tolist(),
        "post_layout": tables["post-layout"].tolist(),
        "pre_layout_pass": pre_ok,
        "post_layout_pass": post_ok,
    })
    print(f"\n저장: {path}\n레이아웃 그림: {png}")


if __name__ == "__main__":
    main()
