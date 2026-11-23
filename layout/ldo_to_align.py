"""RL이 찾은 LDO(SKY130) 크기를 ALIGN 입력 넷리스트로 바꾼다.

사용 예:
    # 1) 기본 예제 설계(이 저장소의 HER 학습이 찾은, 9개 PVT corner 를 모두 통과한 설계)
    python layout/ldo_to_align.py --out layout/work/ldo_rl

    # 2) 학습 로그의 eval 결과에서 성공한 설계를 골라서
    python layout/ldo_to_align.py --log my_train.log --task 25 --out layout/work/ldo_rl

    # 3) 13개 파라미터를 직접 (ldo_sky130.yaml 의 params 순서)
    python layout/ldo_to_align.py --params "4.819 1.789 68.72 0.5 30.12 1.418 71.47 0.5 100 1.4 400 9.39e-13 4.08e-10"

ALIGN의 SKY130 PDK(ALIGN-pdk-sky130)는 "fin" 추상화를 쓴다.
    - 트랜지스터 폭  W = nfin x 0.21um x nf x m       (nfin 은 2..10, nf 는 짝수)
    - 게이트 길이    폴리 폭 0.15um 고정  ->  긴 L 은 stack(직렬 게이트 개수)으로 근사
                    (5V 소자 전용 층/두꺼운 산화막도 그리지 않는다 -> 배치·배선 연습용 레이아웃)
    - 커패시터       1000fF(=1pF) 이하만 지원 -> 큰 값은 단위 cap 여러 개를 병렬로
그래서 RL이 찾은 연속값 W, L 을 가장 가까운 격자값으로 반올림한다. 반올림 결과와 오차를 같이 출력한다.

보상 RC(RFB, CFB) 는 기본적으로 넣지 않고 n_amp 를 핀으로 뺀다. ALIGN-pdk-sky130 의 저항 생성기는
두 단자를 short 시키고, MIM cap 생성기는 핀이 open/격자 밖이라 ALIGN 자체 검사에서 오류가 나기 때문이다
(트랜지스터만 넣으면 오류 0). R/C 까지 그려 보려면 --with_rc.
"""
import argparse
import json
import os
import re

import numpy as np

# ldo_sky130.yaml 의 params 순서 (단위: W/L um, RFB ohm, CFB/CL F)
NAMES = ["W_M1", "L_M1", "W_M3", "L_M3", "W_M5", "L_M5",
         "W_pass", "L_pass", "M_pass", "Vb", "RFB", "CFB", "CL"]

# 기본 예제: LDO HER(seed 42) 학습 후 eval task 25 에서 성공한 설계 (pass 소자가 가장 작은 설계)
DEFAULT = [4.819, 1.789, 68.72, 0.5, 30.12, 1.418, 71.47, 0.5, 100, 1.4, 400, 9.39e-13, 4.082e-10]

FIN_UM = 0.21          # SKY130_PDK/layers.json : Fin pitch 210nm
POLY_L_UM = 0.15       # 폴리(게이트) 폭 150nm
MAX_NFIN = 10          # 한 finger 당 최대 fin 수 (2.1um)
MAX_CAP_F = 1e-12      # design_info.max_size_cap = 1000 fF
NFET = "sky130_fd_pr__nfet_g5v0d10v5"
PFET = "sky130_fd_pr__pfet_g5v0d10v5"


def quantize_w(w_um, nf=2, tol=0.02):
    """W(um) -> (nfin, nf, m).  W_eff = nfin*0.21*nf*m.
    finger 가 넓을수록(nfin 클수록) 단위 셀 수가 줄어 배치가 쉬우므로, nfin=10 부터 내려가며
    오차가 tol(2%) 이내인 첫 값을 쓰고, 없으면 오차가 가장 작은 값을 쓴다."""
    best = None
    for nfin in range(MAX_NFIN, 1, -1):
        m = max(1, round(w_um / (FIN_UM * nfin * nf)))
        err = abs(FIN_UM * nfin * nf * m - w_um) / w_um
        if err <= tol:
            return nfin, nf, m
        if best is None or err < best[0] - 1e-9:
            best = (err, nfin, m)
    _, nfin, m = best
    return nfin, nf, m


def quantize_l(l_um):
    """L(um) -> stack.  ALIGN SKY130 PDK 는 0.15um 폴리만 그리므로, 긴 L 은 0.15um 게이트 stack 개를
    직렬로 이어 근사한다. 그린 길이가 원래 L 보다 짧아지지 않게 올림한다 (0.5um -> 4 x 0.15 = 0.6um)."""
    return max(1, int(np.ceil(l_um / POLY_L_UM - 1e-6)))


def params_from_log(path, task):
    txt = open(path).read()
    pat = rf"eval task {task}: success=(\w+)\ntarget specs: \[[^\]]*\]\ncur specs: \[[^\]]*\]\nparams: \[([^\]]*)\]"
    ms = list(re.finditer(pat, txt))
    if not ms:
        raise SystemExit(f"{path} 에서 eval task {task} 를 찾지 못했습니다.")
    # 학습 중 eval 은 여러 번 돈다 -> 마지막(학습이 끝난 모델)의 결과를 쓴다
    m = ms[-1]
    print(f"[로그] eval task {task}: {len(ms)}번 중 마지막 결과 사용 (success={m.group(1)})")
    if m.group(1) != "True":
        print(f"[주의] eval task {task} 는 success=False 인 설계입니다.")
    return [float(x) for x in m.group(2).split()]


def mos_line(name, d, g, s, b, model, w_um, l_um):
    nfin, nf, m = quantize_w(w_um)
    stack = quantize_l(l_um)
    w_eff = FIN_UM * nfin * nf * m
    line = (f"{name} {d} {g} {s} {b} {model} w={nfin * FIN_UM * 1e-6:.4g} l={POLY_L_UM * 1e-6:.4g} "
            f"nf={nf} m={m} stack={stack}")
    info = (f"{name:3s} W {w_um:9.3f}um -> {w_eff:9.3f}um (nfin={nfin}, nf={nf}, m={m}, "
            f"err {100 * (w_eff - w_um) / w_um:+.1f}%) | L {l_um:.3f}um -> stack {stack} "
            f"= {stack * POLY_L_UM:.2f}um")
    return line, info, w_eff, stack * POLY_L_UM


def build(p, with_rc=False):
    v = dict(zip(NAMES, p))
    w_pass_total = v["W_pass"] * v["M_pass"]
    lines, infos, eff = [], [], {}
    # 오차 증폭기 (NMOS 차동쌍 + PMOS 전류거울 + NMOS 꼬리 전류원)
    for args in [("M1", "n_mir", "vfb", "n_tail", "vss", NFET, v["W_M1"], v["L_M1"]),
                 ("M2", "n_amp", "vref", "n_tail", "vss", NFET, v["W_M1"], v["L_M1"]),
                 ("M3", "n_amp", "n_mir", "vdd", "vdd", PFET, v["W_M3"], v["L_M3"]),
                 ("M4", "n_mir", "n_mir", "vdd", "vdd", PFET, v["W_M3"], v["L_M3"]),
                 ("M5", "n_tail", "vb", "vss", "vss", NFET, v["W_M5"], v["L_M5"]),
                 # 패스 트랜지스터: 원래 넷리스트는 W_pass 짜리를 M_pass 개 병렬 -> 총 폭으로 합쳐서 배치
                 ("M6", "vreg", "n_amp", "vdd", "vdd", PFET, w_pass_total, v["L_pass"])]:
        line, info, w_eff, l_eff = mos_line(*args)
        lines.append(line); infos.append(info); eff[args[0]] = (w_eff, l_eff)
    # 밀러 보상: n_amp -- RFB -- n_rc -- CFB -- vreg
    if with_rc:
        lines.append(f"R1 n_amp n_rc {v['RFB']:.4g}")
        n_cap = int(np.ceil(v["CFB"] / MAX_CAP_F - 1e-9))
        for i in range(n_cap):
            lines.append(f"C{i + 1} n_rc vreg {v['CFB'] / n_cap:.4g}")
        infos.append(f"R1  {v['RFB']:.4g} ohm | CFB {v['CFB'] * 1e15:.1f} fF -> {n_cap} x {v['CFB'] / n_cap * 1e15:.1f} fF"
                     "  [주의: ALIGN SKY130 의 R/C 생성기는 short/open 오류가 있음]")
    else:
        infos.append(f"보상 RC (RFB {v['RFB']:.4g} ohm, CFB {v['CFB'] * 1e15:.1f} fF) 는 넣지 않음 -> "
                     "핀 n_amp, vreg 사이에 SKY130 poly 저항 + MIM cap 을 따로 연결")
    infos.append(f"(레이아웃에 넣지 않는 값) Vb = {v['Vb']:.3g} V (외부 바이어스), CL = {v['CL'] * 1e12:.0f} pF (부하 cap, 칩 밖)")
    return v, lines, infos, eff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="학습 로그 (eval task 출력이 들어 있는 파일)")
    ap.add_argument("--task", type=int, help="--log 에서 꺼낼 eval task 번호")
    ap.add_argument("--params", help="13개 값을 공백으로 (ldo_sky130.yaml params 순서)")
    ap.add_argument("--out", default="layout/work/ldo_rl", help="ALIGN 입력 폴더 (폴더 이름 = 회로 이름)")
    ap.add_argument("--with_rc", action="store_true",
                    help="보상 R/C 도 ALIGN 으로 그린다 (ALIGN SKY130 PDK 의 R/C 생성기 오류 때문에 LVS/DRC 오류가 남)")
    a = ap.parse_args()

    if a.params:
        p = [float(x) for x in a.params.split()]
    elif a.log:
        p = params_from_log(a.log, a.task)
    else:
        p = DEFAULT
    assert len(p) == len(NAMES), f"파라미터는 {len(NAMES)}개여야 합니다: {NAMES}"

    v, lines, infos, eff = build(p, a.with_rc)
    name = os.path.basename(os.path.normpath(a.out))
    os.makedirs(a.out, exist_ok=True)
    sp = os.path.join(a.out, f"{name}.sp")
    with open(sp, "w") as f:
        f.write(f"* LDO (SKY130, 5V devices) sized by RL -- generated by layout/ldo_to_align.py\n")
        f.write("* " + " ".join(f"{k}={v[k]:.4g}" for k in NAMES) + "\n")
        ports = "vdd vss vref vb vfb vreg" + ("" if a.with_rc else " n_amp")
        f.write(f".subckt {name} {ports}\n")
        f.write("\n".join(lines) + "\n")
        f.write(f".ends {name}\n")
    const = [{"constraint": "PowerPorts", "ports": ["VDD"]},
             {"constraint": "GroundPorts", "ports": ["VSS"]},
             # 가로/세로 비율 제한: 없으면 큰 패스 트랜지스터를 한 줄로 늘어놓아 3mm x 50um 같은 띠가 된다
             {"constraint": "AspectRatio", "ratio_low": 0.5, "ratio_high": 2.0}]
    with open(os.path.join(a.out, f"{name}.const.json"), "w") as f:
        json.dump(const, f, indent=2)
    with open(os.path.join(a.out, "quantized.json"), "w") as f:
        json.dump({"rl_params": dict(zip(NAMES, p)),
                   "layout_W_um": {k: e[0] for k, e in eff.items()},
                   "layout_L_um": {k: e[1] for k, e in eff.items()}}, f, indent=2)

    print(f"ALIGN 입력 생성: {sp}")
    for s in infos:
        print("  " + s)


if __name__ == "__main__":
    main()
