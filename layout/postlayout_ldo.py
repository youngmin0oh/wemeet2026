"""레이아웃 후(post-layout) 시뮬레이션: ALIGN 이 뽑은 배선 기생 R/C 를 넣고 9개 PVT corner 를 다시 돈다.

    conda activate analog_sizing
    python layout/postlayout_ldo.py --align_dir layout/work/ldo_rl

하는 일
  1. ALIGN 기생 추출 결과(run/LDO_RL_0.cir, schematic2layout.py -x)를 읽어 net 마다
       - 배선 C  = 그 net 배선의 커패시턴스 합
       - 배선 R  = 그 net 에서 가장 멀리 떨어진 두 점 사이의 실효 저항 (보수적: 실제보다 크게 잡힘)
     를 구한다.
  2. 원래 LDO 테스트벤치(9개 corner 넷리스트)의 ldo 회로에 이 R/C 를 끼워 넣은 post-layout 넷리스트를 만든다.
       - 전원·출력 net (VDD, VSS, VREG) 은 R 을 소자와 핀 사이에 직렬로, C 는 소자 쪽에
       - 증폭기 출력 -> 패스 게이트 (N_AMP) 는 R 직렬 + C 를 양쪽에 반씩 (pi 모델)
       - 그 밖의 net 은 C 만 (전류가 거의 안 흐르는 게이트 net)
  3. 레이아웃에 쓴 폭(반올림값)으로 pre-layout / post-layout 을 모두 시뮬레이션해서 스펙별 최악값을 비교한다.

한계 (결과를 읽을 때 꼭 같이 볼 것)
  - ALIGN 추출은 블록 사이 배선(top-level routing)만 다룬다. 트랜지스터 블록 내부 배선, 결합(coupling) C 는 없다.
  - 트랜지스터는 원래 SKY130 5V 모델(원래 L)을 그대로 쓴다 (ALIGN 이 그린 0.15um x stack 게이트가 아님).
  - 보상 RC(RFB, CFB) 는 레이아웃에 넣지 않았으므로 이상적인 소자로 둔다.
  즉 "배선 기생까지 넣은 1차 확인" 이지, 상용 툴(Calibre/Quantus) 수준의 sign-off 가 아니다.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analog_sizing.config import benchmark_config, load_config, resolve_path  # noqa: E402
from analog_sizing.envs.ngspice_env import ngspice_env_cont  # noqa: E402

sys.path.insert(0, str(REPO / "layout"))
from check_ldo_quantized import DEFAULT_TARGET, worst_specs  # noqa: E402

NODE_RE = re.compile(r"^(\S+)_M(\d)_(_?)(\d+)_(_?)(\d+)$")   # NET_M<층>_<x>_<y>  (음수 좌표는 '_' 가 하나 더)


def parse_pex(path):
    """ALIGN .cir -> {net: {"C": fF 합, "R": [(n1, n2, ohm)], "xy": {node: (x, y)}}}"""
    nets = defaultdict(lambda: {"C": 0.0, "R": [], "xy": {}})

    def node(n):
        m = NODE_RE.match(n)
        if not m:
            return None
        net = m.group(1)
        x = int(m.group(4)) * (-1 if m.group(3) else 1)
        y = int(m.group(6)) * (-1 if m.group(5) else 1)
        nets[net]["xy"][n] = (x, y)
        return net

    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0][0] == "r" and f[0][1:].isdigit():
            net = node(f[1]); node(f[2])
            if net:
                nets[net]["R"].append((f[1], f[2], float(f[3])))
        elif f[0][0] == "c" and f[0][1:].isdigit():
            net = node(f[1])
            if net:
                nets[net]["C"] += float(f[3].rstrip("f"))
    return nets


def _laplacian(net):
    names = list(net["xy"])
    idx = {n: i for i, n in enumerate(names)}
    rows, cols, vals = [], [], []
    for a, b, r in net["R"]:
        g = 1.0 / max(r, 1e-6)
        i, j = idx[a], idx[b]
        rows += [i, j, i, j]; cols += [i, j, j, i]; vals += [g, g, -g, -g]
    n = len(names)
    return names, sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def net_resistances(net):
    """net 하나의 배선 저항 두 가지:
      r_far  : 기하학적으로 가장 먼 두 점 사이 실효 저항 (끝에서 끝, 비관적 상한)
      r_dist : 가장 위 배선층 전체 <-> 가장 아래층(소자 단자 쪽) 전체 사이 저항
               (패스 트랜지스터처럼 전류가 수천 개 finger 로 나뉘어 흐르는 경우의 현실적 값)
    """
    names, lap = _laplacian(net)
    n = len(names)
    if n < 2 or not net["R"]:
        return 0.0, 0.0, int(n > 0)
    ncomp, labels = connected_components(lap != 0, directed=False)
    xy = np.array([net["xy"][k] for k in names], dtype=float)
    r_far = 0.0
    for c in range(ncomp):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue
        pts = xy[members]
        a = members[np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1))]           # 중심에서 가장 먼 점
        b = members[np.argmax(np.linalg.norm(pts - xy[a], axis=1))]                 # 그 점에서 가장 먼 점
        sub = lap[members][:, members].tocsc()
        pos = {m: k for k, m in enumerate(members)}
        keep = [k for k in range(len(members)) if k != pos[b]]                      # b 를 접지로
        rhs = np.zeros(len(members)); rhs[pos[a]] = 1.0                              # a 에 1A 주입
        v = spla.spsolve(sub[keep][:, keep], rhs[keep])
        r_far = max(r_far, float(v[keep.index(pos[a])]))

    layer = np.array([int(NODE_RE.match(k).group(2)) for k in names])
    top, bot = np.where(layer == layer.max())[0], np.where(layer == layer.min())[0]
    if layer.max() == layer.min() or ncomp > 1:
        return r_far, r_far, ncomp
    fixed = np.zeros(n, bool); fixed[top] = True; fixed[bot] = True
    volt = np.zeros(n); volt[top] = 1.0                                              # 위층 1V, 아래층 0V
    free = np.where(~fixed)[0]
    if len(free):
        volt[free] = spla.spsolve(lap[free][:, free].tocsc(), -lap[free][:, fixed] @ volt[fixed])
    current = float((lap @ volt)[top].sum())
    return r_far, (1.0 / current if current > 0 else r_far), ncomp


MOS_TAIL = ("nf=1 ad='int((nf+1)/2) * W/nf * 0.29' as='int((nf+2)/2) * W/nf * 0.29' "
            "pd='2*int((nf+1)/2) * (W/nf + 0.29)' ps='2*int((nf+2)/2) * (W/nf + 0.29)' "
            "nrd='0.29 / W' nrs='0.29 / W' sa=0 sb=0 sd=0")
NFET, PFET = "sky130_fd_pr__nfet_g5v0d10v5", "sky130_fd_pr__pfet_g5v0d10v5"


def ldo_subckt(par):
    """기생 R/C 를 넣은 평탄화된 ldo 서브회로 (원래: ldo + diff_pair). par: {net: (R ohm, C fF)}"""
    R = lambda k: max(par.get(k, (0, 0))[0], 1e-3)       # 0 ohm 저항은 ngspice 가 싫어하므로 1 mohm
    C = lambda k: par.get(k, (0, 0))[1]
    s = [".subckt ldo Vdd Vfb Vref Vb Vss Vreg",
         "* ---- layout parasitics: ALIGN routing extraction, lumped per net (layout/postlayout_ldo.py) ----",
         f"Rpx_vdd Vdd vdd_l {R('VDD'):.6g}", f"Cpx_vdd vdd_l 0 {C('VDD'):.6g}f",
         f"Rpx_vss Vss vss_l {R('VSS'):.6g}", f"Cpx_vss vss_l 0 {C('VSS'):.6g}f",
         f"Rpx_vreg vreg_l Vreg {R('VREG'):.6g}", f"Cpx_vreg vreg_l 0 {C('VREG'):.6g}f",
         f"Rpx_vfb Vfb vfb_l {R('VFB'):.6g}", f"Cpx_vfb vfb_l 0 {C('VFB'):.6g}f",
         f"Rpx_vref Vref vref_l {R('VREF'):.6g}", f"Cpx_vref vref_l 0 {C('VREF'):.6g}f",
         f"Rpx_vb Vb vb_l {R('VB'):.6g}", f"Cpx_vb vb_l 0 {C('VB'):.6g}f",
         f"Rpx_namp n_amp_a n_amp_g {R('N_AMP'):.6g}",
         f"Cpx_namp1 n_amp_a 0 {C('N_AMP') / 2:.6g}f", f"Cpx_namp2 n_amp_g 0 {C('N_AMP') / 2:.6g}f",
         f"Cpx_nmir n_mir 0 {C('N_MIR'):.6g}f", f"Cpx_ntail n_tail 0 {C('N_TAIL'):.6g}f",
         "* ---- devices (same models / parameters as the schematic) ----",
         f"XM1 n_mir vfb_l n_tail vss_l {NFET} L=L_M1 W=W_M1 {MOS_TAIL} mult=1 m=1",
         f"XM2 n_amp_a vref_l n_tail vss_l {NFET} L=L_M2 W=W_M2 {MOS_TAIL} mult=1 m=1",
         f"XM3 n_amp_a n_mir vdd_l vdd_l {PFET} L=L_M3 W=W_M3 {MOS_TAIL} mult=1 m=1",
         f"XM4 n_mir n_mir vdd_l vdd_l {PFET} L=L_M4 W=W_M4 {MOS_TAIL} mult=1 m=1",
         f"XM5 n_tail vb_l vss_l vss_l {NFET} L=L_M5 W=W_M5 {MOS_TAIL} mult=1 m=1",
         f"XM6 vreg_l n_amp_g vdd_l vdd_l {PFET} L=L_pass W=W_pass {MOS_TAIL} mult=M_pass m=M_pass",
         "* ---- compensation RC: not in the ALIGN layout, ideal here ----",
         "Cfb n_rc vreg_l {CFB}",
         "Rfb n_rc n_amp_a {RFB}",
         ".ends"]
    return "\n".join(s) + "\n"


def make_postlayout_netlists(src_yaml, par, out_dir):
    """원래 corner 넷리스트들에서 ldo/diff_pair 서브회로를 기생 포함 버전으로 바꿔 out_dir 에 쓴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(src_yaml)
    new_list = []
    sub = ldo_subckt(par)
    for rel in cfg["dsn_netlist"]:
        src = resolve_path(rel)
        txt = src.read_text()
        # 상대 경로 include/lib 을 절대 경로로 (새 폴더에서도 모델을 찾도록)
        txt = re.sub(r'^(\s*\.(?:include|lib)\s+)"([^"]+)"',
                     lambda m: f'{m.group(1)}"{(src.parent / m.group(2)).resolve()}"'
                     if not os.path.isabs(m.group(2)) else m.group(0), txt, flags=re.M)
        txt, n1 = re.subn(r"^\.subckt ldo .*?^\.ends\s*$", lambda m: sub.rstrip("\n"), txt, flags=re.M | re.S)
        txt, n2 = re.subn(r"^\.subckt diff_pair .*?^\.ends\s*$", "* diff_pair: flattened into ldo (post-layout)",
                          txt, flags=re.M | re.S)
        assert n1 == 1 and n2 == 1, f"{src}: ldo/diff_pair 서브회로를 찾지 못했습니다"
        dst = out_dir / src.name.replace(".cir", "_post.cir")
        dst.write_text(txt)
        new_list.append(str(dst))
    cfg["dsn_netlist"] = tuple(new_list)
    new_yaml = out_dir / "ldo_sky130_postlayout.yaml"
    new_yaml.write_text(yaml.dump(cfg, sort_keys=False))   # params 순서가 바뀌면 값이 엉뚱한 소자로 감
    return new_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align_dir", default="layout/work/ldo_rl")
    ap.add_argument("--target", help="목표 스펙 9개 (기본: 예제 설계의 목표)")
    a = ap.parse_args()
    adir = Path(a.align_dir).resolve()

    cirs = sorted((adir / "run").glob("*.cir"))
    if not cirs:
        raise SystemExit(f"{adir}/run/*.cir 가 없습니다. schematic2layout.py 를 -x 옵션으로 돌렸는지 확인하세요.")
    nets = parse_pex(cirs[0])
    par_dist, par_far = {}, {}
    print(f"[1] 배선 기생 (ALIGN 추출: {cirs[0].name})")
    print(f"    {'net':8s} {'배선 C':>11s} {'R(분산)':>10s} {'R(끝-끝)':>10s}")
    for name in sorted(nets):
        r_far, r_dist, _ = net_resistances(nets[name])
        par_dist[name], par_far[name] = (r_dist, nets[name]["C"]), (r_far, nets[name]["C"])
        print(f"    {name:8s} {nets[name]['C']:8.1f} fF {r_dist:8.2f} ohm {r_far:8.2f} ohm")

    base_yaml = benchmark_config("LDO")["CIR_YAML"]
    yamls = {"pre-layout": base_yaml,
             "post(R 분산)": str(make_postlayout_netlists(base_yaml, par_dist, adir / "postlayout_dist")),
             "post(R 끝-끝)": str(make_postlayout_netlists(base_yaml, par_far, adir / "postlayout_far"))}
    print(f"[2] post-layout 넷리스트: {adir}/postlayout_dist/, {adir}/postlayout_far/ (corner 9개씩)")

    # 레이아웃에 쓴 폭 (L 은 원래 값; check_ldo_quantized.py 와 같은 규칙)
    q = json.load(open(adir / "quantized.json"))
    rl = q["rl_params"]
    lay = dict(rl)
    for k, dev in (("W_M1", "M1"), ("W_M3", "M3"), ("W_M5", "M5")):
        lay[k] = q["layout_W_um"][dev]
    lay["W_pass"] = q["layout_W_um"]["M6"] / rl["M_pass"]
    params = [lay[k] for k in rl]
    target = np.array([float(x) for x in a.target.split()]) if a.target else np.array(DEFAULT_TARGET)

    print("[3] 9개 PVT corner 시뮬레이션")
    worst, score, ids = {}, {}, None
    for label, cir_yaml in yamls.items():
        env = ngspice_env_cont({"CIR_YAML": cir_yaml, "spec_path": benchmark_config("LDO")["spec_path"],
                                "episode_len": 30})
        try:
            env.reset(seed=0)
            env.specs_ideal = target
            worst[label] = worst_specs(env, params)
            score[label] = env.aggregate(worst[label], target)
            ids = env.specs_id
        finally:
            env.close()

    labels = list(yamls)
    print(f"\n{'spec (9 corner 중 최악)':34s} {'목표':>11s} " + " ".join(f"{l:>13s}" for l in labels))
    for i, sid in enumerate(ids):
        op = ">=" if sid.endswith("min") else "<="
        cells = []
        for l in labels:
            v = worst[l][i]
            ok = v >= target[i] if op == ">=" else v <= target[i]
            cells.append(f"{v:11.4g}{' ' if ok else '*'} ")
        print(f"{sid:34s} {op}{target[i]:10.4g} " + " ".join(cells))
    print("(* = 목표를 살짝이라도 넘은 값. 아래 판정은 학습 환경과 같은 기준: 합계 오차 0.02 이내 허용)")
    for l in labels:
        print(f"{l:14s}: {'9개 PVT 모두 통과' if score[l] > 0 else '목표 미달'} (aggregate={score[l]:.3f})")


if __name__ == "__main__":
    main()
