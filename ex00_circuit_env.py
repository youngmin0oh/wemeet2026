"""
ex00 — 회로 환경 둘러보기 (학습 없음): LDO 크기 맞추기를 Gymnasium 게임으로
conda activate analog_sizing ; python ex00_circuit_env.py        (약 10–30초)

이 저장소가 풀려는 문제를 먼저 손으로 만져 본다. ex01 의 reset() / step() 이 회로에서는 무엇인지 보고,
ex08 – ex10 에서 SAC 로 이 환경을 푼다.

    1. 환경 만들기     회로 하나 = YAML 하나 (넷리스트 9개, 크기 13개의 범위와 처음 값, 스펙 이름)
    2. reset()        처음 크기로 시뮬레이션 -> 관측 31개 = 지금 스펙 9 + 목표 스펙 9 + 크기 13
    3. step(action)   크기 13개를 조금씩 바꾸고 다시 시뮬레이션 -> 보상, 끝났는지
    4. PVT            같은 크기를 9개 조건(corner)에서 시뮬레이션해 보통 조건(TT)과 비교

환경은 ex08 – ex10 과 같은 것(ex_ldo_common.make_env)을 쓴다. 목표는 예제 스펙(ex_ldo_common.SPEC).
"""
import argparse

import numpy as np

from ex_ldo_common import SIGN, SPEC, Simulator, make_env, pad, passes, print_pvt_table, worst

# 크기 13개의 뜻 (eval_engines/ngspice/ngspice_inputs/netlist/ldo_cont_sky130_*.cir)
#   오차 증폭기: M1·M2 입력쌍(NMOS), M3·M4 전류 거울(PMOS), M5 꼬리 전류원(NMOS, 게이트 = Vb)
#   패스 트랜지스터 M6(PMOS): 폭 W_pass 짜리를 M_pass 개 병렬
PARAM_MEANING = {
    "W_M1": ("증폭기 입력쌍 M1·M2 폭", "um"),
    "L_M1": ("증폭기 입력쌍 M1·M2 길이", "um"),
    "W_M3": ("증폭기 전류 거울 M3·M4 폭", "um"),
    "L_M3": ("증폭기 전류 거울 M3·M4 길이", "um"),
    "W_M5": ("증폭기 꼬리 전류원 M5 폭", "um"),
    "L_M5": ("증폭기 꼬리 전류원 M5 길이", "um"),
    "W_pass": ("패스 트랜지스터 하나의 폭", "um"),
    "L_pass": ("패스 트랜지스터 길이", "um"),
    "M_pass": ("패스 트랜지스터 병렬 개수", "개"),
    "Vb": ("꼬리 전류원 바이어스 전압", "V"),
    "RFB": ("보상 저항", "ohm"),
    "CFB": ("보상 커패시터", "F"),
    "CL": ("출력 부하 커패시터", "F"),
}
# 표에 줄여 쓰는 스펙 이름 (SPEC 순서)
SHORT = ["Vdrop", "PSRR최대10k", "PSRR최대1M", "PSRR최대>1M", "PSRR최소10k", "PSRR최소1M", "PSRR최소>1M",
         "위상최대", "위상최소"]


def title(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def spec_table(values, target):
    """스펙 9개: 목표와 지금 값, 통과 여부"""
    ok = passes(values, target)
    print(f"  {pad('스펙', 34)} {pad('목표', 14, True)} {pad('지금', 10, True)}  통과")
    for (_, op, _, unit, desc), t, v, good in zip(SPEC, target, values, ok):
        print(f"  {pad(desc, 34)} {op}{t:8.1f} {unit:3s} {v:10.2f}  {'O' if good else 'X'}")
    print(f"  -> 9개 중 {int(ok.sum())}개 통과")


def simulated(base, info):
    if base.tt_only:
        return "TT 1번"
    return "9개 corner 모두 (TT 통과)" if info["tt_done"] else "TT 1번 (TT 에서 떨어져 나머지 8개는 건너뜀)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3, help="무작위 행동으로 해 볼 step 수")
    a = ap.parse_args()

    # ------------------------------------------------------------------ 1
    title("1. 환경 만들기: make_env()  (ex08 – ex10 과 같은 LDO 환경)")
    env = make_env(tt_only=False)          # 9개 corner, SoF(TT 먼저) 켜짐, 목표 = 예제 스펙
    base = env.unwrapped                   # 래퍼 안쪽의 진짜 회로 환경 (ngspice_env_cont)
    sim = Simulator()
    try:
        print("회로: LDO, SKY130 공정 5V 소자 · 전원 2.0V, 기준 1.8V · 부하 10uA – 10mA")
        print(f"corner (넷리스트) {base.num_corners}개: " + ", ".join(sim.corners))
        print(f"\n조절하는 크기 {len(base.params_id)}개 = 행동")
        print(f"  {pad('이름', 8)} {pad('뜻', 28)} {pad('범위', 22)} {pad('처음 값', 10, True)}")
        for name, (lo, hi), init in zip(base.params_id, base.params, base.initial_vals):
            meaning, unit = PARAM_MEANING[name]
            print(f"  {name:8s} {pad(meaning, 28)} {f'{lo:.3g} – {hi:.3g} {unit}':22s} {init:10.4g}")
        print(f"\n맞춰야 하는 스펙 {len(base.specs_id)}개 = 목표 (예제 스펙)")
        for _, op, val, unit, desc in SPEC:
            print(f"  {pad(desc, 34)} {op} {val:g} {unit}")
        print(f"\nobservation_space: {env.observation_space.shape[0]}개 숫자 "
              f"= 지금 스펙 {len(base.specs_id)} + 목표 스펙 {len(base.specs_id)} + 크기 {len(base.params_id)}")
        print(f"action_space     : {env.action_space}")
        print(f"  행동 1.0 = 그 크기를 (범위 최댓값 / {base.episode_len}) 만큼 늘림, -1.0 = 그만큼 줄임")
        print(f"  한 판(에피소드) = 최대 {base.episode_len} step")

        # -------------------------------------------------------------- 2
        title("2. reset(): 처음 크기로 시뮬레이션")
        obs, info = env.reset(seed=a.seed)
        print(f"시뮬레이션: {simulated(base, info)}")
        spec_table(info["cur_specs"], info["target_specs"])
        print(f"보상 {info['reward']:+.3f}  "
              "(TT 에서 떨어지면 -6 ~ -1, TT 는 통과하고 다른 corner 에서 떨어지면 -1 ~ 0, 모두 통과하면 +30)")
        n = len(base.specs_id)
        np.set_printoptions(precision=2, suppress=True, linewidth=110)
        print("\n관측 31개 (신경망이 보는 숫자)")
        print(f"  지금 스펙 9개 (정규화) : {obs[:n]}")
        print(f"  목표 스펙 9개 (정규화) : {obs[n:2 * n]}")
        print(f"  크기 13개 ([-1, 1])     : {obs[2 * n:]}")
        print("  정규화: 스펙마다 기준값과 비교한 상대값 (단위가 달라도 같은 눈금이 되게)")

        # -------------------------------------------------------------- 3
        title("3. step(action): 크기를 바꾸고 다시 시뮬레이션")
        # 첫 step 은 손으로 만든 행동: 패스 트랜지스터 병렬 개수만 +1 (= 2000/30 = 약 67개 늘림)
        action = np.zeros(len(base.params_id))
        action[base.params_id.index("M_pass")] = 1.0
        before = np.array(info["params"], dtype=float)
        specs_before, reward_before = np.array(info["cur_specs"]), info["reward"]
        obs, reward, terminated, truncated, info = env.step(action)
        print("step 1: 행동 = M_pass 만 +1, 나머지 0")
        print(f"  M_pass {before[base.params_id.index('M_pass')]:.0f} -> "
              f"{info['params'][base.params_id.index('M_pass')]:.0f} 개")
        print(f"  시뮬레이션: {simulated(base, info)}")
        spec_table(info["cur_specs"], info["target_specs"])
        print(f"  보상 {reward_before:+.3f} -> {reward:+.3f} | terminated={terminated} | truncated={truncated}")
        change = (np.array(info["cur_specs"]) - specs_before) * SIGN      # + 면 좋아짐
        better = [SHORT[i] for i in np.argsort(-change) if change[i] > 0.5]
        worse = [SHORT[i] for i in np.argsort(change) if change[i] < -0.5]
        print(f"  좋아진 스펙: {', '.join(better) or '없음'} | 나빠진 스펙: {', '.join(worse) or '없음'}")
        print("  -> 크기 하나를 바꿔도 여러 스펙이 함께, 때로는 서로 반대로 움직인다 (그래서 손으로 맞추기 어렵다)")
        print("  -> 보상은 통과한 개수가 아니라 '모자란 만큼' 으로 매긴다 (크게 모자란 스펙이 있으면 낮아짐)")

        env.action_space.seed(a.seed)
        for t in range(2, a.steps + 2):
            action = env.action_space.sample()      # 아무 행동이나 (무작위)
            obs, reward, terminated, truncated, info = env.step(action)
            ok = passes(info["cur_specs"], info["target_specs"])
            print(f"\nstep {t}: 무작위 행동 {np.round(action, 2)}")
            print(f"  시뮬레이션: {simulated(base, info)}")
            print(f"  9개 중 {int(ok.sum())}개 통과 | 보상 {reward:+.3f} | "
                  f"terminated={terminated} | truncated={truncated}")
            if terminated or truncated:
                break
        print("\n무작위로는 목표에 잘 가까워지지 않는다 -> 어떤 상황에서 어떤 크기를 바꿀지 배워야 한다 (ex08)")

        # -------------------------------------------------------------- 4
        title("4. PVT: 처음 크기를 9개 corner 에서 시뮬레이션")
        table = sim.run(base.initial_vals)          # (corner 9개 x 스펙 9개)
        print(f"  {pad('corner', 14)}" + "".join(f" {pad(s, 11, True)}" for s in SHORT))
        for name, row in zip(sim.corners, table):
            print(f"  {pad(name, 14)}" + "".join(f" {v:11.2f}" for v in row))
        w, _ = worst(table)
        print(f"  {pad('TT 와 최악 차이', 14)}" + "".join(f" {v:+11.2f}" for v in w - table[0]))
        print_pvt_table("처음 크기 -> 9개 corner", table, sim.corners)
        print("\n같은 크기라도 공정 · 온도에 따라 값이 달라진다. 합격은 9개 corner 의 '최악값' 으로 판단한다.")
        print("다음: ex08 은 TT 하나만 보고 SAC 로 크기를 찾고, ex09 는 9개 corner 를 모두 보고 찾는다.")
    finally:
        env.close()
        sim.close()


if __name__ == "__main__":
    main()
