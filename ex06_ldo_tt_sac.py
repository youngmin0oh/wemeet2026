"""
ex06 — LDO(SKY130) 크기 찾기 ①: TT corner 만 보고 SAC 로 찾기
conda activate analog_sizing ; python ex06_ldo_tt_sac.py        (약 5–15분)

ex04 의 SAC 를 진짜 회로에 쓴다. 환경 = ngspice 로 LDO 를 시뮬레이션하는 Gymnasium 환경.
    상태(관측)   지금 스펙 9개(정규화) + 목표 스펙 9개(정규화) + 회로 파라미터 13개
    행동         파라미터 13개를 얼마나 바꿀지 ([-1, 1], 한 번에 범위의 최대 1/30)
    보상         목표에 가까울수록 커지고, 목표를 만족하면 +30 과 함께 에피소드 종료
    에피소드     처음 크기(ldo_sky130.yaml 의 init_params)에서 시작, 최대 30 step

이 예제는 TT(typical, 2.0V, 27도) corner 하나만 보고 학습한다 (step 당 시뮬레이션 1번, 빠름).
찾은 설계를 공정·온도가 다른 9개 PVT corner 에서 다시 시뮬레이션해 보면 무슨 일이 생기는지 본다
-> ex07 (9개 corner 를 다 보고 학습) 로 이어진다.

결과: results/ex06_ldo_tt_sac.json
"""
import argparse
import time

from ex_ldo_common import (pad, RESULTS, SPEC, Simulator, make_env, min_margin, params_text,
                           print_pvt_table, rollout, save_json, train_sac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000, help="최대 학습 step")
    ap.add_argument("--stop_after", type=int, default=30, help="성공한 설계를 이만큼 모으면 학습을 멈춤")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = make_env(tt_only=True)                         # ① TT corner 하나만 보는 환경
    params_id = env.unwrapped.params_id
    print("목표(예제 스펙):")
    for sid, op, val, unit, desc in SPEC:
        print(f"  {pad(desc, 34)} {op} {val:g} {unit}")

    t0 = time.time()
    model, found, sims = train_sac(env, a.steps, seed=a.seed, stop_after=a.stop_after, desc="SAC (TT)")
    learned = rollout(model, env)                        # ② 학습된 정책으로 한 번 더 설계
    env.close()
    print(f"\n학습 {time.time() - t0:.0f}초, 시뮬레이션 {sims}번 | 성공한 설계 {len(found)}개"
          + (f" (첫 성공: {found[0]['step']} step)" if found else "")
          + f" | 학습된 정책 단독 설계: {'성공 (' + str(learned['steps_in_episode']) + ' step 만에)' if learned else '실패'}")
    candidates = found + ([learned] if learned else [])
    if not candidates:
        raise SystemExit("목표를 만족한 설계가 없습니다. --steps 를 늘리거나 --seed 를 바꿔 보세요.")

    # ③ 후보 중 TT 에서 여유가 가장 큰 설계
    best = max(candidates, key=lambda d: min_margin(d["specs"]))
    print(f"고른 설계 (TT 여유가 가장 큰 것, {best['step']}):\n  {params_text(params_id, best['params'])}")

    # ④ 그 설계를 9개 PVT corner 에서 다시 시뮬레이션
    sim = Simulator()
    try:
        table = sim.run(best["params"])
        corners = sim.corners
    finally:
        sim.close()
    print_pvt_table("TT 만 보고 찾은 설계 -> TT corner 에서", table[:1], corners[:1])
    ok = print_pvt_table("같은 설계 -> 9개 PVT corner 에서", table, corners)
    if not ok:
        print("\nTT 에서는 목표를 넘었지만 다른 공정·온도 corner 에서 떨어진다."
              "\n-> ex07: 처음부터 9개 corner 를 다 보고 학습한다.")

    path = save_json(RESULTS / "ex06_ldo_tt_sac.json", {
        "example": "ex06: TT only + SAC",
        "spec": {s[0]: [s[1], s[2]] for s in SPEC},
        "params": dict(zip(params_id, best["params"])),
        "chosen_from": best["step"],
        "found": len(found),
        "simulations": int(sims),
        "pvt_corners": corners,
        "pvt_specs": table.tolist(),
        "pvt_pass": ok,
    })
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
