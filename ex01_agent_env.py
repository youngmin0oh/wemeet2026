"""
Env <-> Random Agent 100 step 상호작용 (Gymnasium 기본 환경 사용)
pip install gymnasium
"""
import numpy as np
import gymnasium as gym

# ------------------------------------------------------------------
# reset / step 인터페이스 (Gymnasium)
#
#   obs, info = env.reset(seed=None, options=None)
#       obs  : 초기 관측
#       info : dict (부가 정보)
#
#   obs, reward, terminated, truncated, info = env.step(action)
#       action     : action_space 안의 값
#       obs        : 다음 관측
#       reward     : float, 이번 step의 보상
#       terminated : bool, 목표 달성/실패로 에피소드가 끝남 (MDP의 terminal state)
#       truncated  : bool, 시간 제한 등 외부 이유로 끊김
#       info       : dict (부가 정보)
# ------------------------------------------------------------------

class GridMoveEnv(gym.Env):
    """MultiDiscrete 예시용 커스텀 환경
    5x5 격자, (0,0)에서 출발해 (4,4)에 도착하면 종료
    action = [ax, ay], 각 원소 ∈ {0,1,2} -> 이동량 {-1, 0, +1}
    """

    def __init__(self, size=5, max_steps=50):
        self.size, self.max_steps = size, max_steps
        self.observation_space = gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int64)  # [x, y]
        self.action_space = gym.spaces.MultiDiscrete([3, 3])                               # [ax, ay]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0])
        self.t = 0
        obs, info = self.pos.copy(), {}
        return obs, info

    def step(self, action):
        self.pos = np.clip(self.pos + (action - 1), 0, self.size - 1)
        self.t += 1
        obs = self.pos.copy()
        terminated = bool((self.pos == self.size - 1).all())   # 목표 도착
        truncated = self.t >= self.max_steps                   # 시간 초과
        reward = 1.0 if terminated else -0.1
        info = {}
        return obs, reward, terminated, truncated, info


ENVS = {
    # Continuous   : Box(-2.0, 2.0, (1,))   -> 토크 값
    "continuous":    lambda: gym.make("Pendulum-v1"),
    # Discrete     : Discrete(2)            -> 0: 왼쪽, 1: 오른쪽
    "discrete":      lambda: gym.make("CartPole-v1"),
    # MultiDiscrete: MultiDiscrete([3 3])   -> x, y 이동을 동시에 선택 (커스텀 환경)
    "multidiscrete": lambda: GridMoveEnv(),
}


def run(name, make_env, n_steps=10, seed=0):
    env = make_env()
    print(f"\n===== {name} =====")
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)

    obs, info = env.reset(seed=seed)                      # reset
    for t in range(n_steps):
        action = env.action_space.sample()                # Random Policy
        next_obs, reward, terminated, truncated, info = env.step(action)  # step

        print(f"[{t:3d}] obs={np.round(obs, 2)}  action={np.round(action, 2)}  "
              f"reward={np.round(reward, 2)}  terminated={terminated}  truncated={truncated}")

        obs = next_obs
        if terminated or truncated:                       # 에피소드 끝 -> 직접 reset
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    for name, make_env in ENVS.items():
        run(name, make_env)