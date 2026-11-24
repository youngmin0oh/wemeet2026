"""
ex06 — ex04 의 SAC 를 Stable-Baselines3 로  —  Pendulum-v1
pip install gymnasium torch stable-baselines3 tqdm

ex04 에서 직접 만든 것                       Stable-Baselines3 에서는
    ReplayBuffer                       ->   SAC 안에 들어 있음 (buffer_size)
    Actor, QNet 2개, target network    ->   "MlpPolicy" (크기는 net_arch)
    SACAgent.update()                  ->   model.learn() 이 매 step 호출 (train_freq=1)
    log_alpha 자동 조절                ->   ent_coef="auto" (기본값, 목표 엔트로피 = -|A|)
    evaluate()                         ->   evaluate_policy()

하이퍼파라미터는 ex04 와 같다. 행동 범위([-2, 2])는 SB3 가 tanh 출력([-1, 1])을 알아서 늘려 맞춘다.
"""
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

# 작은 신경망의 CPU 연산은 스레드 1개가 가장 빠르다 (torch 기본값은 코어 수만큼 써서 CPU 에서 훨씬 느렸다)
torch.set_num_threads(1)

# ---------------- Hyper-parameters (ex04 와 같음) ----------------
total_steps       = 20_000
batch_size        = 256
buffer_size       = 100_000
eval_interval     = 2_000
num_eval_episodes = 5
gamma             = 0.99
tau               = 0.005     # target network soft update 비율
lr                = 3e-4


class EvalLog(BaseCallback):
    """진행바를 보여 주고, eval_interval step 마다 평가해서 ex04 와 같은 형식으로 출력한다."""

    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env

    def _on_training_start(self):
        self.bar = tqdm(total=total_steps, desc="SAC (SB3)", unit="step", dynamic_ncols=True)

    def _on_step(self):                       # env.step() 한 번마다 불린다
        self.bar.update(1)
        if self.num_timesteps % eval_interval == 0:
            avg_return, _ = evaluate_policy(self.model, self.eval_env, num_eval_episodes, deterministic=True)
            log = self.logger.name_to_value   # SB3 가 학습하면서 기록한 값 (train/...)
            tqdm.write(f"step {self.num_timesteps:>6} | eval return {avg_return:8.1f} | "
                       f"alpha {log['train/ent_coef']:.3f} | actor_loss {log['train/actor_loss']:+.2f} | "
                       f"critic_loss {log['train/critic_loss']:.3f}")
        return True                           # False 를 돌려주면 학습을 멈춘다

    def _on_training_end(self):
        self.bar.close()


def main():
    env = gym.make("Pendulum-v1")
    eval_env = Monitor(gym.make("Pendulum-v1"))       # Monitor: 에피소드 return 을 기록 (evaluate_policy 용)
    eval_env.reset(seed=1000)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau,
        learning_starts=batch_size,                    # ex04: 버퍼에 batch_size 개가 쌓이면 학습 시작
        policy_kwargs=dict(net_arch=[256, 256]),       # actor, Q1, Q2 모두 256-256 (ex04 의 mlp)
        seed=0, verbose=0,
    )
    tqdm.write(f"device: {model.device}")
    avg_return, _ = evaluate_policy(model, eval_env, num_eval_episodes, deterministic=True)
    tqdm.write(f"step {0:>6} | eval return {avg_return:8.1f} | (학습 전)")

    model.learn(total_timesteps=total_steps, callback=EvalLog(eval_env))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
