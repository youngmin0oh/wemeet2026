"""
ex07 — ex05 의 PPO 를 Stable-Baselines3 로  —  Pendulum-v1
pip install gymnasium torch stable-baselines3 tqdm

ex05 에서 직접 만든 것                        Stable-Baselines3 에서는
    rollout 리스트 + compute_gae()      ->   RolloutBuffer (n_steps, gamma, gae_lambda)
    clipped surrogate, K epoch 재사용   ->   model.learn() (n_epochs, batch_size, clip_range)
    actor_mean, log_std, critic         ->   "MlpPolicy" (net_arch, Tanh, log_std_init=0)
    advantage 정규화, grad clip         ->   normalize_advantage=True (기본값), max_grad_norm
    reward_scale = 1/8                  ->   TransformReward wrapper (학습 env 에만)
    evaluate()                          ->   evaluate_policy()

ex05 와 다른 점
    - actor 와 critic 이 optimizer 하나(learning rate 하나)를 같이 쓴다 (ex05: actor 3e-4, critic 1e-3).
    - 행동 평균을 tanh 로 묶지 않는다. 범위([-2, 2]) 밖의 행동은 env 에 넘길 때 잘라낸다.
"""
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

# 작은 신경망은 CPU 스레드 1개가 가장 빠르다. torch 기본값(코어 수만큼)은 이 예제에서 5–9배 느렸다.
torch.set_num_threads(1)

# ---------------- Hyper-parameters (ex05 와 같음) ----------------
total_steps       = 300_000
rollout_len       = 2048      # 몇 step 모아서 한 번 업데이트할지
num_epochs        = 10        # 같은 rollout을 몇 번 반복 학습할지
minibatch_size    = 64
eval_interval     = 10_000
num_eval_episodes = 5
gamma             = 0.9
gae_lambda        = 0.95
clip_eps          = 0.2
entropy_coef      = 0.0
max_grad_norm     = 0.5
lr                = 3e-4      # actor 와 critic 이 같이 씀 (ex05 의 actor lr)
reward_scale      = 1 / 8


class EvalLog(BaseCallback):
    """진행바를 보여 주고, eval_interval step 마다 평가해서 ex05 와 같은 형식으로 출력한다."""

    def __init__(self, eval_env):
        super().__init__()
        self.eval_env = eval_env

    def _on_training_start(self):
        self.bar = tqdm(total=total_steps, desc="PPO (SB3)", unit="step", dynamic_ncols=True)

    def _on_step(self):
        self.bar.update(1)
        if self.num_timesteps % eval_interval == 0:
            avg_return, _ = evaluate_policy(self.model, self.eval_env, num_eval_episodes, deterministic=True)
            log = self.logger.name_to_value   # 직전 업데이트에서 SB3 가 기록한 값
            tqdm.write(f"step {self.num_timesteps:>7} | eval return {avg_return:8.1f} | "
                       f"std {self.model.policy.log_std.exp().item():.2f} | "
                       f"clip_frac {log['train/clip_fraction']:.3f} | "
                       f"actor_loss {log['train/policy_gradient_loss']:+.3f} | "
                       f"critic_loss {log['train/value_loss']:.4f}")
        return True

    def _on_training_end(self):
        self.bar.close()


def main():
    # 학습 env 는 reward 를 1/8 로 (Critic 안정화, ex05 와 같음). 평가는 원래 reward 그대로.
    env = gym.wrappers.TransformReward(gym.make("Pendulum-v1"), lambda r: r * reward_scale)
    eval_env = Monitor(gym.make("Pendulum-v1"))
    eval_env.reset(seed=1000)

    model = PPO(
        "MlpPolicy", env,
        n_steps=rollout_len, n_epochs=num_epochs, batch_size=minibatch_size,
        gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_eps, ent_coef=entropy_coef,
        max_grad_norm=max_grad_norm, learning_rate=lr,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh),
        device="cpu",                 # 작은 MLP 의 PPO 는 CPU 가 더 빠르다 (SB3 도 CPU 를 권장)
        seed=0, verbose=0,
    )
    avg_return, _ = evaluate_policy(model, eval_env, num_eval_episodes, deterministic=True)
    tqdm.write(f"step {0:>7} | eval return {avg_return:8.1f} | (학습 전)")

    model.learn(total_timesteps=total_steps, callback=EvalLog(eval_env))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
