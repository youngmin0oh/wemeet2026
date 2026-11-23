"""
간단한 Actor-Critic — Continuous action 버전 (On-policy, 1-step TD)  —  Pendulum-v1
pip install gymnasium torch

Discrete 버전(actor_critic.py)과 달라지는 것은 Actor 하나뿐:
    Categorical(logits)  ->  Normal(mean, std)   (Gaussian policy)
    action_space: Box(-2.0, 2.0, (1,))  = 토크

  Critic  V(s; w)      : TD 학습      δ = r + γ V(s') - V(s),   loss = δ²
  Actor   π(a|s; θ)    : Policy Gradient,  loss = -log π(a|s) · δ
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm

# ---------------- Hyper-parameters ----------------
total_steps       = 300_000
update_interval   = 32
eval_interval     = 10_000
num_eval_episodes = 5
gamma             = 0.9       # Pendulum은 짧은 horizon이 더 잘 학습됨
actor_lr          = 1e-3
critic_lr         = 3e-3
reward_scale      = 1 / 8     # Pendulum reward(-16~0)를 작게 → Critic 학습 안정화


# ==================================================
# Agent
# ==================================================
class GaussianActorCriticAgent:
    def __init__(self, obs_dim, act_dim, act_high):
        self.act_high = torch.as_tensor(act_high, dtype=torch.float32)  # 2.0
        # Actor  π(a|s; θ) = N(mean(s), std²)
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))   # 학습되는 탐색 폭 (state와 무관)
        # Critic V(s; w)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor_opt = torch.optim.Adam(
            list(self.actor_mean.parameters()) + [self.log_std], lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def dist(self, s):
        mean = self.act_high * torch.tanh(self.actor_mean(s))   # 평균을 [-2, 2] 안으로
        return torch.distributions.Normal(mean, self.log_std.exp())

    @torch.no_grad()
    def select_action(self, state, evaluation=False):
        d = self.dist(torch.as_tensor(state, dtype=torch.float32))
        action = d.mean if evaluation else d.sample()     # 평가: 평균 / 학습: 샘플링
        return action.numpy()                             # shape (act_dim,)  (범위 밖은 env가 clip)

    def update(self, rollout):
        s, a, r, s_next, done = map(np.array, zip(*rollout))
        s      = torch.as_tensor(s, dtype=torch.float32)          # (B, obs_dim)
        a      = torch.as_tensor(a, dtype=torch.float32)          # (B, act_dim)  ← 실수 벡터
        r      = torch.as_tensor(r, dtype=torch.float32)          # (B,)
        s_next = torch.as_tensor(s_next, dtype=torch.float32)     # (B, obs_dim)
        done   = torch.as_tensor(done, dtype=torch.float32)       # (B,)

        # ---------------- Critic : TD update ----------------
        with torch.no_grad():
            td_target = r + gamma * (1.0 - done) * self.critic(s_next).squeeze(-1)  # r + γV(s')
        value = self.critic(s).squeeze(-1)                                         # V(s)
        td_error = td_target - value                                               # δ

        critic_loss = td_error.pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- Actor : Policy Gradient update ----------------
        advantage = td_error.detach()
        log_prob = self.dist(s).log_prob(a).sum(-1)      # log π(a|s) = Σ_dim log N(a_i | μ_i, σ_i)

        actor_loss = -(log_prob * advantage).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item(), critic_loss.item()


# ==================================================
# Periodic Evaluation (shared)
# ==================================================
def evaluate(agent, eval_env):
    total_reward = 0.0
    for episode in range(num_eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluation=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward                        # 평가는 원래 reward 그대로
    return total_reward / num_eval_episodes


# ==================================================
# On-policy training loop
# ==================================================
def main():
    env, eval_env = gym.make("Pendulum-v1"), gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]   # 3 : [cos θ, sin θ, θ_dot]
    act_dim = env.action_space.shape[0]        # 1 : torque
    act_high = env.action_space.high           # [2.0]

    agent = GaussianActorCriticAgent(obs_dim, act_dim, act_high)
    rollout = []
    state, _ = env.reset(seed=0)
    eval_env.reset(seed=1000)

    tqdm.write(f"step {0:>7} | eval return {evaluate(agent, eval_env):8.1f} | (학습 전)")

    for step in tqdm(range(total_steps), desc="AC-continuous", unit="step", dynamic_ncols=True):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Pendulum은 terminated가 항상 False (200 step에서 truncated만 발생)
        rollout.append((state, action, reward * reward_scale, next_state, terminated))

        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

        if (step + 1) % update_interval == 0:
            actor_loss, critic_loss = agent.update(rollout)
            rollout.clear()

        if (step + 1) % eval_interval == 0:
            avg_return = evaluate(agent, eval_env)
            tqdm.write(f"step {step + 1:>7} | eval return {avg_return:8.1f} | "
                       f"std {agent.log_std.exp().item():.2f} | "
                       f"actor_loss {actor_loss:+.3f} | critic_loss {critic_loss:.4f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()