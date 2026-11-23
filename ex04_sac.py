"""
SAC (Soft Actor-Critic) — Off-policy, Continuous action  —  Pendulum-v1
pip install gymnasium torch

actor_critic_continuous.py (On-policy) 와 비교:
    rollout (쓰고 버림)          ->  replay_buffer (저장 → 샘플링 → 반복 학습)
    V(s) Critic                  ->  Q1(s,a), Q2(s,a) + target network
    loss = -log π · δ            ->  loss = α·log π(a|s) - min Q(s, a),  a ~ π (reparameterization)
    std 파라미터                 ->  entropy 온도 α 자동 조절
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm

# ---------------- Hyper-parameters ----------------
total_steps       = 20_000
batch_size        = 256
buffer_size       = 100_000
eval_interval     = 2_000
num_eval_episodes = 5
gamma             = 0.99
tau               = 0.005     # target network soft update 비율
lr                = 3e-4

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Replay Buffer
# ==================================================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.s  = np.zeros((size, obs_dim), dtype=np.float32)
        self.a  = np.zeros((size, act_dim), dtype=np.float32)
        self.r  = np.zeros(size, dtype=np.float32)
        self.s2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.d  = np.zeros(size, dtype=np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def append(self, s, a, r, s2, d):
        self.s[self.ptr], self.a[self.ptr], self.r[self.ptr] = s, a, r
        self.s2[self.ptr], self.d[self.ptr] = s2, d
        self.ptr = (self.ptr + 1) % self.size          # 가득 차면 오래된 것부터 덮어씀
        self.n = min(self.n + 1, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, size=batch_size)
        return [torch.as_tensor(x[idx], device=device) for x in (self.s, self.a, self.r, self.s2, self.d)]

    def __len__(self):
        return self.n


# ==================================================
# Networks
# ==================================================
def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class QNet(nn.Module):                                # Q(s, a)
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class Actor(nn.Module):                               # π(a|s) = tanh-squashed Gaussian
    def __init__(self, obs_dim, act_dim, act_high):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)          # mean, log_std
        # register_buffer: .to(device) 호출 시 파라미터와 함께 GPU로 이동
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

    def forward(self, s, evaluation=False):
        mean, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        dist = torch.distributions.Normal(mean, log_std.exp())

        u = mean if evaluation else dist.rsample()    # rsample: reparameterization (u = μ + σ·ε)
        action = self.act_high * torch.tanh(u)        # [-2, 2] 범위로 squash

        # tanh 변환에 따른 log 확률 보정:  log π(a) = log N(u) - Σ log |da/du|
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(self.act_high * (1 - torch.tanh(u).pow(2)) + 1e-6).sum(-1)
        return action, log_prob


# ==================================================
# Agent
# ==================================================
class SACAgent:
    def __init__(self, obs_dim, act_dim, act_high):
        self.actor = Actor(obs_dim, act_dim, act_high).to(device)
        self.q1, self.q2 = QNet(obs_dim, act_dim).to(device), QNet(obs_dim, act_dim).to(device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)

        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)   # α = exp(log_alpha)
        self.target_entropy = -float(act_dim)                 # 목표 엔트로피 = -|A|

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def select_action(self, state, evaluation=False):
        action, _ = self.actor(torch.as_tensor(state, dtype=torch.float32, device=device), evaluation)
        return action.cpu().numpy()                    # env는 numpy를 받으므로 CPU로 복사

    def update(self, batch):
        s, a, r, s_next, done = batch
        alpha = self.log_alpha.exp().detach()

        # ---------------- Critic : Soft TD update ----------------
        with torch.no_grad():
            a_next, logp_next = self.actor(s_next)                       # a' ~ π(·|s')  (현재 정책)
            q_next = torch.min(self.q1_target(s_next, a_next),
                               self.q2_target(s_next, a_next))           # 두 Q 중 작은 값 (과대추정 방지)
            td_target = r + gamma * (1 - done) * (q_next - alpha * logp_next)  # r + γ(Q' - α log π)

        q1_loss = (self.q1(s, a) - td_target).pow(2).mean()
        q2_loss = (self.q2(s, a) - td_target).pow(2).mean()
        critic_loss = q1_loss + q2_loss
        self.q_opt.zero_grad()
        critic_loss.backward()
        self.q_opt.step()

        # ---------------- Actor : maximize  Q(s, a) - α log π(a|s) ----------------
        a_new, logp = self.actor(s)                                      # rsample → Q를 거쳐 gradient
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        actor_loss = (alpha * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---------------- α : entropy가 target_entropy 근처로 유지되게 ----------------
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---------------- Target network soft update ----------------
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - tau).add_(tau * p)                       # θ' ← (1-τ)θ' + τθ

        return actor_loss.item(), critic_loss.item(), alpha.item()


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
            total_reward += reward
    return total_reward / num_eval_episodes


# ==================================================
# Off-policy training loop
# ==================================================
def main():
    env, eval_env = gym.make("Pendulum-v1"), gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = env.action_space.high

    torch.manual_seed(0); np.random.seed(0)
    tqdm.write(f"device: {device}")
    agent = SACAgent(obs_dim, act_dim, act_high)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
    state, _ = env.reset(seed=0)
    eval_env.reset(seed=1000)

    tqdm.write(f"step {0:>6} | eval return {evaluate(agent, eval_env):8.1f} | (학습 전)")

    for step in tqdm(range(total_steps), desc="SAC", unit="step", dynamic_ncols=True):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.append(state, action, reward, next_state, terminated)

        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

        if len(replay_buffer) >= batch_size:              # 매 step: 버퍼에서 샘플링해 학습
            batch = replay_buffer.sample(batch_size)
            actor_loss, critic_loss, alpha = agent.update(batch)

        if (step + 1) % eval_interval == 0:
            avg_return = evaluate(agent, eval_env)
            tqdm.write(f"step {step + 1:>6} | eval return {avg_return:8.1f} | alpha {alpha:.3f} | "
                       f"actor_loss {actor_loss:+.2f} | critic_loss {critic_loss:.3f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()