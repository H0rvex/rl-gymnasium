import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


def normalize_obs(obs: np.ndarray, rms: RunningMeanStd, clip=10.0) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    obs = (obs - rms.mean.astype(np.float32)) / rms.std.astype(np.float32)
    return np.clip(obs, -clip, clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def collect_rollout(model, env, obs, obs_rms: RunningMeanStd, num_steps=2048):
    episode_rewards = []
    episode_reward = 0.0

    states, actions, log_probs = [], [], []
    rewards, episode_ends = [], []
    values, next_values = [], []

    for _ in range(num_steps):
        obs_rms.update(obs[None, :])
        obs_n = normalize_obs(obs, obs_rms)
        # Avoid potential aliasing with numpy buffers
        obs_tensor = torch.tensor(obs_n, dtype=torch.float32)

        with torch.no_grad():
            logits, value = model(obs_tensor)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        obs_rms.update(next_obs[None, :])
        next_obs_n = normalize_obs(next_obs, obs_rms)
        next_obs_tensor = torch.tensor(next_obs_n, dtype=torch.float32)

        with torch.no_grad():
            _, next_value = model(next_obs_tensor)

        # bootstrap only if NOT a true terminal
        if terminated:
            next_value = torch.zeros_like(value)

        states.append(obs_tensor)
        actions.append(action)
        log_probs.append(dist.log_prob(action))
        rewards.append(float(reward))
        episode_ends.append(bool(terminated or truncated))
        values.append(value)
        next_values.append(next_value)

        episode_reward += reward

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    return states, actions, log_probs, rewards, episode_ends, values, next_values, episode_rewards, obs


def compute_advantages(rewards, values, next_values, episode_ends, gamma=0.99, lam=0.95):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.stack(values).float()
    next_values = torch.stack(next_values).float()
    episode_ends = torch.tensor(episode_ends, dtype=torch.float32)

    advantages = torch.zeros_like(values)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        mask = 1.0 - episode_ends[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        gae = delta + gamma * lam * mask * gae

        advantages[t] = gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_log_probs,
    old_values,
    advantages,
    returns,
    ent_coef,
    epochs=10,
    batch_size=256,
    clip=0.2,
    vf_coef=0.5,
):
    
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs).detach()
    old_values = torch.stack(old_values).detach()
    advantages = advantages.detach()
    returns = returns.detach()

    n = states.size(0)

    kl_values = []
    num_updates = 0

    for _ in range(epochs):
        indices = torch.randperm(n)
        stop_early = False

        for start in range(0, n, batch_size):
            mb_idx = indices[start:start + batch_size]

            logits, values = model(states[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])

            log_ratio = new_log_probs - old_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            values_old = old_values[mb_idx]
            values_clipped = values_old + torch.clamp(values - values_old, -clip, clip)
            vf_loss1 = (values - returns[mb_idx]).pow(2)
            vf_loss2 = (values_clipped - returns[mb_idx]).pow(2)
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            entropy = dist.entropy().mean()

            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            kl_values.append(approx_kl)
            num_updates += 1

            if approx_kl > 0.015:
                stop_early = True
                break
        
        if stop_early:
            break
    
    return np.mean(kl_values), num_updates

def evaluate(model, obs_rms: RunningMeanStd, env_name="LunarLander-v3", episodes=20, deterministic=True):
    env = gym.make(env_name)
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=12345 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            obs_n = normalize_obs(obs, obs_rms)
            obs_tensor = torch.tensor(obs_n, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = model(obs_tensor)

            if deterministic:
                action = torch.argmax(logits).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
        rewards.append(ep_reward)
    
    env.close()
    return np.mean(rewards), np.std(rewards)

# setup
torch.manual_seed(0)
np.random.seed(0)

env = gym.wrappers.RecordEpisodeStatistics(gym.make("LunarLander-v3"))
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
obs_rms = RunningMeanStd(shape=(obs_dim,))

model = ActorCritic(obs_dim, action_dim)
# PPO-style Adam eps helps stability a bit
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

obs, _ = env.reset(seed=0)
best_eval = -float("inf")

for iteration in range(1000):
    t0 = time.time()
    (
        states,
        actions,
        log_probs,
        rewards,
        episode_ends,
        values,
        next_values,
        episode_rewards,
        obs,
    ) = collect_rollout(model, env, obs, obs_rms=obs_rms, num_steps=2048)

    advantages, returns = compute_advantages(
        rewards, values, next_values, episode_ends, gamma=0.99, lam=0.95
    )

    # Light schedules; keeps learning stable later in training
    frac = 1.0 - iteration / 1000
    ent_coef = max(0.001, 0.01 * frac)
    lr_now = 3e-4 * frac
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    mean_kl, num_updates = ppo_update(
        model,
        optimizer,
        states,
        actions,
        log_probs,
        values,
        advantages,
        returns,
        ent_coef=ent_coef,
        epochs=10,
        batch_size=256,
        clip=0.2,
    )

    roll_mean = float(np.mean(episode_rewards)) if len(episode_rewards) else float("nan")
    roll_n = len(episode_rewards)
    dt = time.time() - t0
    print(
        f"it {iteration:04d} | dt {dt:5.2f}s | lr {lr_now:.2e} | "
        f"KL {mean_kl:.5f} | updates {num_updates:3d} | "
        f"rollout_ep_ret {roll_mean:7.1f} ({roll_n} eps)"
    )

    if iteration % 10 == 0:
        mean_eval_det, std_eval_det = evaluate(model, obs_rms=obs_rms, episodes=20, deterministic=True)
        mean_eval_sto, std_eval_sto = evaluate(model, obs_rms=obs_rms, episodes=20, deterministic=False)

        print(f"Det eval: {mean_eval_det:.1f} ± {std_eval_det:.1f}")
        print(f"Sto eval: {mean_eval_sto:.1f} ± {std_eval_sto:.1f}")
        
        if mean_eval_det > best_eval:
            best_eval = mean_eval_det
            torch.save(model.state_dict(), "best_ppo_lunarlander.pt")
            print(f"New best model saved with eval reward {best_eval:.1f}")
            