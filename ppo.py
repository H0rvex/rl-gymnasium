import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


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


def collect_rollout(model, env, obs, num_steps=2048):
    episode_rewards = []
    episode_reward = 0.0

    states, actions, log_probs = [], [], []
    rewards, episode_ends = [], []
    values, next_values = [], []

    for _ in range(num_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            logits, value = model(obs_tensor)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32)

        with torch.no_grad():
            _, next_value = model(next_obs_tensor)

        # bootstrap only if NOT a true terminal
        if terminated:
            next_value = torch.tensor(0.0)

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

    advantages = torch.zeros_like(values)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]

        if episode_ends[t]:
            gae = delta
        else:
            gae = delta + gamma * lam * gae

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
    advantages,
    returns,
    epochs=10,
    batch_size=256,
    clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
):
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs).detach()
    advantages = advantages.detach()
    returns = returns.detach()

    n = states.size(0)

    for _ in range(epochs):
        indices = torch.randperm(n)

        for start in range(0, n, batch_size):
            mb_idx = indices[start:start + batch_size]

            logits, values = model(states[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])

            ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.functional.mse_loss(values, returns[mb_idx])
            entropy = dist.entropy().mean()

            loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


# setup
torch.manual_seed(0)
np.random.seed(0)

env = gym.wrappers.RecordEpisodeStatistics(gym.make("LunarLander-v3"))
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(obs_dim, action_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

obs, _ = env.reset(seed=0)

for iteration in range(1000):
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
    ) = collect_rollout(model, env, obs, num_steps=2048)

    advantages, returns = compute_advantages(
        rewards, values, next_values, episode_ends, gamma=0.99, lam=0.95
    )

    ppo_update(
        model,
        optimizer,
        states,
        actions,
        log_probs,
        advantages,
        returns,
        epochs=10,
        batch_size=256,
        clip=0.2,
    )

    if iteration % 10 == 0:
        avg_reward = np.mean(episode_rewards) if episode_rewards else float("nan")
        print(f"Iteration {iteration} | Avg Episode Reward: {avg_reward:.1f}")