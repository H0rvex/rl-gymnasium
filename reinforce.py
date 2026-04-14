import torch
import torch.nn as nn
import gymnasium as gym


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


def select_action(policy, obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    probs = policy(obs)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train():
    gamma = 0.99
    lr = 1e-3
    num_episodes = 1000

    env = gym.make("CartPole-v1")
    policy = PolicyNetwork(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            action, log_prob = select_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            if terminated or truncated:
                break

        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-lp * G for lp, G in zip(log_probs, returns))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode} | Total Reward: {sum(rewards):.0f}")


if __name__ == "__main__":
    train()
