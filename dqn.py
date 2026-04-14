import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import random
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train():
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    target_update_freq = 10
    num_episodes = 2000

    env = gym.make("CartPole-v1")
    q_net = QNetwork(4, 2)
    target_net = QNetwork(4, 2)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_sa = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = q_net(next_states).argmax(dim=1)
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target = rewards + gamma * next_q * (1 - dones)

                loss = nn.functional.mse_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if terminated or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 50 == 0:
            print(f"Episode {episode} | Reward: {total_reward:.0f} | Epsilon: {epsilon:.2f}")

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())


if __name__ == "__main__":
    train()
