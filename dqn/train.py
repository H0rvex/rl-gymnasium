import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from common import CsvLogger, resolve_device, seed_all


@dataclass
class TrainConfig:
    env_id: str = "CartPole-v1"

    hidden_dim: int = 128

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64

    buffer_capacity: int = 10_000
    learning_starts: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.999

    target_update_steps: int = 1_000

    eval_interval: int = 50
    eval_episodes: int = 20


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = rng

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(actions), dtype=torch.long, device=device),
            torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(dones), dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def td_loss(
    q_net: QNetwork,
    target_net: QNetwork,
    batch: Tuple[torch.Tensor, ...],
    gamma: float,
    double_dqn: bool,
) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    q_sa = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            # Online net selects the action, target net evaluates it — decorrelates
            # action selection from value estimation, reduces max-bias overestimation.
            next_actions = q_net(next_states).argmax(dim=1)
            next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = target_net(next_states).max(dim=1).values
        target = rewards + gamma * next_q * (1.0 - dones)

    return nn.functional.mse_loss(q_sa, target)


def epsilon_greedy(
    q_net: QNetwork,
    obs: np.ndarray,
    epsilon: float,
    action_space: gym.spaces.Discrete,
    device: torch.device,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return int(action_space.sample())
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return int(q_net(obs_t).argmax().item())


def evaluate(
    q_net: QNetwork,
    cfg: TrainConfig,
    device: torch.device,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    Deterministic eval: argmax over Q. Stochastic eval: softmax-sampled over Q
    (proxy for policy noise; DQN has no explicit stochastic policy).
    """
    env = gym.make(cfg.env_id)
    rewards: list[float] = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=12345 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = q_net(obs_t)
            if deterministic:
                action = int(torch.argmax(q).item())
            else:
                dist = torch.distributions.Categorical(logits=q)
                action = int(dist.sample().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

        rewards.append(ep_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()

    seed_all(args.seed)
    py_rng = random.Random(args.seed)

    device = resolve_device(args.device)
    print(f"Training on device: {device}  |  double_dqn={args.double_dqn}")

    env = gym.make(cfg.env_id)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    target_net = QNetwork(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_capacity, py_rng)

    log_path = Path(__file__).with_name(f"metrics_seed{args.seed}.csv")
    fieldnames = [
        "episode", "env_steps", "dt_sec", "loss_mean", "epsilon", "buffer_size",
        "ep_return", "ep_length", "train_updates",
        "eval_det_mean", "eval_det_std", "eval_sto_mean", "eval_sto_std", "best_eval_det",
    ]

    epsilon = cfg.epsilon_start
    total_env_steps = 0
    steps_since_target_update = 0
    best_eval = -float("inf")

    with CsvLogger(log_path, fieldnames) as logger:
        for episode in range(args.episodes):
            t0 = time.time()
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0
            losses: list[float] = []

            while True:
                action = epsilon_greedy(q_net, obs, epsilon, env.action_space, device, py_rng)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Store `terminated` (not `done`) so truncation at time-limit doesn't
                # zero the bootstrap — CartPole-v1 truncates at 500 steps.
                buffer.add(obs, action, float(reward), next_obs, float(terminated))
                obs = next_obs
                ep_return += float(reward)
                ep_length += 1
                total_env_steps += 1
                steps_since_target_update += 1

                if len(buffer) >= max(cfg.batch_size, cfg.learning_starts):
                    batch = buffer.sample(cfg.batch_size, device)
                    loss = td_loss(q_net, target_net, batch, cfg.gamma, args.double_dqn)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                if steps_since_target_update >= cfg.target_update_steps:
                    target_net.load_state_dict(q_net.state_dict())
                    steps_since_target_update = 0

                if done:
                    break

            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
            dt = time.time() - t0
            loss_mean = float(np.mean(losses)) if losses else float("nan")

            mean_eval_det = float("nan")
            std_eval_det = float("nan")
            mean_eval_sto = float("nan")
            std_eval_sto = float("nan")

            if episode % cfg.eval_interval == 0:
                mean_eval_det, std_eval_det = evaluate(q_net, cfg, device, deterministic=True)
                mean_eval_sto, std_eval_sto = evaluate(q_net, cfg, device, deterministic=False)

                print(
                    f"ep {episode:04d} | steps {total_env_steps:6d} | eps {epsilon:.3f} | "
                    f"return {ep_return:6.1f} | loss {loss_mean:7.4f} | "
                    f"det {mean_eval_det:6.1f}±{std_eval_det:4.1f} | "
                    f"sto {mean_eval_sto:6.1f}±{std_eval_sto:4.1f}"
                )

                if mean_eval_det > best_eval:
                    best_eval = mean_eval_det
                    tag = "double_dqn" if args.double_dqn else "dqn"
                    ckpt_path = Path(__file__).with_name(f"best_{tag}_cartpole_seed{args.seed}.pt")
                    torch.save({"model": q_net.state_dict()}, ckpt_path)

            logger.log({
                "episode": episode,
                "env_steps": total_env_steps,
                "dt_sec": dt,
                "loss_mean": loss_mean,
                "epsilon": epsilon,
                "buffer_size": len(buffer),
                "ep_return": ep_return,
                "ep_length": ep_length,
                "train_updates": len(losses),
                "eval_det_mean": mean_eval_det,
                "eval_det_std": std_eval_det,
                "eval_sto_mean": mean_eval_sto,
                "eval_sto_std": std_eval_sto,
                "best_eval_det": best_eval if best_eval > -float("inf") else float("nan"),
            })

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for torch / numpy / env reset")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device. 'auto' picks cuda if available, else cpu.")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Double DQN target (online-net action, target-net value). "
                             "Pass --no-double-dqn for vanilla DQN.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
