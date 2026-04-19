import argparse
import time
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
    lr: float = 1e-3

    eval_interval: int = 25
    eval_episodes: int = 20


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(
    policy: PolicyNetwork, obs: np.ndarray, device: torch.device
) -> Tuple[int, torch.Tensor]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    logits = policy(obs_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action)


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def run_episode(
    policy: PolicyNetwork, env: gym.Env, device: torch.device
) -> Tuple[list[torch.Tensor], list[float]]:
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    while True:
        action, log_prob = select_action(policy, obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return log_probs, rewards


def policy_gradient_update(
    log_probs: list[torch.Tensor],
    rewards: list[float],
    optimizer: torch.optim.Optimizer,
    gamma: float,
    device: torch.device,
) -> float:
    returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs_t = torch.stack(log_probs)
    loss = -(log_probs_t * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def evaluate(
    policy: PolicyNetwork,
    cfg: TrainConfig,
    device: torch.device,
    deterministic: bool = True,
) -> Tuple[float, float]:
    env = gym.make(cfg.env_id)
    rewards: list[float] = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=12345 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

        rewards.append(ep_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()

    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"Training on device: {device}")

    env = gym.make(cfg.env_id)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    log_path = Path(__file__).with_name(f"metrics_seed{args.seed}.csv")
    fieldnames = [
        "episode", "env_steps", "dt_sec", "loss", "ep_return", "ep_length",
        "eval_det_mean", "eval_det_std", "eval_sto_mean", "eval_sto_std", "best_eval_det",
    ]

    total_env_steps = 0
    best_eval = -float("inf")

    with CsvLogger(log_path, fieldnames) as logger:
        for episode in range(args.episodes):
            t0 = time.time()
            log_probs, rewards = run_episode(policy, env, device)
            loss = policy_gradient_update(log_probs, rewards, optimizer, cfg.gamma, device)

            ep_return = sum(rewards)
            ep_length = len(rewards)
            total_env_steps += ep_length
            dt = time.time() - t0

            mean_eval_det = float("nan")
            std_eval_det = float("nan")
            mean_eval_sto = float("nan")
            std_eval_sto = float("nan")

            if episode % cfg.eval_interval == 0:
                mean_eval_det, std_eval_det = evaluate(policy, cfg, device, deterministic=True)
                mean_eval_sto, std_eval_sto = evaluate(policy, cfg, device, deterministic=False)

                print(
                    f"ep {episode:04d} | return {ep_return:6.1f} | loss {loss:8.2f} | "
                    f"det {mean_eval_det:6.1f}±{std_eval_det:4.1f} | "
                    f"sto {mean_eval_sto:6.1f}±{std_eval_sto:4.1f}"
                )

                if mean_eval_det > best_eval:
                    best_eval = mean_eval_det
                    ckpt_path = Path(__file__).with_name(f"best_reinforce_cartpole_seed{args.seed}.pt")
                    torch.save({"model": policy.state_dict()}, ckpt_path)

            logger.log({
                "episode": episode,
                "env_steps": total_env_steps,
                "dt_sec": dt,
                "loss": loss,
                "ep_return": ep_return,
                "ep_length": ep_length,
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
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device. 'auto' picks cuda if available, else cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
