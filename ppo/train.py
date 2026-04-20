import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from common import CsvLogger, TbLogger, resolve_device, seed_all


@dataclass
class TrainConfig:
    # Environment
    env_id: str = "LunarLander-v3"
    n_envs: int = 8
    rollout_steps: int = 256        # per env; total batch = n_envs * rollout_steps
    
    # Network
    hidden_dim: int = 64

    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    epochs: int = 8
    batch_size: int = 256
    grad_clip: float = 0.5
    kl_threshold: float = 0.015

    # Optimizer
    lr: float = 3e-4
    lr_floor: float = 1e-4
    adam_eps: float = 1e-5

    # Entropy schedule: high → low at iter ent_switch, floor applied throughout
    ent_coef_early: float = 0.01
    ent_coef_late: float = 0.005
    ent_floor_early: float = 0.001
    ent_floor_late: float = 0.0005
    ent_switch: int = 300           # iteration at which late schedule kicks in
    
    # Evaluation
    eval_interval: int = 10
    eval_episodes: int = 20
    obs_clip: float = 10.0

# Type alias for collect_rollout_vec return
RolloutBatch = Tuple[
    torch.Tensor,   # states      [T, N, obs_dim]
    torch.Tensor,   # actions     [T, N]
    torch.Tensor,   # log_probs   [T, N]
    torch.Tensor,   # rewards     [T, N]
    torch.Tensor,   # dones       [T, N]
    torch.Tensor,   # terminateds [T, N]
    torch.Tensor,   # values      [T, N]
    torch.Tensor,   # next_values [T, N]
    np.ndarray,     # finished episode returns
    np.ndarray,     # next obs (for continuing rollout)
]


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
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
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def snapshot(self) -> "RunningMeanStd":
        snap = RunningMeanStd(self.mean.shape, eps=0.0)
        snap.mean = self.mean.copy()
        snap.var = self.var.copy()
        snap.count = float(self.count)
        return snap


def normalize_obs(obs: np.ndarray, rms: RunningMeanStd, clip: float = 10.0) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    obs = (obs - rms.mean.astype(np.float32)) / rms.std.astype(np.float32)
    return np.clip(obs, -clip, clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def collect_rollout_vec(
    model: nn.Module,
    envs: gym.vector.VectorEnv,
    obs: np.ndarray,
    obs_rms: RunningMeanStd,
    num_steps: int,
    device: torch.device,
    cfg: TrainConfig,
) -> RolloutBatch:
    """
    Collect a rollout from vectorized environments. Rollout buffers stay on CPU;
    only the model forward runs on `device`.
    """
    n_envs = obs.shape[0]
    obs_dim = obs.shape[1]

    states = torch.zeros((num_steps, n_envs, obs_dim), dtype=torch.float32)
    actions = torch.zeros((num_steps, n_envs), dtype=torch.int64)
    log_probs = torch.zeros((num_steps, n_envs), dtype=torch.float32)
    rewards = torch.zeros((num_steps, n_envs), dtype=torch.float32)
    dones = torch.zeros((num_steps, n_envs), dtype=torch.float32)
    terminateds = torch.zeros((num_steps, n_envs), dtype=torch.float32)
    values = torch.zeros((num_steps, n_envs), dtype=torch.float32)
    next_values = torch.zeros((num_steps, n_envs), dtype=torch.float32)

    ep_returns = np.zeros((n_envs,), dtype=np.float32)
    finished_ep_returns: list[float] = []

    for t in range(num_steps):
        obs_rms.update(obs)
        obs_n = normalize_obs(obs, obs_rms, clip=cfg.obs_clip)
        obs_tensor = torch.tensor(obs_n, dtype=torch.float32)

        with torch.no_grad():
            logits, value = model(obs_tensor.to(device))
        logits = logits.cpu()
        value = value.cpu()

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        lp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        # For bootstrap, we want V(s_{t+1}) based on the *real* next state.
        # Under SAME_STEP autoreset, gym.vector returns the reset observation in
        # `next_obs` and stores the true terminal observation in `infos["final_obs"]`
        # (renamed from `final_observation` in Gymnasium 1.0). `_final_obs` is the
        # boolean mask of which sub-envs produced a final obs on this step.
        next_obs_for_value = next_obs
        if isinstance(infos, dict) and "final_obs" in infos:
            final_mask = infos.get("_final_obs", done)
            final_obs = infos["final_obs"]
            if np.any(final_mask):
                next_obs_for_value = np.array(next_obs, copy=True)
                for i in np.where(final_mask)[0]:
                    next_obs_for_value[i] = np.asarray(final_obs[i], dtype=next_obs.dtype)

        obs_rms.update(next_obs_for_value)
        next_obs_n = normalize_obs(next_obs_for_value, obs_rms, clip=cfg.obs_clip)
        next_obs_tensor = torch.tensor(next_obs_n, dtype=torch.float32)
        with torch.no_grad():
            _, next_value = model(next_obs_tensor.to(device))
        next_value = next_value.cpu() * torch.tensor(1.0 - terminated.astype(np.float32))

        states[t] = obs_tensor
        actions[t] = action
        log_probs[t] = lp
        rewards[t] = torch.tensor(reward, dtype=torch.float32)
        dones[t] = torch.tensor(done.astype(np.float32))
        terminateds[t] = torch.tensor(terminated.astype(np.float32))
        values[t] = value
        next_values[t] = next_value

        ep_returns += reward.astype(np.float32)
        if np.any(done):
            for i in np.where(done)[0]:
                finished_ep_returns.append(float(ep_returns[i]))
                ep_returns[i] = 0.0

        # Do NOT manually reset here. Vector envs handle resetting based on their autoreset mode.
        obs = next_obs

    return (
        states, actions, log_probs, rewards, dones, terminateds,
        values, next_values,
        np.array(finished_ep_returns, dtype=np.float32), obs,
    )


def compute_advantages_vec(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GAE.

    - `dones`: True for terminated OR truncated (used to reset GAE across episode boundaries)
    - `terminateds`: True only for real terminals (used to disable bootstrap at terminal states)
    """
    rewards = rewards.float()
    values = values.float()
    next_values = next_values.float()
    dones = dones.float()
    terminateds = terminateds.float()

    T, N = rewards.shape
    advantages = torch.zeros((T, N), dtype=torch.float32)
    gae = torch.zeros((N,), dtype=torch.float32)

    for t in reversed(range(T)):
        gae_mask = 1.0 - dones[t]          # reset at episode end (terminated OR truncated)
        boot_mask = 1.0 - terminateds[t]   # bootstrap if NOT a real terminal
        delta = rewards[t] + gamma * next_values[t] * boot_mask - values[t]
        gae = delta + gamma * lam * gae_mask * gae
        advantages[t] = gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    device: torch.device,
    cfg: TrainConfig,
    ent_coef: float,
) -> Tuple[float, int]:
    states = states.detach().to(device)
    actions = actions.detach().to(device)
    old_log_probs = old_log_probs.detach().to(device)
    old_values = old_values.detach().to(device)
    advantages = advantages.detach().to(device)
    returns = returns.detach().to(device)

    n = states.size(0)
    kl_values: list[float] = []
    num_updates = 0

    for _ in range(cfg.epochs):
        indices = torch.randperm(n, device=device)
        stop_early = False

        for start in range(0, n, cfg.batch_size):
            mb_idx = indices[start:start + cfg.batch_size]

            logits, values = model(states[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])

            log_ratio = new_log_probs - old_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages[mb_idx]
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            values_old = old_values[mb_idx]
            values_clipped = values_old + torch.clamp(values - values_old, -cfg.clip, cfg.clip)
            vf_loss1 = (values - returns[mb_idx]).pow(2)
            vf_loss2 = (values_clipped - returns[mb_idx]).pow(2)
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            entropy = dist.entropy().mean()

            loss = actor_loss + cfg.vf_coef * critic_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            kl_values.append(approx_kl)
            num_updates += 1

            if approx_kl > cfg.kl_threshold:
                stop_early = True
                break

        if stop_early:
            break

    return float(np.mean(kl_values)), num_updates


def evaluate(
    model: nn.Module,
    obs_rms: RunningMeanStd,
    device: torch.device,
    cfg: TrainConfig,
    deterministic: bool = True,
) -> Tuple[float, float]:
    env = gym.make(cfg.env_id)
    rewards: list[float] = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=12345 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            obs_n = normalize_obs(obs, obs_rms, clip=cfg.obs_clip)
            obs_tensor = torch.tensor(obs_n, dtype=torch.float32).to(device)
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
    return float(np.mean(rewards)), float(np.std(rewards))


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()

    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"Training on device: {device}")

    def make_env(env_id: str, seed: int):
        def thunk():
            env = gym.make(env_id)
            env.reset(seed=seed)
            return env
        return thunk

    # SAME_STEP autoreset (the old default) matches collect_rollout_vec's
    # `infos["final_observation"]` bootstrap path. Gymnasium >=1.0 defaults
    # to NEXT_STEP, which silently injects a corrupt (terminal_obs, ignored
    # action, reward=0, next_obs=reset_obs, done=False) transition per episode.
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, seed=1000 + args.seed * 1000 + i) for i in range(cfg.n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    model = ActorCritic(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    # PPO-style Adam eps helps stability a bit
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    obs, _ = envs.reset(seed=[args.seed * 1000 + i for i in range(cfg.n_envs)])
    best_eval = -float("inf")

    log_path = Path(__file__).with_name(f"metrics_seed{args.seed}.csv")
    tb_dir = Path(__file__).parent / "runs" / f"seed{args.seed}"
    fieldnames = [
        "iteration", "env_steps", "dt_sec", "lr", "ent_coef",
        "approx_kl", "updates", "rollout_ep_ret_mean", "rollout_ep_ret_n",
        "eval_det_mean", "eval_det_std", "eval_sto_mean", "eval_sto_std", "best_eval_det",
    ]

    with CsvLogger(log_path, fieldnames) as logger, TbLogger(tb_dir) as tb:
        for iteration in range(args.iterations):
            t0 = time.time()
            (
                states, actions, log_probs, rewards, dones, terminateds,
                values, next_values, episode_rewards, obs,
            ) = collect_rollout_vec(model, envs, obs, obs_rms=obs_rms,
                                    num_steps=cfg.rollout_steps, device=device, cfg=cfg)

            advantages, returns = compute_advantages_vec(
                rewards, values, next_values, dones, terminateds,
                gamma=cfg.gamma, lam=cfg.lam,
            )

            # flatten [T, N, ...] -> [B, ...]
            B = cfg.rollout_steps * cfg.n_envs
            states_f    = states.reshape(B, obs_dim)
            actions_f   = actions.reshape(B)
            log_probs_f = log_probs.reshape(B)
            values_f    = values.reshape(B)
            advantages_f = advantages.reshape(B)
            returns_f   = returns.reshape(B)

            # Light schedules; keep exploration early, converge later
            frac = 1.0 - iteration / args.iterations
            if iteration < cfg.ent_switch:
                ent_coef = max(cfg.ent_floor_early, cfg.ent_coef_early * frac)
            else:
                ent_coef = max(cfg.ent_floor_late, cfg.ent_coef_late * frac)

            # Keep an LR floor so we can recover after performance dips
            lr_now = max(cfg.lr_floor, cfg.lr * frac)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            mean_kl, num_updates = ppo_update(
                model, optimizer,
                states_f, actions_f, log_probs_f, values_f, advantages_f, returns_f,
                device=device, cfg=cfg, ent_coef=ent_coef,
            )

            roll_mean = float(np.mean(episode_rewards)) if len(episode_rewards) else float("nan")
            roll_n = len(episode_rewards)
            dt = time.time() - t0
            print(
                f"it {iteration:04d} | dt {dt:5.2f}s | lr {lr_now:.2e} | "
                f"KL {mean_kl:.5f} | updates {num_updates:3d} | "
                f"rollout_ep_ret {roll_mean:7.1f} ({roll_n} eps)"
            )

            mean_eval_det = float("nan")
            std_eval_det  = float("nan")
            mean_eval_sto = float("nan")
            std_eval_sto  = float("nan")

            if iteration % cfg.eval_interval == 0:
                # Freeze normalization stats for consistent evaluation
                obs_rms_eval = obs_rms.snapshot()
                mean_eval_det, std_eval_det = evaluate(model, obs_rms_eval, device, cfg, deterministic=True)
                mean_eval_sto, std_eval_sto = evaluate(model, obs_rms_eval, device, cfg, deterministic=False)

                print(f"Det eval: {mean_eval_det:.1f} ± {std_eval_det:.1f}")
                print(f"Sto eval: {mean_eval_sto:.1f} ± {std_eval_sto:.1f}")

                if mean_eval_det > best_eval:
                    best_eval = mean_eval_det
                    ckpt_path = Path(__file__).with_name(f"best_ppo_lunarlander_seed{args.seed}.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "obs_rms_mean": obs_rms_eval.mean,
                        "obs_rms_var": obs_rms_eval.var,
                        "obs_rms_count": obs_rms_eval.count,
                    }, ckpt_path)
                    print(f"New best model saved with eval reward {best_eval:.1f} -> {ckpt_path.name}")

            env_steps = (iteration + 1) * cfg.rollout_steps * cfg.n_envs
            logger.log({
                "iteration": iteration,
                "env_steps": env_steps,
                "dt_sec": dt, "lr": lr_now, "ent_coef": ent_coef,
                "approx_kl": mean_kl, "updates": num_updates,
                "rollout_ep_ret_mean": roll_mean, "rollout_ep_ret_n": roll_n,
                "eval_det_mean": mean_eval_det, "eval_det_std": std_eval_det,
                "eval_sto_mean": mean_eval_sto, "eval_sto_std": std_eval_sto,
                "best_eval_det": best_eval,
            })
            tb.scalar("train/approx_kl", mean_kl, env_steps)
            tb.scalar("train/rollout_ep_ret_mean", roll_mean, env_steps)
            tb.scalar("train/lr", lr_now, env_steps)
            tb.scalar("train/ent_coef", ent_coef, env_steps)
            tb.scalar("eval/det_mean", mean_eval_det, env_steps)
            tb.scalar("eval/sto_mean", mean_eval_sto, env_steps)
            tb.scalar("eval/best_det", best_eval, env_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for torch / numpy / env reset")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Number of PPO iterations to train for")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device. 'auto' picks cuda if available, else cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
