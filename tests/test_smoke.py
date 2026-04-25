"""
Smoke tests — one per algorithm. Each test:
  - runs the core training loop for a small number of steps
  - writes metrics to a tmp CSV via CsvLogger
  - asserts the CSV was written and loss values are finite (no NaN / Inf)

Modules are loaded via importlib so the algorithm dirs don't need to be
installed packages; only `common` needs `pip install -e .`.
"""

import csv
import importlib.util
import math
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

ROOT = Path(__file__).parent.parent


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


reinforce = _load("reinforce_train", "reinforce/train.py")
dqn = _load("dqn_train", "dqn/train.py")
ppo = _load("ppo_train", "ppo/train.py")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _finite(val: str) -> bool:
    try:
        return math.isfinite(float(val))
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# REINFORCE
# --------------------------------------------------------------------------- #


def test_reinforce_smoke(tmp_path):
    import gymnasium as gym
    from common import CsvLogger, resolve_device, seed_all

    N = 5
    seed_all(0)
    device = resolve_device("cpu")
    cfg = reinforce.TrainConfig()

    env = gym.make(cfg.env_id)
    env.reset(seed=0)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = reinforce.PolicyNetwork(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    log_path = tmp_path / "reinforce.csv"
    fieldnames = ["episode", "loss", "ep_return", "ep_length"]

    with CsvLogger(log_path, fieldnames) as logger:
        for ep in range(N):
            log_probs, rewards = reinforce.run_episode(policy, env, device)
            loss = reinforce.policy_gradient_update(
                log_probs, rewards, optimizer, cfg.gamma, device
            )
            logger.log(
                {
                    "episode": ep,
                    "loss": loss,
                    "ep_return": sum(rewards),
                    "ep_length": len(rewards),
                }
            )

    env.close()

    assert log_path.exists(), "CSV not written"
    rows = _read_csv(log_path)
    assert len(rows) == N, f"expected {N} rows, got {len(rows)}"
    assert all(_finite(r["loss"]) for r in rows), "NaN/Inf in REINFORCE loss"


# --------------------------------------------------------------------------- #
# DQN
# --------------------------------------------------------------------------- #


def test_dqn_smoke(tmp_path):
    import gymnasium as gym
    from common import CsvLogger, resolve_device, seed_all

    N_EPISODES = 15
    seed_all(0)
    py_rng = random.Random(0)
    device = resolve_device("cpu")

    # Reduce learning_starts so training kicks in within a few CartPole episodes
    cfg = dqn.TrainConfig(learning_starts=50, buffer_capacity=200, batch_size=32)

    env = gym.make(cfg.env_id)
    env.reset(seed=0)
    env.action_space.seed(0)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = dqn.QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target_net = dqn.QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    buf = dqn.ReplayBuffer(cfg.buffer_capacity, py_rng)

    log_path = tmp_path / "dqn.csv"
    fieldnames = ["episode", "loss_mean", "ep_return", "epsilon", "buffer_size"]

    epsilon = cfg.epsilon_start
    steps_since_target = 0

    with CsvLogger(log_path, fieldnames) as logger:
        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            ep_return = 0.0
            losses: list[float] = []

            while True:
                action = dqn.epsilon_greedy(q_net, obs, epsilon, env.action_space, device, py_rng)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buf.add(obs, action, float(reward), next_obs, float(terminated))
                obs = next_obs
                ep_return += float(reward)
                steps_since_target += 1

                if len(buf) >= max(cfg.batch_size, cfg.learning_starts):
                    batch = buf.sample(cfg.batch_size, device)
                    loss = dqn.td_loss(q_net, target_net, batch, cfg.gamma, double_dqn=True)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                if steps_since_target >= cfg.target_update_steps:
                    target_net.load_state_dict(q_net.state_dict())
                    steps_since_target = 0

                if done:
                    break

            epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)
            loss_mean = float(np.mean(losses)) if losses else float("nan")
            logger.log(
                {
                    "episode": ep,
                    "loss_mean": loss_mean,
                    "ep_return": ep_return,
                    "epsilon": epsilon,
                    "buffer_size": len(buf),
                }
            )

    env.close()

    assert log_path.exists(), "CSV not written"
    rows = _read_csv(log_path)
    assert len(rows) == N_EPISODES, f"expected {N_EPISODES} rows, got {len(rows)}"

    # Once training begins, every loss must be finite
    training_rows = [r for r in rows if r["loss_mean"] != "nan"]
    assert training_rows, "no training updates occurred — lower learning_starts or raise N_EPISODES"
    assert all(_finite(r["loss_mean"]) for r in training_rows), "NaN/Inf in DQN loss"


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #


def test_ppo_smoke(tmp_path):
    import gymnasium as gym
    from common import CsvLogger, resolve_device, seed_all

    N_ITERS = 3
    seed_all(0)
    device = resolve_device("cpu")

    # Use CartPole to skip the box2d dependency; tiny rollout for speed
    cfg = ppo.TrainConfig(env_id="CartPole-v1", n_envs=2, rollout_steps=32)

    def make_env(seed: int):
        def thunk():
            env = gym.make(cfg.env_id)
            env.reset(seed=seed)
            return env

        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(1000 + i) for i in range(cfg.n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    obs_rms = ppo.RunningMeanStd(shape=(obs_dim,))

    model = ppo.ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    obs, _ = envs.reset(seed=[1000 + i for i in range(cfg.n_envs)])

    log_path = tmp_path / "ppo.csv"
    fieldnames = ["iteration", "env_steps", "approx_kl", "updates", "rollout_ep_ret_mean"]

    with CsvLogger(log_path, fieldnames) as logger:
        for it in range(N_ITERS):
            (
                states,
                actions,
                log_probs,
                rewards,
                dones,
                terminateds,
                values,
                next_values,
                episode_rewards,
                obs,
            ) = ppo.collect_rollout_vec(
                model,
                envs,
                obs,
                obs_rms=obs_rms,
                num_steps=cfg.rollout_steps,
                device=device,
                cfg=cfg,
            )

            advantages, returns = ppo.compute_advantages_vec(
                rewards,
                values,
                next_values,
                dones,
                terminateds,
                gamma=cfg.gamma,
                lam=cfg.lam,
            )

            B = cfg.rollout_steps * cfg.n_envs
            mean_kl, num_updates = ppo.ppo_update(
                model,
                optimizer,
                states.reshape(B, obs_dim),
                actions.reshape(B),
                log_probs.reshape(B),
                values.reshape(B),
                advantages.reshape(B),
                returns.reshape(B),
                device=device,
                cfg=cfg,
                ent_coef=cfg.ent_coef_early,
            )

            roll_mean = float(np.mean(episode_rewards)) if len(episode_rewards) else float("nan")
            logger.log(
                {
                    "iteration": it,
                    "env_steps": (it + 1) * cfg.rollout_steps * cfg.n_envs,
                    "approx_kl": mean_kl,
                    "updates": num_updates,
                    "rollout_ep_ret_mean": roll_mean,
                }
            )

    envs.close()

    assert log_path.exists(), "CSV not written"
    rows = _read_csv(log_path)
    assert len(rows) == N_ITERS, f"expected {N_ITERS} rows, got {len(rows)}"
    assert all(_finite(r["approx_kl"]) for r in rows), "NaN/Inf in PPO approx_kl"
