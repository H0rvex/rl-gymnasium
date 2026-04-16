# rl-gymnasium

From-scratch implementations of foundational RL algorithms in PyTorch + Gymnasium. Each project is self-contained, runnable, and instrumented with metrics + plots.

## About

A learning portfolio building toward reinforcement learning for **robotics / embodied AI**. Algorithms are implemented from scratch — no Stable-Baselines3, no RLLib — to develop deep understanding of the standard machinery (clipped surrogates, GAE, target networks, replay buffers) before transitioning to framework-based work in Isaac Lab and beyond.

## Projects

| Status | Algorithm | Environment | Folder |
|---|---|---|---|
| ✅ | REINFORCE | CartPole-v1 | [`reinforce/`](reinforce/) |
| ✅ | DQN | CartPole-v1 | [`dqn/`](dqn/) |
| ✅ | PPO (discrete) | LunarLander-v3 | [`ppo/`](ppo/) |
| 📋 | PPO (continuous) | HalfCheetah-v4 (MuJoCo) | _next_ |
| 📋 | SAC | HalfCheetah-v4 (MuJoCo) | _planned_ |
| 📋 | PPO @ scale | Isaac Lab humanoid locomotion | _planned_ |

Each project's README documents its design decisions, hyperparameters, multi-seed results, and reproduction commands.

## Setup

Tested with Python 3.11.

```bash
pip install -r requirements.txt
# For LunarLander Box2D physics:
pip install -r requirements-box2d.txt
```

## Run

From the repo root:

```bash
python reinforce/train.py
python dqn/train.py
python ppo/train.py --seed 0
```

The PPO trainer accepts `--seed` and `--iterations` for multi-seed runs.

## Plotting

Shared plotting utility supports single runs, multi-seed mean ± std bands, and rolling-window smoothing:

```bash
# Single run
python scripts/plot_csv.py --csv ppo/metrics_seed0.csv --ys eval_det_mean,eval_sto_mean

# Multi-seed (glob expands to all matching CSVs; mean +/- std band drawn automatically)
python scripts/plot_csv.py --csv "ppo/metrics_seed*.csv" \
    --ys rollout_ep_ret_mean --smooth 50 --out ppo/plots/rollout_return.png
```

## Notes

- Model checkpoints (`*.pt`) are gitignored. Per-run metrics CSVs (`metrics_seed*.csv`) are tracked so plots in each project's README are reproducible from committed data alone.
