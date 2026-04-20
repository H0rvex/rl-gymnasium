# rl-gymnasium

From-scratch implementations of foundational RL algorithms in PyTorch + Gymnasium. Each project is self-contained, runnable, and instrumented with metrics and plots.

## About

A learning portfolio building toward reinforcement learning for **robotics / embodied AI**. Algorithms are implemented from scratch — no Stable-Baselines3, no RLLib — to develop deep understanding of the standard machinery (clipped surrogates, GAE, target networks, replay buffers) before transitioning to framework-based work in Isaac Lab and beyond.

## Projects

| Algorithm | Environment | Results | Folder |
|---|---|---|---|
| REINFORCE | CartPole-v1 | Solves in ≤450 eps across 3 seeds | [`reinforce/`](reinforce/) |
| Double DQN | CartPole-v1 | All 3 seeds hit 500; post-solve regression documented | [`dqn/`](dqn/) |
| PPO (discrete) | LunarLander-v3 | 257 ± 14 det. eval across 3 seeds | [`ppo/`](ppo/) |

Each project's README documents design decisions, hyperparameters, multi-seed results, training curves, and reproduction commands.

## Setup

```bash
git clone <repo>
cd rl-gymnasium

# core deps (REINFORCE + DQN)
pip install -e .

# add Box2D physics for LunarLander (PPO)
pip install -e ".[box2d]"
```

Requires Python ≥3.10. Pinned dependency floors are in `pyproject.toml`.

## Run

```bash
# REINFORCE — CartPole-v1
python reinforce/train.py --seed 0

# Double DQN — CartPole-v1
python dqn/train.py --seed 0

# PPO — LunarLander-v3
python ppo/train.py --seed 0
```

Pass `--help` to any trainer for the full flag list (`--episodes`, `--iterations`, `--device`, `--double-dqn`/`--no-double-dqn`).

## Plotting

Shared plotting utility in `scripts/plot_csv.py` supports single runs, multi-seed mean ± std bands, and rolling-window smoothing:

```bash
# single run
python scripts/plot_csv.py --csv reinforce/metrics_seed0.csv \
    --x episode --ys eval_det_mean,eval_sto_mean

# multi-seed mean ± std band
python scripts/plot_csv.py --csv "ppo/metrics_seed*.csv" \
    --ys rollout_ep_ret_mean --smooth 50 --out ppo/plots/rollout_return.png
```

## Notes

- Checkpoints (`*.pt`) are gitignored. Per-run metrics CSVs (`metrics_seed*.csv`) are tracked so plots in each README are reproducible from committed data alone.
- `common/` is a minimal shared package (device selection, seeding, CSV logger). Networks, update rules, and eval loops stay in each algorithm's `train.py`.
