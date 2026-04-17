"""
Render a trained PPO agent to a GIF.

Usage:
    python ppo/render.py --checkpoint ppo/best_ppo_lunarlander_seed0.pt \
                         --metrics   ppo/metrics_seed0.csv \
                         --out       ppo/rollout.gif

Requires imageio:  pip install imageio
"""

import argparse
import csv
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym

from train import ActorCritic, RunningMeanStd, TrainConfig, normalize_obs


def load_rms_from_csv(csv_path: Path, obs_dim: int) -> RunningMeanStd:
    """
    Reconstruct a RunningMeanStd snapshot from the last row of a metrics CSV.

    We don't persist the RMS directly, so this rebuilds it by replaying the
    obs stream — fast approximation: read the stored mean/var columns if present,
    otherwise return unit stats and warn.
    """
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = [h.strip() for h in lines[0].split(",")]

    rms = RunningMeanStd(shape=(obs_dim,))
    mean_cols = [f"obs_mean_{i}" for i in range(obs_dim)]
    var_cols  = [f"obs_var_{i}"  for i in range(obs_dim)]

    if mean_cols[0] in header and var_cols[0] in header:
        last = dict(zip(header, lines[-1].split(",")))
        rms.mean = np.array([float(last[c]) for c in mean_cols], dtype=np.float64)
        rms.var  = np.array([float(last[c]) for c in var_cols],  dtype=np.float64)
        rms.count = 1e6
    else:
        print(
            "Warning: metrics CSV does not contain obs_mean/obs_var columns. "
            "Using unit normalisation — rendered behaviour may differ from training."
        )
    return rms


def render_episode(
    model: torch.nn.Module,
    obs_rms: RunningMeanStd,
    cfg: TrainConfig,
    device: torch.device,
    seed: int = 0,
) -> tuple[list[np.ndarray], float]:
    env = gym.make(cfg.env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    done = False

    while not done:
        frames.append(env.render())
        obs_n = normalize_obs(obs, obs_rms, clip=cfg.obs_clip)
        obs_t = torch.tensor(obs_n, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

    env.close()
    return frames, total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--metrics",    required=True, help="Path to metrics_seedN.csv")
    parser.add_argument("--out",        default="ppo/rollout.gif", help="Output GIF path")
    parser.add_argument("--seed",       type=int, default=0, help="Env seed for the episode")
    parser.add_argument("--fps",        type=int, default=30, help="GIF frames per second")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="Device to run the model on (cpu recommended for single inference)")
    args = parser.parse_args()

    cfg    = TrainConfig()
    device = torch.device(args.device)

    env_probe = gym.make(cfg.env_id)
    obs_dim   = env_probe.observation_space.shape[0]
    action_dim = env_probe.action_space.n
    env_probe.close()

    model = ActorCritic(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    obs_rms = load_rms_from_csv(Path(args.metrics), obs_dim)

    print(f"Rendering episode (seed={args.seed})…")
    frames, total_reward = render_episode(model, obs_rms, cfg, device, seed=args.seed)
    print(f"Episode return: {total_reward:.1f}  |  {len(frames)} frames")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
