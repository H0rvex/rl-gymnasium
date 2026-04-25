"""
Render a trained PPO agent to a GIF.

Usage (new-format checkpoint, bundled RMS):
    python ppo/render.py --checkpoint ppo/best_ppo_lunarlander_seed0.pt \
                         --out ppo/plots/rollout.gif

Usage (legacy checkpoint, no saved RMS):
    python ppo/render.py --checkpoint ppo/best_ppo_lunarlander_seed0.pt \
                         --warmup-steps 5000 \
                         --out ppo/plots/rollout.gif

Requires imageio:  pip install imageio
"""

import argparse
import pickle
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym

from train import ActorCritic, RunningMeanStd, TrainConfig, normalize_obs


def load_checkpoint(
    path: Path,
    obs_dim: int,
    action_dim: int,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, RunningMeanStd | None]:
    """Return (model, obs_rms | None). None means legacy format — caller must warm up RMS."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(obs_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
        obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
        obs_rms.count = float(ckpt["obs_rms_count"])
    else:
        model.load_state_dict(ckpt)
        obs_rms = None

    model.eval()
    return model, obs_rms


def warmup_rms(
    model: torch.nn.Module, cfg: TrainConfig, device: torch.device, n_steps: int
) -> RunningMeanStd:
    """
    Run the policy for n_steps to estimate the observation distribution.
    A converged policy visits the same obs distribution as late training,
    so ~5k steps gives a good approximation of the training-time RMS.
    """
    print(f"Warming up RMS over {n_steps} steps...")
    env = gym.make(cfg.env_id)
    obs_rms = RunningMeanStd(shape=(env.observation_space.shape[0],))
    obs, _ = env.reset(seed=99)
    steps = 0

    while steps < n_steps:
        obs_rms.update(obs[None])
        obs_n = normalize_obs(obs, obs_rms, clip=cfg.obs_clip)
        obs_t = torch.tensor(obs_n, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, _ = model(obs_t)
        action = torch.argmax(logits).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        steps += 1

    env.close()
    return obs_rms


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
    parser.add_argument("--out", default="ppo/plots/rollout.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Steps to run before rendering to estimate obs RMS. "
        "Use ~5000 for legacy checkpoints without saved RMS.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device(args.device)

    env_probe = gym.make(cfg.env_id)
    obs_dim = env_probe.observation_space.shape[0]
    action_dim = env_probe.action_space.n
    env_probe.close()

    model, obs_rms = load_checkpoint(Path(args.checkpoint), obs_dim, action_dim, cfg, device)

    if obs_rms is None:
        warmup = args.warmup_steps if args.warmup_steps > 0 else 5000
        print(f"Legacy checkpoint: no saved RMS. Running {warmup}-step warmup.")
        obs_rms = warmup_rms(model, cfg, device, n_steps=warmup)

    print(f"Rendering episode (seed={args.seed})...")
    frames, total_reward = render_episode(model, obs_rms, cfg, device, seed=args.seed)
    print(f"Episode return: {total_reward:.1f}  |  {len(frames)} frames")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
