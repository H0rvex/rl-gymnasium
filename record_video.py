import argparse
import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parent


def _load_ppo_render():
    ppo_dir = ROOT / "ppo"
    sys.path.insert(0, str(ppo_dir))
    try:
        spec = importlib.util.spec_from_file_location("ppo_render", ppo_dir / "render.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load ppo/render.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ppo_render"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(ppo_dir))
        except ValueError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained PPO LunarLander policy.")
    parser.add_argument("--algorithm", choices=["ppo"], default="ppo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="ppo/plots/rollout.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render = _load_ppo_render()
    cfg = render.TrainConfig()
    device = torch.device(args.device)

    import gymnasium as gym
    import imageio

    env_probe = gym.make(cfg.env_id)
    obs_dim = env_probe.observation_space.shape[0]
    action_dim = env_probe.action_space.n
    env_probe.close()

    model, obs_rms = render.load_checkpoint(Path(args.checkpoint), obs_dim, action_dim, cfg, device)
    if obs_rms is None:
        warmup = args.warmup_steps if args.warmup_steps > 0 else 5000
        obs_rms = render.warmup_rms(model, cfg, device, n_steps=warmup)

    frames, total_reward = render.render_episode(model, obs_rms, cfg, device, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps)
    print(f"Saved {out_path} | return={total_reward:.1f} | frames={len(frames)}")


if __name__ == "__main__":
    main()
