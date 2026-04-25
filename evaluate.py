import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).parent


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_state_dict(checkpoint: Path, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved RL checkpoint.")
    parser.add_argument("--algorithm", choices=["reinforce", "dqn", "ppo"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", default="", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)

    if args.algorithm == "reinforce":
        module = _load_module("reinforce_train", ROOT / "reinforce" / "train.py")
        cfg = module.TrainConfig(eval_episodes=args.episodes)
        env = gym.make(cfg.env_id)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        model = module.PolicyNetwork(obs_dim, action_dim, cfg.hidden_dim).to(device)
        model.load_state_dict(_load_state_dict(checkpoint, device))
        det_mean, det_std = module.evaluate(model, cfg, device, deterministic=True)
        sto_mean, sto_std = module.evaluate(model, cfg, device, deterministic=False)
        env_id = cfg.env_id
    elif args.algorithm == "dqn":
        module = _load_module("dqn_train", ROOT / "dqn" / "train.py")
        cfg = module.TrainConfig(eval_episodes=args.episodes)
        env = gym.make(cfg.env_id)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        model = module.QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(device)
        model.load_state_dict(_load_state_dict(checkpoint, device))
        det_mean, det_std = module.evaluate(model, cfg, device, deterministic=True)
        sto_mean, sto_std = module.evaluate(model, cfg, device, deterministic=False)
        env_id = cfg.env_id
    else:
        module = _load_module("ppo_train", ROOT / "ppo" / "train.py")
        cfg = module.TrainConfig(eval_episodes=args.episodes)
        env = gym.make(cfg.env_id)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        try:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        except pickle.UnpicklingError:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model = module.ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        model.load_state_dict(ckpt["model"])
        obs_rms = module.RunningMeanStd(shape=(obs_dim,))
        obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
        obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
        obs_rms.count = float(ckpt["obs_rms_count"])
        det_mean, det_std = module.evaluate(model, obs_rms, device, cfg, deterministic=True)
        sto_mean, sto_std = module.evaluate(model, obs_rms, device, cfg, deterministic=False)
        env_id = cfg.env_id

    summary = {
        "algorithm": args.algorithm,
        "env_id": env_id,
        "checkpoint": str(checkpoint),
        "eval_episodes": args.episodes,
        "deterministic": {"mean": det_mean, "std": det_std},
        "stochastic": {"mean": sto_mean, "std": sto_std},
    }
    out_path = Path(args.out) if args.out else checkpoint.parent / "eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
